import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward, DWTInverse
from advertorch.bpda import BPDAWrapper
from torch.nn.parameter import Parameter
from skimage.restoration import denoise_nl_means, estimate_sigma
from pykalman import KalmanFilter
from scipy.ndimage import gaussian_filter
import numpy as np


# import PyEMD

class Sabre(nn.Module):
    def __init__(self, eps: float = 8. / 255, wave='coif1', use_rand: bool = True, n_variants: int = 10):
        super().__init__()

        # Initialization of parameters and setting up the base configuration
        self.eps = eps  # Perturbation magnitude
        self.use_rand = use_rand  # Whether to use randomness
        self.n_variants = n_variants  # Number of variants for randomness
        self.wave = wave  # Wavelet type
        self.mode = 'symmetric'  # Padding mode for DWT
        self.max_levels = 1  # Default max level for wavelet transforms

        self.annealing_coefficient = 1e-12
        self.error_coefficient = 1e-3

        # Placeholders for forward and inverse wavelet transform functions
        self.fwt = None
        self.iwt = None

        self.lambda_r = Parameter(torch.ones(1))
        self.coef = Parameter(torch.ones(1))

    def forward(self, x):
        """Defines the forward pass of the module."""
        return self.transform(x)

    def build_transforms(self, x):
        """Initializes wavelet transforms."""
        if self.fwt is not None:
            return

        _, _, h, w = x.shape
        max_levels = int(math.ceil(math.sqrt(math.log(1 + h * w))))
        self.max_levels = max_levels

        self.annealing_coefficient = torch.tensor(self.annealing_coefficient, dtype=torch.float32).to(x.device)

        # Initialize DWT and IDWT with the calculated max levels and specified configurations
        self.fwt = DWTForward(J=max_levels, wave=self.wave, mode=self.mode).to(x.device)
        self.iwt = DWTInverse(wave=self.wave, mode=self.mode).to(x.device)

    def reshape_inputs(self, x):
        """Reshapes inputs to have dimensions as squares if necessary."""
        b, c, h, w = x.shape
        target_dim = int(math.ceil(math.sqrt(h * w)))
        if h * w < target_dim ** 2:
            padding_size = target_dim ** 2 - h * w
            x_padded = torch.cat([x.view(b, c, -1), torch.zeros(b, c, padding_size, device=x.device)], dim=2)
            x = x_padded.view(b, c, target_dim, target_dim)
        return x

    # def transform(self, x):
    #     original_shape = x.shape[-2:]
    #     x = self.reshape_inputs(x)
    #
    #     x = self.apply_random_noise(x)
    #     #y = self.apply_wavelet_processing(x * 255) / 255 ##消融
    #     #y = self.apply_nlm_denoising(y)  # 确保这个函数的输出在计算图中
    #     y = self.restore_shape(x, original_shape)
    #     return y

    def transform(self, x):
        original_shape = x.shape[-2:]
        x = self.reshape_inputs(x)

        x = self.apply_random_noise(x)
        y = self.apply_wavelet_processing(x * 255) / 255
        #y = self.apply_nlm_denoising(y)  # 确保这个函数的输出在计算图中
        y = self.restore_shape(y, original_shape)
        return y
    def apply_random_noise(self, x):
        """Applies random noise to the input tensor."""
        if (not self.use_rand) or (self.n_variants <= 1):
            return x

        b, c, m, n = x.shape
        xs = x.clone().unsqueeze(2).tile(1, 1, self.n_variants, 1, 1)
        noise = (2 * torch.rand_like(xs) - 1) * self.eps  # Generate noise in the range [-eps, eps]
        noise_coefficients = torch.arange(self.n_variants, device=x.device) / (self.n_variants - 1)
        noise_coefficients = noise_coefficients.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).tile(b, c, 1, m, n)
        noise = noise * noise_coefficients
        xs_noisy = xs + noise.to(xs.device)
        xs_noisy = torch.clamp(xs_noisy, 0, 1)

        stds = xs_noisy.reshape(b, c, self.n_variants, -1).std(dim=3)
        best_variant = torch.argmin(stds, dim=2)
        batch_indices = torch.arange(b)[:, None].expand(-1, c).to(best_variant.device)
        channel_indices = torch.arange(c)[None, :].expand(b, -1).to(best_variant.device)
        x = xs_noisy[batch_indices, channel_indices, best_variant].reshape(b, c, m, n)

        return x

    def estimate_threshold(self, energies):
        """ Heuristic for estimating the denoising threshold based on spectral energies."""
        b, c, num_bands = energies.shape

        # Calculate mean and std for each band
        mean_energies = energies.mean(dim=2, keepdim=True)
        std_energies = energies.std(dim=2, keepdim=True)

        # Calculate the threshold for each band separately
        thresholds = (1 / self.max_levels) * torch.exp(
            -self.lambda_r * mean_energies / (std_energies + self.error_coefficient))
        thresholds = torch.nan_to_num(thresholds, nan=1., posinf=1., neginf=1.)

        # Ensure lambda_r is valid
        if torch.isnan(self.lambda_r):
            self.lambda_r.data = torch.ones(self.max_levels, device=self.lambda_r.device)

        # Reshape to match the dimensions required for filtering
        thresholds = thresholds.squeeze(2)
        return thresholds


    def apply_wavelet_processing(self, x):
        """Applies spectral processing and reconstructs the signal."""
        b, c, m, n = x.shape

        self.build_transforms(x)
        self.fwt.to(x.device)

        x_approx, x_details = self.fwt(x)
        bands = [x_approx.unsqueeze(2).to(x.device)] + x_details

        energies = torch.cat([(band ** 2).reshape(b, c, -1).sum(dim=2, keepdim=True) for band in bands], dim=2)
        thresholds = self.estimate_threshold(energies)
        thresholds = thresholds.unsqueeze(2).unsqueeze(3).unsqueeze(4).tile(1, 1, 3, 1, 1)

        y_details = []
        for i, coeff in enumerate(x_details):
            band_size = coeff.shape[-1] * coeff.shape[-2]
            band_weight = torch.sqrt(2 * torch.log(torch.tensor(band_size, dtype=torch.float32)))

            _coeff = coeff.abs().view(b, c, 3, -1)
            _coeff = torch.median(_coeff, dim=-1, keepdim=True).values.view(b, c, 3, 1, 1) / 0.6745
            cutoff = (self.coef * thresholds + band_weight * _coeff) / 2

            band_coeff = coeff.clone()
            coeff = torch.sign(coeff) * torch.maximum(
                coeff.abs() - cutoff,
                coeff.abs() * self.annealing_coefficient
            )
            coeff[coeff.abs() > self.error_coefficient] = band_coeff[coeff.abs() > self.error_coefficient]
            y_details.append(self.apply_additional_filters(coeff.float()))
            # y_details.append(coeff)

        outputs = self.iwt((x_approx, y_details))

        return outputs

    def apply_additional_filters(self, coeff):
        """Applies Gaussian and EMD filtering to the wavelet coefficients."""
        b, c, d, h, w = coeff.shape

        # Apply Gaussian filter
        coeff_gaussian = gaussian_filter(coeff.cpu().detach().numpy(), sigma=1)
        coeff_gaussian = torch.tensor(coeff_gaussian, device=coeff.device, dtype=coeff.dtype)

        # # Apply EMD
        # emd = PyEMD.EMD()
        # coeff_emd_list = []
        # for i in range(b):
        #     for j in range(c):
        #         for k in range(d):
        #             # Flatten each spatial component to a 1D array before applying EMD
        #             coeff_flat = coeff[i, j, k].cpu().detach().numpy().flatten()
        #             imfs = emd(coeff_flat)
        #             coeff_emd = np.sum(imfs, axis=0).reshape(h, w)
        #             coeff_emd_list.append(coeff_emd)

        # # Reshape the EMD results to match the original shape
        # coeff_emd = torch.tensor(np.array(coeff_emd_list).reshape(b, c, d, h, w)).to(coeff.device)

        # Combine the results (this is a simple example, more sophisticated fusion methods can be used)
        coeff_combined = (coeff.float() + coeff_gaussian) / 2
        return coeff_combined

    def apply_nlm_denoising(self, x):
        """Applies Non-Local Means (NLM) denoising to the input tensor."""
        sigma_est = estimate_sigma(x.cpu().detach().numpy(), multichannel=True, average_sigmas=True)
        patch_kw = dict(patch_size=5, patch_distance=6, multichannel=True)
        denoised = denoise_nl_means(x.cpu().detach().numpy(), h=1.15 * sigma_est, fast_mode=True, **patch_kw)
        return torch.tensor(denoised, device=x.device, dtype=x.dtype)

    def restore_shape(self, x, original_shape):
        """Reshapes the output to match the original input shape."""
        # x = x.unsqueeze(1)
        b, c, m, n = x.shape
        _m, _n = original_shape

        min_ = x.reshape(*x.shape[:2], -1).min(dim=2)[0].unsqueeze(2).unsqueeze(3).tile(1, 1, m, n)
        max_ = x.reshape(*x.shape[:2], -1).max(dim=2)[0].unsqueeze(2).unsqueeze(3).tile(1, 1, m, n)
        x[max_ > min_] = (x[max_ > min_] - min_[max_ > min_]) / (max_[max_ > min_] - min_[max_ > min_])

        x = torch.clamp(x, 0, 1)
        dim = math.sqrt(_m * _n)
        if dim > int(dim):
            x = x.reshape(b, c, -1)[:, :, :_m * _n].reshape(b, c, _m, _n)

        return x


class BPDAWrapper(nn.Module):
    def __init__(self, fn):
        super(BPDAWrapper, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class SabreWrapper(nn.Module):
    def __init__(self, eps: float = 8. / 255, wave='coif1', use_rand: bool = False, n_variants: int = 10,
                 base_model: nn.Module = None, use_bpda=True):
        super(SabreWrapper, self).__init__()
        model = Sabre(eps=eps, wave=wave, use_rand=use_rand, n_variants=n_variants)
        base_model.set_eps(1.25 * eps)
        self.core = model
        self.base_model = base_model
        self.use_bpda = use_bpda
        self.transform_fn = model.transform
        self.transform_bpda = BPDAWrapper(self.transform_fn)
        self.base_model.denoise.preprocessing = self.transform

    def transform(self, x):
        if self.use_bpda:
            return self.transform_bpda(x)
        return self.transform_fn(x)

    @property
    def lambda_r(self):
        return self.core.lambda_r

    def reconstruct(self, x, ctx):
        return self.base_model.reconstruct(x, ctx)

    def preprocessing(self, x):
        b, c, m, n = x.shape

        ctx = x.clone()

        denoised = self.transform(x)

        if hasattr(self.base_model, "normalize_data"):
            denoised = self.base_model.normalize_data(denoised)
            self.base_model.denoise.normalize_data = self.base_model.normalize_data

        self.base_model.denoise.classifier = self.base_model.classifier
        x = self.reconstruct(denoised, ctx)

        return denoised.reshape(b, c, m, n), x.reshape(b, c, m, n)

    def classify(self, x):
        return self.base_model(x)

    def features_logits(self, x):
        return self.base_model.features_logits(x)

    def forward(self, x):
        return self.classify(self.preprocessing(x)[1])

    def set_base_model(self, base_model):
        self.base_model = base_model

# class SabreWrapper(nn.Module):
#     def __init__(self, eps: float = 8. / 255, wave='coif1', use_rand: bool = True, n_variants: int = 10,
#                  base_model: nn.Module = None):
#         super(SabreWrapper, self).__init__()
#         self.core = Sabre(eps=eps, wave=wave, use_rand=use_rand, n_variants=n_variants)
#         self.base_model = base_model  # 用于分类和对抗样本生成
#         self.transform_fn = self.core.transform  # 小波去噪函数
#         self.use_bpda = False  # 是否使用 BPDA
#
#     def transform(self, x):
#         return self.transform_fn(x)
#
#     def preprocessing(self, x):
#         """对输入进行小波去噪预处理"""
#         denoised = self.transform(x)
#         return denoised
#
#     def classify(self, x):
#         """使用 base_model 进行分类"""
#         return self.base_model(x)
#
#     def forward(self, x):
#         # 对输入进行小波去噪预处理
#         denoised = self.preprocessing(x)
#         # 传递给 base_model 进行分类
#         return self.classify(denoised)