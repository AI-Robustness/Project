import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


def reshape_inputs(x):
    """Reshapes inputs to have dimensions as squares if necessary."""
    b, c, h, w = x.shape
    target_dim = int(math.ceil(math.sqrt(h * w)))
    if h * w < target_dim ** 2:
        padding_size = target_dim ** 2 - h * w
        x_padded = torch.cat([x.view(b, c, -1), torch.zeros(b, c, padding_size, device=x.device)], dim=2)
        x = x_padded.view(b, c, target_dim, target_dim)
    return x


class DenoisingCNN(nn.Module):
    def __init__(self, in_channels=1, num_layers=17, num_features=64, classifier=None, eps=1, normalized=False):
        super(DenoisingCNN, self).__init__()
        self.in_channels = in_channels
        self.denoising_model = DenoisingAutoencoder()
        layers = [nn.Sequential(nn.Conv2d(3 * in_channels, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1))
        if not normalized:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        self.classifier = classifier
        self.eps = eps
        self.num_steps = 1
        self.alpha = eps / self.num_steps
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        self.preprocessing = None
        self.normalize_data = None

    def forward(self, inputs, ctx):
        b, c, m, n = inputs.shape
        inputs = self.denoising_model(inputs, ctx)
        initial_logits = self.classifier(inputs.clone())
        initial_preds = initial_logits.max(1)[1]

        ctx_fwd = ctx.clone().detach()
        ctx_bwd = ctx.clone().detach()

        for _ in range(self.num_steps):
            ctx_fwd.requires_grad = True
            ctx_bwd.requires_grad = True
            with torch.enable_grad():
                logits = self.classifier(ctx_fwd)
                loss = F.cross_entropy(logits, initial_preds)
                logits_back = self.classifier(ctx_bwd)
                loss_back = -F.cross_entropy(logits_back, initial_preds)

            grads = torch.autograd.grad(loss, ctx_fwd)[0]
            grads_back = torch.autograd.grad(loss_back, ctx_bwd)[0]
            with torch.no_grad():
                ctx_fwd += self.alpha * grads.sign()
                eta = torch.clamp(ctx_fwd - ctx, min=-self.eps, max=self.eps)
                ctx_fwd = torch.clamp(ctx + eta, min=0, max=1).detach()

                ctx_bwd += self.alpha * grads_back.sign()
                eta = torch.clamp(ctx_bwd - ctx, min=-self.eps, max=self.eps)
                ctx_bwd = torch.clamp(ctx + eta, min=0, max=1).detach()

        inputs = reshape_inputs(inputs)

        ctx_fwd = reshape_inputs(ctx_fwd)
        ctx_bwd = reshape_inputs(ctx_bwd)

        if self.normalize_data is not None:
            ctx_fwd = self.normalize_data(ctx_fwd)
            ctx_bwd = self.normalize_data(ctx_bwd)

        outputs = torch.cat((inputs, ctx_fwd, ctx_bwd), dim=1)

        outputs = self.layers(outputs)

        if math.sqrt(m * n) > int(math.sqrt(m * n)):
            outputs = outputs.reshape(b, c, -1)[:, :, :m * n].reshape(b, c, m, n)

        return outputs


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Define the encoder
        self.ecn1 = nn.Conv2d(6,64, kernel_size=3, stride=2, padding=1)  # [512, 64, 14, 14]
        self.relu = nn.ReLU(True)
        # self.ecn2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [512, 128, 7, 7]
        # self.ecn3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [512, 256, 4, 4]
        self.naf1 = NAFBlock(64)#NFA消融
        # self.naf2 = NAFBlock(64)
        self.feat1 = DoubleConv(64, 64)#消融DoubleConv
        # self.feat2 = DoubleConv(64,64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.jpu1 = JPU([64, 64, 64])
        # Define the decoder
        self.sam = PSAModule(64, 64)#消融PSA
        self.decoder = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # [512, 64, 16, 16]
        self.sig = nn.Sigmoid()

    def forward(self, x, x_ori):
        x = torch.cat([x, x_ori], dim=1)
        x = self.relu(self.ecn1(x))
        x = self.naf1(x)#NFA
        # x = self.naf2(x)
        x = self.feat1(x)#doubleconv消融
        x = self.sam(x)#PSA消融
        # x = self.feat2(x)
        # _, _, _, x = self.jpu1(x ,x2, x3)

        x = self.sig(self.decoder(self.up(x)))
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=32, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs
        # 256
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            # norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        # 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            # norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        # 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            # norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(128, in_channels[0], 1)
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            #    norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            #    norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            #    norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            #    norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bilinear', align_corners=False)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='bilinear', align_corners=False)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)
        feat = self.conv(feat)
        return inputs[0], inputs[1], inputs[2], feat


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        # MFFE第一层
        self.myConv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # MFFE第二层
        self.myConv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # MFFE第三层
        self.myConv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # MFFE第四层
        self.myConv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.double_conv(x)
        # 分别通过MFFE的不从尺度层，并将特征整合
        x1 = self.myConv1(x)
        x2 = self.myConv2(x)
        x3 = self.myConv3(x)
        x4 = self.myConv4(x)
        x_all = x + x1 + x2 + x3 + x4
        x_all = self.drop(x_all)
        return x_all


# stylegan2 classes
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=8):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 2, 4, 8]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, int(planes / 4), kernel_size=conv_kernels[0], padding=int(conv_kernels[0] / 2),
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, int(planes / 4), kernel_size=conv_kernels[1], padding=int(conv_kernels[1] / 2),
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, int(planes / 4), kernel_size=conv_kernels[2], padding=int(conv_kernels[2] / 2),
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, int(planes / 4), kernel_size=conv_kernels[3], padding=int(conv_kernels[3] / 2),
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(int(planes / 4))
        self.split_channel = int(planes / 4)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout()

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = self.drop(feats)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        x_se = self.drop(x_se)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out