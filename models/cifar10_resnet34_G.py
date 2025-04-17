import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from pytorch_wavelets import DWTForward
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import math

from .denoise import DenoisingCNN
from torch.nn import Parameter

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
        num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                    weight, self.bias, self.stride,
                    _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)

class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, dilation=1, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, dilation=1, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, dilation=1, padding_mode='reflect')

        # Simple Pixel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        #x = self.CondConv2D(x)
        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicBlock_new(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_new, self).__init__()
        if planes < 256:
            self.conv1 = conv3x3(in_planes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        else:
            self.conv1 = CondConv2D(in_planes, planes, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = CondConv2D(planes, planes, padding=1)
            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    CondConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FDResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_C=12, wave='sym17', mode='append'):
        super(FDResNet, self).__init__()
        self.wave = wave
        self.DWT = DWTForward(J=1, wave=self.wave, mode='symmetric').cuda()
        self.FDmode = mode
        self.in_planes = 64
        self.conv1 = conv3x3(in_C, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.w = Parameter(torch.Tensor(4))

        # self.ResidualAttention1 = ResidualAttention(64,10)
        # self.ResidualAttention2 = ResidualAttention(128,10)
        # self.ResidualAttention3 = ResidualAttention(256,10)
        # self.ResidualAttention4 = ResidualAttention(512,10)

        #self.MixStructureBlock1 = MixStructureBlock(64)
        #self.MixStructureBlock2 = MixStructureBlock(128)
        #self.MixStructureBlock3 = MixStructureBlock(256)
        self.MixStructureBlock4 = MixStructureBlock(512)
        #self.condconv = CondConv2D(512, 10, kernel_size=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.wavelets(x, self.FDmode)
        #print(f"Shape of input to conv1 after DWT: {x.shape}")
        # x = self.w[0]*x[:,0:3,:]+self.w[1]*x[:,3:6,:]+self.w[2]*x[:,6:9,:]+self.w[3]*x[:,9:12,:]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        #out = self.MixStructureBlock1(out)
        # out1 = self.ResidualAttention1(out)
        out = self.layer2(out)
        #out = self.MixStructureBlock2(out)
        # out2 = self.ResidualAttention2(out)
        out = self.layer3(out)
       # out = self.MixStructureBlock3(out)
        # out3 = self.ResidualAttention3(out)
        out = self.layer4(out)
        out = self.MixStructureBlock4(out)
        # out4 = self.ResidualAttention4(out)
        # y_list = [out3, out4]
        # y = torch.sum(torch.stack(y_list), dim=0)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        #y = self.condconv(out)

        return y.squeeze(-1).squeeze(-1)

    # function to extact the multiple features
    def feature_list(self, x):
        # 对输入先做wavelets转换，保持与 forward() 中一致
        x = self.wavelets(x, self.FDmode)
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = self.MixStructureBlock4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return out_list,y

    # # function to extact a specific feature
    # def intermediate_forward(self, x, layer_index):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     if layer_index == 1:
    #         out = self.layer1(out)
    #     elif layer_index == 2:
    #         out = self.layer1(out)
    #         out = self.layer2(out)
    #     elif layer_index == 3:
    #         out = self.layer1(out)
    #         out = self.layer2(out)
    #         out = self.layer3(out)
    #     elif layer_index == 4:
    #         out = self.layer1(out)
    #         out = self.layer2(out)
    #         out = self.layer3(out)
    #         out = self.layer4(out)
    #     return out

    # def features_logits(self, x: torch.Tensor):
    #     # 确保输入在相同设备上
    #     x = x.to(self.conv1.weight.device)
    #     # 对输入做 wavelets 预处理
    #     x = self.wavelets(x, self.FDmode)
    #
    #     # 逐步提取各层特征
    #     f1 = F.relu(self.bn1(self.conv1(x)))
    #     f2 = self.layer1(f1)
    #     f3 = self.layer2(f2)
    #     f4 = self.layer3(f3)
    #     f5 = self.layer4(f4)
    #
    #     # 将各层特征存入列表中（按需返回中间特征）
    #     features = [f1, f2, f3, f4, f5]
    #
    #     # 对最后一层特征做全局池化和展平
    #     pooled = F.avg_pool2d(f5, 4)
    #     flattened = pooled.view(pooled.size(0), -1)
    #
    #     # 计算 logits
    #     logits = self.linear(flattened)
    #
    #     return features, logits

    # function to extact the penultimate features
    # def penultimate_forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     penultimate = self.layer4(out)
    #     out = F.avg_pool2d(penultimate, 4)
    #     out = out.view(out.size(0), -1)
    #     y = self.linear(out)
    #     return y, penultimate

    def wavelets(self, x, FDmode):
        # 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',\n
        # 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'\n\n
        x = x.cuda().reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        Yl, Yh = self.DWT(x)
        output = self.plugdata(x, Yl, Yh, FDmode)
        return output

    def plugdata(self, x, Yl, Yh, mode):
        if mode == 'append':
            output = torch.zeros(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0:3, :] = Yl[:, :, :]
            output[:, 3:6, :] = Yh[0][:, 0, :, :]
            output[:, 6:9, :] = Yh[0][:, 1, :, :]
            output[:, 9:12, :] = Yh[0][:, 2, :, :]
            output = output.reshape(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        elif mode == 'avg':
            output = torch.zeros(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0, :] = torch.mean(Yl[:, :, :], axis=1)
            output[:, 1, :] = torch.mean(Yh[0][:, 0, :, :], axis=1)
            output[:, 2, :] = torch.mean(Yh[0][:, 1, :, :], axis=1)
            output[:, 3, :] = torch.mean(Yh[0][:, 2, :, :], axis=1)
            output = output.reshape(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        return output



def FDResNet34(num_c):
    return FDResNet(BasicBlock_new, [3,4,6,3], num_classes=num_c,in_C=12)

class Cifar10Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar10Model, self).__init__()
        classifier = FDResNet34(num_classes)

        self.classifier = classifier
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(-1, 1, 1)
        self.stds = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(-1, 1, 1)
        self.eps = 8. / 255
        self.normalize = True

        self.denoise = DenoisingCNN(in_channels=3, num_layers=5, classifier=self.classify, eps=self.eps,
                                    normalized=self.normalize)

    def set_eps(self, eps):
        self.eps = eps
        self.denoise.eps = eps

    def normalize_data(self, x):
        if self.normalize:
            x = (x - self.mean.to(x.device)) / self.stds.to(x.device)
        return x

    def reconstruct(self, x, ctx):
        return self.denoise(x, ctx)

    def features_logits(self, x):
        # Using penultimate features and logits (if needed)
        return self.classifier.feature_list(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        return self.classify(x)
