import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoise import DenoisingCNN


class LeNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = lambda x: F.relu(x)
        self.pooling = lambda x: F.max_pool2d(x, 2)
        # #归一化
        # self.mean = torch.tensor([0.1307])
        # self.stds = torch.tensor([0.3081])
    def features(self, x):
        x = x.to(self.conv1.weight.device)
        x_layer1 = self.pooling(self.relu(self.conv1(x)))
        x_layer2 = self.pooling(self.relu(self.conv2(x_layer1)))
        x_layer2 = x_layer2.view(x_layer2.size(0), -1)
        x_layer3 = self.relu(self.fc1(x_layer2))
        x_layer4 = self.relu(self.fc2(x_layer3))

        return x_layer1, x_layer2, x_layer3, x_layer4

    def features_logits(self, x: torch.Tensor):
        x = x.to(self.conv1.weight.device)
        features = self.features(x)
        logits = self.fc3(features[-1])
        return features, logits

    def forward(self, x):
        features, logits = self.features_logits(x)
        return logits


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        classifier = LeNetModel(num_classes=10)

        self.classifier = classifier
        self.features = classifier.features

        self.mean = torch.tensor([0.1307])
        self.stds = torch.tensor([0.3081])
        self.eps = 80./255
        self.normalize = False

        self.denoise = DenoisingCNN(in_channels=1, num_layers=3, classifier=self.classify, eps=self.eps)

    def set_eps(self, eps):
        self.eps = eps
        self.denoise.eps = eps

    def reconstruct(self, x, ctx):
        return self.denoise(x, ctx)

    def features_logits(self, x):
        return self.classifier.features_logits(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x):
        return self.classify(x)
