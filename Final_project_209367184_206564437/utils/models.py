import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform, xavier_normal

class ConvClassifier(nn.Module):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims


        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def initialize_flags(self, BatchNorm=False, Dropout=False):
        self.BatchNorm = BatchNorm
        self.Dropout = Dropout


    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        for i, f in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, f, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_h //= 2
                in_w //= 2
            in_channels = f
        self.out_dim = (in_channels, in_h, in_w)
        return nn.Sequential(*layers)

    def _make_classifier(self):
        in_channels, in_h, in_w = self.out_dim
        flat_dim = in_channels * in_h * in_w
        layers = []
        prev_dim = flat_dim
        for next_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.ReLU())
            prev_dim = next_dim
        layers.append(nn.Linear(prev_dim, self.out_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)

        # Flatten the features
        x = torch.flatten(x, start_dim=1)

        # Pass the flattened features through the classifier
        out = self.classifier(x)

        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims,BatchNorm=False,Dropout=False):
        self.initialize_flags(BatchNorm, Dropout)
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w = tuple(self.in_size)
        layers = []
        for i, f in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, f, kernel_size=3, padding=1))
            if self.BatchNorm:
                layers.append(nn.BatchNorm2d(f))  # Batch Normalization after Conv layer
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                if self.Dropout:
                    layers.append(nn.Dropout(p=0.5))  # Dropout after Max Pooling layer
                in_h //= 2
                in_w //= 2

            in_channels = f
        self.out_dim = (in_channels, in_h, in_w)
        return nn.Sequential(*layers)

