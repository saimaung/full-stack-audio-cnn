import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # support output of the first convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # by default empty sequence that does nothing for the shortcut
        self.short_cut = nn.Sequential()
        # shortcut layer:
        #   1. stride != 1: down sample input data
        #   2. in_channels != out_channels: need to transform so that both out_channels and in_channels are in the same shape
        #   the output matrix of the parent layer won't equal
        #   to input matric of the child layer
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            # sequential so that we can add multiple layers
            # making deeper networks
            self.short_cut = nn.Sequential(
                # conv layer because this transformation is applied to input data such that
                # we can add input data to the output of the above layers
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # input -> first Neuron
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.short_cut(x) if self.use_shortcut else x
        out = torch.relu(out + shortcut)
        return out


class AudioCNN(nn.Module):
    # ES50 dataset has 50 classes
    def __init__(self, num_classes=50):
        super().__init__()
        self.initial_conv = nn.Sequential(
            # gray scale mel spectrum
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            # inplace=True applies directly to the output of BatchNorm2D
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer_1 = nn.ModuleList([
            ResidualBlock(in_channels=64, out_channels=64) for _ in range(3)
        ])
        self.layer_2 = nn.ModuleList([
            # first iteration in_channels tensor need t match layer_1 output tensor
            ResidualBlock(in_channels=64 if i == 0 else 128, out_channels=128, stride=2 if i == 0 else 1) for i in range(4)
        ])
        self.layer_3 = nn.ModuleList([
            # make layers compatible with each other
            ResidualBlock(in_channels=128 if i == 0 else 256, out_channels=256, stride=2 if i == 0 else 1) for i in range(6)
        ])
        self.layer_4 = nn.ModuleList([
            ResidualBlock(in_channels=256 if i == 0 else 512, out_channels=512, stride=2 if i == 0 else 1) for i in range(3)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        # pass tensors through even ResidualBlock
        # Process through all residual layers sequentially
        for layer in [self.layer_1, self.layer_2, self.layer_3, self.layer_4]:
            for residual_block in layer:
                x = residual_block(x)
        x = self.avg_pool(x)
        # Flatten layer: reshape tensor without changing the values
        # first dimension is batch size - keep it as it is
        # second dimension is -1 means infer the dimension automatically
        # such that the total number of elements remains the same
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # linear layer
        x = self.fc(x)
        return x
