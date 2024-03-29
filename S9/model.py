import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolution Block 1 - Standard Convolution
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=2, stride=2),  # Reduced channels
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2, stride=2),  # Reduced channels
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, stride=2),  # Adjusted channels
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )  # Receptive field: 9
        
        # Convolution Block 2 - Standard Convolution
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2, stride=2, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, stride=2, groups=128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2, stride=2, groups=128),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value),
        )  # Receptive field: 14

        # Convolution Block 3 - Depthwise Separable Convolution with Dilation
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=2, stride=2, groups=256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=4, stride=2, groups=256, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_value),
        )  # Receptive field: 32 (approx, considering dilations)
        
        # Convolution Block 4 - Depthwise Separable Convolution
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=4, stride=2, groups=256, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),  # Pointwise
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=2, stride=2, groups=64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2, groups=256), # Stride 2 here
            

            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),  # Pointwise
            # nn.ReLU(),
            # nn.BatchNorm2d(64)
        )  # Receptive field: 44 (approx)

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
