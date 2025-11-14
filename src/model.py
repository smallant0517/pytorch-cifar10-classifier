import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 第一層：輸入 3 (RGB) → 32 個特徵圖
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二層：32 → 64 特徵圖
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 當經過兩次 pool 後尺寸變：
        # 32x32 -> 16x16 -> 8x8
        self.flatten = nn.Flatten()

        # 全連接層 (MLP)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # -> (B, 32, 16, 16)
        x = self.pool2(self.relu2(self.conv2(x)))  # -> (B, 64, 8, 8)
        x = self.flatten(x)                        # -> (B, 4096)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def test_model():
    model = SimpleCNN()
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    test_model()