import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size: int = 64):
    """回傳 CIFAR-10 的 train / test DataLoader"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),   # 每個 channel 的平均值
            (0.5, 0.5, 0.5)    # 每個 channel 的標準差
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return trainloader, testloader

if __name__ == "__main__":
    trainloader, testloader = get_dataloaders(batch_size=64)
    images, labels = next(iter(trainloader))
    print("images shape:", images.shape)   # 期待 [64, 3, 32, 32]
    print("labels shape:", labels.shape)   # 期待 [64]