import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import SimpleCNN


def train_model(num_epochs=5, batch_size=64, lr=1e-3, device="cpu"):
    # Device
    device = torch.device(device)

    # Data
    trainloader, testloader = get_dataloaders(batch_size)

    # Model
    model = SimpleCNN().to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "cifar10_cnn.pth")
    print("模型已儲存為 cifar10_cnn.pth")

    return model


if __name__ == "__main__":
    # 自動使用 GPU（如果有）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(num_epochs=5, device=device)