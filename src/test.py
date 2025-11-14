import torch
from src.dataset import get_dataloaders
from src.model import SimpleCNN


def evaluate(device="cpu"):
    # Device
    device = torch.device(device)

    # Load data
    _, testloader = get_dataloaders(batch_size=64)

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(device=device)