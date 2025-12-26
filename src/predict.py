import os
import sys
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from model import CIFAR10CNN
from utils import get_device, load_model_weights

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

@torch.no_grad()
def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    cfg = Config()
    device = get_device(cfg.device)

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)  # (1, 3, 32, 32)

    model = CIFAR10CNN(num_classes=cfg.num_classes).to(device)
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.best_ckpt_name)
    load_model_weights(ckpt_path, model, map_location=device)

    model.eval()
    logits = model(x)
    pred = logits.argmax(dim=1).item()

    print(f"Prediction: {pred} ({CIFAR10_CLASSES[pred]})")

if __name__ == "__main__":
    main()