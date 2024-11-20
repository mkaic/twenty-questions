import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm

from ..src.model import Questioner

HIDDEN_DIM = 128
QUESTION_VECTOR_SIZE = 32
CONTEXT_VECTOR_SIZE = 32
QUESTIONS_PER_LAYER = 16
NUM_LAYERS = 16
NUM_CLASSES = 10


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-4
COMPILE = False
SAVE = False


print(
    f"""
{BATCH_SIZE=}, 
{EPOCHS=}
"""
)

if not Path("twenty-questions/weights").exists():
    Path("twenty-questions/weights").mkdir(parents=True)

model = Questioner(
    input_channels=3,
    num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS,
    questions_per_layer=QUESTIONS_PER_LAYER,
    question_vector_size=QUESTION_VECTOR_SIZE,
    context_vector_size=CONTEXT_VECTOR_SIZE,
    hidden_dim=HIDDEN_DIM,
)
model = model.to(DEVICE)
model = model.to(DTYPE)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")


# Load the MNIST dataset
if NUM_CLASSES == 100:
    train = CIFAR100(
        root="./twenty-questions/data", train=True, download=True, transform=ToTensor()
    )
    test = CIFAR100(
        root="./twenty-questions/data", train=False, download=True, transform=ToTensor()
    )
elif NUM_CLASSES == 10:
    train = CIFAR10(
        root="./twenty-questions/data", train=True, download=True, transform=ToTensor()
    )
    test = CIFAR10(
        root="./twenty-questions/data", train=False, download=True, transform=ToTensor()
    )
else:
    raise ValueError("NUM_CLASSES must be 10 or 100")

train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=8
)

loss_fn = nn.CrossEntropyLoss()

# Train the model
optimizer = Adam(model.parameters(), lr=LR)

test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, leave=False)

    for images, labels in pbar:
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, labels = images.to(DTYPE), labels.to(torch.long)
        images = images.permute(0, 2, 3, 1)  # (b, c, h, w) -> (b, h, w, c)

        output = model(images)

        loss = loss_fn(output, labels)

        loss.backward()

        optimizer.step()

        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    model.eval()
    if SAVE:
        torch.save(model.state_dict(), f"twenty-questions/weights/{epoch:03d}.ckpt")

    total = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, leave=False)):

            images: torch.Tensor
            labels: torch.Tensor

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = images.to(DTYPE), labels.to(torch.long)
            images = images.permute(0, 2, 3, 1)  # (b, c, h, w) -> (b, h, w, c)

            random_locations = random.random() < 0.5

            predictions = model(tokens=images)
            predicted = torch.max(predictions, dim=1)[1]

            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
