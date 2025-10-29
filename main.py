import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

import trackio
import argparse
import time

from zmq import device

from moe import ToyMoE, SimpleConvNet

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path="best_model.pth"):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # in case validation changes the state
        model.train()

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero grad
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, aux_loss = model(inputs)
            loss = criterion(outputs, labels)
            total_loss = loss + aux_loss

            trackio.log({
                "epoch": epoch,
                "batch": i,
                "loss": loss.item(),
                "aux_loss": aux_loss.item(),
                "total_loss": total_loss.item()
            })

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.5f}    aux_loss: {aux_loss.item():.5f}    total_loss: {total_loss.item():.5f}")
                running_loss = 0.0

        # validate after each epoch
        val_acc = eval(model, val_loader, device, test=False)

        # Track the best validation accuracy and save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
            }, save_path)
            print(f"✅ Saved new best model (epoch {epoch + 1}, val acc {val_acc:.2f}%)")

        trackio.log({
            "epoch": epoch + 1,
            "val_accuracy": val_acc,
            "best_val_accuracy": best_val_acc,
        })

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.2f}%")


def eval(model, test_loader, device, test=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    if test:
        print(f"Test Accuracy: {acc:.2f}%")
        trackio.log({"test_accuracy": acc})
    else:
        print(f"Validation Accuracy: {acc:.2f}%")
        trackio.log({"val_accuracy": acc})
    return acc


def main(args):
    trackio.init(
        project="MoE-Toy-Example",
        name=f"ConvMoE-CIFAR10-{time.strftime('%Y%m%d-%H%M%S')}",
        config=vars(args)
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Split into train/val (e.g. 45k/5k)
    train_size = int(0.9 * len(full_train))  # 45,000
    val_size = len(full_train) - train_size  # 5,000
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = ToyMoE(
        in_channels=3,
        input_size=512,
        output_size=10,
        hidden_size=128,
        num_experts=args.num_experts,
        noisy_gating=True,
        k=args.k
    )

    # print model and parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model size (MB, float32): {total_params * 4 / (1024**2):.2f}")
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.save_dir + "/best_model.pth")

    print("Finished Training. Loading best checkpoint...")
    checkpoint = torch.load(args.save_dir + "/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with val acc {checkpoint['val_accuracy']:.2f}%")

    eval(model, test_loader, device)

    trackio.finish()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--epochs", type=int, default=30)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--num_experts", type=int, default=10)
    argparser.add_argument("--k", type=int, default=1)
    argparser.add_argument("--save_dir", type=str, default="./ckpts")
    args = argparser.parse_args()

    main(args)
