import argparse
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16, VGG16_Weights
from pre_process import ImageDataset


import wandb

# Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 8e-6
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 1e-4

## Just to name exp
def get_experiment_name(args: argparse.Namespace) -> str:
    """
    Generate the experiment name based on the provided arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        str: Experiment name.
    """
    experiment_name = "VGG16_IMAGENETA"
    if args.use_rp:
        experiment_name += f"_RP_lambda_{args.lambda_value}"
    return experiment_name

## Actually, RM
class RanPACLayer(nn.Module):
    """
    A randomized projection layer with optional normalization.

    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        lambda_value (Optional[float]): Lambda scaling value for the projection matrix.
        norm_type (str): Normalization type, either "batch" or "layer".
    """
    def __init__(self, input_dim: int, output_dim: int, lambda_value: Optional[float] = None, norm_type: str = "batch"):
        super(RanPACLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False) 
        self.projection.weight.requires_grad = False  
        nn.init.normal_(self.projection.weight, mean=0, std=1.0) 
        self.lambda_param = lambda_value if lambda_value else nn.Parameter(torch.FloatTensor([1e-3])) 
        self.norm = nn.BatchNorm1d(output_dim) if norm_type == "batch" else nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RanPAC layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        x = self.projection(x) * self.lambda_param
        x = nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.norm(x)
        return x

## VGG16
class VGG16(nn.Module):
    """
    Modified VGG16 model with optional randomized projection.

    Args:
        num_classes (int): Number of output classes.
        use_rp (bool): Whether to use the RanPACLayer.
        lambda_value (Optional[float]): Lambda scaling value for RanPACLayer.
    """
    def __init__(self, num_classes: int, use_rp: bool = False, lambda_value: Optional[float] = None):
        super().__init__()
        self.model = vgg16(weights=VGG16_Weights.DEFAULT)  
        self.features = nn.Sequential(
            *self.model.features,    
            self.model.avgpool,   
        )
        self.features2 = nn.Sequential(*list(self.model.classifier[:-1]) )    
        num_features = self.model.classifier[6].in_features  
        self.use_rp = use_rp
        if use_rp:       
            self.rp1 = RanPACLayer(25088, 25088, lambda_value)      
            self.rp2 = RanPACLayer(num_features, num_features, lambda_value)  
            self.fc = nn.Linear(num_features , num_classes) 
        else:
            self.fc = nn.Linear(num_features, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VGG16 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.flatten(self.features(x), 1) 
        if self.use_rp:
            x = self.rp1(x)
            x = self.features2(x)
            x = self.rp2(x)
        else:
            x = self.features2(x)
        return self.fc(x)


def train_one_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device, writer: SummaryWriter, epoch: int) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device for computation.
        writer (SummaryWriter): TensorBoard SummaryWriter.
        epoch (int): Current epoch number.

    Returns:
        Tuple[float, float]: Training loss and accuracy.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # Log batch metrics
        if writer:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + batch_idx)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader, criterion: nn.Module,
             device: torch.device, writer: SummaryWriter, epoch: int) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): Model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for computation.
        writer (SummaryWriter): TensorBoard SummaryWriter.
        epoch (int): Current epoch number.

    Returns:
        Tuple[float, float]: Validation loss and accuracy.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    if writer:
        writer.add_scalar("Test/Loss", epoch_loss, epoch)
        writer.add_scalar("Test/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


def main(args: argparse.Namespace) -> None:
    """
    Main training loop.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    experiment_name = get_experiment_name(args)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    # Wandb setup
    wandb.login(key="18867541319386f8b2e1362741174bd50968c3f3")
    
    wandb.init(
        project="vgg16-ImageNetA-rp-exp",  # Replace with your project name
    )

    # Transforms & Dataset
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=8),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageDataset(mode="train", transform=train_transform)
    test_dataset = ImageDataset(mode="test", transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    ################
    model = VGG16(num_classes=100, use_rp=args.use_rp, lambda_value=args.lambda_value).to(device)
    criterion = nn.CrossEntropyLoss() ## cross-entropy
    lr = args.learning_rate if args.learning_rate != None else LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2), eps=EPSILON, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) ###

    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, writer, epoch)

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Test/Loss", val_loss, epoch)
        writer.add_scalar("Test/Accuracy", val_acc, epoch)

        writer.add_histogram("Model/Weights", model.fc.weight, epoch)

        scheduler.step()
        if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), f"{experiment_name}_{epoch+1}.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{experiment_name}_best.pth")

        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        wandb.log({
                "epoch": epoch,
                "Train/Loss": train_loss,
                "Train/Acc": train_acc,
                "Test/Loss":val_loss,
                "Test/Acc":val_acc,
                }, commit=False)
        if args.use_rp == True and args.lambda_value == None:
            logger.info(f"Lambda 1 Value: {model.rp1.lambda_param.item()}")
            logger.info(f"Lambda 2 Value: {model.rp2.lambda_param.item()}")
            wandb.log({
                "Lambda1":model.rp1.lambda_param.item(),
                "Lambda2":model.rp2.lambda_param.item(),
            }, commit=True)
        else:
            wandb.log({
            }, commit=True)

    logger.info(f"Training finished. Best validation accuracy: {best_acc:.2f}%")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_rp", type=bool, default=False)
    parser.add_argument("--lambda_value", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()
    main(args)

