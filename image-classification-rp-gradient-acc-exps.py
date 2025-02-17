"""
Script for training and evaluating image classification models with Random Projecto=ion variations.
Supports multiple architectures (ResNet50, VGG16, ViT) and Cifar100, ImageNet-A and OmniBenchmark datasets.

Example usage:
    python image-classification-rp-exps.py --model ResNet50 --dataset CIFAR100 --use_rp True --lambda_value 1e-3
"""

import argparse
import logging
import os
from typing import Optional, Tuple
from enum import Enum
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, vgg16, ResNet50_Weights, VGG16_Weights
from transformers import ViTModel, ViTConfig, ViTForImageClassification
import neptune

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model architectures."""
    RESNET50 = "ResNet50"
    VGG16 = "VGG16"
    VIT = "ViT"


class DatasetType(Enum):
    """Supported datasets."""
    CIFAR100 = "CIFAR100" #100
    IMAGENET_A = "ImageNet-A" #1000
    OMNIBENCHMARK = "OmniBenchmark" #1623


class DatasetManager:
    """Handles dataset creation and data loading."""

    def __init__(self, dataset_type: DatasetType):
        """
        Args:
            dataset_type: Type of dataset to use
        """
        self.dataset_type = dataset_type

    def get_datasets(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Creates and returns training and testing datasets."""
        if self.dataset_type == DatasetType.CIFAR100:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
            train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True,
                                                          download=True, transform=transform_train)
            test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                         download=True, transform=transform_test)
        elif self.dataset_type == DatasetType.IMAGENET_A:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            train_dataset = torchvision.datasets.ImageNet(root="./data", split="train", transform=transform)
            test_dataset = torchvision.datasets.ImageNet(root="./data", split="val", transform=transform)
        elif self.dataset_type == DatasetType.OMNIBENCHMARK:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            train_dataset = torchvision.datasets.Omniglot(root="./data", background=True, download=True,
                                                          transform=transform)
            test_dataset = torchvision.datasets.Omniglot(root="./data", background=False, download=True,
                                                         transform=transform)
        else:
            raise ValueError(f"Dataset type {self.dataset_type} is not supported.")
        return train_dataset, test_dataset

    def get_loaders(self, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns DataLoader instances."""
        train_dataset, test_dataset = self.get_datasets()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader


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
    
class CNNRandomProjection(nn.Module):
    def __init__(self, C, H, W, lambda_value = None):
        super(CNNRandomProjection, self).__init__()
        # Tạo ma trận random với kích thước (C, H, W)
        self.random_projection = nn.ModuleList()
        for i in range(C):
            linear_matrix = nn.Linear(H, H, bias=False)
            linear_matrix.weight.requires_grad = False 
            nn.init.normal_(linear_matrix.weight, mean=0, std=1.0)
            self.random_projection.append(linear_matrix)
        if lambda_value:
            self.sqrt_d = H
            self.lambda_param = lambda_value  
        else:
            self.sqrt_d = 1
            self.lambda_param = nn.Parameter(torch.FloatTensor([1e-3])) 
        self.batch_norm = nn.BatchNorm2d(C)
        self.W = W

    def forward(self, x):
        x_new = x.clone()
        for i in range(len(self.random_projection)):
            for j in range(self.W):
                proj_val = self.random_projection[i](x[:, i, :, j])
                x_new[:, i, :, j] = proj_val
        x_new = self.lambda_param * self.sqrt_d * x_new    
        x_new = nn.functional.leaky_relu(x_new, negative_slope=0.2)
        x_new = self.batch_norm(x_new)
        return x_new


class ClassificationModel(nn.Module):
    """Classification model with configurable architecture."""
    
    def __init__(self, model_type: ModelType, num_classes: int, use_rp: bool = False,
                 lambda_value: Optional[float] = None, use_cnn_rp: bool = False,
                 cnn_lambda_value: Optional[float] = None, num_input_channels: int = 3):
        """
        Args:
            model_type: Type of base model architecture
            num_classes: Number of output classes
            use_rp: Whether to use randomized projection
            lambda_value: Scaling factor for RP layer
        """
        super().__init__()
        self.use_rp = use_rp
        self.lambda_value = lambda_value
        self.use_cnn_rp = use_cnn_rp
        self.cnn_lambda_value = cnn_lambda_value

        if model_type == ModelType.RESNET50:
            if num_input_channels == 3:
                base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                base_model.conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            elif num_input_channels == 1:
                base_model = resnet50()
                base_model.conv1 = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            base_model.maxpool = nn.Identity()
            self.features = nn.Sequential(
                base_model.conv1,    
                base_model.bn1,
                base_model.relu,        
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
            )
            self.features2 = nn.Sequential(
                base_model.avgpool
            )
            self.feature_dim = base_model.fc.in_features

        elif model_type == ModelType.VGG16:
            if num_input_channels == 3:
                base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                base_model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.features = nn.Sequential(
                    base_model.features,
                    nn.AdaptiveAvgPool2d((1, 1))
                )
            elif num_input_channels == 1:
                base_model = vgg16()
                base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
                self.features = nn.Sequential(
                    base_model.features,
                    nn.AdaptiveAvgPool2d((1, 1))
                )
           
            self.features = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.feature_dim = 512
        elif model_type == ModelType.VIT:
            self.config = ViTConfig(
                image_size=32,
                patch_size=4,
                num_channels=num_input_channels,
                num_labels=num_classes,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02,
            )
            pretrained = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                ignore_mismatched_sizes=True,
                num_labels=num_classes
            )
            self.features = ViTModel(self.config)
            with torch.no_grad():
                self.features.embeddings.patch_embeddings.weight.data = \
                    pretrained.vit.embeddings.patch_embeddings.weight.data[:, :, :4, :4]
                self.features.embeddings.cls_token.data = pretrained.vit.embeddings.cls_token.data
                self.features.encoder = pretrained.vit.encoder
            self.feature_dim = self.config.hidden_size
        else:
            raise ValueError(f"Model type {model_type} is not supported.")

        if use_rp:
            self.rp = RanPACLayer(self.feature_dim, self.feature_dim, self.lambda_value)
        if use_cnn_rp:
            self.cnn_rp = CNNRandomProjection(2048,4,4,self.cnn_lambda_value)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.features(x)
        if self.use_cnn_rp:
            x = self.cnn_rp(x)

        if isinstance(self.features, ViTModel):
            x = x.last_hidden_state[:, 0, :] ### Take the first element, [CLS] token
        else:
            x = self.features2(x)
            x = torch.flatten(x, 1)

        if self.use_rp:
            x = self.rp(x)
        return self.fc(x)


class Trainer:
    """Handles model training and evaluation."""
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                 device: torch.device, exp_dir: str, args: argparse.Namespace,
                 neptune_run: Optional[neptune.Run] = None):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to run on
            exp_dir: Experiment directory for saving outputs
            args: Training arguments and hyperparameters
            neptune_run: Neptune.ai run instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.exp_dir = exp_dir
        self.args = args
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.neptune_run = neptune_run
        self.checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=args.initial_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.initial_lr,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)

    def train_one_epoch(self, writer: SummaryWriter, epoch: int) -> Tuple[float, float]:
        """Trains model for one epoch and returns (loss, accuracy)."""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if epoch < self.args.warmup_epochs:
                scale = min(1., float(epoch * len(self.train_loader) + i + 1) / 
                          float(self.args.warmup_epochs * len(self.train_loader)))
                for group in self.optimizer.param_groups:
                    group["lr"] = scale * self.args.initial_lr

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            # Scale loss by accumulation steps.
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * inputs.size(0) * self.gradient_accumulation_steps
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)

            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/Accuracy", epoch_acc, epoch)
        if self.neptune_run:
            self.neptune_run["Train/Loss"].append(epoch_loss)
            self.neptune_run["Train/Accuracy"].append(epoch_acc)
        return epoch_loss, epoch_acc

    def evaluate(self, writer: SummaryWriter, epoch: int) -> Tuple[float, float]:
        """Evaluates model and returns (loss, accuracy)."""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                correct += outputs.max(1)[1].eq(labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        writer.add_scalar("Test/Loss", epoch_loss, epoch)
        writer.add_scalar("Test/Accuracy", epoch_acc, epoch)
        if self.neptune_run:
            self.neptune_run["Test/Loss"].append(epoch_loss)
            self.neptune_run["Test/Accuracy"].append(epoch_acc)
        return epoch_loss, epoch_acc

    def train(self, writer: SummaryWriter, epochs: int):
        """Trains the model for specified number of epochs."""
        best_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(writer, epoch)
            val_loss, val_acc = self.evaluate(writer, epoch)

            logger.info(f"Epoch {epoch+1}/{epochs}: "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)


def get_experiment_name(args: argparse.Namespace) -> str:
    """
    Generate a unique experiment name based on configuration.

    Args:
        args: Command line arguments

    Returns:
        str: Formatted experiment name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model.value}_{args.dataset.value}"
    if args.use_rp:
        exp_name += f"_RP{args.lambda_value}"
    exp_name += f"_lr{args.initial_lr}_bs{args.batch_size}_g{args.gradient_accumulation_steps}_{timestamp}"
    return exp_name


def setup_experiment_folders(exp_name: str) -> str:
    """
    Create folders for experiment outputs.

    Args:
        exp_name: Experiment name

    Returns:
        str: Path to experiment directory
    """
    base_dir = os.path.join("experiments", exp_name)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    return base_dir


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser()
    # Model and dataset arguments
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="Model type")
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), required=True, help="Dataset type")
    parser.add_argument("--use_rp", type=bool, default=False, help="Use randomized projection")
    parser.add_argument("--lambda_value", type=float, default=None, help="Lambda value for RP")
    parser.add_argument("--use_cnn_rp", type=bool, default=False, help="Use randomized projection for CNN layer")
    parser.add_argument("--cnn_lambda_value", type=float, default=None, help="Lambda value for RP in CNN layer")
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--initial_lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay coefficient")
    parser.add_argument("--nesterov", type=bool, default=True, help="Use Nesterov momentum")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes")
    args = parser.parse_args()

    exp_name = get_experiment_name(args)
    exp_dir = setup_experiment_folders(exp_name)
    logger.info(f"Starting experiment: {exp_name}")
    
    config = {
        "model_type": args.model.value,
        "dataset_type": args.dataset.value,
        "use_rp": args.use_rp,
        "lambda_value": args.lambda_value,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "virtual_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.initial_lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "nesterov": args.nesterov,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs
    }
    
    neptune_run = neptune.init_run(
        project="phuca1tt1bn/RAMA",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODZlNDU0Yy1iMDk0LTQ5MDEtOGNiYi00OTZlYTY4ODI0MzgifQ==",
        name=exp_name
    )
    neptune_run["config"] = config

    dataset_manager = DatasetManager(args.dataset)
    train_loader, test_loader = dataset_manager.get_loaders(batch_size=args.batch_size)

    model = ClassificationModel(
        model_type=args.model,
        num_classes=args.num_classes,
        use_rp=args.use_rp,
        lambda_value=args.lambda_value,
        use_cnn_rp = args.use_cnn_rp,
        cnn_lambda_value = args.cnn_lambda_value,
        num_input_channels= 1 if args.dataset.value == "OmniBenchmark" else 3
    )
    print(model)
    logger.info(f"Model initialized: {model}")
    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        exp_dir=exp_dir,
        args=args,
        neptune_run=neptune_run
    )
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs"))
    trainer.train(writer, epochs=args.epochs)
    writer.close()
    if neptune_run:
        neptune_run.stop()


if __name__ == "__main__":
    main()
