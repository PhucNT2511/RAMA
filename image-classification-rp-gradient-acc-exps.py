"""
Script for training and evaluating image classification models with Random Projecto=ion variations.
Supports multiple architectures (ResNet18, VGG16, ViT) and Cifar100, ImageNet-A and OmniBenchmark datasets.

Example usage:
    python image-classification-rp-exps.py --model ResNet18 --dataset CIFAR100 --use_rp True --lambda_value 1e-3
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
from torchvision.models import resnet18, vgg16, ResNet18_Weights, VGG16_Weights
from transformers import ViTModel, ViTConfig, ViTForImageClassification
import torch.nn.init as init
import neptune
import random
import numpy as np
import math

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model architectures."""
    RESNET18 = "ResNet18"
    VGG16 = "VGG16"
    VIT = "ViT"


class DatasetType(Enum):
    """Supported datasets."""
    CIFAR100 = "CIFAR100" #100
    IMAGENET_A = "ImageNet-A" #1000
    OMNIBENCHMARK = "OmniBenchmark" #1623

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        num_projection (int): Number of RAMA projection
    """
    def __init__(self, input_dim: int, output_dim: int, lambda_value: Optional[float] = None, non_linearities: str = 'leaky_relu', norm_type: str = "batch", negative_slope_leaky_relu: float = 0.2, num_projection: int = 1):
        super(RanPACLayer, self).__init__()
        self.projection = nn.Linear(input_dim, num_projection*output_dim, bias=False) 
        self.projection.weight.requires_grad = False
        nn.init.normal_(self.projection.weight, mean=0, std=1.0)  ## randn
        if lambda_value:
            self.sqrt_d = math.sqrt(input_dim)
            self.lambda_param = lambda_value  
            self.clamp = False
        else:
            self.sqrt_d = math.sqrt(input_dim)
            self.lambda_param = nn.Parameter(torch.tensor(0.01))  ########
            self.clamp = True
        self.norm = nn.BatchNorm1d(output_dim*num_projection) if norm_type == "batch" else nn.LayerNorm(output_dim*num_projection)
        self.non_linearities = non_linearities
        self.negative_slope_leaky_relu = negative_slope_leaky_relu 
        self.output_dim = output_dim
        self.num_projection = num_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RanPAC layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        # Điều chỉnh giá trị lambda
        '''
        #Keep computational graph
        lambda_clamp = self.lambda_param.clone()
        if self.clamp:
            lambda_clamp = torch.clamp(lambda_clamp, min=0.01, max=1.0)
        '''
        # Break graph
        if self.clamp:
            self.lambda_param.data.clamp_(0.005, 0.3)

        x = self.projection(x) * self.lambda_param  * self.sqrt_d
        x = self.norm(x)
        
        if self.non_linearities == 'leaky_relu':
            x_new = nn.functional.leaky_relu(x, negative_slope=self.negative_slope_leaky_relu)
        elif self.non_linearities == 'sigmoid':
            x_new = nn.functional.sigmoid(x)
        elif self.non_linearities == 'tanh':
            x_new = nn.functional.tanh(x)
        elif self.non_linearities == 'exp':
            x_new = torch.exp(x)

        # Reshape projections to separate each projection for averaging
        x_new = x_new.reshape(x_new.shape[0], self.num_projection, self.output_dim)
        x_avg = x_new.mean(dim=1)

        #x = self.norm(x)
        return x_avg
    
############### Riêng phần này nếu để học đang gặp vấn đề rất lớn. Lý do ở đây là ta muốn nó hội tụ dần về tầm 0.001;
#  nhưng grad của nó lớn hơn nhiều so với giá trị lambda, do được scale sqrt(dim) nên dù có nhân với lr thì cx ko đủ đô.


'''   
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
            self.sqrt_d = math.sqrt(H)
            self.lambda_param = lambda_value  
        else:
            self.sqrt_d = 1 ##########
            self.lambda_param = nn.Parameter(torch.FloatTensor([0.001])) ### 0.2
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
        #x_new = self.batch_norm(x_new)
        return x_new
'''

class CNNRandomProjection(nn.Module):
    def __init__(self, C, H, W, lambda_value=None, resemble="partial", vector_based = 'column', non_linearities = 'leaky_relu', negative_slope_leaky_relu: float = 0.2, num_projection: int = 1):
        '''
        vector_based: decomposition vectors, i.e use column vectors
        resemble: use the same matrix U for all channels: full, partial, separate
        '''
        super(CNNRandomProjection, self).__init__()
        self.resemble = resemble
        self.base = vector_based
        self.C, self.H, self.W = C, H, W

        # Chọn kích thước của ma trận A: W nếu nhân theo hàng, H nếu nhân theo cột, C nếu theo channel
        if self.base == 'row':
            size = W 
            K1 = C
            K2 = H
        elif  self.base == 'column':
            size = H
            K1 = C
            K2 = W
        elif self.base == 'channel':
            size = C
            K1 = H
            K2 = W

        # Nếu resemble=full: dùng chung 1 ma trận A, nếu separate: mỗi kênh có ma trận A riêng, partial thì chung theo một chiều
        if resemble == "full":
            A = torch.randn(size, size)  
        elif resemble == "partial":
            A = torch.randn(K1, size, size)
        elif resemble == "separate":
            A = torch.randn(K1, K2, size, size)

        A.requires_grad = False
        self.sqrt_d = math.sqrt(size)
        self.A = A

        if lambda_value is not None:
            self.lambda_param = lambda_value
            self.clamp = False
        else:
            self.lambda_param = nn.Parameter(torch.tensor(0.3))
            self.clamp = False

        self.batch_norm = nn.BatchNorm2d(C)
        self.non_linearities = non_linearities
        self.negative_slope_leaky_relu = negative_slope_leaky_relu 

    def forward(self, x):
        """
        """
        self.A = self.A.to(x.device)
        if self.base == "column":
            # Nhân theo cột (N,C,:,W)
            if self.resemble == "full":
                x_new = torch.einsum('ih,nchw->nciw', self.A, x)
            elif self.resemble == "partial":
                x_new = torch.einsum('cih,nchw->nciw', self.A, x)
            elif self.resemble == "separate":
                x_new = torch.einsum('cwih,nchw->nciw', self.A, x)
        elif self.base == "row":
            # Nhân theo hàng (N,C,H,:) - permute
            if self.resemble == "full":
                x_new = torch.einsum('ih,nchw->nciw', self.A, x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            elif self.resemble == "partial":
                x_new = torch.einsum('cih,nchw->nciw', self.A, x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            elif self.resemble == "separate":
                x_new = torch.einsum('cwih,nchw->nciw', self.A, x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        elif self.base == "channel":
            # Nhân theo channel (N,C,H,:) - permute
            if self.resemble == "full":
                x_new = torch.einsum('ih,nchw->nciw', self.A, x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            elif self.resemble == "partial":
                x_new = torch.einsum('cih,nchw->nciw', self.A, x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            elif self.resemble == "separate":
                x_new = torch.einsum('cwih,nchw->nciw', self.A, x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Điều chỉnh giá trị lambda
        '''
        lambda_clamp = self.lambda_param.clone()
        if self.clamp:
            lambda_clamp = torch.clamp(lambda_clamp, min=0.01, max=0.5)
        '''
         # Break graph
        if self.clamp:
            self.lambda_param.data.clamp_(0.2, 0.5)

        # Áp dụng scale, kích hoạt và batch normalization
        x_new = x_new * self.lambda_param * self.sqrt_d
        if self.non_linearities == 'leaky_relu':
            x_new = nn.functional.leaky_relu(x_new, negative_slope=self.negative_slope_leaky_relu)
        elif self.non_linearities == 'sigmoid':
            x_new = nn.functional.sigmoid(x_new)
        elif self.non_linearities == 'tanh':
            x_new = nn.functional.tanh(x_new)
        elif self.non_linearities == 'exp':
            x_new = torch.exp(x_new)

        #x_new = self.batch_norm(x_new)
        
        return x_new
######################## channel_based vectors --> learnable lambda (0.01 -- 0.1) khởi tạo 0.05

### Torch sẽ tự khởi tạo, ko cần thiết
def init_weights(m):
    """Hàm khởi tạo trọng số cho các layer."""
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

class ClassificationModel(nn.Module):
    """Classification model with configurable architecture."""
    
    def __init__(self, model_type: ModelType, num_classes: int, use_rp: bool = False,
                 lambda_value: Optional[float] = None, use_cnn_rp: bool = False,
                 cnn_lambda_value: Optional[float] = None, num_input_channels: int = 3,
                 resemble: str = "partial", vector_based: str = 'column', pretrained: bool = False,
                 non_linearities: str = 'leaky_relu', negative_slope_leaky_relu: float = 0.2,
                 num_projection: int = 1):
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

        if model_type == ModelType.RESNET18:
            if num_input_channels == 3:
                if pretrained:
                    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                else:
                    base_model = resnet18()
                base_model.conv1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), ## No stride 2
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            elif num_input_channels == 1:
                base_model = resnet18()
                base_model.conv1 = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            base_model.maxpool = nn.Identity()   ## Maxpool --> Identity
            
            self.features = nn.Sequential(
                base_model.conv1,    
                base_model.bn1,
                base_model.relu,        
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,           
            )

            if use_cnn_rp:
                self.cnn_rp = CNNRandomProjection(256,8,8,lambda_value=self.cnn_lambda_value,resemble=resemble, 
                                                  vector_based= vector_based, non_linearities = non_linearities, 
                                                  negative_slope_leaky_relu= negative_slope_leaky_relu, num_projection = num_projection)
            
            self.features2 = nn.Sequential(
                base_model.layer4,
                base_model.avgpool
            )
            self.feature_dim = base_model.fc.in_features

            if use_rp:
                self.rp = RanPACLayer(self.feature_dim, self.feature_dim, lambda_value=self.lambda_value, 
                                      non_linearities = non_linearities, negative_slope_leaky_relu = negative_slope_leaky_relu, 
                                      num_projection = num_projection)
            
            self.features3 = nn.Sequential(
                nn.Linear(self.feature_dim, num_classes)
            )

        elif model_type == ModelType.VGG16:
            if num_input_channels == 3:
                if pretrained:
                    base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                else:
                    base_model = vgg16()
                base_model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)   
            elif num_input_channels == 1:
                base_model = vgg16()
                base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
           
            self.features = nn.Sequential(
                *list(base_model.features[:28])
            )
            self.features2 = nn.Sequential(
                *list(base_model.features[28:]),
                base_model.avgpool,
                nn.Flatten(),
                *list(base_model.classifier[:-1])
            )
            self.feature_dim = base_model.classifier[6].in_features  

            if use_rp:
                self.rp = RanPACLayer(self.feature_dim, self.feature_dim, self.lambda_value)
            if use_cnn_rp:
                self.cnn_rp = CNNRandomProjection(512,2,2,self.cnn_lambda_value,resemble=resemble, vector_based=vector_based)
            self.features3 = nn.Sequential(
                nn.BatchNorm1d(self.feature_dim),
                nn.Linear(self.feature_dim, num_classes)
            )

        ########### Chưa điều chỉnh gì ViT
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.features(x)
        if self.use_cnn_rp:
            x = self.cnn_rp(x)

        if isinstance(self.features, ViTModel):
            x = x.last_hidden_state[:, 0, :] ### Take the first element, [CLS] token
        else:
            x = self.features2(x) ## avgpool
            x = torch.flatten(x, 1) 

        if self.use_rp:
            x = self.rp(x) ###
        return self.features3(x)


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
        self.args = args ################
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.neptune_run = neptune_run
        self.checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        self.criterion = nn.CrossEntropyLoss()
        if self.args.optim == 'SGD':
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
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1) ## epoch-based


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
            if self.args.optim == 'SGD':
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

            if (i==len(self.train_loader)-1) and (self.neptune_run):
                if self.args.use_rp == True and self.args.lambda_value == None:
                    self.neptune_run["Lambda/Linear"].append(self.model.rp.lambda_param)
                    self.neptune_run["Grad_Lambda/Linear"].append(self.model.rp.lambda_param.grad)
                if self.args.use_cnn_rp == True and self.args.cnn_lambda_value == None:
                    self.neptune_run["Lambda/CNN"].append(self.model.cnn_rp.lambda_param)
                    self.neptune_run["Grad_Lambda/CNN"].append(self.model.cnn_rp.lambda_param.grad)

            if (i + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.args.optim == 'SGD':
                    self.scheduler.step()
                self.optimizer.zero_grad()

        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.args.optim == 'SGD':
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
            if self.args.optim == "Adam":
                self.scheduler.step()
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            '''
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
            '''


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
    if args.use_cnn_rp:
        exp_name += f"_CNN_RP{args.cnn_lambda_value}"
    exp_name += f"_lr{args.initial_lr}_optim{args.optim}_resemble{args.resemble}_vector_based{args.vector_based}_pretrained{args.pretrained}_bs{args.batch_size}_non_linearities{args.non_linearities}_num_proj{args.num_projection}_g{args.gradient_accumulation_steps}"
    if args.non_linearities == "leaky_relu":
        exp_name += f"_negative_slope_leaky_relu_{args.negative_slope_leaky_relu}"
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
    set_seed(42)

    """Main entry point for training script."""
    parser = argparse.ArgumentParser()
    # Model and dataset arguments
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="Model type")
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), required=True, help="Dataset type")
    parser.add_argument("--use_rp", type=bool, default=False, help="Use randomized projection")
    parser.add_argument("--lambda_value", type=float, default=None, help="Lambda value for RP")
    parser.add_argument("--num_projection", type=int, default=1, help="Number of projection for RP")
    parser.add_argument("--use_cnn_rp", type=bool, default=False, help="Use randomized projection for CNN layer")
    parser.add_argument("--cnn_lambda_value", type=float, default=None, help="Lambda value for RP in CNN layer")
    parser.add_argument("--resemble", type=str, default="partial", help="Same U matrix for RAMA in CNN layer")
    parser.add_argument("--vector_based", type=str, default='column', help="Vectorization followed columns or not")
    parser.add_argument("--pretrained", type=bool, default=False, help="Use pre-trained weights")
    parser.add_argument("--non_linearities", type=str, default='leaky_relu', help="Choose non-linear function")
    parser.add_argument("--negative_slope_leaky_relu", type=float, default=0.2, help="Choose value for negative_slope_leaky_relu")
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--initial_lr", type=float, default=0.1, help="Initial learning rate") 
    ###LEARNING_RATE = 1e-05 for Adam, 5e-3 for SGD 
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")

    #Optimization
    parser.add_argument("--optim", type=str, default='SGD', help="Choose optimization algorithm")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay coefficient")
    # SGD Optimization
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--nesterov", type=bool, default=True, help="Use Nesterov momentum")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    # Adam Optimization
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 in Adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 in Adam")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="epsilon in Adam")
    
    args = parser.parse_args()

    exp_name = get_experiment_name(args)
    exp_dir = setup_experiment_folders(exp_name)
    logger.info(f"Starting experiment: {exp_name}")
    
    config = {
        "model_type": args.model.value,
        "dataset_type": args.dataset.value,
        "use_rp": args.use_rp,
        "lambda_value": args.lambda_value,
        "use_cnn_rp": args.use_cnn_rp,
        "cnn_lambda_value": args.cnn_lambda_value,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "virtual_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.initial_lr,
        "epochs": args.epochs,
        "non_linearities": args.non_linearities,
        "num_projection": args.num_projection,
    }
    if args.optim == 'SGD':
        config.update({     "momentum": args.momentum,
                            "weight_decay": args.weight_decay,
                            "nesterov": args.nesterov,
                            "warmup_epochs": args.warmup_epochs,
                            "optim": args.optim
                    })
    else:
        args.weight_decay = 1e-5
        config.update({   
                            "weight_decay": args.weight_decay,
                            "beta1": args.beta1,
                            "beta2": args.beta2,
                            "epsilon": args.epsilon,
                            "optim": args.optim
                    })

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
        num_input_channels= 1 if args.dataset.value == "OmniBenchmark" else 3,
        resemble = args.resemble,
        vector_based = args.vector_based,
        pretrained = args.pretrained,
        non_linearities = args.non_linearities,
        negative_slope_leaky_relu = args.negative_slope_leaky_relu,
        num_projection = args.num_projection,
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

#### log lambda, survey  --> clip lambda
#### different position - lay 3, lay 2, lay 1
#### smaller architecture -- resnet18
#### column + row -- just column
#### Adam

############ Cần sửa lớp features cho VGG16 - ko có batchnorm -- exploding khi train lại từ đầu
