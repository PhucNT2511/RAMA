import argparse
import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import neptune
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from bayes_opt import BayesianOptimization, acquisition
from tqdm import tqdm
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
#MAX_lambda_value = 10
#MIN_lambda_value = 1e-3
NEPTUNE_PRJ_NAME = "phuca1tt1bn/RAMA"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODZlNDU0Yy1iMDk0LTQ5MDEtOGNiYi00OTZlYTY4ODI0MzgifQ=="

##### RAMA with Gaussian for U matrix instead of norm distribution
class GaussianRAMALayer(nn.Module):
    """
    A RAMA layer using Gaussian distribution for random projections.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        use_normalization (bool): Whether to apply layer normalization after projection.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
    """
    def __init__(self, input_dim, output_dim, lambda_value=1.0, 
                 use_normalization=True, activation="relu", sqrt_dim=False):
        super(GaussianRAMALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value

        self.sqrt_d = 1
        if sqrt_dim == True:
            self.sqrt_d = math.sqrt(input_dim)

        self.linear = nn.Linear(input_dim, input_dim*2, bias=False)
        
        '''
        projection1 = torch.randn(input_dim, input_dim//2)
        self.projection1 = nn.Parameter(projection1, requires_grad=False)

        projection2 = torch.randn(input_dim//2, output_dim)
        self.projection2 = nn.Parameter(projection2, requires_grad=False)

        # Add layer normalization for stabilizing the output distribution.
        if use_normalization:
            self.norm1 = nn.LayerNorm(input_dim//2)
            self.norm2 = nn.LayerNorm(output_dim)
        '''

        '''
        #Orthodological projection using QR decomposition. This ensures the projection matrix is orthogonal, which can help with stability.
        # Giả sử input_dim ≥ output_dim
        M = torch.randn(input_dim, input_dim)
        Q, R = torch.qr(M)                      # Q: [input_dim, input_dim], orthogonal
        projection = Q[:, :output_dim].contiguous()  # Lấy  output_dim cột đầu của Q
        '''

        projection = torch.randn(input_dim*2, output_dim*2)
        self.projection = nn.Parameter(projection, requires_grad=False)

        if use_normalization:
            self.norm = nn.LayerNorm(output_dim*2)
        
        self.lambda2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # Learnable lambda for the second part of the projection


    def forward(self, x, lambda_value):
        """
        Forward pass through the Gaussian RAMA layer.
        
        Args:
            x: Input tensor
            lambda_value: lambda
        """

        out = self.linear(x)
               
        out = out @ self.projection

        out *= self.sqrt_d * self.lambda_value

        if self.use_normalization:
            out = self.norm(out)

        # Apply activation function
        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "leaky_relu":
            out = F.leaky_relu(out, negative_slope=0.01)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        elif self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "silu":
            out = torch.nn.functional.silu(out)
        elif self.activation == "gelu":
            out = torch.nn.functional.gelu(out)
            
        #return out
        
        out1, out2 = torch.chunk(out, 2, dim=-1)
        lambda2 = torch.clamp(self.lambda2, 0.2, 0.8)
        return out1 - lambda2 * out2
        

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for ResNet architecture.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for convolution. Default: 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv_block(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Modified ResNet architecture with Gaussian RAMA layers at multiple positions.
    
    Args:
        block (nn.Module): Block type to use for the network.
        num_blocks (List[int]): Number of blocks in each layer.
        num_classes (int): Number of output classes. Default: 100.
        use_rama (bool): Whether to use RAMA layers. Default: False.
        rama_config (dict): Configuration for RAMA layers. Default: None.
    """
    def __init__(self, block, num_blocks, num_classes=100, use_rama=False, rama_config=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_rama = use_rama
        
        if rama_config is None:
            rama_config = {
                    "lambda_value": 1.0,  # This lambda_value for Gaussian
                    "activation": "relu",
                    "use_normalization": False,
                    "sqrt_dim": False,
                }
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Create standard ResNet blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.feature_dim = 512
        
        # Create Gaussian RAMA layers for different positions in the network
        if use_rama:
            # RAMA after layer2 (128 features)
            self.rama_layer2 = GaussianRAMALayer(
                128, 
                128, 
                rama_config['lambda_value'], 
                rama_config.get('use_normalization', False),
                rama_config.get('activation', 'relu'),
                rama_config.get('sqrt_dim', False),
            )
            
            # RAMA after layer3 (256 features)
            self.rama_layer3 = GaussianRAMALayer(
                256, 
                256, 
                rama_config['lambda_value'], 
                rama_config.get('use_normalization', False),
                rama_config.get('activation', 'relu'),
                rama_config.get('sqrt_dim', False),
            )
            
            # RAMA before final classification
            self.rama_layer4 = GaussianRAMALayer(
                self.feature_dim, 
                self.feature_dim, 
                rama_config['lambda_value'], 
                rama_config.get('use_normalization', False),
                rama_config.get('activation', 'relu'),
                rama_config.get('sqrt_dim', False),
            )
            
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize hooks for feature extraction
        self.hooks = []
        self.before_rama_features = None
        self.after_rama_features = None

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with the specified number of blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, lambda_value):
        """Forward pass through the ResNet model with Gaussian RAMA layers."""
        out = self.conv1(x)
        out = self.layer1(out)
        
        # After layer 1
        out = self.layer2(out)
        
        # # Apply RAMA after layer2 (early in the network)
        # if self.use_rama:
        #     # Need to flatten, apply RAMA, then reshape back
        #     batch_size, channels, h, w = out.shape
        #     out_flat = out.view(batch_size, channels, -1).mean(dim=2)  # Global avg pooling per channel
        #     out_flat = self.rama_layer2(out_flat, lambda_value)
        #     # Broadcast back to spatial dimensions
        #     out = out * out_flat.view(batch_size, channels, 1, 1)

        out = self.layer3(out)
        
        # # Apply RAMA after layer3 (middle of the network)
        # if self.use_rama:
        #     batch_size, channels, h, w = out.shape
        #     out_flat = out.view(batch_size, channels, -1).mean(dim=2)
        #     out_flat = self.rama_layer3(out_flat, lambda_value)
        #     out = out * out_flat.view(batch_size, channels, 1, 1)
            
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        # Store features before RAMA for evaluation
        if self.use_rama:
            self.before_rama_features = out.detach().clone()
            
        # Apply RAMA before final classification (original position)
        if self.use_rama:
            out = self.rama_layer4(out, lambda_value)
            # out = self.rama_layer4(out, None)
            # Store features after RAMA for evaluation
            self.after_rama_features = out.detach().clone()
        out = self.fc(out)
        return out

    def forward_with_features(self, x, lambda_value):
        """
        Forward pass that returns both output and features before/after RAMA.
        Useful for analyzing feature quality.
        """
        outputs = self.forward(x, lambda_value)
        if self.use_rama:
            return outputs, self.before_rama_features, self.after_rama_features
        else:
            return outputs, None, None


class DataManager:
    """
    Manager for CIFAR-100 dataset preparation and loading.
    
    Args:
        data_dir (str): Directory to store/load dataset.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading. Default: 2.
    """
    def __init__(self, data_dir, batch_size, num_workers=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), 
                                 (0.2673, 0.2564, 0.2762))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), 
                                 (0.2673, 0.2564, 0.2762))
        ])
        
    def get_loaders(self):
        """
        Get data loaders for training and testing.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

        testset = torchvision.datasets.CIFAR100(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        return trainloader, testloader


class Trainer:
    """
    Improved trainer class for model training and evaluation with feature quality metrics.
    
    Args:
        model (nn.Module): Model to train.
        trainloader (DataLoader): Training data loader.
        testloader (DataLoader): Testing data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use for training.
        checkpoint_dir (str): Directory to save checkpoints.
        bayes_opt_config (dict): Configuration for Bayesian optimization.
        neptune_run: Neptune.ai run instance
    """
    def __init__(self, model, trainloader, testloader, criterion, optimizer, 
                 device, checkpoint_dir, lambda_value=0.01, bayes_opt_config=None, use_rama: bool = False,
                 use_hyperparameter_optimization: bool = False,
                 neptune_run: Optional[neptune.Run] = None, writer: Optional[SummaryWriter] = None):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.best_acc = 0
        self.neptune_run = neptune_run
        self.writer = writer
        self.use_rama = use_rama
        self.use_hyperparameter_optimization = use_hyperparameter_optimization
        self.best_lambda = None
        self.lambda_ = lambda_value
        
        if self.use_rama:
            # Default Bayesian optimization configuration
            self.bayes_opt_config = bayes_opt_config

            # Initialize Bayesian optimizer with correct bounds
            if self.bayes_opt_config["acq"] == "ei":
                acq = acquisition.ExpectedImprovement(xi=self.bayes_opt_config["xi"])
            elif self.bayes_opt_config["acq"] == "poi":
                acq = acquisition.ProbabilityOfImprovement(xi=self.bayes_opt_config["xi"])
            elif self.bayes_opt_config["acq"] == "ucb":
                acq = acquisition.UpperConfidenceBound(kappa=self.bayes_opt_config["kappa"])
            else:
                raise ValueError("Invalid acquisition function specified.")
                
            self.bayesian_optimizer = BayesianOptimization(
                f=self.evaluate_lambda,
                acquisition_function=acq,
                pbounds={"lambda_value": (self.bayes_opt_config["lambda_min"], self.bayes_opt_config["lambda_max"])},
                random_state=42,
                verbose=2
            )

    def optimize_lambda(self, n_warmup=None, n_iter=None):
        """
        Run Bayesian optimization to find the best lambda value.
        
        Args:
            n_warmup (int): Number of random points to evaluate. Default: from config.
            n_iter (int): Number of iterations. Default: from config.
            
        Returns:
            tuple: (best_lambda, best_score, optimization_results)
        """
        if n_warmup is None:
            n_warmup = self.bayes_opt_config["init_points"]
        if n_iter is None:
            n_iter = self.bayes_opt_config["n_iter"]
            
        logger.info(f"Running Bayesian optimization with {n_warmup} initialization points and {n_iter} iterations...")
        
        # Run optimization
        self.bayesian_optimizer.maximize(
            init_points=n_warmup,
            n_iter=n_iter
        )
        
        best_lambda = self.bayesian_optimizer.max["params"]["lambda_value"]
        best_score = self.bayesian_optimizer.max["target"]
        return best_lambda, best_score, self.bayesian_optimizer.res

    def load_optimizer_state(self, path):
        """
        Load Bayesian optimizer state from a file.
        
        Args:
            path (str): Path to the state file.
        """
        if os.path.exists(path):
            logger.info(f"Loading BayesOpt state from {path}")
            self.bayesian_optimizer.load_state(path)
            logger.info(f"Loaded max value: {self.bayesian_optimizer.max}")

    def train_one_epoch(self, lambda_value=None):
        """
        Train the model for one epoch.
        
        Returns:
            tuple: (train_loss, train_accuracy)
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.trainloader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs, lambda_value=lambda_value)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({
                "loss": train_loss / (batch_idx + 1),
                "acc": 100. * correct / total
            })
        return train_loss / len(self.trainloader), 100. * correct / total

    def evaluate(self, lambda_value=None):
        """
        Basic evaluation of the model on the test set.
        
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(self.testloader, desc="Testing")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model.forward(inputs, lambda_value=lambda_value)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': test_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        return test_loss / len(self.testloader), 100. * correct / total

    def evaluate_lambda(self, lambda_value):
        """
        Evaluation function for Bayesian optimization.
        Returns a single scalar value (accuracy).
        """
        _, test_acc = self.evaluate(lambda_value)
        return test_acc  # Return only accuracy for optimization

    def evaluate_with_metrics(self, lambda_value=None):
        """
        Evaluate the model with additional metrics to understand RAMA impact.
        
        Returns:
            dict: Evaluation metrics including feature quality
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize containers for feature analysis
        features_original = []
        features_after_rama = []
        class_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Use the forward pass that returns features
                if self.use_rama:
                    outputs, before_features, after_features = self.model.forward_with_features(inputs, lambda_value)
                    if before_features is not None and after_features is not None:
                        features_original.append(before_features.cpu())
                        features_after_rama.append(after_features.cpu())
                        class_labels.append(targets.cpu())
                else:
                    # outputs = self.model.forward(inputs, lambda_value)
                    outputs, before_features, after_features = self.model.forward_with_features(inputs, None)
                    if before_features is not None and after_features is not None:
                        features_original.append(before_features.cpu())
                        features_after_rama.append(after_features.cpu())
                        class_labels.append(targets.cpu())
                
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate standard metrics
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.testloader)
        
        # Analyze features if we captured them
        feature_metrics = {}
        if features_original and features_after_rama:
            # Concatenate batches
            features_original = torch.cat(features_original, dim=0)
            features_after_rama = torch.cat(features_after_rama, dim=0)
            class_labels = torch.cat(class_labels, dim=0)
            
            # Calculate feature separability before and after RAMA
            feature_metrics = self._calculate_feature_metrics(
                features_original, features_after_rama, class_labels)
            
            # Log feature metrics
            if self.neptune_run:
                for key, value in feature_metrics.items():
                    self.neptune_run[f"Feature/{key}"].append(value)
            
            if self.writer:
                for key, value in feature_metrics.items():
                    self.writer.add_scalar(f"Feature/{key}", value)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'feature_metrics': feature_metrics
        }

    def _calculate_feature_metrics(self, features_before, features_after, labels):
        """
        Calculate metrics to assess feature quality.
        
        Args:
            features_before: Features before RAMA application
            features_after: Features after RAMA application
            labels: Class labels
            
        Returns:
            dict: Feature quality metrics
        """
        metrics = {}
        
        # 1. Calculate intra-class and inter-class distances before RAMA
        intra_dist_before, inter_dist_before = self._calculate_class_distances(features_before, labels)
        
        # 2. Calculate intra-class and inter-class distances after RAMA
        intra_dist_after, inter_dist_after = self._calculate_class_distances(features_after, labels)
        
        # 3. Fisher's criterion (inter-class separation / intra-class spread)
        fisher_before = inter_dist_before / (intra_dist_before + 1e-8)
        fisher_after = inter_dist_after / (intra_dist_after + 1e-8)
        
        metrics['intra_class_distance_before'] = intra_dist_before
        metrics['inter_class_distance_before'] = inter_dist_before
        metrics['intra_class_distance_after'] = intra_dist_after
        metrics['inter_class_distance_after'] = inter_dist_after
        metrics['fisher_ratio_before'] = fisher_before
        metrics['fisher_ratio_after'] = fisher_after
        metrics['fisher_improvement'] = fisher_after / fisher_before
        return metrics
    
    def _calculate_class_distances(self, features, labels):
        """
        Calculate average intra-class and inter-class distances.
        
        Args:
            features: Feature vectors
            labels: Class labels
            
        Returns:
            tuple: (average intra-class distance, average inter-class distance)
        """
        unique_classes = torch.unique(labels)
        class_means = []
        intra_class_distances = []
        
        # Calculate class centroids and intra-class distances
        for cls in unique_classes:
            cls_features = features[labels == cls]
            cls_mean = cls_features.mean(dim=0)
            class_means.append(cls_mean)
            
            # Average distance from each point to its class centroid
            dists = torch.norm(cls_features - cls_mean, dim=1).mean()
            intra_class_distances.append(dists.item())
        
        # Convert to tensor for easier computation
        class_means = torch.stack(class_means)
        
        # Calculate inter-class distances (between centroids)
        n_classes = len(unique_classes)
        inter_dists = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                dist = torch.norm(class_means[i] - class_means[j])
                inter_dists.append(dist.item())
        
        avg_intra_dist = np.mean(intra_class_distances)
        avg_inter_dist = np.mean(inter_dists) if inter_dists else 0
        return avg_intra_dist, avg_inter_dist

    def save_checkpoint(self, state, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            state (dict): State to save.
            is_best (bool): Whether this is the best model so far.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state, checkpoint_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            logger.info(f"Saving best model to {best_path}")
            torch.save(state, best_path)
    
    def train(self, epochs, start_epoch=0):
        """
        Train the model for multiple epochs with feature quality evaluation.
        
        Args:
            epochs (int): Number of epochs to train.
            start_epoch (int): Starting epoch. Default: 0.
            
        Returns:
            float: Best test accuracy.
        """
        # First, run initial optimization to find a good p
        if self.use_rama and self.use_hyperparameter_optimization:
            if start_epoch == 0:
                logger.info("Running initial Bayesian optimization...")
                self.best_lambda, _, _ = self.optimize_lambda()
            else:
                # Try to load previous optimization state
                bayes_opt_path = os.path.join(self.checkpoint_dir, 'bayes_opt_state.json')
                self.load_optimizer_state(bayes_opt_path)
                if self.bayesian_optimizer.max:
                    self.best_lambda = self.bayesian_optimizer.max["params"]["lambda_value"]
                    logger.info(f"Loaded best lambda from previous run: {self.best_lambda:.6f}")
                else:
                    logger.info("No previous optimization state found, running initial optimization...")
                    self.best_lambda, _, _ = self.optimize_lambda()

        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch: {epoch+1}/{epochs}")

            self.lambda_ = (3.0 - 0.01) / epochs * (epochs - 1 - epoch) / (epochs - 1) + 0.01

            # Train with best p
            #train_loss, train_acc = self.train_one_epoch(lambda_value=self.best_lambda)
            train_loss, train_acc = self.train_one_epoch(lambda_value=self.lambda_)
            
            # Basic evaluation
            #test_loss, test_acc = self.evaluate(lambda_value=self.best_lambda)
            test_loss, test_acc = self.evaluate(lambda_value=self.lambda_)
            
            # Detailed evaluation with feature metrics (once every 5 epochs to save time)
            # if epoch % 5 == 0 or epoch == epochs - 1:
            #metrics = self.evaluate_with_metrics(lambda_value=self.best_lambda)
            metrics = self.evaluate_with_metrics(lambda_value=self.lambda_)
            if 'feature_metrics' in metrics and metrics['feature_metrics']:
                feature_metrics = metrics['feature_metrics']
                logger.info(f"Feature metrics at epoch {epoch+1}:")
                logger.info(f"  Fisher ratio before RAMA: {feature_metrics['fisher_ratio_before']:.4f}")
                logger.info(f"  Fisher ratio after RAMA: {feature_metrics['fisher_ratio_after']:.4f}")
                logger.info(f"  Fisher improvement: {feature_metrics['fisher_improvement']:.4f}x")

            # Perform Bayesian optimization periodically.
            if (self.use_rama and self.use_hyperparameter_optimization and
                    epoch % self.bayes_opt_config["optimize_every"] == 0 and epoch > 0):
                logger.info(f"Running Bayesian optimization at epoch {epoch+1}...")
                # Use fewer iterations for subsequent optimizations.
                lambda_value, bayesian_score, results = self.optimize_lambda(
                    n_warmup=max(2, self.bayes_opt_config["init_points"] // 2), 
                    n_iter=max(5, self.bayes_opt_config["n_iter"] // 2)
                )
                if bayesian_score >= test_acc:
                    self.best_lambda = lambda_value
                    logger.info(f"Updated best lambda: {self.best_lambda:.4f} with accuracy: {bayesian_score:.2f}%")
                    
                    # Update the search bounds based on the best lambda found.
                    lambda_min_distance = abs(lambda_value - self.bayes_opt_config["lambda_min"])
                    lambda_max_distance = abs(lambda_value - self.bayes_opt_config["lambda_max"])
                    
                    # If closer to min bound, expand upper bound.
                    if lambda_min_distance < lambda_max_distance:
                        new_max = min(self.best_lambda * 1.5, self.bayes_opt_config["lambda_max"] * 2)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"lambda_value": (self.bayes_opt_config["lambda_min"], new_max)}
                        )
                    # If closer to max bound, expand lower bound.
                    else:
                        new_min = max(self.best_lambda * 0.5, self.bayes_opt_config["lambda_min"] * 0.5)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"lambda_value": (new_min, self.bayes_opt_config["lambda_max"])}
                        )

            # Log metrics.
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            if self.use_rama and self.use_hyperparameter_optimization:
                logger.info(f"Current lambda value: {self.best_lambda:.6f}")

            # Log to Neptune if available.
            if self.neptune_run:
                self.neptune_run["Train/Loss"].append(train_loss)
                self.neptune_run["Train/Accuracy"].append(train_acc)
                self.neptune_run["Test/Loss"].append(test_loss)
                self.neptune_run["Test/Accuracy"].append(test_acc)
                if self.use_rama and self.use_hyperparameter_optimization:
                    self.neptune_run["RAMA_LAMBDA"].append(self.best_lambda)
                '''
                if self.use_rama:
                    self.neptune_run["Lambda2"].append(self.model.rama_layer4.lambda2.item())
                '''
            # Log to TensorBoard if available.
            if self.writer:
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Accuracy", test_acc, epoch)
                if self.use_rama and self.use_hyperparameter_optimization:
                    self.writer.add_scalar("RAMA_LAMBDA", self.best_lambda, epoch)

            # Save checkpoint if best model.
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
            self.save_checkpoint({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "best_lambda": self.best_lambda,
            }, is_best)
        logger.info(f"Best test accuracy: {self.best_acc:.2f}%")
        return self.best_acc


def resnet18(num_classes=100, use_rama=False, rama_config=None):
    """
    Create a ResNet-18 model with optional Gaussian RAMA layers.
    
    Args:
        num_classes (int): Number of output classes. Default: 100.
        use_rama (bool): Whether to use RAMA layers. Default: False.
        rama_config (dict): Configuration for RAMA layers. Default: None.
        
    Returns:
        ResNet: ResNet-18 model.
    """
    model = ResNet(
        ResidualBlock, 
        [2, 2, 2, 2], 
        num_classes=num_classes,
        use_rama=use_rama,
        rama_config=rama_config
    )
    return model


def get_experiment_name(args: argparse.Namespace) -> str:
    """Generate a unique experiment name based on configuration."""
    exp_name = "ResNet18"
    exp_name += "_GaussianRAMA" if args.use_rama else "_NoRAMA"
    
    if args.use_rama:
        exp_name += "_norm" if args.use_normalization else "_nonorm"
        exp_name += f"_{args.activation}"
        exp_name += "_sqrt_d_True" if args.sqrt_dim else "_sqrt_d_False"

        
    exp_name += f"_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}"
    
    if args.use_rama:
        exp_name += f"_lambda_{args.lambda_value}" 
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


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training with Resnet and Gaussian RAMA Layers')
    
    # Training parameters
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Gaussian RAMA configuration
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA layers')
    parser.add_argument('--use-hyperparameter-optimization', action='store_true', help='whether to use Bayesian optimization for p-value')
    parser.add_argument('--lambda-value', default=0.01, type=float, help='Lambda_value for RAMA')
    parser.add_argument('--sqrt-dim', default= False, help='Whether multiply with sqrt(d) or not')
    parser.add_argument('--use-normalization', action='store_true', help='use layer normalization in RAMA layers')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'silu', 'gelu'],
                        help='activation function for RAMA layers')
    
    # Bayesian optimization parameters - adjusted for probability range
    parser.add_argument('--lambda-min', default=0.001, type=float, help='minimum Lambda value for optimization')
    parser.add_argument('--lambda-max', default=10, type=float, help='maximum Lambda value for optimization')
    parser.add_argument('--bayes-init-points', default=5, type=int, help='number of initial points for Bayesian optimization')
    parser.add_argument('--bayes-n-iter', default=15, type=int, help='number of iterations for Bayesian optimization')
    parser.add_argument('--bayes-acq', default="ei", choices=["ucb", "ei", "poi"], help='acquisition function for Bayesian optimization')
    parser.add_argument('--bayes-xi', default=0.01, type=float, help='exploration-exploitation parameter for ei/poi')
    parser.add_argument('--bayes-kappa', default=2.5, type=float, help='exploration-exploitation parameter for ucb')
    parser.add_argument('--optimize-every', default=5, type=int, help='optimize Lambda every N epochs')
    return parser.parse_args()


def main():
    """Main function for training and evaluating the model."""
    args = parse_args()
    logger.info(f"Starting training with arguments: {args}")
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        logger.info(f"Created checkpoint directory: {args.checkpoint_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare data
    data_manager = DataManager(args.data_dir, args.batch_size, args.num_workers)
    trainloader, testloader = data_manager.get_loaders()
    
    # Gaussian RAMA configuration
    rama_config = {
        "lambda_value": args.lambda_value,  # This lambda_value for Gaussian
        "activation": args.activation,
        "use_normalization": args.use_normalization,
        "sqrt_dim": args.sqrt_dim,
    }
    
    # Create model
    model = resnet18(
        num_classes=100, 
        use_rama=args.use_rama,
        rama_config=rama_config
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=5e-4
    ) 

    '''
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    '''
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            logger.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            logger.warning(f"No checkpoint found at '{checkpoint_path}'")

    # Bayesian optimization configuration - adjusted for probability bounds
    bayes_opt_config = {
        "init_points": args.bayes_init_points,
        "n_iter": args.bayes_n_iter,
        "acq": args.bayes_acq,
        "xi": args.bayes_xi,
        "kappa": args.bayes_kappa,
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "optimize_every": args.optimize_every,
    }

    # Set up experiment tracking
    exp_name = get_experiment_name(args)
    if NEPTUNE_PRJ_NAME and NEPTUNE_API_TOKEN:
        neptune_run = neptune.init_run(project=NEPTUNE_PRJ_NAME, api_token=NEPTUNE_API_TOKEN, name=exp_name)
        neptune_run["config"] = vars(args)
        neptune_run["bayes_config"] = bayes_opt_config
        neptune_run["rama_config"] = rama_config
    else:
        neptune_run = None
    
    # Set up TensorBoard
    exp_dir = setup_experiment_folders(exp_name)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs"))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        lambda_value=args.lambda_value,
        use_hyperparameter_optimization =args.use_hyperparameter_optimization,
        checkpoint_dir=args.checkpoint_dir,
        bayes_opt_config=bayes_opt_config,
        use_rama=args.use_rama,
        neptune_run=neptune_run,
        writer=writer
    )
    
    # Set best accuracy if resuming
    if args.resume and best_acc > 0:
        trainer.best_acc = best_acc
        
    # Train model
    best_acc = trainer.train(args.epochs, start_epoch)
    
    # Clean up
    writer.close()
    if neptune_run:
        neptune_run.stop()
        
    logger.info(f"Training completed! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
