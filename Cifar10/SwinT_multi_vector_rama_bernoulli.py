import argparse
import logging
import os
import random
from typing import Optional, List, Tuple

import numpy as np
import torch
import neptune
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.models import swin_t
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
MAX_p_value = 1
MIN_p_value = 1e-2  # Expanded lower bound for p-value
NEPTUNE_PRJ_NAME = "phuca1tt1bn/RAMA"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODZlNDU0Yy1iMDk0LTQ5MDEtOGNiYi00OTZlYTY4ODI0MzgifQ=="

class BernoulliRAMALayer(nn.Module):
    """
    An improved RAMA layer using Bernoulli distribution for random projections.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension (can be smaller than input_dim for dimension reduction).
        p_value (float): Controls the Bernoulli parameter p.
        values (str): Values for Bernoulli: '0_1' for {0,1} or '-1_1' for {-1,1}.
        use_normalization (bool): Whether to apply layer normalization after projection.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
        lambda_value (float): Scaling factor for projection output.
        sqrt_dim (bool): Whether to normalize by sqrt(input_dim).
        dim_reduction_factor (float): Factor to reduce dimensions by (only used if output_dim is None).
    """
    def __init__(self, input_dim, output_dim=None, p_value=0.5, values='0_1', use_normalization=True, 
                 activation="relu", lambda_value=1.0, sqrt_dim=False, dim_reduction_factor=1.0):
        super(BernoulliRAMALayer, self).__init__()
        self.input_dim = input_dim
        
        # Apply dimension reduction if output_dim is None
        if output_dim is None:
            self.output_dim = max(int(input_dim * dim_reduction_factor), 32)  # Ensure minimum dimension
        else:
            self.output_dim = output_dim
            
        self.values = values
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value
        self.sqrt_dim = sqrt_dim
        if sqrt_dim:
            self.sqrt_d = math.sqrt(input_dim)
        else:
            self.sqrt_d = 1

        # Initialize Bernoulli projection matrix
        if values == '0_1':
            projection = (torch.rand(input_dim, self.output_dim) < p_value).float()
        elif values == '-1_1':
            projection = 2 * (torch.rand(input_dim, self.output_dim) < p_value).float() - 1
        else:
            raise ValueError(f"Unknown values: {values}. Use '0_1' or '-1_1'")

        self.projection = nn.Parameter(projection, requires_grad=False)
        
        # Add layer normalization for stabilizing the output distribution
        if use_normalization:
            self.norm = nn.LayerNorm(self.output_dim)
            
        self.current_mask = None  # For fixed mask per N epochs
        self.current_p = None

    def update_mask(self, p_value):
        """Update the fixed mask for the current epoch."""
        if self.values == '0_1':
            mask = (torch.rand_like(self.projection) < p_value).float()
        elif self.values == '-1_1':
            mask = 2 * (torch.rand_like(self.projection) < p_value).float() - 1
        else:
            raise ValueError(f"Unknown values: {self.values}. Use '0_1' or '-1_1'")
        self.current_mask = mask
        self.current_p = p_value

    def forward(self, x, p_value):
        """
        Forward pass through the Bernoulli RAMA layer.
        
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, channels, height, width)
            p_value: Value controlling the Bernoulli parameter p
        """
        original_shape = x.shape
        
        # Handle 4D tensors (batch_size, channels, height, width)
        if len(original_shape) == 4:
            # Reshape to 2D: (batch_size, channels*height*width)
            batch_size = original_shape[0]
            x = x.view(batch_size, -1)
        
        # Generate a dynamic Bernoulli mask based on p_value
        if p_value is not None and self.training:
            # Use the fixed mask if available and p_value matches, else update
            if self.current_mask is None or self.current_p != p_value:
                self.update_mask(p_value)
            out = x @ self.current_mask
        else:
            out = x @ self.projection

        # Apply correct scaling
        out = out * self.lambda_value
        if self.sqrt_dim:
            # Normalize by sqrt(input_dim)
            out = out / self.sqrt_d
        
        # Apply normalization if specified
        if self.use_normalization:
            out = self.norm(out)

        # Apply activation function
        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "leaky_relu":
            out = F.leaky_relu(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        elif self.activation == "sigmoid":
            out = torch.sigmoid(out)
            
        # For 4D tensors, reshape back to proper spatial dimensions
        if len(original_shape) == 4:
            # Calculate new spatial dimensions to maintain tensor size
            C, H, W = original_shape[1], original_shape[2], original_shape[3]
            total_elements = self.output_dim
            
            # Find approximate spatial dimensions preserving aspect ratio
            ratio = W / H
            new_h = int(np.sqrt(total_elements / ratio))
            new_w = int(total_elements / new_h)
            
            # Adjust to ensure exactly output_dim elements
            while new_h * new_w != total_elements:
                if new_h * new_w < total_elements:
                    new_w += 1
                else:
                    new_w -= 1
                
                if new_w <= 0:
                    new_h -= 1
                    new_w = int(total_elements / new_h)
            
            # Reshape to (batch_size, channels, new_height, new_width)
            # For simplicity, we'll use channel=1
            out = out.view(batch_size, 1, new_h, new_w)
        
        return out


class SwinTRAMAAdapter(nn.Module):
    """
    Specialized adapter module for integrating RAMA with SwinT architecture.
    This preserves the hierarchical structure of SwinT features while applying
    RAMA in a way that's compatible with shifted window attention mechanisms.
    
    Args:
        in_features (int): Number of input features/channels
        out_features (int): Number of output features/channels (if None, uses reduction factor)
        rama_config (dict): Configuration for RAMA layers
        stage_idx (int): Index of the SwinT stage (0-3) this adapter is applied to
    """
    def __init__(self, in_features, out_features=None, rama_config=None, stage_idx=0):
        super().__init__()
        
        if rama_config is None:
            rama_config = {}
            
        if out_features is None:
            # More aggressive dimension reduction for earlier stages
            reduction_factor = rama_config.get('dim_reduction_factor', 1.0)
            # Apply slightly different reduction factors based on stage
            out_features = max(int(in_features * reduction_factor), 16)
        
        self.in_features = in_features
        self.out_features = out_features
        self.stage_idx = stage_idx
        
        # Create RAMA layer for feature dimension reduction
        self.rama_layer = BernoulliRAMALayer(
            input_dim=in_features,
            output_dim=out_features,
            p_value=rama_config.get('p_value', 0.5),
            values=rama_config.get('values', '0_1'),
            use_normalization=rama_config.get('use_normalization', True),
            activation=rama_config.get('activation', 'relu'),
            lambda_value=rama_config.get('lambda_value', 1.0),
            sqrt_dim=rama_config.get('sqrt_dim', False),
        )
        
        # Add projection layer to maintain compatibility with SwinT architecture
        self.proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, p_value):
        """
        Forward pass applying RAMA to SwinT feature maps.
        
        Args:
            x: Input tensor from SwinT stage (batch_size, C, H, W)
            p_value: Value for the RAMA layer
        """
        batch_size, H, W, C = x.shape
        
        # Step 1: Apply RAMA to each token's feature vector
        # Reshape to (B*seq_len, C) for RAMA processing
        x = x.reshape(batch_size, -1, C)
        x = self.rama_layer(x, p_value)
        
        # Step 3: Reshape back to sequence form with new feature dimension
        x = x.reshape(batch_size, H * W, self.out_features)
        
        # Step 4: Apply projection and normalization (similar to SwinT's approach)
        x = self.proj(x)
        x = self.norm(x)

        # Step 5: Reshape back to feature map form for subsequent SwinT stages
        # From (B, H*W, C_new) to (B, C_new, H, W)
        x = x.transpose(1, 2).reshape(batch_size, H, W, self.out_features)
        return x


class ImprovedSwinT(nn.Module):
    """
    Improved SwinT architecture with Bernoulli RAMA layers at configurable positions.
    
    Args:
        num_classes (int): Number of output classes
        use_rama (bool): Whether to use RAMA layers
        rama_config (dict): Configuration for RAMA layers
    """
    def __init__(self, num_classes=10, use_rama=False, rama_config=None):
        super().__init__()
        self.use_rama = use_rama
        self.num_classes = num_classes

        if rama_config is None:
            rama_config = {
                "p_value": 0.5,
                "values": '0_1',
                "activation": "leaky_relu",
                "use_normalization": True,
                'lambda_value': 1.0,
                'sqrt_dim': False,
                'dim_reduction_factor': 1.0,
                'positions': ['stage1', 'stage2', 'stage3', 'stage4', 'final']  # Default: apply to all positions
            }
            
        # Parse positions if provided as string
        if 'positions' in rama_config and isinstance(rama_config['positions'], str):
            rama_config['positions'] = rama_config['positions'].split(',')

        # Initialize backbone
        self.backbone = swin_t(weights=None)
        self.feature_dim = self.backbone.head.in_features  # Should be 768 for SwinT
        
        # Extract layers from SwinT backbone for more granular control
        self.patch_embed = self.backbone.features[0]
        
        # Stage 1: Blocks operating on 56x56 with 96 channels
        self.stage1 = self.backbone.features[1]
        # Stage 2: Blocks operating on 28x28 with 192 channels
        self.patch_merging1 = self.backbone.features[2]
        self.stage2 = self.backbone.features[3]
        # Stage 3: Blocks operating on 14x14 with 384 channels
        self.patch_merging2 = self.backbone.features[4]
        self.stage3 = self.backbone.features[5]
        # Stage 4: Blocks operating on 7x7 with 768 channels
        self.patch_merging3 = self.backbone.features[6]
        self.stage4 = self.backbone.features[7]
        
        self.norm = self.backbone.features[8] if len(self.backbone.features) > 8 else nn.Identity()
        
        # Track which positions have RAMA layers
        self.rama_positions = rama_config.get('positions', [])
        
        # Create RAMA layers for multiple stages if enabled
        if use_rama:
            # Define dimensions for each stage - SwinT stages
            stage_dims = [96, 192, 384, 768]  # Channel sizes for SwinT stages
            
            # Create RAMA stages
            self.rama_stages = nn.ModuleList()
            for i, dim in enumerate(stage_dims):
                stage_name = f'stage{i+1}'
                if stage_name in self.rama_positions:
                    # Create SwinTRAMAAdapter for this stage
                    stage_rama = SwinTRAMAAdapter(
                        in_features=dim,
                        out_features=dim,
                        rama_config=rama_config,
                        stage_idx=i
                    )
                else:
                    # Use identity module if RAMA not applied at this position
                    stage_rama = nn.Identity()
                    
                self.rama_stages.append(stage_rama)
            
            # Create final RAMA layer if specified
            if 'final' in self.rama_positions:
                self.rama_final = BernoulliRAMALayer(
                    input_dim=self.feature_dim,
                    output_dim=self.feature_dim,
                    p_value=rama_config['p_value'],
                    values=rama_config.get('values', '0_1'),
                    use_normalization=rama_config.get('use_normalization', True),
                    activation=rama_config.get('activation', 'relu'),
                    lambda_value=rama_config.get('lambda_value', 1.0),
                    sqrt_dim=rama_config.get('sqrt_dim', False),
                )
            else:
                self.rama_final = nn.Identity()
        else:
            # Create dummy identity layers for consistency when RAMA is disabled
            self.rama_stages = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.rama_final = nn.Identity()
        
        # Classification head
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # For feature analysis
        self.intermediate_features = []
        self.final_features = None

    def forward(self, x, p_value=None):
        """
        Forward pass through the improved SwinT model with RAMA layers at selected positions.
        
        Args:
            x: Input tensor (B, C, H, W)
            p_value: Value controlling the Bernoulli parameter p
        """
        # Clear intermediate features for this forward pass
        self.intermediate_features = []
        
        # Initial patch embedding: (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = self.patch_embed(x)
        
        # Stage 1: (B, 96, 56, 56) -> (B, 96, 56, 56)
        x = self.stage1(x)
        # Store features before RAMA
        self.intermediate_features.append(x.detach().clone())
        # Apply RAMA if enabled for stage1
        if self.use_rama and 'stage1' in self.rama_positions:
            # SwinT format is (B, C, H, W) but RAMA adapter expects (B, H, W, C)
            x = self.rama_stages[0](x, p_value)

        # Patch Merging 1: (B, 96, 56, 56) -> (B, 192, 28, 28)
        x = self.patch_merging1(x)
        
        # Stage 2: (B, 192, 28, 28) -> (B, 192, 28, 28)
        x = self.stage2(x)
        self.intermediate_features.append(x.detach().clone())
        if self.use_rama and 'stage2' in self.rama_positions:
            x = self.rama_stages[1](x, p_value)
        
        # Patch Merging 2: (B, 192, 28, 28) -> (B, 384, 14, 14)
        x = self.patch_merging2(x)
        
        # Stage 3: (B, 384, 14, 14) -> (B, 384, 14, 14)
        x = self.stage3(x)
        self.intermediate_features.append(x.detach().clone())
        if self.use_rama and 'stage3' in self.rama_positions:
            x = self.rama_stages[2](x, p_value)
        
        # Patch Merging 3: (B, 384, 14, 14) -> (B, 768, 7, 7)
        x = self.patch_merging3(x)
        
        # Stage 4: (B, 768, 7, 7) -> (B, 768, 7, 7)
        x = self.stage4(x)
        self.intermediate_features.append(x.detach().clone())
        if self.use_rama and 'stage4' in self.rama_positions:
            x = self.rama_stages[3](x, p_value)
        
        # Apply final norm and global average pooling
        x = self.norm(x)
        x = x.mean(dim=[1, 2])  # (B, 1, 1, 768) -> (B, 768)
        
        # Store final features before RAMA
        self.final_features = x.detach().clone()
        
        # Apply final RAMA layer if enabled
        if self.use_rama and 'final' in self.rama_positions:
            x = self.rama_final(x, p_value)
        
        # Classification
        x = self.fc(x)  # (B, 768) -> (B, num_classes)
        return x

    def get_features(self):
        """Return intermediate and final features for analysis."""
        return self.intermediate_features, self.final_features


class DataManager:
    """
    Manager for CIFAR-10 dataset preparation and loading.
    
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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    def get_loaders(self):
        """
        Get data loaders for training and testing.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Training dataset.
        trainset = torchvision.datasets.CIFAR10(
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

        # Testing dataset.
        testset = torchvision.datasets.CIFAR10(
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
    Improved trainer class with better hyperparameter optimization for RAMA layers.
    
    Args:
        model (nn.Module): Model to train.
        trainloader (DataLoader): Training data loader.
        testloader (DataLoader): Testing data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use for training.
        checkpoint_dir (str): Directory to save checkpoints.
        bayes_opt_config (dict): Configuration for Bayesian optimization.
        use_rama (bool): Whether the model uses RAMA layers.
        use_hyperparameter_optimization (bool): Whether to use Bayesian optimization.
        neptune_run (neptune.Run): Neptune run for logging.
        writer (SummaryWriter): TensorBoard writer.
    """
    def __init__(self, model, trainloader, testloader, criterion, optimizer, 
                 device, checkpoint_dir, bayes_opt_config=None, use_rama=False,
                 use_hyperparameter_optimization=False, 
                 neptune_run=None, writer=None):
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
        self.best_p = None
        self.optimization_history = []  # Track optimization history
        
        if self.use_rama and self.use_hyperparameter_optimization:
            # Bayesian optimization configuration
            self.bayes_opt_config = bayes_opt_config
            
            # Set up acquisition function
            if self.bayes_opt_config["acq"] == "ei":
                acq = acquisition.ExpectedImprovement(xi=self.bayes_opt_config["xi"])
            elif self.bayes_opt_config["acq"] == "poi":
                acq = acquisition.ProbabilityOfImprovement(xi=self.bayes_opt_config["xi"])
            elif self.bayes_opt_config["acq"] == "ucb":
                acq = acquisition.UpperConfidenceBound(kappa=self.bayes_opt_config["kappa"])
            else:
                raise ValueError("Invalid acquisition function specified.")
            
            # Initialize Bayesian optimizer with expanded bounds
            self.bayesian_optimizer = BayesianOptimization(
                f=self.evaluate_p,
                acquisition_function=acq,
                pbounds={"p_value": (self.bayes_opt_config["p_min"], self.bayes_opt_config["p_max"])},
                random_state=42,
                verbose=2
            )

    def optimize_p(self, n_warmup=None, n_iter=None):
        """
        Run Bayesian optimization to find the best p value with more thorough exploration.
        
        Args:
            n_warmup (int): Number of random points to evaluate.
            n_iter (int): Number of iterations.
            
        Returns:
            tuple: (best_p, best_score, optimization_results)
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
        
        best_p = self.bayesian_optimizer.max["params"]["p_value"]
        best_score = self.bayesian_optimizer.max["target"]
        
        # Record optimization results
        self.optimization_history.append({
            "best_p": best_p,
            "best_score": best_score,
            "all_results": self.bayesian_optimizer.res
        })
        
        return best_p, best_score, self.bayesian_optimizer.res

    def save_bayes_opt_state(self, path):
        """Save Bayesian optimizer state to a JSON file."""
        if not hasattr(self, 'bayesian_optimizer'):
            return
            
        state = {
            'X': self.bayesian_optimizer.X.tolist() if len(self.bayesian_optimizer.X) > 0 else [],
            'y': self.bayesian_optimizer.Y.tolist() if len(self.bayesian_optimizer.Y) > 0 else [],
            'max': self.bayesian_optimizer.max if hasattr(self.bayesian_optimizer, 'max') else None,
            'bounds': {k: list(v) for k, v in self.bayesian_optimizer._pbounds.items()},
            'random_state': self.bayesian_optimizer._random_state
        }
        
        with open(path, 'w') as f:
            import json
            json.dump(state, f)
        logger.info(f"Saved BayesOpt state to {path}")

    def load_bayes_opt_state(self, path):
        """Load Bayesian optimizer state from a JSON file."""
        if not os.path.exists(path) or not hasattr(self, 'bayesian_optimizer'):
            return False
            
        try:
            import json
            with open(path, 'r') as f:
                state = json.load(f)
                
            # Restore points
            if state['X'] and state['y']:
                import numpy as np
                X = np.array(state['X'])
                y = np.array(state['y'])
                
                # Register all points with the optimizer
                for i in range(len(X)):
                    params = {k: X[i][j] for j, k in enumerate(self.bayesian_optimizer._space.keys())}
                    self.bayesian_optimizer.register(params=params, target=y[i])
                
            # Restore max value if it exists
            if state['max'] is not None:
                self.bayesian_optimizer.max = state['max']
                
            logger.info(f"Loaded BayesOpt state from {path}")
            if hasattr(self.bayesian_optimizer, 'max') and self.bayesian_optimizer.max:
                logger.info(f"Loaded max value: {self.bayesian_optimizer.max}")
            return True
        except Exception as e:
            logger.error(f"Error loading BayesOpt state: {str(e)}")
            return False

    def load_optimizer_state(self, path):
        """Load Bayesian optimizer state from a file."""
        return self.load_bayes_opt_state(path)

    def update_rama_masks(self, p_value):
        """
        Update all RAMA layer masks in the model at active positions.
        
        Args:
            p_value (float): New p-value for RAMA layers.
        """
        if not self.use_rama:
            return
        
        # Get positions that have RAMA layers
        active_positions = []
        if hasattr(self.model, "rama_positions"):
            active_positions = self.model.rama_positions
        
        # Update the final RAMA layer if active
        if hasattr(self.model, "rama_final") and 'final' in active_positions:
            if not isinstance(self.model.rama_final, nn.Identity):
                self.model.rama_final.update_mask(p_value)
        
        # Update intermediate RAMA layers in active stages
        if hasattr(self.model, "rama_stages"):
            for i, stage_rama in enumerate(self.model.rama_stages):
                stage_name = f'stage{i+1}'
                if stage_name in active_positions and not isinstance(stage_rama, nn.Identity):
                    if hasattr(stage_rama, "rama_layer"):
                        stage_rama.rama_layer.update_mask(p_value)

    def train_one_epoch(self, p_value=None):
        """
        Train the model for one epoch.
        
        Args:
            p_value (float): Value for RAMA layers.
            
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
            outputs = self.model(inputs, p_value=p_value)
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

    def evaluate(self, p_value=None):
        """
        Evaluate the model on the test set.
        
        Args:
            p_value (float): Value for RAMA layers.
            
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
                outputs = self.model(inputs, p_value=p_value)
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

    def evaluate_p(self, p_value):
        """
        Evaluation function for Bayesian optimization.
        Returns a single scalar value (accuracy).
        
        Args:
            p_value (float): Value to evaluate.
            
        Returns:
            float: Evaluation score (accuracy).
        """
        # Update RAMA masks with the new p_value
        self.update_rama_masks(p_value)
        
        # Evaluate with this p_value
        _, test_acc = self.evaluate(p_value)
        
        # Log this evaluation
        if self.neptune_run:
            self.neptune_run["BayesOpt/p_value"].append(p_value)
            self.neptune_run["BayesOpt/accuracy"].append(test_acc)
        
        if self.writer:
            # Use a global step counter for TensorBoard
            if not hasattr(self, "_eval_counter"):
                self._eval_counter = 0
            self.writer.add_scalar("BayesOpt/p_value", p_value, self._eval_counter)
            self.writer.add_scalar("BayesOpt/accuracy", test_acc, self._eval_counter)
            self._eval_counter += 1
            
        return test_acc

    def evaluate_with_metrics(self, p_value=None):
        """
        Comprehensive model evaluation with feature quality metrics.
        
        Args:
            p_value (float): Value for RAMA layers.
            
        Returns:
            dict: Evaluation metrics.
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        # For feature analysis
        all_targets = []
        stage_features = [[] for _ in range(4)]  # For 4 stages
        final_features = []
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, p_value)
                loss = self.criterion(outputs, targets)
                
                # Get intermediate features
                intermediate, final = self.model.get_features()
                
                # Store features and targets for analysis
                for i, features in enumerate(intermediate):
                    if i < len(stage_features):
                        # For spatial features, apply global average pooling for simplicity
                        if len(features.shape) == 4:
                            features = features.mean(dim=[2, 3])
                        stage_features[i].append(features.cpu())
                
                if final is not None:
                    final_features.append(final.cpu())
                all_targets.append(targets.cpu())
                
                # Calculate accuracy
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate basic metrics
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.testloader)
        
        # Calculate feature metrics if available
        feature_metrics = {}
        if all_targets and final_features:
            # Concatenate batches
            all_targets = torch.cat(all_targets, dim=0)
            final_features = torch.cat(final_features, dim=0)
            
            # Analyze feature quality
            for i, stage_feat in enumerate(stage_features):
                if stage_feat:
                    stage_feat = torch.cat(stage_feat, dim=0)
                    metrics = self._calculate_feature_metrics(stage_feat, all_targets)
                    
                    # Add stage prefix to metrics
                    for key, value in metrics.items():
                        feature_metrics[f"stage{i+1}_{key}"] = value
            
            # Final feature metrics
            final_metrics = self._calculate_feature_metrics(final_features, all_targets)
            for key, value in final_metrics.items():
                feature_metrics[f"final_{key}"] = value
            
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

    def _calculate_feature_metrics(self, features, labels):
        """
        Calculate metrics to assess feature quality.
        
        Args:
            features: Feature vectors
            labels: Class labels
            
        Returns:
            dict: Feature quality metrics
        """
        metrics = {}
        
        # Calculate class-wise statistics
        unique_classes = torch.unique(labels)
        class_means = []
        intra_class_distances = []
        
        # Calculate class centroids and intra-class distances
        for cls in unique_classes:
            cls_features = features[labels == cls]
            if cls_features.shape[0] == 0:
                continue
                
            cls_mean = cls_features.mean(dim=0)
            class_means.append(cls_mean)
            
            # Average distance from each point to its class centroid
            dists = torch.norm(cls_features - cls_mean, dim=1).mean()
            intra_class_distances.append(dists.item())
        
        # Convert to tensor for easier computation
        if not class_means:
            return metrics
            
        class_means = torch.stack(class_means)
        
        # Calculate inter-class distances (between centroids)
        n_classes = len(class_means)
        inter_dists = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                dist = torch.norm(class_means[i] - class_means[j])
                inter_dists.append(dist.item())
        
        # Calculate metrics
        avg_intra_dist = np.mean(intra_class_distances) if intra_class_distances else 0
        avg_inter_dist = np.mean(inter_dists) if inter_dists else 0
        
        # Fisher's criterion (inter-class separation / intra-class spread)
        fisher_ratio = avg_inter_dist / (avg_intra_dist + 1e-8)
        
        # Store metrics
        metrics['intra_class_distance'] = avg_intra_dist
        metrics['inter_class_distance'] = avg_inter_dist
        metrics['fisher_ratio'] = fisher_ratio
        
        return metrics

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
        
        # Save Bayesian optimization state if applicable
        if self.use_rama and self.use_hyperparameter_optimization:
            bayes_opt_path = os.path.join(self.checkpoint_dir, 'bayes_opt_state.json')
            self.save_bayes_opt_state(bayes_opt_path)
    
    def train(self, epochs, start_epoch=0):
        """
        Train the model for multiple epochs with improved RAMA optimization.
        
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
                # More thorough initial optimization
                self.best_p, _, _ = self.optimize_p(
                    n_warmup=self.bayes_opt_config["init_points"]*2,  # Double the initial points
                    n_iter=self.bayes_opt_config["n_iter"]*2  # Double the iterations
                )
            else:
                # Try to load previous optimization state
                bayes_opt_path = os.path.join(self.checkpoint_dir, 'bayes_opt_state.json')
                self.load_optimizer_state(bayes_opt_path)
                if self.bayesian_optimizer.max:
                    self.best_p = self.bayesian_optimizer.max["params"]["p_value"]
                    logger.info(f"Loaded best p from previous run: {self.best_p:.6f}")
                else:
                    logger.info("No previous optimization state found, running initial optimization...")
                    self.best_p, _, _ = self.optimize_p()

        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch: {epoch+1}/{epochs}")

            # Update all RAMA masks with the current best p value
            if self.use_rama:
                self.update_rama_masks(self.best_p if self.best_p is not None else 0.5)

            # Train with best p
            train_loss, train_acc = self.train_one_epoch(p_value=self.best_p)
            
            # Basic evaluation
            test_loss, test_acc = self.evaluate(p_value=self.best_p)
            
            # Detailed evaluation with feature metrics (once every 5 epochs to save time)
            if epoch % 5 == 0 or epoch == epochs - 1:
                metrics = self.evaluate_with_metrics(p_value=self.best_p)
                if 'feature_metrics' in metrics and metrics['feature_metrics']:
                    logger.info(f"Feature metrics at epoch {epoch+1}:")
                    for key, value in metrics['feature_metrics'].items():
                        if 'fisher_ratio' in key:
                            logger.info(f"  {key}: {value:.4f}")

            # More frequent optimization in early epochs, less frequent later
            should_optimize = False
            if self.use_rama and self.use_hyperparameter_optimization:
                # Dynamic optimization schedule
                if epoch < 10:
                    # More frequent in early stages
                    should_optimize = epoch % 2 == 0
                elif epoch < 30:
                    # Less frequent in middle stages
                    should_optimize = epoch % 5 == 0
                else:
                    # Even less frequent in later stages
                    should_optimize = epoch % 10 == 0
            
            # Perform Bayesian optimization when scheduled
            if should_optimize and epoch > 0:
                logger.info(f"Running Bayesian optimization at epoch {epoch+1}...")
                
                # Adaptive exploration based on training progress
                explore_factor = min(1.0, 0.5 + 0.5 * (epochs - epoch) / epochs)
                
                # More thorough optimization in early stages
                n_warmup = max(2, int(self.bayes_opt_config["init_points"] * explore_factor))
                n_iter = max(5, int(self.bayes_opt_config["n_iter"] * explore_factor))
                
                p_value, bayesian_score, results = self.optimize_p(
                    n_warmup=n_warmup,
                    n_iter=n_iter
                )
                
                if bayesian_score >= test_acc:
                    self.best_p = p_value
                    logger.info(f"Updated best p: {self.best_p:.4f} with accuracy: {bayesian_score:.2f}%")
                    
                    # Adaptively expand the search bounds based on the best p found
                    p_min_distance = abs(p_value - self.bayes_opt_config["p_min"])
                    p_max_distance = abs(p_value - self.bayes_opt_config["p_max"])
                    
                    # Expand in the direction that has more space to explore
                    if p_min_distance < p_max_distance and p_value > self.bayes_opt_config["p_min"] * 2:
                        # Closer to min bound, expand lower bound
                        new_min = max(p_value * 0.5, MIN_p_value)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"p_value": (new_min, self.bayes_opt_config["p_max"])}
                        )
                    elif p_value < self.bayes_opt_config["p_max"] * 0.5:
                        # Closer to max bound, expand upper bound
                        new_max = min(p_value * 2.0, MAX_p_value)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"p_value": (self.bayes_opt_config["p_min"], new_max)}
                        )

            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            if self.use_rama and self.best_p is not None:
                logger.info(f"Current p value: {self.best_p:.6f}")

            # Log to Neptune if available
            if self.neptune_run:
                self.neptune_run["Train/Loss"].append(train_loss)
                self.neptune_run["Train/Accuracy"].append(train_acc)
                self.neptune_run["Test/Loss"].append(test_loss)
                self.neptune_run["Test/Accuracy"].append(test_acc)
                if self.use_rama and self.best_p is not None:
                    self.neptune_run["RAMA_P"].append(self.best_p)

            # Log to TensorBoard if available
            if self.writer:
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Accuracy", test_acc, epoch)
                if self.use_rama and self.best_p is not None:
                    self.writer.add_scalar("RAMA_P", self.best_p, epoch)

            # Save checkpoint if best model
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
            
            self.save_checkpoint({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "best_p": self.best_p,
                "optimization_history": self.optimization_history,
            }, is_best)
            
        logger.info(f"Best test accuracy: {self.best_acc:.2f}%")
        return self.best_acc


def get_experiment_name(args):
    """Generate a unique experiment name based on configuration."""
    exp_name = "ImprovedSwinT"
    exp_name += "_RAMA" if args.use_rama else "_NoRAMA"
    
    if args.use_rama:
        # Include RAMA positions in name
        positions = args.rama_positions.replace(',', '_')
        exp_name += f"_pos({positions})"
        
        exp_name += f"_{args.bernoulli_values}"  # Add Bernoulli value type (0/1 or -1/1)
        exp_name += "_norm" if args.use_normalization else "_nonorm"
        exp_name += "_sqrt_d" if args.sqrt_dim else "_no_sqrt_d"
        exp_name += f"_{args.activation}"
        exp_name += f"_dimred{args.dim_reduction_factor:.2f}"
        
    exp_name += f"_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}"
    
    if args.use_rama:
        exp_name += f"_p{args.p_value:.2f}"
        exp_name += f"_lambda{args.lambda_value:.2f}"

    return exp_name


def setup_experiment_folders(exp_name):
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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with Improved SwinT and RAMA Layers')
    
    # Training parameters
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs (increased)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # RAMA configuration
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA layers')
    parser.add_argument('--use-hyperparameter-optimization', action='store_true', help='whether to use Bayesian optimization for p-value')
    parser.add_argument('--rama-positions', default='stage1,stage2,stage3,stage4,final',
                       type=str, help='comma-separated list of positions to apply RAMA (options: stage1,stage2,stage3,stage4,final)')
    parser.add_argument('--p-value', default=0.5, type=float, help='Bernoulli probability parameter (p-value)')
    parser.add_argument('--lambda-value', default=1.0, type=float, help='Lambda_value for RAMA')
    parser.add_argument('--sqrt-dim', action='store_true', help='Whether to normalize by sqrt(input_dim)')
    parser.add_argument('--bernoulli-values', default='0_1', choices=['0_1', '-1_1'],
                      type=str, help='values for Bernoulli distribution (0/1 or -1/1)')
    parser.add_argument('--use-normalization', action='store_true', help='use layer normalization in RAMA layers')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'],
                        help='activation function for RAMA layers')
    parser.add_argument('--dim-reduction-factor', default=1.0, type=float, 
                       help='factor to reduce dimensions by in RAMA layers')
    
    # Bayesian optimization parameters - with expanded range
    parser.add_argument('--p-min', default=0.01, type=float, help='minimum P value (p-value) for optimization (decreased)')
    parser.add_argument('--p-max', default=0.99, type=float, help='maximum P value (p-value) for optimization')
    parser.add_argument('--bayes-init-points', default=10, type=int, help='number of initial points for Bayesian optimization (increased)')
    parser.add_argument('--bayes-n-iter', default=30, type=int, help='number of iterations for Bayesian optimization (increased)')
    parser.add_argument('--bayes-acq', default="ei", choices=["ucb", "ei", "poi"], help='acquisition function for Bayesian optimization')
    parser.add_argument('--bayes-xi', default=0.01, type=float, help='exploration-exploitation parameter for ei/poi')
    parser.add_argument('--bayes-kappa', default=2.5, type=float, help='exploration-exploitation parameter for ucb')
    parser.add_argument('--optimize-every', default=5, type=int, help='baseline optimization frequency (in epochs)')
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
    
    # RAMA configuration with positions parsing
    rama_config = {
        "p_value": args.p_value,
        "values": args.bernoulli_values,
        "activation": args.activation,
        "use_normalization": args.use_normalization,
        "lambda_value": args.lambda_value,
        "sqrt_dim": args.sqrt_dim,
        "dim_reduction_factor": args.dim_reduction_factor,
        "positions": args.rama_positions.split(',')  # Parse positions from command line
    }
    
    # Create model
    model = ImprovedSwinT(
        num_classes=10, 
        use_rama=args.use_rama,
        rama_config=rama_config
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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

    # Bayesian optimization configuration
    bayes_opt_config = {
        "init_points": args.bayes_init_points,
        "n_iter": args.bayes_n_iter,
        "acq": args.bayes_acq,
        "xi": args.bayes_xi,
        "kappa": args.bayes_kappa,
        "p_min": args.p_min,
        "p_max": args.p_max,
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
        checkpoint_dir=args.checkpoint_dir,
        bayes_opt_config=bayes_opt_config,
        use_rama=args.use_rama,
        use_hyperparameter_optimization=args.use_hyperparameter_optimization,
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