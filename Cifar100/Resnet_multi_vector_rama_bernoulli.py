import argparse
import logging
import os
import random
from typing import Optional
import sys

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


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from Cifar10.common.attacks import fgsm_attack, pgd_attack

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MAX_p_value = 10
MIN_p_value = 1e-3
NEPTUNE_PRJ_NAME = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")

##### RAMA with Bernoulli for U matrix instead of norm distribution
class BernoulliRAMALayer(nn.Module):
    """
    A RAMA layer using Bernoulli distribution for random projections.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        p_value (float): Controls the Bernoulli parameter p.
        values (str): Values for Bernoulli: '0_1' for {0,1} or '-1_1' for {-1,1}.
        use_normalization (bool): Whether to apply layer normalization after projection.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
    """
    def __init__(self, input_dim, output_dim, p_value, values='0_1', use_normalization=True, 
                 activation="relu", lambda_value=1.0, sqrt_dim=False):
        super(BernoulliRAMALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.values = values
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value

        self.sqrt_d = 1
        if sqrt_dim == True:
            self.sqrt_d = math.sqrt(input_dim)
        
        # Initialize Bernoulli projection matrix
        # For 0/1 values, we use a threshold of 0.5 initially
        # For -1/1 values, we use a threshold of 0.5 but convert to -1/1
        if values == '0_1':
            projection = (torch.rand(input_dim, output_dim) < p_value).float()
        elif values == '-1_1':
            projection = 2 * (torch.rand(input_dim, output_dim) < p_value).float() - 1
        else:
            raise ValueError(f"Unknown values: {values}. Use '0_1' or '-1_1'")

        self.projection = nn.Parameter(projection, requires_grad=False)
        # Add layer normalization for stabilizing the output distribution.
        if use_normalization:
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, p_value):
        """
        Forward pass through the Bernoulli RAMA layer.
        
        Args:
            x: Input tensor
            p_value: Value controlling the Bernoulli parameter p
        """
        # Generate a dynamic Bernoulli mask based on p_value
        # For inference or when p_value is None, use the stored projection
        if p_value is not None and self.training:
            # Clamp p_value between 0.01 and 0.99 to avoid extreme values
            p = max(0.01, min(0.99, p_value))
            
            # Generate a new Bernoulli mask
            if self.values == '0_1':
                mask = (torch.rand_like(self.projection) < p).float()
            elif self.values == '-1_1':
                mask = 2 * (torch.rand_like(self.projection) < p).float() - 1
                
            out = x @ mask
        else:
            out = x @ self.projection

        out = out * self.lambda_value * self.sqrt_d

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
        return out


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
    Modified ResNet architecture with Bernoulli RAMA layers at multiple positions.
    
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
                "p_value": 0.5,  # default Bernoulli probability
                "values": '0_1',      # default to 0/1 values
                "activation": "leaky_relu",
                "use_normalization": True,
                'lambda_value': 1.0,
                'sqrt_dim': False,
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
        
        # Create Bernoulli RAMA layers for different positions in the network
        if use_rama:
            # RAMA after layer2 (128 features)
            self.rama_layer2 = BernoulliRAMALayer(
                128, 
                128, 
                rama_config['p_value'], 
                rama_config.get('values', '0_1'),
                rama_config.get('use_normalization', True),
                rama_config.get('activation', 'relu'),
                rama_config.get('lambda_value', 1.0),
                rama_config.get('sqrt_dim', False),
            )
            
            # RAMA after layer3 (256 features)
            self.rama_layer3 = BernoulliRAMALayer(
                256, 
                256, 
                rama_config['p_value'], 
                rama_config.get('values', '0_1'),
                rama_config.get('use_normalization', True),
                rama_config.get('activation', 'relu'),
                rama_config.get('lambda_value', 1.0),
                rama_config.get('sqrt_dim', False),
            )
            
            # RAMA before final classification
            self.rama_layer4 = BernoulliRAMALayer(
                self.feature_dim, 
                self.feature_dim, 
                rama_config['p_value'], 
                rama_config.get('values', '0_1'),
                rama_config.get('use_normalization', True),
                rama_config.get('activation', 'relu'),
                rama_config.get('lambda_value', 1.0),
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

    def forward(self, x, p_value):
        """Forward pass through the ResNet model with Bernoulli RAMA layers."""
        out = self.conv1(x)
        out = self.layer1(out)
        
        # After layer 1
        out = self.layer2(out)
        
        # # Apply RAMA after layer2 (early in the network)
        # if self.use_rama:
        #     # Need to flatten, apply RAMA, then reshape back
        #     batch_size, channels, h, w = out.shape
        #     out_flat = out.view(batch_size, channels, -1).mean(dim=2)  # Global avg pooling per channel
        #     out_flat = self.rama_layer2(out_flat, p_value)
        #     # Broadcast back to spatial dimensions
        #     out = out * out_flat.view(batch_size, channels, 1, 1)

        out = self.layer3(out)
        
        # # Apply RAMA after layer3 (middle of the network)
        # if self.use_rama:
        #     batch_size, channels, h, w = out.shape
        #     out_flat = out.view(batch_size, channels, -1).mean(dim=2)
        #     out_flat = self.rama_layer3(out_flat, p_value)
        #     out = out * out_flat.view(batch_size, channels, 1, 1)
            
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        # Store features before RAMA for evaluation
        if self.use_rama:
            self.before_rama_features = out.detach().clone()
            
        # Apply RAMA before final classification (original position)
        if self.use_rama:
            out = self.rama_layer4(out, p_value)
            # out = self.rama_layer4(out, None)
            # Store features after RAMA for evaluation
            self.after_rama_features = out.detach().clone()
        out = self.fc(out)
        return out

    def forward_with_features(self, x, p_value):
        """
        Forward pass that returns both output and features before/after RAMA.
        Useful for analyzing feature quality.
        """
        outputs = self.forward(x, p_value)
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
        use_hyperparameter_optimization: bool = False ---> TURN ON when you want optimize p value, else OFF
    """
    def __init__(self, model, trainloader, testloader, criterion, optimizer, 
                 device, checkpoint_dir, bayes_opt_config=None, use_rama: bool = False,
                 use_hyperparameter_optimization: bool = False,
                 neptune_run: Optional[neptune.Run] = None, writer: Optional[SummaryWriter] = None, args: Optional[argparse.Namespace] = None):
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
        self.args = args
        
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
                f=self.evaluate_p,
                acquisition_function=acq,
                pbounds={"p_value": (self.bayes_opt_config["p_min"], self.bayes_opt_config["p_max"])},
                random_state=42,
                verbose=2
            )

    def optimize_p(self, n_warmup=None, n_iter=None):
        """
        Run Bayesian optimization to find the best p value.
        
        Args:
            n_warmup (int): Number of random points to evaluate. Default: from config.
            n_iter (int): Number of iterations. Default: from config.
            
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
        return best_p, best_score, self.bayesian_optimizer.res

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

    def train_one_epoch(self, p_value=None):
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
            
            if self.args and self.args.adversarial_training:
                self.model.eval() # Set model to eval mode for attack generation, then back to train
                # Generate adversarial examples
                current_p_for_at = p_value # This is self.best_p from the train loop
                inputs_for_attack = inputs.clone().detach()

                # Wrapper for attack function, ensuring RAMA p_value is used if RAMA is active
                attack_model_wrapper_at = lambda imgs_for_attack: self.model.forward(imgs_for_attack, p_value=current_p_for_at)

                if self.args.at_attack == 'pgd':
                    adv_inputs = pgd_attack(attack_model_wrapper_at, inputs_for_attack, targets,
                                            self.args.at_epsilon, self.args.at_alpha, self.args.at_iter,
                                            self.device, clamp_min=-10.0, clamp_max=10.0) 
                elif self.args.at_attack == 'fgsm':
                    adv_inputs = fgsm_attack(attack_model_wrapper_at, inputs_for_attack, targets,
                                             self.args.at_epsilon, self.device)
                else: # Default to clean inputs if attack type is somehow wrong (should be caught by argparse choices)
                    adv_inputs = inputs
                
                self.model.train() # Set model back to train mode for training step
                self.optimizer.zero_grad()
                outputs = self.model.forward(adv_inputs, p_value=current_p_for_at) # Train on adversarial examples
                loss = self.criterion(outputs, targets)
            else:
                # Standard training
                self.model.train() # Ensure model is in train mode
            self.optimizer.zero_grad()
            outputs = self.model.forward(inputs, p_value=p_value)
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
                outputs = self.model.forward(inputs, p_value=p_value)
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
        """
        _, test_acc = self.evaluate(p_value)
        return test_acc  # Return only accuracy for optimization

    def evaluate_with_metrics(self, p_value=None, epoch=None, test_acc=None, total_epochs=None):
        """
        Evaluate the model with additional metrics to understand RAMA impact.
        
        Returns:
            dict: Evaluation metrics including feature quality
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        correct_fgsm, total_fgsm = 0, 0
        correct_pgd, total_pgd = 0, 0
        
        # Initialize containers for feature analysis
        features_original = []
        features_after_rama = []
        class_labels = []
        
        with torch.no_grad():
            pbar_eval = tqdm(self.testloader, desc=f"Epoch {epoch} Evaluation" if epoch is not None else "Evaluation")
            for batch_idx, (inputs, targets) in enumerate(pbar_eval):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Use the forward pass that returns features
                current_p_value_for_eval = p_value if self.use_rama else None
                outputs, before_features, after_features = self.model.forward_with_features(inputs, current_p_value_for_eval)
                
                if self.use_rama and before_features is not None and after_features is not None: # Only collect if RAMA is used and features are returned
                        features_original.append(before_features.cpu())
                        features_after_rama.append(after_features.cpu())
                        class_labels.append(targets.cpu())
                
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
                # Define a model wrapper for attack functions that handles p_value
                # The attack functions expect model(images) to return logits
                attack_model_wrapper = lambda imgs_for_attack: self.model.forward(imgs_for_attack, p_value=current_p_value_for_eval)

                if test_acc > self.best_acc and (epoch % 15 == 0 or epoch == total_epochs - 1):
                    # FGSM Attack Evaluation
                    if self.args and self.args.eval_fgsm:
                        adv_images_fgsm = fgsm_attack(attack_model_wrapper, inputs.clone(), targets, self.args.epsilon, self.device)
                        outputs_fgsm = self.model.forward(adv_images_fgsm, p_value=current_p_value_for_eval)
                        _, predicted_fgsm = outputs_fgsm.max(1)
                        total_fgsm += targets.size(0)
                        correct_fgsm += predicted_fgsm.eq(targets).sum().item()

                    # PGD Attack Evaluation
                    if self.args and self.args.eval_pgd:
                        adv_images_pgd = pgd_attack(attack_model_wrapper, inputs.clone(), targets, 
                                                    self.args.epsilon, self.args.pgd_alpha, self.args.pgd_iter, 
                                                    self.device, clamp_min=-10.0, clamp_max=10.0) # Wide clamps for normalized data
                        outputs_pgd = self.model.forward(adv_images_pgd, p_value=current_p_value_for_eval)
                        _, predicted_pgd = outputs_pgd.max(1)
                        total_pgd += targets.size(0)
                        correct_pgd += predicted_pgd.eq(targets).sum().item()
                
                eval_desc = f"Clean Acc: {100.*correct/total:.2f}%"
                if self.args and self.args.eval_fgsm:
                    eval_desc += f" | FGSM Acc: {100.*correct_fgsm/total_fgsm if total_fgsm > 0 else 0:.2f}%"
                if self.args and self.args.eval_pgd:
                    eval_desc += f" | PGD Acc: {100.*correct_pgd/total_pgd if total_pgd > 0 else 0:.2f}%"
                pbar_eval.set_postfix_str(eval_desc)

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
                    self.neptune_run[f"Test/Feature/{key}"].append(value) # Appending for time series
            
            if self.writer and epoch is not None:
                for key, value in feature_metrics.items():
                    self.writer.add_scalar(f"Test/Feature/{key}", value, epoch)
        
        eval_results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'feature_metrics': feature_metrics
        }

        if self.args and self.args.eval_fgsm and total_fgsm > 0:
            fgsm_accuracy = 100. * correct_fgsm / total_fgsm
            eval_results['fgsm_accuracy'] = fgsm_accuracy
            logger.info(f"FGSM Robust Accuracy (eps={self.args.epsilon:.3f}): {fgsm_accuracy:.2f}%")
            if self.neptune_run:
                self.neptune_run[f"Test/FGSM_Accuracy_eps{self.args.epsilon}"].append(fgsm_accuracy)
            if self.writer and epoch is not None:
                self.writer.add_scalar(f"Test/FGSM_Accuracy_eps{self.args.epsilon}", fgsm_accuracy, epoch)

        if self.args and self.args.eval_pgd and total_pgd > 0:
            pgd_accuracy = 100. * correct_pgd / total_pgd
            eval_results['pgd_accuracy'] = pgd_accuracy
            logger.info(f"PGD Robust Accuracy (eps={self.args.epsilon:.3f}, alpha={self.args.pgd_alpha:.3f}, iter={self.args.pgd_iter}): {pgd_accuracy:.2f}%")
            if self.neptune_run:
                self.neptune_run[f"Test/PGD_Accuracy_eps{self.args.epsilon}_alpha{self.args.pgd_alpha}_iter{self.args.pgd_iter}"].append(pgd_accuracy)
            if self.writer and epoch is not None:
                self.writer.add_scalar(f"Test/PGD_Accuracy_eps{self.args.epsilon}_alpha{self.args.pgd_alpha}_iter{self.args.pgd_iter}", pgd_accuracy, epoch)
                
        return eval_results

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
                self.best_p, _, _ = self.optimize_p()
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

            # Train with best p
            train_loss, train_acc = self.train_one_epoch(p_value=self.best_p)
            
            # Basic evaluation
            test_loss, test_acc = self.evaluate(p_value=self.best_p)
            
            # Detailed evaluation with feature metrics (once every 5 epochs to save time)
            # if epoch % 5 == 0 or epoch == epochs - 1:
            metrics = self.evaluate_with_metrics(p_value=self.best_p, epoch=epoch, test_acc=test_acc, total_epochs=epochs)
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
                p_value, bayesian_score, results = self.optimize_p(
                    n_warmup=max(2, self.bayes_opt_config["init_points"] // 2), 
                    n_iter=max(5, self.bayes_opt_config["n_iter"] // 2)
                )
                if bayesian_score >= test_acc:
                    self.best_p = p_value
                    logger.info(f"Updated best p: {self.best_p:.4f} with accuracy: {bayesian_score:.2f}%")
                    
                    # Update the search bounds based on the best p found.
                    p_min_distance = abs(p_value - self.bayes_opt_config["p_min"])
                    p_max_distance = abs(p_value - self.bayes_opt_config["p_max"])
                    
                    # If closer to min bound, expand upper bound.
                    if p_min_distance < p_max_distance:
                        new_max = min(self.best_p * 1.5, self.bayes_opt_config["p_max"] * 2)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"p_value": (self.bayes_opt_config["p_min"], new_max)}
                        )
                    # If closer to max bound, expand lower bound.
                    else:
                        new_min = max(self.best_p * 0.5, self.bayes_opt_config["p_min"] * 0.5)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"p_value": (new_min, self.bayes_opt_config["p_max"])}
                        )

            # Log metrics.
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            if self.use_rama and self.use_hyperparameter_optimization:
                logger.info(f"Current p value: {self.best_p:.6f}")

            # Log to Neptune if available.
            if self.neptune_run:
                self.neptune_run["Train/Loss"].append(train_loss)
                self.neptune_run["Train/Accuracy"].append(train_acc)
                self.neptune_run["Test/Loss"].append(test_loss)
                self.neptune_run["Test/Accuracy"].append(test_acc)
                if self.use_rama and self.use_hyperparameter_optimization:
                    self.neptune_run["RAMA_P"].append(self.best_p)

            # Log to TensorBoard if available.
            if self.writer:
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Accuracy", test_acc, epoch)
                if self.use_rama and self.use_hyperparameter_optimization:
                    self.writer.add_scalar("RAMA_P", self.best_p, epoch)

            # Save checkpoint if best model.
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
            self.save_checkpoint({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "best_p": self.best_p,
            }, is_best)
        logger.info(f"Best test accuracy: {self.best_acc:.2f}%")
        return self.best_acc


def resnet18(num_classes=100, use_rama=False, rama_config=None):
    """
    Create a ResNet-18 model with optional Bernoulli RAMA layers.
    
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
    exp_name += "_BernoulliRAMA" if args.use_rama else "_NoRAMA"
    
    if args.use_rama:
        exp_name += f"_{args.bernoulli_values}"  # Add Bernoulli value type (0/1 or -1/1)
        exp_name += "_norm" if args.use_normalization else "_nonorm"
        exp_name += "_sqrt_d_True" if args.sqrt_dim else "_sqrt_d_False"
        exp_name += f"_{args.activation}"
        
    exp_name += f"_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}"
    
    if args.use_rama:
        exp_name += f"_p{args.p_value:.2f}"
        exp_name += f"_lambda{args.lambda_value:.4f}"

    if args.adversarial_training:
        exp_name += f"_AT-{args.at_attack}"

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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training with ResNet-18 and Bernoulli RAMA Layers')
    
    # Training parameters
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Attack evaluation parameters
    parser.add_argument('--eval-fgsm', action='store_true', help='evaluate with FGSM attack')
    parser.add_argument('--eval-pgd', action='store_true', help='evaluate with PGD attack')
    parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon for FGSM and PGD attacks') # Typical 8/255 ~ 0.031
    parser.add_argument('--pgd-alpha', default=0.01, type=float, help='alpha (step size) for PGD attack') # Typical 2/255 ~ 0.0078
    parser.add_argument('--pgd-iter', default=7, type=int, help='number of iterations for PGD attack')
    
    # Bernoulli RAMA configuration
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA layers')
    parser.add_argument('--use-hyperparameter-optimization', action='store_true', help='whether to use Bayesian optimization for p-value')
    parser.add_argument('--p-value', default=0.5, type=float, help='Bernoulli probability parameter (p-value)')
    parser.add_argument('--lambda-value', default=1.0, type=float, help='Lambda_value for RAMA')
    parser.add_argument('--sqrt-dim', default= False, help='Whether multiply with sqrt(d) or not')
    parser.add_argument('--bernoulli-values', default='0_1', choices=['0_1', '-1_1'],
                      type=str, help='values for Bernoulli distribution (0/1 or -1/1)')
    parser.add_argument('--use-normalization', action='store_true', help='use layer normalization in RAMA layers')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'],
                        help='activation function for RAMA layers')
    
    # Bayesian optimization parameters - adjusted for probability range
    parser.add_argument('--p-min', default=0.1, type=float, help='minimum P value (p-value) for optimization')
    parser.add_argument('--p-max', default=1, type=float, help='maximum P value (p-value) for optimization')
    parser.add_argument('--bayes-init-points', default=5, type=int, help='number of initial points for Bayesian optimization')
    parser.add_argument('--bayes-n-iter', default=15, type=int, help='number of iterations for Bayesian optimization')
    parser.add_argument('--bayes-acq', default="ei", choices=["ucb", "ei", "poi"], help='acquisition function for Bayesian optimization')
    parser.add_argument('--bayes-xi', default=0.01, type=float, help='exploration-exploitation parameter for ei/poi')
    parser.add_argument('--bayes-kappa', default=2.5, type=float, help='exploration-exploitation parameter for ucb')
    parser.add_argument('--optimize-every', default=5, type=int, help='optimize P every N epochs')

    # Adversarial Training (AT) parameters
    parser.add_argument('--adversarial-training', '--at', action='store_true', help='Enable adversarial training')
    parser.add_argument('--at-attack', default='pgd', choices=['fgsm', 'pgd'], help='Attack type for adversarial training')
    parser.add_argument('--at-epsilon', default=0.03, type=float, help='Epsilon for adversarial training attack')
    parser.add_argument('--at-alpha', default=0.01, type=float, help='Alpha for PGD adversarial training attack')
    parser.add_argument('--at-iter', default=7, type=int, help='Iterations for PGD adversarial training attack')
    
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
    
    # Bernoulli RAMA configuration
    rama_config = {
        "p_value": args.p_value,  # This is now the p-value for Bernoulli
        "values": args.bernoulli_values,    # 0/1 or -1/1
        "activation": args.activation,
        "use_normalization": args.use_normalization,
        "lambda_value": args.lambda_value,
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
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    
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
        writer=writer,
        args=args
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
