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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MAX_LAMBDA_VALUE = 10
MIN_LAMBDA_VALUE = 1e-3
NEPTUNE_PRJ_NAME = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")


class RAMALayer(nn.Module):
    """
    A basic randomized projection layer with scaling.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        lambda_value (float): Lambda scaling value for the projection matrix.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
    """
    def __init__(self, input_dim, output_dim, lambda_value, activation="relu"):
        super(RAMALayer, self).__init__()
        self.activation = activation
        # self.param = nn.Parameter(torch.tensor(lambda_value))
        self.projection = nn.Parameter(
            # torch.randn(input_dim, output_dim) / (np.sqrt(output_dim) * 3),  # 0.014731391274719738
            # torch.randn(input_dim, output_dim) * np.sqrt(output_dim),
            torch.randn(input_dim, output_dim),
            requires_grad=False
        )

    def forward(self, x, lambda_value):
        """Forward pass through the RAMA layer."""
        if lambda_value is not None:
            out = x @ self.projection * lambda_value
        else:
            out = x @ self.projection

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
    ResNet architecture for image classification with optional RAMA layers.
    
    Args:
        block (nn.Module): Block type to use for the network.
        num_blocks (List[int]): Number of blocks in each layer.
        num_classes (int): Number of output classes. Default: 10.
        use_rama (bool): Whether to use RAMA layer before classification. Default: False.
        rama_config (dict): Configuration for RAMA layer. Default: None.
    """
    def __init__(self, block, num_blocks, num_classes=10, use_rama=False, rama_config=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_rama = use_rama
        if rama_config is None:
            rama_config = {
                "lambda_value": 5e-2,
                "lambda_init": 1e-2,
                "activation": "leaky_relu",
            }
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.feature_dim = 512
        if use_rama:
            self.rama_layer = RAMALayer(self.feature_dim, self.feature_dim, rama_config['lambda_value'], rama_config['activation'])
            # self.rama_layer = torch.randn(self.feature_dim, self.feature_dim).cuda() / (np.sqrt(self.feature_dim) * 3)  # 3
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with the specified number of blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, lambda_value):
        """Forward pass through the ResNet model."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if self.use_rama:
            out = self.rama_layer.forward(out, lambda_value)
            # out = F.relu(out @ self.rama_layer)
        out = self.fc(out)
        return out


def resnet18(num_classes=10, use_rama=False, rama_config=None):
    """
    Create a ResNet-18 model with optional RAMA layer.
    
    Args:
        num_classes (int): Number of output classes. Default: 10.
        use_rama (bool): Whether to use RAMA layer. Default: False.
        rama_config (dict): Configuration for RAMA layer. Default: None.
        
    Returns:
        ResNet: ResNet-18 model.
    """
    return ResNet(
        ResidualBlock, 
        [2, 2, 2, 2], 
        num_classes=num_classes,
        use_rama=use_rama,
        rama_config=rama_config
    )


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
    Trainer class for model training and evaluation.
    
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
                 device, checkpoint_dir, bayes_opt_config=None, use_rama: bool = False,
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
        self.best_lambda = None
        if self.use_rama:
            # self.bayesian_optimizer = BayesianOptimization(
            #     f=self.evaluate_lambda,
            #     pbounds={"lambda_value": (1e-2, 0.5)},
            #     random_state=42,
            # )
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
            tuple: (best_lambda, best_score)
        """
        if n_warmup is None:
            n_warmup = self.bayes_opt_config["init_points"]
        if n_iter is None:
            n_iter = self.bayes_opt_config["n_iter"]
            
        logger.info(f"Running Bayesian optimization with {n_warmup} initialization points and {n_iter} iterations...")
        
        # Run optimization
        # self.bayesian_optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)
        self.bayesian_optimizer.maximize(
            init_points=n_warmup,
            n_iter=n_iter
        )
        
        best_lambda = self.bayesian_optimizer.max["params"]["lambda_value"]
        best_score = self.bayesian_optimizer.max["target"]
        # logger.info(f"Best lambda found: {best_lambda:.6f} with accuracy: {best_score:.2f}%")
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
        """,
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
        Evaluate the model on the test set.
        
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
        Train the model for multiple epochs.
        
        Args:
            epochs (int): Number of epochs to train.
            start_epoch (int): Starting epoch. Default: 0.
            
        Returns:
            float: Best test accuracy.
        """
        # lambda_value = 0.1 # 0.01473
        # First, run initial optimization to find a good lambda
        if self.use_rama:
            if start_epoch == 0:
                logger.info("Running initial Bayesian optimization...")
                self.best_lambda, _, _ = self.optimize_lambda()
            else:
                # Try to load previous optimization state
                bayes_opt_path = os.path.join(self.checkpoint_dir, 'bayes_opt_state.json')
                self.load_optimizer_state(bayes_opt_path)
                if self.bayesian_optimizer.max:
                    self.best_lambda = self.bayesian_optimizer.max["params"]["lambda_value"]
                    bayesian_score = 0
                    logger.info(f"Loaded best lambda from previous run: {self.best_lambda:.6f}")
                else:
                    logger.info("No previous optimization state found, running initial optimization...")
                    self.best_lambda, bayesian_score, _ = self.optimize_lambda()

        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch: {epoch+1}/{epochs}")

            # # Perform Bayesian optimization periodically
            # if (epoch % self.bayes_opt_config["optimize_every"] == 0) and (epoch > 0):
            #     logger.info(f"Running Bayesian optimization at epoch {epoch+1}...")
            #     # Use fewer iterations for subsequent optimizations
            #     self.best_lambda, _, _ = self.optimize_lambda(n_warmup=2, n_iter=3)

            # # Train and evaluate with best lambda.
            train_loss, train_acc = self.train_one_epoch(lambda_value=self.best_lambda)
            test_loss, test_acc = self.evaluate(lambda_value=self.best_lambda)

            # Perform Bayesian optimization periodically
            if self.use_rama and epoch % self.bayes_opt_config["optimize_every"] == 0 and (epoch > 0):
                logger.info(f"Running Bayesian optimization at epoch {epoch+1}...")
                # Use fewer iterations for subsequent optimizations.
                lambda_value, bayesian_score, results = self.optimize_lambda()
                if bayesian_score >= test_acc:
                    self.best_lambda = lambda_value
                    logger.info(f"Best lambda found: {self.best_lambda:.4f} with accuracy: {bayesian_score:.2f}%")

                    # sorted_results = sorted(results, key=lambda kk: kk["target"], reverse=True)
                    # self.bayesian_optimizer.set_bounds(new_bounds={"lambda_value": (min(sorted_results[5]["params"]["lambda_value"], self.best_lambda),
                    #                                                                 max(sorted_results[5]["params"]["lambda_value"], self.best_lambda))})
                    lambda_min_distance = abs(lambda_value - self.bayes_opt_config["lambda_min"])
                    lambda_max_distance = abs(lambda_value - self.bayes_opt_config["lambda_max"])
                    if lambda_min_distance < lambda_max_distance:
                        self.bayesian_optimizer.set_bounds(new_bounds={"lambda_value": (self.bayes_opt_config["lambda_min"], self.best_lambda*1.5)})
                        # self.bayesian_optimizer.set_bounds(new_bounds={"lambda_value": (self.bayes_opt_config["lambda_min"],
                        #                                                                 self.best_lambda if self.best_lambda > sorted_results[1]["params"]["lambda_value"] else sorted_results[1]["params"]["lambda_value"])})
                    else:
                        self.bayesian_optimizer.set_bounds(new_bounds={"lambda_value": (self.best_lambda*0.5, self.bayes_opt_config["lambda_max"])})
                        # self.bayesian_optimizer.set_bounds(new_bounds={"lambda_value": (self.best_lambda if self.best_lambda < sorted_results[1]["params"]["lambda_value"] else sorted_results[1]["params"]["lambda_value"],
                        #                                                                 self.bayes_opt_config["lambda_max"])})

            # # Bayesian optimization step.
            # logger.info("Running Bayesian optimization for lambda...")
            # self.bayesian_optimizer.maximize(init_points=3, n_iter=15)
            # bayes_score = self.bayesain_optimizer.max["target"]
            # if bayes_score > test_acc:
            #     lambda_value = self.bayesian_optimizer.max["params"]["lambda_value"]
            #     logger.info(f"Best lambda found: {lambda_value:.4f} with accuracy: {bayes_score:.2f}%")

            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            if self.neptune_run:
                self.neptune_run["Train/Loss"].append(train_loss)
                self.neptune_run["Train/Accuracy"].append(train_acc)
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, epoch)

            if self.neptune_run:
                self.neptune_run["Test/Loss"].append(test_loss)
                self.neptune_run["Test/Accuracy"].append(test_acc)
            self.writer.add_scalar("Train/Loss", test_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", test_acc, epoch)

            if self.use_rama and self.neptune_run:
                self.neptune_run["RAMA_Lambda"].append(self.best_lambda)
                self.writer.add_scalar("RAMA_Lambda", self.best_lambda, epoch)

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


def get_experiment_name(args: argparse.Namespace) -> str:
    exp_name = "ResNet"
    exp_name += "_RAMA" if args.use_rama else "_NoRAMA"
    exp_name += f"_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}_lambda{args.lambda_value}"
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
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with ResNet-18')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs')  # 15, 20
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA')
    parser.add_argument('--lambda-value', default=0.01, type=float, help='lambda value for basic RAMA')
    parser.add_argument('--lambda-init', default=0.01, type=float, help='lambda init for enhanced RAMA')

    parser.add_argument('--lambda-min', default=0.01, type=float, help='minimum lambda value for optimization')
    parser.add_argument('--lambda-max', default=0.1, type=float, help='maximum lambda value for optimization')
    parser.add_argument('--bayes-init-points', default=3, type=int, help='number of initial points for Bayesian optimization')  # 3
    parser.add_argument('--bayes-n-iter', default=5, type=int, help='number of iterations for Bayesian optimization')  # 20
    parser.add_argument('--bayes-acq', default="ei", choices=["ucb", "ei", "poi"], help='acquisition function for Bayesian optimization')
    parser.add_argument('--bayes-xi', default=0.01, type=float, help='exploration-exploitation parameter for ei/poi')
    parser.add_argument('--bayes-kappa', default=2.5, type=float, help='exploration-exploitation parameter for ucb')
    parser.add_argument('--optimize-every', default=1, type=int, help='optimize lambda every N epochs')  # 4

    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'], help='activation for enhanced RAMA')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def main():
    """Main function for training and evaluating the model."""
    args = parse_args()
    logger.info(f"Starting training with arguments: {args}")
    
    # Set random seeds for reproducibility.
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        logger.info(f"Created checkpoint directory: {args.checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    data_manager = DataManager(args.data_dir, args.batch_size, args.num_workers)
    trainloader, testloader = data_manager.get_loaders()
    rama_config = {
        "lambda_value": args.lambda_value,
        "lambda_init": args.lambda_init,
        "activation": args.activation,
    }
    model = resnet18(
        num_classes=10, 
        use_rama=args.use_rama,
        rama_config=rama_config
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
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
        "lambda_min": args.lambda_min,
        "lambda_max": args.lambda_max,
        "optimize_every": args.optimize_every,
    }

    exp_name = get_experiment_name(args)
    if NEPTUNE_PRJ_NAME and NEPTUNE_API_TOKEN:
        neptune_run = neptune.init_run(project=NEPTUNE_PRJ_NAME, api_token=NEPTUNE_API_TOKEN, name=exp_name)
        neptune_run["config"] = args
        neptune_run["bayes_config"] = bayes_opt_config
        neptune_run["rama_config"] = rama_config
    else:
        neptune_run = None
    
    exp_dir = setup_experiment_folders(exp_name)
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "logs"))
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
        neptune_run=neptune_run,
        writer=writer
    )
    if args.resume and best_acc > 0:
        trainer.best_acc = best_acc
    best_acc = trainer.train(args.epochs, start_epoch)
    writer.close()
    if neptune_run:
        neptune_run.stop()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
