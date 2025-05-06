import argparse
import os
import neptune
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import common modules
from common.data import DataManager
from common.trainer import Trainer
from common.swint import ImprovedSwinT
from common.utils import set_seed, setup_experiment_folders, setup_logging

# Environment variables
NEPTUNE_PRJ_NAME = "phuca1tt1bn/RAMA"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODZlNDU0Yy1iMDk0LTQ5MDEtOGNiYi00OTZlYTY4ODI0MzgifQ=="


def get_experiment_name(args):
    """Generate a unique experiment name based on configuration."""
    exp_name = "ImprovedSwinT"
    exp_name += "_GaussianRAMA" if args.use_rama else "_NoRAMA"
    
    if args.use_rama:
        # Include RAMA positions in name
        positions = args.rama_positions.replace(',', '_')
        exp_name += f"_pos({positions})"
        
        exp_name += f"_mu{args.mu:.1f}_sigma{args.sigma:.1f}"
        exp_name += "_norm" if args.use_normalization else "_nonorm"
        exp_name += "_sqrt_d" if args.sqrt_dim else "_no_sqrt_d"
        exp_name += f"_{args.activation}"
        exp_name += f"_dimred{args.dim_reduction_factor:.2f}"
        
    exp_name += f"_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}"
    
    if args.use_rama:
        exp_name += f"_lambda{args.lambda_value:.2f}"

    return exp_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with Improved SwinT and Gaussian RAMA Layers')
    
    # Training parameters
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # RAMA configuration
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA layers')
    parser.add_argument('--rama-positions', default='stage1,stage2,stage3,stage4,final',
                       type=str, help='comma-separated list of positions to apply RAMA (options: stage1,stage2,stage3,stage4,final)')
    parser.add_argument('--mu', default=0.0, type=float, help='Mean for Gaussian distribution')
    parser.add_argument('--sigma', default=1.0, type=float, help='Standard deviation for Gaussian distribution')
    parser.add_argument('--lambda-value', default=1.0, type=float, help='Lambda_value for RAMA')
    parser.add_argument('--sqrt-dim', action='store_true', help='Whether to normalize by sqrt(input_dim)')
    parser.add_argument('--use-normalization', action='store_true', help='use layer normalization in RAMA layers')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid'],
                        help='activation function for RAMA layers')
    parser.add_argument('--dim-reduction-factor', default=1.0, type=float, 
                       help='factor to reduce dimensions by in RAMA layers')
    
    return parser.parse_args()


def main():
    """Main function for training and evaluating the model."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging("training.log")
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
    
    # RAMA configuration
    rama_config = {
        "mu": args.mu,
        "sigma": args.sigma,
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
        rama_config=rama_config,
        rama_type='gaussian'
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

    # Set up experiment tracking
    exp_name = get_experiment_name(args)
    if NEPTUNE_PRJ_NAME and NEPTUNE_API_TOKEN:
        neptune_run = neptune.init_run(
            project=NEPTUNE_PRJ_NAME, 
            api_token=NEPTUNE_API_TOKEN, 
            name=exp_name
        )
        neptune_run["config"] = vars(args)
        neptune_run["rama_config"] = rama_config
    else:
        neptune_run = None
    
    # Set up experiment directories and TensorBoard
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
        checkpoint_dir=os.path.join(exp_dir, "checkpoints"),
        use_rama=args.use_rama,
        use_hyperparameter_optimization=False,  # Gaussian version doesn't use hyperparameter optimization
        neptune_run=neptune_run,
        writer=writer,
        scheduler=scheduler
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
