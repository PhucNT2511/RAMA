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
from common.resnet import resnet18
from common.utils import set_seed, setup_experiment_folders, setup_logging, parse_rama_positions

# Environment variables
NEPTUNE_PRJ_NAME = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")


def get_experiment_name(args):
    """Generate a unique experiment name based on configuration."""
    exp_name = "ResNet18"
    exp_name += "_BernoulliRAMA" if args.use_rama else "_NoRAMA"
    
    if args.use_rama:
        exp_name += f"_{args.bernoulli_values}"  # Add Bernoulli value type (0/1 or -1/1)
        exp_name += "_norm" if args.use_normalization else "_nonorm"
        exp_name += "_sqrt_d_True" if args.sqrt_dim else "_sqrt_d_False"
        exp_name += f"_{args.activation}"
        exp_name += f"_positions_{args.rama_positions.replace(',', '-')}"
        
    exp_name += f"_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}"
    exp_name += f"_sched_{args.lr_scheduler}"
    
    if args.use_rama:
        exp_name += f"_p{args.p_value:.2f}"
        exp_name += f"_lambda{args.lambda_value:.2f}"
    return exp_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with ResNet-18 and Bernoulli RAMA Layers')
    
    # Training parameters
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--data-dir', default='./data', help='data directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num-workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # LR scheduler parameters
    parser.add_argument('--lr-scheduler', default='cosine', choices=['step', 'cosine', 'plateau'], 
                        help='Learning rate scheduler type')
    parser.add_argument('--lr-step-size', default=30, type=int, help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='Gamma for StepLR scheduler')
    parser.add_argument('--lr-warmup-epochs', default=5, type=int, help='Warm-up epochs for learning rate')
    
    # Bernoulli RAMA configuration
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA layers')
    parser.add_argument('--use-hyperparameter-optimization', action='store_true', help='whether to use Bayesian optimization for p-value')
    parser.add_argument('--p-value', default=0.5, type=float, help='Bernoulli probability parameter (p-value)')
    parser.add_argument('--lambda-value', default=0.3, type=float, help='Lambda_value for RAMA')
    parser.add_argument('--sqrt-dim', action='store_true', help='Whether to divide by sqrt(d)')
    parser.add_argument('--bernoulli-values', default='0_1', choices=['0_1', '-1_1'],
                      type=str, help='values for Bernoulli distribution (0/1 or -1/1)')
    parser.add_argument('--use-normalization', action='store_true', help='use layer normalization in RAMA layers')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'silu'],
                        help='activation function for RAMA layers')
    parser.add_argument('--evolution-rate', default=0.1, type=float, help='Rate at which RAMA mask evolves')
    
    # RAMA position configuration
    parser.add_argument('--rama-positions', default='layer1,layer2,layer3,final',
                       type=str, help='comma-separated list of positions to apply RAMA (options: layer1,layer2,layer3,layer4,final)')
    
    # Bayesian optimization parameters
    parser.add_argument('--p-min', default=0.1, type=float, help='minimum P value for optimization')
    parser.add_argument('--p-max', default=0.9, type=float, help='maximum P value for optimization')
    parser.add_argument('--bayes-init-points', default=5, type=int, help='number of initial points for Bayesian optimization')
    parser.add_argument('--bayes-n-iter', default=15, type=int, help='number of iterations for Bayesian optimization')
    parser.add_argument('--bayes-acq', default="ei", choices=["ucb", "ei", "poi"], help='acquisition function for Bayesian optimization')
    parser.add_argument('--bayes-xi', default=0.01, type=float, help='exploration-exploitation parameter for ei/poi')
    parser.add_argument('--bayes-kappa', default=2.5, type=float, help='exploration-exploitation parameter for ucb')
    parser.add_argument('--optimize-every', default=10, type=int, help='optimize P every N epochs')
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
    
    # Configure RAMA positions
    rama_positions = parse_rama_positions(args.rama_positions)
    
    # Bernoulli RAMA configuration
    rama_config = {
        "p_value": args.p_value,
        "values": args.bernoulli_values,
        "activation": args.activation,
        "use_normalization": args.use_normalization,
        "lambda_value": args.lambda_value,
        "sqrt_dim": args.sqrt_dim,
    }
    
    # Create model
    model = resnet18(
        num_classes=10, 
        use_rama=args.use_rama,
        rama_config=rama_config,
        rama_positions=rama_positions,
        rama_type='bernoulli'
    ).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    
    # Setup learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.lr_step_size, 
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=args.lr_gamma, 
            patience=10
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
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
        neptune_run = neptune.init_run(
            project=NEPTUNE_PRJ_NAME, 
            api_token=NEPTUNE_API_TOKEN, 
            name=exp_name
        )
        neptune_run["config"] = vars(args)
        neptune_run["bayes_config"] = bayes_opt_config
        neptune_run["rama_config"] = rama_config
        neptune_run["rama_positions"] = rama_positions
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
        bayes_opt_config=bayes_opt_config if args.use_hyperparameter_optimization else None,
        use_rama=args.use_rama,
        use_hyperparameter_optimization=args.use_hyperparameter_optimization,
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
