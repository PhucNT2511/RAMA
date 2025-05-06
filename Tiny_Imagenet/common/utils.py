import os
import random
import logging
import torch
import numpy as np


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_experiment_folders(exp_name):
    """
    Create folders for experiment outputs.

    Args:
        exp_name (str): Experiment name

    Returns:
        str: Path to experiment directory
    """
    base_dir = os.path.join("experiments", exp_name)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    return base_dir


def setup_logging(log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_file (str, optional): Path to log file. If None, will only log to console.
        
    Returns:
        logging.Logger: Configured logger
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def parse_rama_positions(positions_str):
    """
    Parse comma-separated positions string into a dictionary for RAMA layer positions.
    
    Args:
        positions_str (str): Comma-separated string of positions (e.g. "layer1,layer2,final")
        
    Returns:
        dict: Dictionary mapping position names to boolean values
    """
    valid_positions = ['layer1', 'layer2', 'layer3', 'layer4', 'final']
    positions_dict = {pos: False for pos in valid_positions}
    
    if positions_str:
        selected_positions = [pos.strip() for pos in positions_str.split(',')]
        for pos in selected_positions:
            if pos in valid_positions:
                positions_dict[pos] = True
            else:
                logging.warning(f"Invalid RAMA position: {pos}. Skipping.")
    
    return positions_dict
