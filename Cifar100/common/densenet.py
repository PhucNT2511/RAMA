import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

from common.rama_layers import BernoulliRAMALayer, GaussianRAMALayer


class DenseNetRAMAAdapter(nn.Module):
    """
    Specialized adapter module for integrating RAMA with DenseNet architecture.
    
    Args:
        in_features (int): Number of input features/channels
        out_features (int): Number of output features/channels (if None, uses reduction factor)
        rama_config (dict): Configuration for RAMA layers
        rama_type (str): Type of RAMA layer to use ('bernoulli' or 'gaussian')
        block_idx (int): Index of the DenseNet block this adapter is applied to
    """
    def __init__(self, in_features, out_features=None, rama_config=None, rama_type='bernoulli', block_idx=0):
        super().__init__()
        
        if rama_config is None:
            rama_config = {}
            
        if out_features is None:
            # Apply dimension reduction factor if specified
            reduction_factor = rama_config.get('dim_reduction_factor', 1.0)
            out_features = max(int(in_features * reduction_factor), 16)
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_idx = block_idx
        self.rama_type = rama_type
        
        # Create RAMA layer based on type
        if rama_type == 'bernoulli':
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
        else:  # gaussian
            self.rama_layer = GaussianRAMALayer(
                input_dim=in_features,
                output_dim=out_features,
                mu=rama_config.get('mu', 0.0),
                sigma=rama_config.get('sigma', 1.0),
                use_normalization=rama_config.get('use_normalization', True),
                activation=rama_config.get('activation', 'relu'),
                lambda_value=rama_config.get('lambda_value', 1.0),
                sqrt_dim=rama_config.get('sqrt_dim', False),
            )
        
        # Projection layer to handle dimension changes if needed
        self.proj = nn.Identity() if in_features == out_features else nn.Linear(out_features, in_features)
    
    def forward(self, x, p_value=None):
        """
        Forward pass applying RAMA to feature maps.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            p_value: Value for the RAMA layer
        """
        # Save original shape for reshaping back
        orig_shape = x.shape
        batch_size = orig_shape[0]
        
        # Reshape to (batch_size * height * width, channels) for RAMA processing
        x_flat = x.reshape(batch_size * orig_shape[2] * orig_shape[3], self.in_features)
        
        # Apply RAMA to each spatial position
        x_flat = self.rama_layer(x_flat, p_value)
        
        # Reshape back to original format with projection if needed
        if self.in_features != self.out_features:
            x_flat = self.proj(x_flat)
            
        x = x_flat.reshape(orig_shape)
        return x
        
    def update_mask(self, p_value):
        """
        Update the RAMA mask (for Bernoulli type only).
        
        Args:
            p_value: New p-value for the RAMA layer
        """
        if self.rama_type == 'bernoulli' and hasattr(self.rama_layer, "update_mask"):
            self.rama_layer.update_mask(p_value)


class DenseNet121(nn.Module):
    """
    DenseNet121 architecture with RAMA layers at configurable positions.
    
    Args:
        num_classes (int): Number of output classes
        use_rama (bool): Whether to use RAMA layers
        rama_config (dict): Configuration for RAMA layers
        rama_type (str): Type of RAMA layer to use ('bernoulli' or 'gaussian')
    """
    def __init__(self, num_classes=100, use_rama=False, rama_config=None, rama_type='bernoulli'):
        super().__init__()
        self.use_rama = use_rama
        self.num_classes = num_classes
        self.rama_type = rama_type

        if rama_config is None:
            if rama_type == 'bernoulli':
                rama_config = {
                    "p_value": 0.5,
                    "values": '0_1',
                    "activation": "leaky_relu",
                    "use_normalization": True,
                    'lambda_value': 1.0,
                    'sqrt_dim': False,
                    'dim_reduction_factor': 1.0,
                    'positions': ['block0', 'block1', 'block2', 'block3', 'final']
                }
            else:  # gaussian
                rama_config = {
                    "mu": 0.0,
                    "sigma": 1.0,
                    "activation": "leaky_relu",
                    "use_normalization": True,
                    'lambda_value': 1.0,
                    'sqrt_dim': False,
                    'dim_reduction_factor': 1.0,
                    'positions': ['block0', 'block1', 'block2', 'block3', 'final']
                }
            
        # Parse positions if provided as string
        if 'positions' in rama_config and isinstance(rama_config['positions'], str):
            rama_config['positions'] = rama_config['positions'].split(',')

        # Initialize the DenseNet121 model
        self.base_model = densenet121(weights=None)
        
        # Modify the classifier for CIFAR-10 (10 classes)
        self.base_model.classifier = nn.Linear(1024, num_classes)
        
        # The feature dimension before the classifier
        self.feature_dim = 1024
        
        # Extract features and blocks from DenseNet for easier access
        self.features = self.base_model.features
        
        # For DenseNet, we want to extract different dense blocks for RAMA insertion
        # Initial convolution and norm
        self.initial_layers = nn.Sequential(
            self.features.conv0,
            self.features.norm0,
            self.features.relu0,
            self.features.pool0
        )
        
        # Dense blocks and transition layers
        self.denseblock1 = self.features.denseblock1
        self.transition1 = self.features.transition1
        self.denseblock2 = self.features.denseblock2
        self.transition2 = self.features.transition2
        self.denseblock3 = self.features.denseblock3
        self.transition3 = self.features.transition3
        self.denseblock4 = self.features.denseblock4
        
        # Final batch norm and pooling
        self.final_norm = self.features.norm5
        
        # Track which positions have RAMA layers
        self.rama_positions = rama_config.get('positions', [])
        
        # Feature dimensions after each block in DenseNet121
        block_dims = [256, 512, 1024, 1024]  # Approximate channel counts after each dense block
        
        # Create RAMA layers if enabled
        if use_rama:
            # Create RAMA for intermediate blocks
            self.rama_blocks = nn.ModuleDict()
            
            # For blocks 0-3 (corresponding to dense blocks in DenseNet)
            for i, dim in enumerate(block_dims):
                block_name = f'block{i}'
                if block_name in self.rama_positions:
                    # Create RAMA adapter for this block
                    self.rama_blocks[block_name] = DenseNetRAMAAdapter(
                        in_features=dim,
                        out_features=dim,
                        rama_config=rama_config,
                        rama_type=rama_type,
                        block_idx=i
                    )
            
            # Create final RAMA layer if specified (after global pooling)
            if 'final' in self.rama_positions:
                if rama_type == 'bernoulli':
                    self.rama_final = BernoulliRAMALayer(
                        input_dim=self.feature_dim,
                        output_dim=self.feature_dim,
                        p_value=rama_config.get('p_value', 0.5),
                        values=rama_config.get('values', '0_1'),
                        use_normalization=rama_config.get('use_normalization', True),
                        activation=rama_config.get('activation', 'relu'),
                        lambda_value=rama_config.get('lambda_value', 1.0),
                        sqrt_dim=rama_config.get('sqrt_dim', False),
                    )
                else:  # gaussian
                    self.rama_final = GaussianRAMALayer(
                        input_dim=self.feature_dim,
                        output_dim=self.feature_dim,
                        mu=rama_config.get('mu', 0.0),
                        sigma=rama_config.get('sigma', 1.0),
                        use_normalization=rama_config.get('use_normalization', True),
                        activation=rama_config.get('activation', 'relu'),
                        lambda_value=rama_config.get('lambda_value', 1.0),
                        sqrt_dim=rama_config.get('sqrt_dim', False),
                    )
            else:
                self.rama_final = nn.Identity()
        
        # For feature extraction and analysis
        self.intermediate_features = []
        self.final_features = None

    def forward(self, x, p_value=None):
        """
        Forward pass through the DenseNet121 model with RAMA layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            p_value: Value controlling the RAMA parameter
        """
        # Clear intermediate features for this forward pass
        self.intermediate_features = []
        
        # Initial layers
        x = self.initial_layers(x)
        
        # Block 1
        x = self.denseblock1(x)
        if self.use_rama and 'block0' in self.rama_positions:
            if hasattr(self, 'rama_blocks') and 'block0' in self.rama_blocks:
                x = self.rama_blocks['block0'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        x = self.transition1(x)
        
        # Block 2
        x = self.denseblock2(x)
        if self.use_rama and 'block1' in self.rama_positions:
            if hasattr(self, 'rama_blocks') and 'block1' in self.rama_blocks:
                x = self.rama_blocks['block1'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        x = self.transition2(x)
        
        # Block 3
        x = self.denseblock3(x)
        if self.use_rama and 'block2' in self.rama_positions:
            if hasattr(self, 'rama_blocks') and 'block2' in self.rama_blocks:
                x = self.rama_blocks['block2'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        x = self.transition3(x)
        
        # Block 4
        x = self.denseblock4(x)
        if self.use_rama and 'block3' in self.rama_positions:
            if hasattr(self, 'rama_blocks') and 'block3' in self.rama_blocks:
                x = self.rama_blocks['block3'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        
        # Final normalization
        x = self.final_norm(x)
        x = F.relu(x, inplace=True)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Store features before final RAMA
        self.final_features = x.detach().clone()
        
        # Apply final RAMA if enabled
        if self.use_rama and 'final' in self.rama_positions:
            x = self.rama_final(x, p_value)
        
        # Apply classifier
        x = self.base_model.classifier(x)
        return x

    def get_features(self):
        """Return intermediate and final features for analysis."""
        return self.intermediate_features, self.final_features
            
    def update_rama_masks(self, p_value=None):
        """
        Update all RAMA masks (for Bernoulli type only).
        
        Args:
            p_value: Value for RAMA parameter
        """
        if not self.use_rama or self.rama_type != 'bernoulli':
            return

        # Update intermediate RAMA layers
        if hasattr(self, 'rama_blocks'):
            for block_name, block in self.rama_blocks.items():
                if hasattr(block, "update_mask"):
                    block.update_mask(p_value)
        
        # Update final RAMA layer if active
        if 'final' in self.rama_positions and hasattr(self, 'rama_final') and hasattr(self.rama_final, "update_mask"):
            self.rama_final.update_mask(p_value)
