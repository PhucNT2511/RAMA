import torch.nn as nn
import torch.nn.functional as F

from common.rama_layers import BernoulliRAMALayer, GaussianRAMALayer, BernoulliRAMA4D, RandConv


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for ResNet architecture.
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
    Modified ResNet architecture with RAMA layers at configurable positions.
    
    Args:
        block: ResNet block type
        num_blocks: List of number of blocks per layer
        num_classes: Number of output classes
        use_rama: Whether to use RAMA layers
        rama_config: Configuration for RAMA layers
        rama_positions: Dictionary of positions to apply RAMA
        rama_type: Type of RAMA to use ('bernoulli' or 'gaussian')
    """
    def __init__(self, block, num_blocks, num_classes=10, use_rama=False, 
                 rama_config=None, rama_positions=None, rama_type='bernoulli'):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.use_rama = use_rama
        self.rama_type = rama_type
        self.feature_dim = 512  # Set this before creating RAMA layers
        
        # Default RAMA positions
        if rama_positions is None:
            self.rama_positions = {
                'layer1': False,
                'layer2': True,
                'layer3': True,
                'layer4': False,
                'final': True
            }
        else:
            self.rama_positions = rama_positions
        
        # Default RAMA configuration
        if rama_config is None:
            if rama_type == 'bernoulli':
                rama_config = {
                    "p_value": 0.5,
                    "values": '0_1',
                    "activation": "leaky_relu",
                    "use_normalization": True,
                    'lambda_value': 0.3,
                    'sqrt_dim': False,
                }
            else:  # gaussian
                rama_config = {
                    "mu": 0.0,
                    "sigma": 1.0,
                    "activation": "leaky_relu",
                    "use_normalization": True,
                    'lambda_value': 0.3,
                    'sqrt_dim': False,
                }
        
        self.rama_config = rama_config
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
        
        # Create RAMA layers based on positions
        if use_rama:
            self._create_rama_layers()
        
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize hooks for feature extraction
        self.before_rama_features = None
        self.after_rama_features = None

    def _create_rama_layers(self):
        """Create RAMA layers based on specified positions and type"""
        if self.rama_type == 'bernoulli':
            # Extract configuration without p_value for creating layers
            config = {
                "values": self.rama_config.get('values', '0_1'),
                "use_normalization": self.rama_config.get('use_normalization', True),
                "activation": self.rama_config.get('activation', 'relu'),
                "lambda_value": self.rama_config.get('lambda_value', 0.3),
                "sqrt_dim": self.rama_config.get('sqrt_dim', False),
            }
            p_value = self.rama_config.get('p_value', 0.5)
            
            # Create 4D Bernoulli RAMA layers for convolutional layers
            if self.rama_positions.get('layer1', False):
                self.rama_layer1 = BernoulliRAMA4D(
                    64, 64, p_value, **config
                )
                
            if self.rama_positions.get('layer2', False):
                self.rama_layer2 = BernoulliRAMA4D(
                    128, 128, p_value, **config
                )
                
            if self.rama_positions.get('layer3', False):
                self.rama_layer3 = BernoulliRAMA4D(
                    256, 256, p_value, **config
                )
                
            if self.rama_positions.get('layer4', False):
                self.rama_layer4 = BernoulliRAMA4D(
                    512, 512, p_value, **config
                )
                
            if self.rama_positions.get('final', False):
                self.rama_final = BernoulliRAMALayer(
                    self.feature_dim, self.feature_dim, p_value, **config
                )
                
        elif self.rama_type == 'gaussian':
            config = {
                "use_normalization": self.rama_config.get('use_normalization', True),
                "activation": self.rama_config.get('activation', 'relu'),
                "lambda_value": self.rama_config.get('lambda_value', 0.3),
                "sqrt_dim": self.rama_config.get('sqrt_dim', False),
            }
            mu = self.rama_config.get('mu', 0.0)
            sigma = self.rama_config.get('sigma', 1.0)

            # Configuration for RandConv
            randconv_config = {
                "p_random": 0.5,  # Can be adjusted based on experiments
                "distribution": "normal",
                "std": sigma,  # Use the same sigma as GaussianRAMA
                "use_residual": True,
                "activation": config["activation"],
                "batchnorm": config["use_normalization"]
            }
            
            if self.rama_positions.get('layer1', False):
                self.rama_layer1 = RandConv(
                    in_channels=64, 
                    out_channels=64, 
                    **randconv_config
                )
                
            if self.rama_positions.get('layer2', False):
                self.rama_layer2 = RandConv(
                    in_channels=128, 
                    out_channels=128, 
                    **randconv_config
                )
                
            if self.rama_positions.get('layer3', False):
                self.rama_layer3 = RandConv(
                    in_channels=256, 
                    out_channels=256, 
                    **randconv_config
                )
                
            if self.rama_positions.get('layer4', False):
                self.rama_layer4 = RandConv(
                    in_channels=512, 
                    out_channels=512, 
                    **randconv_config
                )
                
            # Keep GaussianRAMALayer for final layer (operates on flattened features)
            if self.rama_positions.get('final', False):
                self.rama_final = GaussianRAMALayer(
                    self.feature_dim, self.feature_dim, mu, sigma, **config
                )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Create a ResNet layer with the specified number of blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, p_value=None):
        """Forward pass through the ResNet model with RAMA layers."""
        out = self.conv1(x)
        out = self.layer1(out)

        # Apply RAMA after layer1 (if enabled)
        if self.use_rama and self.rama_positions.get('layer1', False):
            if hasattr(self, 'rama_layer1'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_layer1(out, p_value)
                else:  # gaussian (now using RandConv)
                    out = self.rama_layer1(out)  # RandConv doesn't need p_value

        out = self.layer2(out)

        # Apply RAMA after layer2 (if enabled)
        if self.use_rama and self.rama_positions.get('layer2', False):
            if hasattr(self, 'rama_layer2'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_layer2(out, p_value)
                else:  # gaussian (now using RandConv)
                    out = self.rama_layer2(out)  # RandConv doesn't need p_value

        out = self.layer3(out)
        
        # Apply RAMA after layer3 (if enabled)
        if self.use_rama and self.rama_positions.get('layer3', False):
            if hasattr(self, 'rama_layer3'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_layer3(out, p_value)
                else:  # gaussian (now using RandConv)
                    out = self.rama_layer3(out)  # RandConv doesn't need p_value
            
        out = self.layer4(out)
        
        # Apply RAMA after layer4, before pooling (if enabled)
        if self.use_rama and self.rama_positions.get('layer4', False):
            if hasattr(self, 'rama_layer4'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_layer4(out, p_value)
                else:  # gaussian (now using RandConv)
                    out = self.rama_layer4(out)  # RandConv doesn't need p_value
                
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # Apply RAMA after pooling, before FC (if enabled)
        if self.use_rama and self.rama_positions.get('final', False):
            self.before_rama_features = out.detach().clone()
            if hasattr(self, 'rama_final'):
                if self.rama_type == 'bernoulli' or self.rama_type == 'gaussian':
                    # Both BernoulliRAMALayer and GaussianRAMALayer need p_value
                    out = self.rama_final(out, p_value)
            self.after_rama_features = out.detach().clone()

        out = self.fc(out)
        return out

    def forward_with_features(self, x, p_value=None):
        """
        Forward pass that returns both output and features before/after RAMA.
        Useful for analyzing feature quality.
        """
        outputs = self.forward(x, p_value)
        if self.use_rama and self.rama_positions.get('final', False):
            return outputs, self.before_rama_features, self.after_rama_features
        else:
            return outputs, None, None
            
    def update_rama_masks(self, p_value=None):
        """
        Update all RAMA masks in the model
        
        Args:
            p_value: Value for RAMA parameter (used for Bernoulli RAMA)
        """
        if not self.use_rama:
            return
            
        if self.rama_type == 'bernoulli':
            # Check each position and update the corresponding RAMA layer if it exists
            if self.rama_positions.get('layer1', False) and hasattr(self, 'rama_layer1'):
                self.rama_layer1.update_mask(p_value)
                
            if self.rama_positions.get('layer2', False) and hasattr(self, 'rama_layer2'):
                self.rama_layer2.update_mask(p_value)
                
            if self.rama_positions.get('layer3', False) and hasattr(self, 'rama_layer3'):
                self.rama_layer3.update_mask(p_value)
                
            if self.rama_positions.get('layer4', False) and hasattr(self, 'rama_layer4'):
                self.rama_layer4.update_mask(p_value)
                
            if self.rama_positions.get('final', False) and hasattr(self, 'rama_final'):
                self.rama_final.update_mask(p_value)
        elif self.rama_type == 'gaussian':
            # Only update the final layer which is still using GaussianRAMALayer
            if self.rama_positions.get('final', False) and hasattr(self, 'rama_final'):
                self.rama_final.update_mask(sigma=p_value)


def resnet18(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    """
    Create a ResNet-18 model with optional RAMA layers.
    """
    return ResNet(
        ResidualBlock, 
        [2, 2, 2, 2], 
        num_classes=num_classes,
        use_rama=use_rama,
        rama_config=rama_config,
        rama_positions=rama_positions,
        rama_type=rama_type
    )