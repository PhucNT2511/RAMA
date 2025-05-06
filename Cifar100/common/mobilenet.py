import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from common.rama_layers import BernoulliRAMALayer, GaussianRAMALayer


class MobileNetV3RAMAAdapter(nn.Module):
    """
    Specialized adapter module for integrating RAMA with MobileNetV3 architecture.
    
    Args:
        in_features (int): Number of input features/channels
        out_features (int): Number of output features/channels (if None, uses reduction factor)
        rama_config (dict): Configuration for RAMA layers
        rama_type (str): Type of RAMA layer to use ('bernoulli' or 'gaussian')
        layer_idx (int): Index of the layer this adapter is applied to
    """
    def __init__(self, in_features, out_features=None, rama_config=None, rama_type='bernoulli', layer_idx=0):
        super().__init__()
        
        if rama_config is None:
            rama_config = {}
            
        if out_features is None:
            # Apply dimension reduction factor if specified
            reduction_factor = rama_config.get('dim_reduction_factor', 1.0)
            out_features = max(int(in_features * reduction_factor), 16)
        
        self.in_features = in_features
        self.out_features = out_features
        self.layer_idx = layer_idx
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
        
        # Reshape to (batch_size, -1, channels) for RAMA processing
        x_flat = x.view(batch_size, -1, self.in_features)
        
        # Apply RAMA to each spatial position
        x_flat = self.rama_layer(x_flat.reshape(-1, self.in_features), p_value)
        
        # Reshape back to original format
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


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3Small architecture with RAMA layers at configurable positions.
    
    Args:
        num_classes (int): Number of output classes
        use_rama (bool): Whether to use RAMA layers
        rama_config (dict): Configuration for RAMA layers
        rama_type (str): Type of RAMA layer to use ('bernoulli' or 'gaussian')
    """
    def __init__(self, num_classes=10, use_rama=False, rama_config=None, rama_type='bernoulli'):
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
                    'positions': ['layer0', 'layer1', 'layer2', 'layer3', 'final']
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
                    'positions': ['layer0', 'layer1', 'layer2', 'layer3', 'final']
                }
            
        # Parse positions if provided as string
        if 'positions' in rama_config and isinstance(rama_config['positions'], str):
            rama_config['positions'] = rama_config['positions'].split(',')

        # Initialize the MobileNetV3Small model
        self.base_model = mobilenet_v3_small(weights=None)
        
        # Modify the classifier for CIFAR-10 (10 classes)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )
        
        # The feature dimension before the classifier
        self.feature_dim = 576
        
        # Extract the features section of the model for more granular control
        self.features = self.base_model.features
        
        # Track which positions have RAMA layers
        self.rama_positions = rama_config.get('positions', [])
        
        # Layer dimensions for MobileNetV3Small
        # These are approximate channel counts at different stages
        layer_dims = [16, 24, 40, 576]  # Channels at key points in the network
        
        # Create RAMA layers if enabled
        if use_rama:
            # Create RAMA for intermediate layers
            self.rama_layers = nn.ModuleDict()
            
            # For layers 0-3 (corresponding to key stages in MobileNetV3Small)
            for i, dim in enumerate(layer_dims):
                layer_name = f'layer{i}'
                if layer_name in self.rama_positions:
                    # Create RAMA adapter for this layer
                    self.rama_layers[layer_name] = MobileNetV3RAMAAdapter(
                        in_features=dim,
                        out_features=dim,
                        rama_config=rama_config,
                        rama_type=rama_type,
                        layer_idx=i
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
        Forward pass through the MobileNetV3Small model with RAMA layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            p_value: Value controlling the RAMA parameter
        """
        # Clear intermediate features for this forward pass
        self.intermediate_features = []
        
        # Apply features in stages for RAMA integration
        # Stage 0: Initial layers (first conv and first block)
        x = self.features[0:4](x)  # First conv + first inverted residual block
        if self.use_rama and 'layer0' in self.rama_positions:
            if hasattr(self, 'rama_layers') and 'layer0' in self.rama_layers:
                x = self.rama_layers['layer0'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        
        # Stage 1: Second set of inverted residual blocks
        x = self.features[4:7](x)
        if self.use_rama and 'layer1' in self.rama_positions:
            if hasattr(self, 'rama_layers') and 'layer1' in self.rama_layers:
                x = self.rama_layers['layer1'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        
        # Stage 2: Third set of inverted residual blocks
        x = self.features[7:10](x)
        if self.use_rama and 'layer2' in self.rama_positions:
            if hasattr(self, 'rama_layers') and 'layer2' in self.rama_layers:
                x = self.rama_layers['layer2'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        
        # Stage 3: Final inverted residual blocks
        x = self.features[10:](x)
        if self.use_rama and 'layer3' in self.rama_positions:
            if hasattr(self, 'rama_layers') and 'layer3' in self.rama_layers:
                x = self.rama_layers['layer3'](x, p_value)
        self.intermediate_features.append(x.detach().clone())
        
        # Global average pooling
        x = x.mean([2, 3])
        
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
        if hasattr(self, 'rama_layers'):
            for layer_name, layer in self.rama_layers.items():
                if hasattr(layer, "update_mask"):
                    layer.update_mask(p_value)
        
        # Update final RAMA layer if active
        if 'final' in self.rama_positions and hasattr(self, 'rama_final') and hasattr(self.rama_final, "update_mask"):
            self.rama_final.update_mask(p_value)
