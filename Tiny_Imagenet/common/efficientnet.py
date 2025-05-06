import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2

from common.rama_layers import BernoulliRAMALayer, GaussianRAMALayer


class EfficientNet(nn.Module):
    """
    Modified EfficientNet-B2 architecture with RAMA layers.
    
    Args:
        num_classes (int): Number of output classes. Default: 10.
        use_rama (bool): Whether to use RAMA layers. Default: False.
        rama_config (dict): Configuration for RAMA layers. Default: None.
        rama_type (str): Type of RAMA layer to use ('bernoulli' or 'gaussian'). Default: 'bernoulli'.
    """
    def __init__(self, num_classes=10, use_rama=False, rama_config=None, rama_type='bernoulli'):
        super().__init__()
        self.use_rama = use_rama
        self.rama_type = rama_type
        if rama_config is None:
            if rama_type == 'bernoulli':
                rama_config = {
                    "p_value": 0.5,  # default Bernoulli probability
                    "values": '0_1',  # default to 0/1 values
                    "activation": "leaky_relu",
                    "use_normalization": True,
                    'lambda_value': 1.0,
                    'sqrt_dim': False,
                }
            else:  # gaussian
                rama_config = {
                    "mu": 0.0,  # default mean
                    "sigma": 1.0,  # default std
                    "activation": "leaky_relu",
                    "use_normalization": True,
                    'lambda_value': 1.0,
                    'sqrt_dim': False,
                }

        self.backbone = efficientnet_b2(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.features_1 = nn.Sequential(*list(self.backbone.children())[:-1]) 

        # Create RAMA layer before the linear layer in the network
        if use_rama:
            if rama_type == 'bernoulli':
                self.rama_layer = BernoulliRAMALayer(
                    self.feature_dim, 
                    self.feature_dim, 
                    rama_config['p_value'], 
                    rama_config.get('values', '0_1'),
                    rama_config.get('use_normalization', True),
                    rama_config.get('activation', 'relu'),
                    rama_config.get('lambda_value', 1.0),
                    rama_config.get('sqrt_dim', False),
                )
            else:  # gaussian
                self.rama_layer = GaussianRAMALayer(
                    self.feature_dim, 
                    self.feature_dim, 
                    rama_config.get('mu', 0.0),
                    rama_config.get('sigma', 1.0),
                    rama_config.get('use_normalization', True),
                    rama_config.get('activation', 'relu'),
                    rama_config.get('lambda_value', 1.0),
                    rama_config.get('sqrt_dim', False),
                )

        # Dropout and Linear
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.features_2 = nn.Sequential(self.dropout, self.fc)

        # Initialize hooks for feature extraction
        self.hooks = []
        self.before_rama_features = None
        self.after_rama_features = None

    def forward(self, x, p_value=None):
        """Forward pass through the model with optional RAMA layer."""
        out = self.features_1(x)
        out = torch.flatten(out, 1)

        # Store features before RAMA for evaluation.
        if self.use_rama:
            self.before_rama_features = out.detach().clone()
            out = self.rama_layer(out, p_value)
            self.after_rama_features = out.detach().clone()

        out = self.features_2(out)
        return out

    def forward_with_features(self, x, p_value=None):
        """
        Forward pass that returns both output and features before/after RAMA.
        Useful for analyzing feature quality.
        """
        outputs = self.forward(x, p_value)
        if self.use_rama:
            return outputs, self.before_rama_features, self.after_rama_features
        else:
            return outputs, None, None
            
    def update_rama_masks(self, p_value=None):
        """
        Update RAMA masks for all RAMA layers in the model.
        
        Args:
            p_value: Value to use for Bernoulli RAMA layers
        """
        if not self.use_rama:
            return
            
        if self.rama_type == 'bernoulli' and hasattr(self, 'rama_layer'):
            self.rama_layer.update_mask(p_value)