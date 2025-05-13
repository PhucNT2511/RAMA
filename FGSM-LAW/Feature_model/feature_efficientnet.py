import torch
import torch.nn as nn  
import torch.nn.functional as F
from torchvision.models import efficientnet_b2
import math

class GaussianRAMALayer(nn.Module):
    """
    A RAMA layer using Gaussian distribution for random projections.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        use_normalization (bool): Whether to apply layer normalization after projection.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
    """
    def __init__(self, input_dim, output_dim, lambda_value=1.0, 
                 use_normalization=False, activation="relu", sqrt_dim=False):
        super(GaussianRAMALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value

        self.sqrt_d = 1
        if sqrt_dim == True:
            self.sqrt_d = math.sqrt(input_dim)
        

        projection = torch.randn(input_dim, output_dim)
        self.projection = nn.Parameter(projection, requires_grad=False)

        # Add layer normalization for stabilizing the output distribution.
        if use_normalization:
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, lambda_value=None):
        """
        Forward pass through the Gaussian RAMA layer.
        
        Args:
            x: Input tensor
            lambda_value: Optional scaling factor for the output.
        """
        
        if lambda_value is not None:
            self.lambda_value = lambda_value
        
        out = x @ self.projection

        out *= self.sqrt_d * self.lambda_value

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
        elif self.activation == "silu":
            out = torch.nn.functional.silu(out)
        elif self.activation == "gelu":
            out = torch.nn.functional.gelu(out)
        return out

class feature_EfficientNet(nn.Module):
    """
    Modified EfficientNet-B2 architecture with Gaussian RAMA layers at multiple positions.
    
    Args:
        num_classes (int): Number of output classes. Default: 10.
        use_rama (bool): Whether to use RAMA layers. Default: False.
        rama_config (dict): Configuration for RAMA layers. Default: None.
    """
    def __init__(self, num_classes=10, use_rama=False, rama_config=None, rama_type='gaussian'):
        super().__init__()
        
        self.use_rama = use_rama
        
        if rama_config is None:
            rama_config = {
                    "lambda_value": 1.0,  # This lambda_value for Gaussian
                    "activation": "relu",
                    "use_normalization": False,
                    "sqrt_dim": False,
                }
            
        self.backbone = efficientnet_b2(weights=None)
        self.feature_dim = self.backbone.classifier[1].in_features

        self.features_1 = nn.Sequential(*list(self.backbone.children())[:-1]) 
        
        # Create Gaussian RAMA layer before the linear layer in the network
        if use_rama:
            self.rama_linearLayer = GaussianRAMALayer(
                self.feature_dim, 
                self.feature_dim, 
                rama_config['lambda_value'], 
                rama_config.get('use_normalization', False),
                rama_config.get('activation', 'relu'),
                rama_config.get('sqrt_dim', False),
            )

        # Dropout và Linear
        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.features_2 = nn.Sequential(
            self.dropout,
            self.fc
        )
        
        # Initialize hooks for feature extraction
        self.hooks = []
        self.before_rama_features = None
        self.after_rama_features = None

    def forward(self, x, lambda_value=None):
        """Forward pass through the EfficientNet-B2 model with Gaussian RAMA layers."""
        out = self.features_1(x)
        out = out.view(out.size(0), -1)
        feature_out = out

        # Store features before RAMA for evaluation
        if self.use_rama:
            self.before_rama_features = out.detach().clone()
            
        # Apply RAMA before final classification (original position)
        if self.use_rama:
            out = self.rama_linearLayer(out, lambda_value)
            self.after_rama_features = out.detach().clone()

        out = self.features_2(out)
        
        return out, feature_out
    
def Feature_EfficientNet(num_classes=10, use_rama=False, rama_config=None):
    return feature_EfficientNet(num_classes, use_rama, rama_config)