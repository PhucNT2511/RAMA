import torch
import torch.nn as nn
import torch.nn.functional as F
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

def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class feature_EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10, use_rama=False, rama_config=None, rama_type='gaussian'):
        super(feature_EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

        self.use_rama = use_rama
        if rama_config is None:
            rama_config = {
                    "lambda_value": 1.0,  # This lambda_value for Gaussian
                    "activation": "relu",
                    "use_normalization": False,
                    "sqrt_dim": False,
                }
        # Create Gaussian RAMA layer before the linear layer in the network
        if use_rama:
            self.rama_linearLayer = GaussianRAMALayer(
                cfg['out_channels'][-1], 
                cfg['out_channels'][-1], 
                rama_config['lambda_value'], 
                rama_config.get('use_normalization', False),
                rama_config.get('activation', 'relu'),
                rama_config.get('sqrt_dim', False),
            )

        # Initialize hooks for feature extraction
        self.hooks = []
        self.before_rama_features = None
        self.after_rama_features = None

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, lambda_value=None):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        feature_out = out.clone()
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)

        if self.use_rama:
            self.before_rama_features = out.detach().clone()
            out = self.rama_linearLayer(out, lambda_value)
            self.after_rama_features = out.detach().clone()

        out = self.linear(out)
        return out, feature_out

def Feature_EfficientNetB0(num_classes=10, use_rama=False, rama_config=None):
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return feature_EfficientNet(cfg, num_classes, use_rama, rama_config)

def Feature_EfficientNetB2(num_classes=10, use_rama=False, rama_config=None):
    cfg = {
        'num_blocks':        [2, 3, 3, 4, 4, 5, 2],
        'expansion':         [1, 6, 6, 6, 6, 6, 6],
        'out_channels':      [16, 24, 48, 88, 120, 208, 352],
        'kernel_size':       [3, 3, 5, 3, 5, 5, 3],
        'stride':            [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate':      0.3,
        'drop_connect_rate': 0.2,
    }
    return feature_EfficientNet(cfg, num_classes, use_rama, rama_config)