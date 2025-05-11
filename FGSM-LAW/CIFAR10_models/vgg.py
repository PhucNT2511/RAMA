'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from Feature_model.rama_layers import BernoulliRAMALayer, GaussianRAMALayer, BernoulliRAMA4D, GaussianRAMA4D


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
        super(VGG, self).__init__()
        self.rama_type = rama_type
        self.feature_dim = 512
        self.use_rama = use_rama
        
        # Set up RAMA positions (where to apply RAMA layers)
        if rama_positions is None:
            self.rama_positions = {
                'block1': False,  # After first block (64 channels) 
                'block2': True,   # After second block (128 channels)
                'block3': True,   # After third block (256 channels)
                'block4': False,  # After fourth block (512 channels)
                'block5': False,  # After fifth block (512 channels)
                'final': True     # After flattening, before classifier
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
        
        # Create VGG blocks with separation points for RAMA insertion
        self.block1, end_channels = self._make_block(cfg[vgg_name], 0, 64, in_channels=3)
        self.block2, end_channels = self._make_block(cfg[vgg_name], end_channels, 128, prev_channels=end_channels)
        self.block3, end_channels = self._make_block(cfg[vgg_name], end_channels, 256, prev_channels=end_channels)
        self.block4, end_channels = self._make_block(cfg[vgg_name], end_channels, 512, prev_channels=end_channels)
        self.block5 = self._make_final_block(cfg[vgg_name], end_channels)
        
        # Create RAMA layers if enabled
        if use_rama:
            self._create_rama_layers()
        
        self.classifier = nn.Linear(512, 10)

    def _make_block(self, cfg, start_idx, target_channel, in_channels=3, prev_channels=None):
        """Create a VGG block ending at the first occurrence of target_channel"""
        layers = []
        
        # Use previous channels as starting channels if provided
        if prev_channels is not None:
            in_channels = prev_channels
            
        # Find the index where we reach the target channel
        end_idx = start_idx
        end_channels = in_channels
        
        while end_idx < len(cfg):
            x = cfg[end_idx]
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(inplace=True)]
                in_channels = x
                end_channels = x
                
            end_idx += 1
            
            # Stop when we reach the target channel or the end of config
            if end_channels == target_channel and x != 'M':
                break
                
        return nn.Sequential(*layers), end_idx
    
    def _make_final_block(self, cfg, start_idx):
        """Create the final block of VGG"""
        layers = []
        in_channels = cfg[start_idx-1] if start_idx > 0 and cfg[start_idx-1] != 'M' else 512
        
        for x in cfg[start_idx:]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                          nn.BatchNorm2d(x),
                          nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _create_rama_layers(self):
        """Create RAMA layers based on specified positions and type"""
        if self.rama_type == 'bernoulli':
            # Extract configuration for creating layers
            config = {
                "values": self.rama_config.get('values', '0_1'),
                "use_normalization": self.rama_config.get('use_normalization', True),
                "activation": self.rama_config.get('activation', 'relu'),
                "lambda_value": self.rama_config.get('lambda_value', 0.3),
                "sqrt_dim": self.rama_config.get('sqrt_dim', False),
            }
            p_value = self.rama_config.get('p_value', 0.5)
            
            # Create 4D Bernoulli RAMA layers for convolutional layers
            if self.rama_positions.get('block1', False):
                self.rama_block1 = BernoulliRAMA4D(
                    64, 64, p_value, **config
                )
                
            if self.rama_positions.get('block2', False):
                self.rama_block2 = BernoulliRAMA4D(
                    128, 128, p_value, **config
                )
                
            if self.rama_positions.get('block3', False):
                self.rama_block3 = BernoulliRAMA4D(
                    256, 256, p_value, **config
                )
                
            if self.rama_positions.get('block4', False):
                self.rama_block4 = BernoulliRAMA4D(
                    512, 512, p_value, **config
                )
                
            if self.rama_positions.get('block5', False):
                self.rama_block5 = BernoulliRAMA4D(
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

            # Configuration for Gaussian RAMA
            randconv_config = {
                "p_random": 0.5,
                "distribution": "normal",
                "std": sigma,
                "use_residual": True,
                "activation": config["activation"],
                "batchnorm": config["use_normalization"]
            }
            
            if self.rama_positions.get('block1', False):
                self.rama_block1 = GaussianRAMA4D(
                    in_channels=64, 
                    out_channels=64, 
                    **randconv_config
                )
                
            if self.rama_positions.get('block2', False):
                self.rama_block2 = GaussianRAMA4D(
                    in_channels=128, 
                    out_channels=128, 
                    **randconv_config
                )
                
            if self.rama_positions.get('block3', False):
                self.rama_block3 = GaussianRAMA4D(
                    in_channels=256, 
                    out_channels=256, 
                    **randconv_config
                )
                
            if self.rama_positions.get('block4', False):
                self.rama_block4 = GaussianRAMA4D(
                    in_channels=512, 
                    out_channels=512, 
                    **randconv_config
                )
                
            if self.rama_positions.get('block5', False):
                self.rama_block5 = GaussianRAMA4D(
                    in_channels=512, 
                    out_channels=512, 
                    **randconv_config
                )
                
            # Use GaussianRAMALayer for final layer (operates on flattened features)
            if self.rama_positions.get('final', False):
                self.rama_final = GaussianRAMALayer(
                    self.feature_dim, self.feature_dim, mu, sigma, **config
                )

    def forward(self, x, p_value=None):
        # Block 1
        out = self.block1(x)
        if self.use_rama and self.rama_positions.get('block1', False):
            if hasattr(self, 'rama_block1'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block1(out, p_value)
                else:  # gaussian
                    out = self.rama_block1(out)
        
        # Block 2
        out = self.block2(out)
        if self.use_rama and self.rama_positions.get('block2', False):
            if hasattr(self, 'rama_block2'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block2(out, p_value)
                else:  # gaussian
                    out = self.rama_block2(out)
        
        # Block 3
        out = self.block3(out)
        if self.use_rama and self.rama_positions.get('block3', False):
            if hasattr(self, 'rama_block3'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block3(out, p_value)
                else:  # gaussian
                    out = self.rama_block3(out)
        
        # Block 4
        out = self.block4(out)
        if self.use_rama and self.rama_positions.get('block4', False):
            if hasattr(self, 'rama_block4'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block4(out, p_value)
                else:  # gaussian
                    out = self.rama_block4(out)
        
        # Block 5
        out = self.block5(out)
        if self.use_rama and self.rama_positions.get('block5', False):
            if hasattr(self, 'rama_block5'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block5(out, p_value)
                else:  # gaussian
                    out = self.rama_block5(out)
        
        # Flatten features
        out = out.view(out.size(0), -1)
        feature_out = out  # Save features before final classifier
        
        # Apply RAMA after pooling, before FC (if enabled)
        if self.use_rama and self.rama_positions.get('final', False):
            self.before_rama_features = out.detach().clone()
            if hasattr(self, 'rama_final'):
                if self.rama_type == 'bernoulli' or self.rama_type == 'gaussian':
                    out = self.rama_final(out, p_value)
            self.after_rama_features = out.detach().clone()
        
        # Apply classifier
        out = self.classifier(out)
        
        # Return both class outputs and features if using RAMA
        if self.use_rama:
            return out, feature_out
        else:
            return out


def VGG11(use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return VGG('VGG11', use_rama, rama_config, rama_positions, rama_type)


def VGG13(use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return VGG('VGG13', use_rama, rama_config, rama_positions, rama_type)


def VGG16(use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return VGG('VGG16', use_rama, rama_config, rama_positions, rama_type)


def VGG19(use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return VGG('VGG19', use_rama, rama_config, rama_positions, rama_type)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
