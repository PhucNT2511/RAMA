import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Feature_model.rama_layers import BernoulliRAMALayer, GaussianRAMALayer, BernoulliRAMA4D, GaussianRAMA4D


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, use_rama=False, rama_config=None,
                 rama_positions=None, rama_type='bernoulli'):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        
        # Track if we're using RAMA
        self.use_rama = use_rama
        self.rama_type = rama_type
        self.feature_dim = nChannels[3]  # Final feature dimension

        # Set up RAMA positions
        if rama_positions is None:
            self.rama_positions = {
                'block1': False,  # After first network block
                'block2': True,   # After second network block
                'block3': True,   # After third network block
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
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Create RAMA layers if enabled
        if use_rama:
            self._create_rama_layers(nChannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def _create_rama_layers(self, nChannels):
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
                    nChannels[1], nChannels[1], p_value, **config
                )
                
            if self.rama_positions.get('block2', False):
                self.rama_block2 = BernoulliRAMA4D(
                    nChannels[2], nChannels[2], p_value, **config
                )
                
            if self.rama_positions.get('block3', False):
                self.rama_block3 = BernoulliRAMA4D(
                    nChannels[3], nChannels[3], p_value, **config
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
                    nChannels[1], nChannels[1], mu, sigma, **config
                )
                
            if self.rama_positions.get('block2', False):
                self.rama_block2 = GaussianRAMA4D(
                    nChannels[2], nChannels[2], mu, sigma, **config
                )
                
            if self.rama_positions.get('block3', False):
                self.rama_block3 = GaussianRAMA4D(
                    nChannels[3], nChannels[3], mu, sigma, **config
                )
                
            # Use GaussianRAMALayer for final layer (operates on flattened features)
            if self.rama_positions.get('final', False):
                self.rama_final = GaussianRAMALayer(
                    self.feature_dim, self.feature_dim, mu, sigma, **config
                )

    def forward(self, x, p_value=None):
        out = self.conv1(x)
        
        # Block 1
        out = self.block1(out)
        if self.use_rama and self.rama_positions.get('block1', False):
            if hasattr(self, 'rama_block1'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block1(out, p_value)
                else:  # gaussian
                    out = self.rama_block1(out, p_value if self.rama_type == 'gaussian' else None)
        
        # Block 2
        out = self.block2(out)
        if self.use_rama and self.rama_positions.get('block2', False):
            if hasattr(self, 'rama_block2'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block2(out, p_value)
                else:  # gaussian
                    out = self.rama_block2(out, p_value if self.rama_type == 'gaussian' else None)
        
        # Block 3
        out = self.block3(out)
        if self.use_rama and self.rama_positions.get('block3', False):
            if hasattr(self, 'rama_block3'):
                if self.rama_type == 'bernoulli':
                    out = self.rama_block3(out, p_value)
                else:  # gaussian
                    out = self.rama_block3(out, p_value if self.rama_type == 'gaussian' else None)
        
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        
        # Save the features before the final classifier
        feature_out = out.clone()
        
        # Apply RAMA after pooling, before FC (if enabled)
        if self.use_rama and self.rama_positions.get('final', False):
            self.before_rama_features = out.detach().clone()
            if hasattr(self, 'rama_final'):
                if self.rama_type == 'bernoulli' or self.rama_type == 'gaussian':
                    out = self.rama_final(out, p_value)
            self.after_rama_features = out.detach().clone()
        
        # Apply classifier
        out = self.fc(out)
        
        # Return both class outputs and features if using RAMA
        if self.use_rama:
            return out, feature_out
        else:
            return out


def Wide_ResNet28_10(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3,
                      use_rama=use_rama, rama_config=rama_config, rama_positions=rama_positions, rama_type=rama_type)


def Wide_ResNet40_10(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=10, dropRate=0.3,
                      use_rama=use_rama, rama_config=rama_config, rama_positions=rama_positions, rama_type=rama_type)


def Wide_ResNet16_8(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.3,
                      use_rama=use_rama, rama_config=rama_config, rama_positions=rama_positions, rama_type=rama_type)


def Wide_ResNet34_10(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.3,
                      use_rama=use_rama, rama_config=rama_config, rama_positions=rama_positions, rama_type=rama_type)


def Wide_ResNet10_2(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return WideResNet(depth=10, num_classes=num_classes, widen_factor=2, dropRate=0.3,
                      use_rama=use_rama, rama_config=rama_config, rama_positions=rama_positions, rama_type=rama_type)


def Wide_ResNet18_2(num_classes=10, use_rama=False, rama_config=None, rama_positions=None, rama_type='bernoulli'):
    return WideResNet(depth=18, num_classes=num_classes, widen_factor=2, dropRate=0.3,
                      use_rama=use_rama, rama_config=rama_config, rama_positions=rama_positions, rama_type=rama_type)


def test():
    net = Wide_ResNet28_10()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
