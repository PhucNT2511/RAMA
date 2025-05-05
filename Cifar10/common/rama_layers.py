import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BernoulliRAMALayer(nn.Module):
    """
    A RAMA layer using Bernoulli distribution for random projections.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        p_value (float): Controls the Bernoulli parameter p.
        values (str): Values for Bernoulli: '0_1' for {0,1} or '-1_1' for {-1,1}.
        use_normalization (bool): Whether to apply layer normalization after projection.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
        lambda_value (float): Scaling factor for the output.
        sqrt_dim (bool): Whether to normalize by sqrt(input_dim).
    """
    def __init__(self, input_dim, output_dim, p_value, values='0_1', use_normalization=True, 
                 activation="relu", lambda_value=1.0, sqrt_dim=False):
        super(BernoulliRAMALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.values = values
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value
        self.sqrt_dim = sqrt_dim
        if sqrt_dim:
            self.sqrt_d = math.sqrt(input_dim)
        else:
            self.sqrt_d = 1

        # Initialize Bernoulli projection matrix
        if values == '0_1':
            projection = (torch.rand(input_dim, output_dim) < p_value).float()
        elif values == '-1_1':
            projection = 2 * (torch.rand(input_dim, output_dim) < p_value).float() - 1
        else:
            raise ValueError(f"Unknown values: {values}. Use '0_1' or '-1_1'")

        self.projection = nn.Parameter(projection, requires_grad=False)
        
        # Add layer normalization for stabilizing the output distribution
        if use_normalization:
            self.norm = nn.LayerNorm(output_dim)
            
        self.current_mask = projection.clone()  # Initialize with starting projection
        self.current_p = p_value

    def update_mask(self, p_value):
        """Update mask gradually by evolving a percentage of elements"""
        device = self.projection.device

        # Create new random mask based on p_value
        if self.values == '0_1':
            new_elements = (torch.rand_like(self.projection, device=device) < p_value).float()
        elif self.values == '-1_1':
            new_elements = 2 * (torch.rand_like(self.projection, device=device) < p_value).float() - 1

        self.current_mask = new_elements
        self.current_p = p_value

    def forward(self, x, p_value):
        """
        Forward pass through the Bernoulli RAMA layer.
        
        Args:
            x: Input tensor
            p_value: Value controlling the Bernoulli parameter p
        """
        # For element-wise multiplication, we need the mask to have the same shape as x
        # This requires reshaping since x is [batch_size, input_dim] and mask is [input_dim, output_dim]
        batch_size = x.size(0)
        
        # Generate a dynamic Bernoulli mask based on p_value
        # For inference or when p_value is None, use the stored projection
        # Apply element-wise multiplication (Hadamard product)
        if p_value is not None and self.training:
            if self.current_p != p_value:
                self.update_mask(p_value)
            mask = torch.diagonal(self.current_mask).unsqueeze(0).expand(batch_size, -1)
            out = x * mask
        else:
            proj = torch.diagonal(self.projection).unsqueeze(0).expand(batch_size, -1)
            out = x * proj

        # Apply correct scaling - multiply by lambda and normalize by sqrt_d if needed.
        out = out * self.lambda_value
        if self.sqrt_dim:
            # Correct scaling: multiply by 1/sqrt(d) for stability
            out = out / self.sqrt_d

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
            out = out * torch.sigmoid(out)
        return out


class BernoulliRAMA4D(nn.Module):
    """
    RAMA layer that directly applies 4D Bernoulli random projections to convolutional features.
    Useful for applying RAMA to intermediate layers in CNNs.
    
    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        p_value (float): Controls the Bernoulli parameter p.
        kernel_size (int): Size of the convolutional kernel.
        values (str): Values for Bernoulli: '0_1' for {0,1} or '-1_1' for {-1,1}.
        use_normalization (bool): Whether to apply batch normalization after projection.
        activation (str): Activation function to use.
        lambda_value (float): Scaling factor for the output.
        sqrt_dim (bool): Whether to normalize by sqrt(input_channels * kernel_size^2).
    """
    def __init__(self, input_channels, output_channels, p_value, kernel_size=3, 
                 values='0_1', use_normalization=True, activation="relu", 
                 lambda_value=0.3, sqrt_dim=False):
        super(BernoulliRAMA4D, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.values = values
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value
        padding = kernel_size // 2
        self.padding = padding
        
        # Calculate scaling factor
        if sqrt_dim:
            self.scale = math.sqrt(input_channels * kernel_size * kernel_size)
        else:
            self.scale = 1.0

        # Initialize 4D Bernoulli projection tensor
        if values == '0_1':
            projection = (torch.rand(output_channels, input_channels, kernel_size, kernel_size) < p_value).float()
        elif values == '-1_1':
            projection = 2 * (torch.rand(output_channels, input_channels,
                                         kernel_size, kernel_size) < p_value).float() - 1
        
        self.projection = nn.Parameter(projection, requires_grad=False)
        
        if use_normalization:
            self.norm = nn.BatchNorm2d(output_channels)
        
        self.current_mask = projection.clone()
        self.current_p = p_value

    def update_mask(self, p_value):
        """Update mask gradually by evolving a percentage of elements"""
        device = self.projection.device
        
        # Create new random mask based on p_value
        if self.values == '0_1':
            new_elements = (torch.rand(self.projection.size(), device=device) < p_value).float()
        elif self.values == '-1_1':
            new_elements = 2 * (torch.rand(self.projection.size(), device=device) < p_value).float() - 1
        self.current_mask = new_elements
        self.current_p = p_value

    def forward(self, x, p_value):
        """Forward pass using 4D convolution with Bernoulli random weights"""
        device = x.device
        
        # Select projection weights
        if p_value is not None and self.training:
            if self.current_p != p_value:
                self.update_mask(p_value)
            weights = self.current_mask.to(device)
        else:
            weights = self.projection.to(device)

        # Apply convolution with the random Bernoulli weights
        out = F.conv2d(x, weights, padding=self.padding)
        
        # Apply scaling
        out = out * (self.lambda_value / self.scale)
        
        # Apply normalization if specified
        if self.use_normalization:
            out = self.norm(out)
        
        # Apply activation
        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "leaky_relu":
            out = F.leaky_relu(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        elif self.activation == "sigmoid":
            out = torch.sigmoid(out)
        
        # Apply residual connection
        return x + out


class GaussianRAMALayer(nn.Module):
    """
    A RAMA layer using Gaussian distribution for random projections.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        use_normalization (bool): Whether to apply layer normalization after projection.
        activation (str): Activation function to use. Options: relu, leaky_relu, tanh, sigmoid.
        lambda_value (float): Scaling factor for the output.
        sqrt_dim (bool): Whether to normalize by sqrt(input_dim).
    """
    def __init__(self, input_dim, output_dim, mu=0.0, sigma=1.0, use_normalization=True, 
                 activation="relu", lambda_value=1.0, sqrt_dim=False):
        super(GaussianRAMALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mu = mu
        self.sigma = sigma
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value
        self.sqrt_dim = sqrt_dim
        
        if sqrt_dim:
            self.sqrt_d = math.sqrt(input_dim)
        else:
            self.sqrt_d = 1
            
        # Initialize Gaussian projection matrix
        projection = torch.randn(input_dim, output_dim) * sigma + mu
        self.projection = nn.Parameter(projection, requires_grad=False)
        
        # Add layer normalization for stabilizing the output distribution
        if use_normalization:
            self.norm = nn.LayerNorm(output_dim)
            
        # Initialize current mask for potential dynamic updates
        self.current_mask = projection.clone()
        self.current_sigma = sigma
        self.current_mu = mu

    def update_mask(self, sigma=None, mu=None):
        """Update mask with new Gaussian parameters if provided"""
        device = self.projection.device
        
        # Use current values if not provided
        sigma = sigma if sigma is not None else self.current_sigma
        mu = mu if mu is not None else self.current_mu
        
        # Create new random mask with updated parameters
        new_elements = torch.randn(self.input_dim, self.output_dim, device=device) * sigma + mu
        self.current_mask = new_elements
        self.current_sigma = sigma
        self.current_mu = mu

    def forward(self, x, sigma=None):  # Keeps API compatible with Bernoulli variant
        """
        Forward pass through the Gaussian RAMA layer using element-wise multiplication.
        
        Args:
            x: Input tensor
            sigma: Optional parameter to dynamically update the random projection (unused by default)
        """
        # Get batch size for reshaping
        batch_size = x.size(0)
        
        # Apply element-wise multiplication (Hadamard product)
        if sigma is not None and self.training:
            if sigma != self.current_sigma:
                self.update_mask(sigma=sigma)
            mask = torch.diagonal(self.current_mask).unsqueeze(0).expand(batch_size, -1)
            out = x * mask
        else:
            proj = torch.diagonal(self.projection).unsqueeze(0).expand(batch_size, -1)
            out = x * proj

        # Apply correct scaling
        out = out * self.lambda_value
        if self.sqrt_dim:
            out = out / self.sqrt_d
            
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
            out = out * torch.sigmoid(out)
        return out


class GaussianRAMA4D(nn.Module):
    """
    RAMA layer that directly applies 4D Gaussian random projections to convolutional features.
    
    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        kernel_size (int): Size of the convolutional kernel.
        use_normalization (bool): Whether to apply batch normalization.
        activation (str): Activation function to use.
        lambda_value (float): Scaling factor for the output.
        sqrt_dim (bool): Whether to normalize by sqrt(input_channels * kernel_size^2).
    """
    def __init__(self, input_channels, output_channels, mu=0.0, sigma=1.0, kernel_size=3,
                 use_normalization=True, activation="relu", lambda_value=0.3, sqrt_dim=False):
        super(GaussianRAMA4D, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.mu = mu
        self.sigma = sigma
        self.activation = activation
        self.use_normalization = use_normalization
        self.lambda_value = lambda_value
        
        padding = kernel_size // 2
        self.padding = padding
        
        # Calculate scaling factor
        if sqrt_dim:
            self.scale = math.sqrt(input_channels * kernel_size * kernel_size)
        else:
            self.scale = 1.0
            
        # Initialize 4D Gaussian projection tensor
        projection = torch.randn(output_channels, input_channels, kernel_size, kernel_size) * sigma + mu
        self.projection = nn.Parameter(projection, requires_grad=False)
        
        if use_normalization:
            self.norm = nn.BatchNorm2d(output_channels)
            
        # Initialize current mask for potential dynamic updates
        self.current_mask = projection.clone()
        self.current_sigma = sigma
        self.current_mu = mu

    def update_mask(self, sigma=None, mu=None):
        """Update mask with new Gaussian parameters if provided"""
        device = self.projection.device
        
        # Use current values if not provided
        sigma = sigma if sigma is not None else self.current_sigma
        mu = mu if mu is not None else self.current_mu
        
        # Create new random mask with updated parameters
        new_elements = torch.randn(self.output_channels, self.input_channels, 
                                   self.kernel_size, self.kernel_size, 
                                   device=device) * sigma + mu
        self.current_mask = new_elements
        self.current_sigma = sigma
        self.current_mu = mu

    def forward(self, x, sigma=None):  # sigma parameter for API consistency
        """Forward pass using element-wise multiplication with Gaussian random weights"""
        device = x.device
        batch_size, channels, height, width = x.shape
        
        # Select projection weights
        if sigma is not None and self.training:
            if sigma != self.current_sigma:
                self.update_mask(sigma=sigma)
            weights = self.current_mask.to(device)
        else:
            weights = self.projection.to(device)
            
        # Create a channel-wise mask for element-wise multiplication
        # We'll use the center pixel of each kernel as the multiplicative factor
        center = self.kernel_size // 2
        channel_mask = weights[:, :, center, center]  # [output_channels, input_channels]
        
        # Reshape for broadcasting: [1, output_channels, 1, 1]
        channel_mask = channel_mask.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        
        # Apply element-wise multiplication across channels
        out = x * channel_mask
        
        # Apply scaling
        out = out * (self.lambda_value / self.scale)
        
        # Apply normalization if specified
        if self.use_normalization:
            out = self.norm(out)
        
        # Apply activation
        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "leaky_relu":
            out = F.leaky_relu(out)
        elif self.activation == "tanh":
            out = torch.tanh(out)
        elif self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "silu":
            out = out * torch.sigmoid(out)
        
        # Apply residual connection
        return x + out


class ProgressiveRandomConv(nn.Module):
    """
    Progressive Random Convolution layer based on CVPR 2023 paper:
    "Progressive Random Convolutions for Single Domain Generalization"
    
    The key idea is to apply random convolution operations with frozen weights
    to improve model robustness and generalization to unseen domains.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to all sides of the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        weight_init (str): Weight initialization method: 'normal', 'uniform', or 'kaiming'.
        scale_factor (float): Scaling factor for weights.
        progressive_scale (bool): Whether to progressively adjust the scale during training.
        use_residual (bool): Whether to use residual connection.
        use_norm (bool): Whether to apply batch normalization.
        activation (str): Activation function to use.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        weight_init='normal',
        scale_factor=0.1,
        progressive_scale=True,
        use_residual=True,
        use_norm=True,
        activation="relu"
    ):
        super(ProgressiveRandomConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale_factor = scale_factor
        self.progressive_scale = progressive_scale
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.activation = activation
        self.current_scale = scale_factor

        # Initialize weights using specified method
        if weight_init == 'normal':
            weights = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        elif weight_init == 'uniform':
            weights = torch.rand(out_channels, in_channels // groups, kernel_size, kernel_size) * 2 - 1
        elif weight_init == 'kaiming':
            weights = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
            weights = weights * math.sqrt(2. / (in_channels * kernel_size * kernel_size))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}. Use 'normal', 'uniform', or 'kaiming'")
        
        # Scale the weights
        weights = weights * scale_factor
        
        # Register weights as parameter (but frozen)
        self.weight = nn.Parameter(weights, requires_grad=False)
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # Batch normalization for stabilizing output distribution
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        
        # Additional random weights for progressive mixing
        self.mix_ratio = 0.5  # Default mixing ratio
        self.alternate_weight = None
    
    def update_progressive_scale(self, scale):
        """Update the scale factor for the random weights"""
        if self.progressive_scale:
            self.current_scale = scale
    
    def update_mix_ratio(self, ratio):
        """Update the mixing ratio between original and alternate weights"""
        self.mix_ratio = ratio
        
    def generate_alternate_weights(self):
        """Generate a different set of random weights for mixing"""
        if self.alternate_weight is None:
            device = self.weight.device
            alt_weights = torch.randn_like(self.weight, device=device) * self.current_scale
            self.alternate_weight = alt_weights
        return self.alternate_weight
        
    def forward(self, x, scale=None, mix_ratio=None):
        """
        Forward pass using random convolution.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            scale: Optional scale factor to use during this forward pass
            mix_ratio: Optional mixing ratio for progressive random convolution
        """
        device = x.device
        
        # Update scale if provided
        if scale is not None and self.progressive_scale:
            self.current_scale = scale
            
        # Update mix_ratio if provided
        if mix_ratio is not None:
            self.mix_ratio = mix_ratio
            
        # Scale the weights dynamically
        if self.current_scale != self.scale_factor:
            scaled_weight = self.weight * (self.current_scale / self.scale_factor)
        else:
            scaled_weight = self.weight
            
        # Progressive random mixing for domain randomization
        if self.training and self.mix_ratio < 1.0:
            if self.alternate_weight is None:
                self.generate_alternate_weights()
            
            # Mix original weights with alternate weights
            mixed_weight = self.mix_ratio * scaled_weight + (1 - self.mix_ratio) * self.alternate_weight
        else:
            mixed_weight = scaled_weight
            
        # Apply convolution with the random weights
        out = F.conv2d(
            x,
            mixed_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        # Apply normalization if specified
        if self.use_norm:
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
            out = out * torch.sigmoid(out)
        
        # Apply residual connection if specified
        if self.use_residual and self.in_channels == self.out_channels:
            out = x + out
            
        return out

    def progressive_update(self, epoch, total_epochs, warmup_epochs=5, schedule='linear'):
        """
        Update parameters progressively based on training epoch.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            schedule: Scheduling strategy: 'linear', 'cosine', 'step'
        """
        if not self.progressive_scale:
            return
            
        if epoch < warmup_epochs:
            # Warmup phase - gradually increase scale
            progress = epoch / warmup_epochs
            new_scale = self.scale_factor * progress
        else:
            # Main training phase - adjust according to schedule
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            
            if schedule == 'linear':
                new_scale = self.scale_factor * (1.0 - 0.5 * progress)
            elif schedule == 'cosine':
                new_scale = self.scale_factor * (0.5 + 0.5 * math.cos(math.pi * progress))
            elif schedule == 'step':
                if progress < 0.3:
                    new_scale = self.scale_factor
                elif progress < 0.6:
                    new_scale = self.scale_factor * 0.7
                else:
                    new_scale = self.scale_factor * 0.5
            else:
                new_scale = self.scale_factor
                
        self.current_scale = max(new_scale, self.scale_factor * 0.1)  # Ensure minimum scale
        
        # Also update mixing ratio - increase original weight influence over time
        self.mix_ratio = min(0.5 + 0.5 * (epoch / total_epochs), 1.0)


class RandConv(nn.Module):
    """
    RandConv layer as described in the paper:
    "Robust and Generalizable Visual Representation Learning via Random Convolutions"
    (https://arxiv.org/pdf/2007.13003)
    
    This layer applies random convolutions to input features to improve robustness 
    and generalization. Random convolutions act as a form of regularization and
    help models learn more domain-invariant features.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels, defaults to same as in_channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to input.
        p_random (float): Probability of applying random convolution during training.
        distribution (str): Distribution for random weights: 'normal', 'uniform', or 'kaiming'.
        reinit (bool): Whether to re-initialize weights at each forward pass.
        use_residual (bool): Whether to add a residual connection.
        std (float): Standard deviation for normal distribution.
        activation (str): Activation function to use.
        batchnorm (bool): Whether to apply batch normalization.
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        padding=1,
        p_random=0.5,
        distribution='normal',
        reinit=False,
        use_residual=True,
        std=0.1,
        activation='relu',
        batchnorm=True
    ):
        super(RandConv, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.p_random = p_random
        self.distribution = distribution
        self.reinit = reinit
        self.use_residual = use_residual
        self.std = std
        self.activation = activation
        
        # Create both a standard learnable convolution and a random fixed convolution
        # Standard learnable convolution (used with probability 1-p_random)
        self.conv_standard = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Initialize random convolution weights (used with probability p_random)
        random_weights = self._initialize_random_weights()
        self.conv_random_weight = nn.Parameter(random_weights, requires_grad=False)
        
        # Batch normalization
        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
            
        # Flag to determine if we use random or standard convolution
        self.use_random = False
        
    def _initialize_random_weights(self):
        """Initialize the random convolution weights based on the specified distribution"""
        if self.distribution == 'normal':
            weights = torch.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            ) * self.std
        elif self.distribution == 'uniform':
            weights = torch.rand(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            ) * 2 - 1  # Scale to [-1, 1]
            weights = weights * self.std
        elif self.distribution == 'kaiming':
            weights = torch.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            )
            # Kaiming initialization scaling factor
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            weights = weights * math.sqrt(2.0 / fan_in) * self.std
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        return weights
        
    def _maybe_reinit_random_weights(self):
        """Re-initialize random weights if reinit flag is set"""
        if self.reinit:
            with torch.no_grad():
                random_weights = self._initialize_random_weights()
                self.conv_random_weight.copy_(random_weights)
        
    def forward(self, x):
        """
        Forward pass through the RandConv layer.
        
        During training, with probability p_random, applies random convolution,
        otherwise applies the standard learnable convolution.
        During evaluation, always uses the standard convolution.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor [batch_size, out_channels, height, width]
        """
        # Determine whether to use random convolution (only in training)
        self.use_random = self.training and torch.rand(1).item() < self.p_random
        
        if self.use_random:
            # Maybe re-initialize the random weights
            self._maybe_reinit_random_weights()
            
            # Apply random convolution
            out = F.conv2d(
                x,
                self.conv_random_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding
            )
        else:
            # Apply standard learnable convolution
            out = self.conv_standard(x)
        
        # Apply batch normalization if enabled
        if self.batchnorm:
            out = self.bn(out)
        
        # Apply activation function
        if self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'leaky_relu':
            out = F.leaky_relu(out)
        elif self.activation == 'tanh':
            out = torch.tanh(out)
        elif self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.activation == 'silu':
            out = out * torch.sigmoid(out)
        
        # Apply residual connection if enabled and dimensions match
        if self.use_residual and self.in_channels == self.out_channels and self.stride == 1:
            out = x + out
        return out


class RandConvModule(nn.Module):
    """
    A module that wraps multiple RandConv layers to create a replaceable building block 
    for neural networks. This can be used to replace standard convolutional blocks in 
    various architectures.
    
    As described in the paper:
    "Robust and Generalizable Visual Representation Learning via Random Convolutions"
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for the final layer.
        mid_channels (int): Number of channels in middle layers.
        depth (int): Number of RandConv layers.
        p_random (float): Probability of applying random convolution.
        distribution (str): Weight distribution for random convolutions.
        reinit (bool): Whether to re-initialize random weights in each forward pass.
        use_residual (bool): Whether to use residual connections.
        downsample (bool): Whether to downsample the spatial dimensions (stride=2 in first layer).
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        depth=2,
        p_random=0.5,
        distribution='normal',
        reinit=False,
        use_residual=True,
        downsample=False
    ):
        super(RandConvModule, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        layers = []
        
        # First layer (can downsample if needed)
        stride = 2 if downsample else 1
        layers.append(
            RandConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                p_random=p_random,
                distribution=distribution,
                reinit=reinit,
                use_residual=use_residual and in_channels == mid_channels and stride == 1
            )
        )
        
        # Middle layers
        for i in range(depth - 2):
            layers.append(
                RandConv(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    p_random=p_random,
                    distribution=distribution,
                    reinit=reinit,
                    use_residual=use_residual
                )
            )
            
        # Last layer
        if depth > 1:
            layers.append(
                RandConv(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    p_random=p_random,
                    distribution=distribution,
                    reinit=reinit,
                    use_residual=use_residual and mid_channels == out_channels
                )
            )
            
        # Downsample shortcut for residual connections if input and output dimensions differ
        self.downsample = None
        if downsample or (use_residual and (in_channels != out_channels)):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
            
        self.layers = nn.Sequential(*layers)
        self.use_residual = use_residual
        
    def forward(self, x):
        """
        Forward pass through the RandConvModule.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor [batch_size, out_channels, height, width]
        """
        out = self.layers(x)
        
        # Apply residual connection with potential shortcut if dimensions differ
        if self.use_residual and self.downsample is not None:
            residual = self.downsample(x)
            out = out + residual
            
        return out