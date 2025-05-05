import torch
import torch.nn as nn
from torchvision.models import swin_t

from common.rama_layers import BernoulliRAMALayer, GaussianRAMALayer


class SwinTRAMAAdapter(nn.Module):
    """
    Specialized adapter module for integrating RAMA with SwinT architecture.
    This preserves the hierarchical structure of SwinT features while applying
    RAMA in a way that's compatible with shifted window attention mechanisms.
    
    Args:
        in_features (int): Number of input features/channels
        out_features (int): Number of output features/channels (if None, uses reduction factor)
        rama_config (dict): Configuration for RAMA layers
        rama_type (str): Type of RAMA layer to use ('bernoulli' or 'gaussian')
        stage_idx (int): Index of the SwinT stage (0-3) this adapter is applied to
    """
    def __init__(self, in_features, out_features=None, rama_config=None, rama_type='bernoulli', stage_idx=0):
        super().__init__()
        
        if rama_config is None:
            rama_config = {}
            
        if out_features is None:
            # More aggressive dimension reduction for earlier stages
            reduction_factor = rama_config.get('dim_reduction_factor', 1.0)
            # Apply slightly different reduction factors based on stage
            out_features = max(int(in_features * reduction_factor), 16)
        
        self.in_features = in_features
        self.out_features = out_features
        self.stage_idx = stage_idx
        self.rama_type = rama_type
        
        # Create RAMA layer for feature dimension reduction
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
        
        # Add projection layer to maintain compatibility with SwinT architecture
        self.proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
    
    def forward(self, x, p_value=None):
        """
        Forward pass applying RAMA to SwinT feature maps.
        
        Args:
            x: Input tensor from SwinT stage (batch_size, H, W, C)
            p_value: Value for the RAMA layer
        """
        batch_size, H, W, C = x.shape
        
        # Step 1: Apply RAMA to each token's feature vector
        # Reshape to (B*H*W, C) for RAMA processing
        x_flat = x.reshape(batch_size * H * W, C)
        x_flat = self.rama_layer(x_flat, p_value)
        
        # Step 2: Reshape back to sequence form with new feature dimension
        x = x_flat.reshape(batch_size, H * W, self.out_features)
        
        # Step 3: Apply projection and normalization (similar to SwinT's approach)
        x = self.proj(x)
        x = self.norm(x)

        # Step 4: Reshape back to feature map form for subsequent SwinT stages
        x = x.reshape(batch_size, H, W, self.out_features)
        return x
        
    def update_mask(self, p_value):
        """
        Update the RAMA mask (for Bernoulli type only).
        
        Args:
            p_value: New p-value for the RAMA layer
        """
        if self.rama_type == 'bernoulli' and hasattr(self.rama_layer, "update_mask"):
            self.rama_layer.update_mask(p_value)


class ImprovedSwinT(nn.Module):
    """
    Improved SwinT architecture with RAMA layers at configurable positions.
    
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
                    'positions': ['stage1', 'stage2', 'stage3', 'stage4', 'final']
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
                    'positions': ['stage1', 'stage2', 'stage3', 'stage4', 'final']
                }
            
        # Parse positions if provided as string
        if 'positions' in rama_config and isinstance(rama_config['positions'], str):
            rama_config['positions'] = rama_config['positions'].split(',')

        # Initialize backbone
        self.backbone = swin_t(weights=None)
        self.feature_dim = self.backbone.head.in_features  # Should be 768 for SwinT
        
        # Extract layers from SwinT backbone for more granular control
        self.patch_embed = self.backbone.features[0]
        
        # Stage 1: Blocks operating on 56x56 with 96 channels
        self.stage1 = self.backbone.features[1]
        # Stage 2: Blocks operating on 28x28 with 192 channels
        self.patch_merging1 = self.backbone.features[2]
        self.stage2 = self.backbone.features[3]
        # Stage 3: Blocks operating on 14x14 with 384 channels
        self.patch_merging2 = self.backbone.features[4]
        self.stage3 = self.backbone.features[5]
        # Stage 4: Blocks operating on 7x7 with 768 channels
        self.patch_merging3 = self.backbone.features[6]
        self.stage4 = self.backbone.features[7]
        
        self.norm = self.backbone.features[8] if len(self.backbone.features) > 8 else nn.Identity()
        
        # Track which positions have RAMA layers
        self.rama_positions = rama_config.get('positions', [])
        
        # Create RAMA layers for multiple stages if enabled
        if use_rama:
            # Define dimensions for each stage - SwinT stages
            stage_dims = [96, 192, 384, 768]  # Channel sizes for SwinT stages
            
            # Create RAMA stages
            self.rama_stages = nn.ModuleList()
            for i, dim in enumerate(stage_dims):
                stage_name = f'stage{i+1}'
                if stage_name in self.rama_positions:
                    # Create SwinTRAMAAdapter for this stage
                    stage_rama = SwinTRAMAAdapter(
                        in_features=dim,
                        out_features=dim,
                        rama_config=rama_config,
                        rama_type=rama_type,
                        stage_idx=i
                    )
                else:
                    # Use identity module if RAMA not applied at this position
                    stage_rama = nn.Identity()
                    
                self.rama_stages.append(stage_rama)
            
            # Create final RAMA layer if specified
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
        else:
            # Create dummy identity layers for consistency when RAMA is disabled
            self.rama_stages = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.rama_final = nn.Identity()
        
        # Classification head
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # For feature analysis
        self.intermediate_features = []
        self.final_features = None

    def forward(self, x, p_value=None):
        """
        Forward pass through the improved SwinT model with RAMA layers at selected positions.
        
        Args:
            x: Input tensor (B, C, H, W)
            p_value: Value controlling the RAMA parameter
        """
        # Clear intermediate features for this forward pass
        self.intermediate_features = []
        
        # Initial patch embedding: (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = self.patch_embed(x)
        
        # Stage 1: (B, 96, 56, 56) -> (B, 96, 56, 56)
        x = self.stage1(x)
        # Store features before RAMA
        if isinstance(x, torch.Tensor):
            self.intermediate_features.append(x.detach().clone())
        # Apply RAMA if enabled for stage1
        if self.use_rama and 'stage1' in self.rama_positions:
            # SwinT format is (B, H, W, C)
            x = self.rama_stages[0](x, p_value)

        # Patch Merging 1: (B, 96, 56, 56) -> (B, 192, 28, 28)
        x = self.patch_merging1(x)
        
        # Stage 2: (B, 192, 28, 28) -> (B, 192, 28, 28)
        x = self.stage2(x)
        if isinstance(x, torch.Tensor):
            self.intermediate_features.append(x.detach().clone())
        if self.use_rama and 'stage2' in self.rama_positions:
            x = self.rama_stages[1](x, p_value)
        
        # Patch Merging 2: (B, 192, 28, 28) -> (B, 384, 14, 14)
        x = self.patch_merging2(x)
        
        # Stage 3: (B, 384, 14, 14) -> (B, 384, 14, 14)
        x = self.stage3(x)
        if isinstance(x, torch.Tensor):
            self.intermediate_features.append(x.detach().clone())
        if self.use_rama and 'stage3' in self.rama_positions:
            x = self.rama_stages[2](x, p_value)
        
        # Patch Merging 3: (B, 384, 14, 14) -> (B, 768, 7, 7)
        x = self.patch_merging3(x)
        
        # Stage 4: (B, 768, 7, 7) -> (B, 768, 7, 7)
        x = self.stage4(x)
        if isinstance(x, torch.Tensor):
            self.intermediate_features.append(x.detach().clone())
        if self.use_rama and 'stage4' in self.rama_positions:
            x = self.rama_stages[3](x, p_value)
        
        # Apply final norm and global average pooling
        x = self.norm(x)
        x = x.mean(dim=[1, 2])  # Global average pooling
        
        # Store final features before RAMA
        self.final_features = x.detach().clone()
        
        # Apply final RAMA layer if enabled
        if self.use_rama and 'final' in self.rama_positions:
            x = self.rama_final(x, p_value)
        
        # Classification
        x = self.fc(x)  # (B, 768) -> (B, num_classes)
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

        # Update the final RAMA layer if active
        if 'final' in self.rama_positions and hasattr(self.rama_final, "update_mask"):
            self.rama_final.update_mask(p_value)
        
        # Update intermediate RAMA layers in active stages
        for i, stage_rama in enumerate(self.rama_stages):
            stage_name = f'stage{i+1}'
            if stage_name in self.rama_positions and hasattr(stage_rama, "update_mask"):
                stage_rama.update_mask(p_value)