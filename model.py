import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision.models import vit_b_32, ViT_B_32_Weights

# --------------------------------------------------------
# 1. Core Vision Transformer Blocks (From Scratch)
# --------------------------------------------------------

class TokenizationLayer(nn.Module):
    def __init__(self, dim, patch_dim, patch_height, patch_width):
        super().__init__()
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.norm1 = nn.LayerNorm(patch_dim)
        self.fc1 = nn.Linear(patch_dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.to_patch(x)
        x = self.norm1(x)
        x = self.fc1(x)
        return self.norm2(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = nn.functional.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.final_linear(out)
        return self.dropout(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        return self.dropout(self.fc2(x))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, PositionwiseFeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x)
        return self.ff(x)

# --------------------------------------------------------
# 2. End-to-End Segmentation Model using ViT Backbone
# --------------------------------------------------------

class SegmentationViT(nn.Module):
    """
    Semantic Segmentation model leveraging a pretrained ViT image encoder 
    and a custom transpose-convolution decoder for dense prediction.
    """
    def __init__(self, num_classes=151): 
        super().__init__()
        # Load a pretrained ViT backbone
        self.encoder = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        self.encoder.heads = nn.Identity() # Remove classification head
        
        # Simple Custom Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        n = x.shape[0]
        
        # 1. Process via ViT patch embedding
        x = self.encoder._process_input(x)
        batch_class_token = self.encoder.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 2. Extract latent sequence from transformer blocks
        x = self.encoder.encoder(x)
        
        # 3. Drop CLS token and reshape into 2D feature map
        x = x[:, 1:] 
        x = x.transpose(1, 2).reshape(n, 768, 7, 7) 
        
        # 4. Decode into segmentation mask
        out = self.decoder(x)
        
        # 5. Interpolate back to full spatial dimensions (224x224)
        out = nn.functional.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out