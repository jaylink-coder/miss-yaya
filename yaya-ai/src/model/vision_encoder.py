"""Phase 12: Vision Encoder for Yaya — image to visual token embeddings."""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection  = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return x


class VisionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        return x + self.mlp(self.norm2(x))


class YayaVisionEncoder(nn.Module):
    """
    Converts images to visual token embeddings for Yaya's language model.
    Output: (batch, num_visual_tokens, language_hidden_size)
    These tokens are prepended to the text sequence.
    """
    def __init__(self, image_size=224, patch_size=16, vision_dim=768,
                 vision_layers=12, vision_heads=12, language_dim=768, num_visual_tokens=64):
        super().__init__()
        self.patch_embed     = PatchEmbedding(image_size, patch_size, 3, vision_dim)
        self.transformer     = nn.Sequential(*[VisionBlock(vision_dim, vision_heads) for _ in range(vision_layers)])
        self.norm            = nn.LayerNorm(vision_dim)
        self.num_visual_tokens = num_visual_tokens
        self.pool            = nn.AdaptiveAvgPool1d(num_visual_tokens)
        self.projection      = nn.Linear(vision_dim, language_dim)

    def forward(self, images):
        x = self.patch_embed(images)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.projection(x)


def preprocess_image(image_path, image_size=224):
    try:
        from PIL import Image
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("pip install Pillow torchvision")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
