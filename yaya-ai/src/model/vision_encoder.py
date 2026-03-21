"""Phase 12: Vision Encoder for Yaya — image to visual token embeddings."""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 *, hidden_size=None):
        super().__init__()
        dim = hidden_size if hidden_size is not None else embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection  = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x


class VisionAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out


class VisionMLP(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VisionTransformerBlock(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4.0,
                 *, intermediate_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn  = VisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_dim = intermediate_size or int(hidden_size * mlp_ratio)
        self.mlp   = VisionMLP(hidden_size, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Alias for backward compatibility
VisionBlock = VisionTransformerBlock


class VisionEncoder(nn.Module):
    """Converts images to visual token embeddings for Yaya's language model.

    Accepts either a VisionConfig dataclass or explicit keyword arguments.
    Output: (batch, num_patches, vision_hidden_size)
    """

    def __init__(self, config=None, *, image_size=224, patch_size=16, vision_dim=768,
                 vision_layers=12, vision_heads=12, language_dim=None, num_visual_tokens=None):
        super().__init__()
        if config is not None:
            image_size = getattr(config, "image_size", image_size)
            patch_size = getattr(config, "patch_size", patch_size)
            vision_dim = getattr(config, "vision_hidden_size", vision_dim)
            vision_layers = getattr(config, "vision_layers", vision_layers)
            vision_heads = getattr(config, "vision_heads", vision_heads)

        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, vision_dim)
        self.blocks = nn.Sequential(
            *[VisionTransformerBlock(vision_dim, vision_heads) for _ in range(vision_layers)]
        )
        self.norm = nn.LayerNorm(vision_dim)

        # Optional projection to language dim and pooling
        self._project = None
        self._pool = None
        if language_dim is not None and language_dim != vision_dim:
            self._project = nn.Linear(vision_dim, language_dim)
        if num_visual_tokens is not None:
            self._pool = nn.AdaptiveAvgPool1d(num_visual_tokens)

    def forward(self, images):
        x = self.patch_embed(images)
        x = self.blocks(x)
        x = self.norm(x)
        if self._pool is not None:
            x = self._pool(x.transpose(1, 2)).transpose(1, 2)
        if self._project is not None:
            x = self._project(x)
        return x


# Backward-compatible alias
YayaVisionEncoder = VisionEncoder


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
