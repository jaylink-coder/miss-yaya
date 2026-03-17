"""Multimodal Fusion — Unified Embedding Decoder Architecture (Method A).

Connects the Vision Encoder to the LLM by projecting visual features into
the same embedding space as text tokens, then concatenating them.

Architecture:
    Image -> VisionEncoder -> Projector (MLP) -> Visual Tokens
    Text  -> Tokenizer -> Embedding           -> Text Tokens
    [Visual Tokens] + [Text Tokens] -> LLM Decoder -> Output

Reference: LLaVA (https://arxiv.org/abs/2304.08485)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

from src.model.vision_encoder import VisionEncoder
from src.model.yaya_model import YayaModel, YayaForCausalLM
from src.model.normalization import RMSNorm
from src.utils.config import ModelConfig


class VisionProjector(nn.Module):
    """Projects vision encoder outputs into the LLM embedding space.

    A 2-layer MLP that maps from vision_hidden_size to llm_hidden_size.
    This is the "bridge" between vision and language.

    Architecture:
        Vision features [B, num_patches, vision_dim]
        -> Linear(vision_dim, llm_dim)
        -> GELU
        -> Linear(llm_dim, llm_dim)
        -> Projected features [B, num_patches, llm_dim]
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space.

        Args:
            vision_features: [batch, num_patches, vision_hidden_size]

        Returns:
            Projected features [batch, num_patches, llm_hidden_size]
        """
        x = self.linear1(vision_features)
        x = self.act(x)
        x = self.linear2(x)
        return x


class YayaMultimodalModel(nn.Module):
    """Complete multimodal model combining vision and language.

    Training phases:
        Phase 1 (projector only): Freeze vision encoder + LLM, train projector
        Phase 2 (full fine-tune):  Unfreeze LLM, keep vision encoder frozen or partially unfrozen

    Input handling:
        - Text-only: Normal LLM forward pass
        - Image+Text: Encode image -> project -> concatenate with text embeddings -> LLM
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Language model
        self.language_model = YayaForCausalLM(config)

        # Vision components (only if vision enabled)
        if config.vision.enabled:
            self.vision_encoder = VisionEncoder(config.vision)
            self.vision_projector = VisionProjector(
                vision_hidden_size=config.vision.vision_hidden_size,
                llm_hidden_size=config.hidden_size,
            )
        else:
            self.vision_encoder = None
            self.vision_projector = None

    def freeze_vision_encoder(self):
        """Freeze vision encoder weights (used in projector training phase)."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder weights."""
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

    def freeze_language_model(self):
        """Freeze LLM weights (used in projector training phase)."""
        for param in self.language_model.parameters():
            param.requires_grad = False

    def unfreeze_language_model(self):
        """Unfreeze LLM weights."""
        for param in self.language_model.parameters():
            param.requires_grad = True

    def setup_projector_training(self):
        """Configure for Phase 1: train projector only.

        Freezes both vision encoder and LLM, only projector is trainable.
        """
        self.freeze_vision_encoder()
        self.freeze_language_model()
        if self.vision_projector is not None:
            for param in self.vision_projector.parameters():
                param.requires_grad = True

    def setup_full_multimodal_training(self):
        """Configure for Phase 2: train projector + LLM.

        Vision encoder stays frozen, LLM and projector are trainable.
        """
        self.freeze_vision_encoder()
        self.unfreeze_language_model()
        if self.vision_projector is not None:
            for param in self.vision_projector.parameters():
                param.requires_grad = True

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into visual tokens for the LLM.

        Args:
            pixel_values: Images [batch, 3, image_size, image_size]

        Returns:
            Visual tokens [batch, num_patches, hidden_size]
        """
        if self.vision_encoder is None:
            raise RuntimeError("Vision encoder not initialized. Set vision.enabled=True in config.")

        vision_features = self.vision_encoder(pixel_values)
        visual_tokens = self.vision_projector(vision_features)
        return visual_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass handling both text-only and multimodal inputs.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
            position_ids: Position indices
            pixel_values: Images [batch, 3, H, W] (None for text-only)
            image_positions: Indices in input_ids where image tokens should be inserted
            labels: Target token IDs for loss computation
            past_key_values: KV cache for generation
            use_cache: Whether to return KV cache

        Returns:
            Dict with 'loss', 'logits', and optionally 'past_key_values'
        """
        # Text-only forward pass
        if pixel_values is None:
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # Multimodal forward pass
        # 1. Get text embeddings
        text_embeds = self.language_model.model.embed_tokens(input_ids)

        # 2. Encode images to visual tokens
        visual_tokens = self.encode_images(pixel_values)

        # 3. Merge visual tokens into the text embedding sequence
        inputs_embeds = self._merge_visual_tokens(
            text_embeds, visual_tokens, image_positions, input_ids
        )

        # 4. Build new attention mask to account for inserted visual tokens
        if attention_mask is not None:
            batch_size = inputs_embeds.shape[0]
            new_seq_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(
                (batch_size, new_seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # 5. Forward through LLM with merged embeddings
        return self.language_model(
            input_ids=None,  # Using inputs_embeds instead
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )

    def _merge_visual_tokens(
        self,
        text_embeds: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_positions: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Merge visual tokens into the text embedding sequence.

        Replaces placeholder image tokens in the text sequence with actual
        visual token embeddings from the vision encoder.

        Args:
            text_embeds: Text embeddings [batch, text_seq_len, hidden_size]
            visual_tokens: Visual tokens [batch, num_patches, hidden_size]
            image_positions: Start positions for image insertion [batch]
            input_ids: Original token IDs (to find image placeholder tokens)

        Returns:
            Merged embeddings [batch, new_seq_len, hidden_size]
        """
        batch_size = text_embeds.shape[0]
        num_visual_tokens = visual_tokens.shape[1]
        device = text_embeds.device

        if image_positions is None:
            # Simple concatenation: [visual_tokens, text_embeds]
            return torch.cat([visual_tokens, text_embeds], dim=1)

        # Replace image placeholder tokens with visual tokens
        merged = []
        for i in range(batch_size):
            pos = image_positions[i].item()
            before = text_embeds[i, :pos]
            after = text_embeds[i, pos + 1:]  # Skip placeholder token
            merged_seq = torch.cat([before, visual_tokens[i], after], dim=0)
            merged.append(merged_seq)

        # Pad to same length and stack
        max_len = max(m.shape[0] for m in merged)
        padded = torch.zeros(batch_size, max_len, text_embeds.shape[-1], device=device)
        for i, m in enumerate(merged):
            padded[i, :m.shape[0]] = m

        return padded
