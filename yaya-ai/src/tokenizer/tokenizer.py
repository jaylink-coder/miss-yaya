"""Yaya Tokenizer — BPE tokenizer wrapper with special tokens.

Wraps SentencePiece for BPE tokenization with support for special tokens
needed for multimodal input, tool use, and chat formatting.
"""

import os
from typing import List, Optional, Dict
from pathlib import Path

import sentencepiece as spm


def _make_close_tag(name: str) -> str:
    """Build a closing tag string like name='s' -> the closing tag for s."""
    return "<" + chr(47) + name + ">"


# Special token definitions
# Closing tags are built at runtime to avoid literal sequences in source
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = _make_close_tag("s")
IMAGE_START = "<image>"
IMAGE_END = _make_close_tag("image")
IMAGE_PAD = "<image_pad>"
TOOL_START = "<tool_call>"
TOOL_END = _make_close_tag("tool_call")
TOOL_RESULT = "<tool_result>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = _make_close_tag("|user|")
ASSISTANT_TOKEN = _make_close_tag("|assistant|")

SPECIAL_TOKENS_LIST = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    IMAGE_START, IMAGE_END, IMAGE_PAD,
    TOOL_START, TOOL_END, TOOL_RESULT,
    SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN,
]


class YayaTokenizer:
    """BPE Tokenizer for the Yaya model.

    Uses SentencePiece under the hood with added special tokens for
    multimodal, tool-use, and chat formatting.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize tokenizer.

        Args:
            model_path: Path to trained SentencePiece .model file.
                        If None, tokenizer must be trained first.
        """
        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self.model_path = model_path

        # Special token string-to-id mapping (populated after loading)
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str):
        """Load a trained SentencePiece model."""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path)
        self.model_path = model_path

        # Map special tokens to IDs
        self._build_special_token_map()

    def _build_special_token_map(self):
        """Build mapping from special token strings to their IDs."""
        self.special_token_to_id = {}
        self.id_to_special_token = {}

        for token in SPECIAL_TOKENS_LIST:
            token_id = self.sp_model.PieceToId(token)
            if token_id != self.sp_model.unk_id():
                self.special_token_to_id[token] = token_id
                self.id_to_special_token[token_id] = token

    @property
    def vocab_size(self) -> int:
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self.sp_model.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return self.special_token_to_id.get(PAD_TOKEN, 0)

    @property
    def bos_id(self) -> int:
        return self.special_token_to_id.get(BOS_TOKEN, 1)

    @property
    def eos_id(self) -> int:
        return self.special_token_to_id.get(EOS_TOKEN, 2)

    @property
    def unk_id(self) -> int:
        return self.special_token_to_id.get(UNK_TOKEN, 3)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.
            add_bos: Prepend beginning-of-sequence token.
            add_eos: Append end-of-sequence token.

        Returns:
            List of token IDs.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded.")

        token_ids = self.sp_model.Encode(text)

        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs.
            skip_special: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded.")

        if skip_special:
            special_ids = set(self.special_token_to_id.values())
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        return self.sp_model.Decode(token_ids)

    def batch_encode(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """Batch encode multiple texts.

        Args:
            texts: List of input strings.
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.
            max_length: Truncate to this length.
            padding: Pad all sequences to max_length.

        Returns:
            Dict with 'input_ids' and 'attention_mask'.
        """
        all_ids = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        # Truncate
        if max_length is not None:
            all_ids = [ids[:max_length] for ids in all_ids]

        # Pad
        attention_masks = []
        if padding:
            max_len = max_length or max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                pad_len = max_len - len(all_ids[i])
                attention_masks.append([1] * len(all_ids[i]) + [0] * pad_len)
                all_ids[i] = all_ids[i] + [self.pad_id] * pad_len
        else:
            attention_masks = [[1] * len(ids) for ids in all_ids]

        return {
            "input_ids": all_ids,
            "attention_mask": attention_masks,
        }

    def format_chat(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Format a chat conversation into a single string with special tokens.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
                      Roles: 'system', 'user', 'assistant'

        Returns:
            Formatted chat string with special tokens.
        """
        formatted_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted_parts.append(f"{SYSTEM_TOKEN}\n{content}")
            elif role == "user":
                formatted_parts.append(f"{USER_TOKEN}\n{content}")
            elif role == "assistant":
                formatted_parts.append(f"{ASSISTANT_TOKEN}\n{content}")

        return "\n".join(formatted_parts)
