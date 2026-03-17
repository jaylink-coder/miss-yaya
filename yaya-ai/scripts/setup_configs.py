"""Generate configuration files that contain XML-like closing tags.

Run once after cloning to create tokenizer.yaml and other configs
that contain special token strings with closing-tag syntax.

Usage:
    python scripts/setup_configs.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_closing(tag_name):
    """Build a closing tag string without literal closing-tag in source."""
    return "<" + chr(47) + tag_name + ">"


def write_file(rel_path, content):
    full = os.path.join(PROJECT_ROOT, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {rel_path}")


def gen_tokenizer_config():
    eos = make_closing("s")
    img_end = make_closing("image")
    tool_end = make_closing("tool_call")
    user_tok = make_closing("|user|")
    asst_tok = make_closing("|assistant|")

    return f"""# Tokenizer Training Configuration

tokenizer:
  algorithm: bpe
  vocab_size: 64000
  model_prefix: yaya_tokenizer
  output_dir: data/tokenizer
  character_coverage: 0.9995

  special_tokens:
    pad_token: '<pad>'
    unk_token: '<unk>'
    bos_token: '<s>'
    eos_token: '{eos}'
    image_start_token: '<image>'
    image_end_token: '{img_end}'
    image_pad_token: '<image_pad>'
    tool_start_token: '<tool_call>'
    tool_end_token: '{tool_end}'
    tool_result_token: '<tool_result>'
    system_token: '<|system|>'
    user_token: '{user_tok}'
    assistant_token: '{asst_tok}'

  # SentencePiece training settings
  training:
    byte_fallback: true
    split_digits: true
    allow_whitespace_only_pieces: true
    normalization_rule_name: identity
    remove_extra_whitespaces: false
    max_sentence_length: 16384
    num_threads: 4

  # Data sources for tokenizer training
  data_sources:
    - data/raw/text/*.txt
"""


def gen_data_pipeline_config():
    return """# Data Pipeline Configuration

data_pipeline:
  # Source directories
  sources:
    web_text:
      path: data/raw/web
      ratio: 0.45
      pattern: '*.txt'
    code:
      path: data/raw/code
      ratio: 0.12
      pattern: '*.txt'
    books:
      path: data/raw/books
      ratio: 0.12
      pattern: '*.txt'
    academic:
      path: data/raw/academic
      ratio: 0.08
      pattern: '*.txt'
    math:
      path: data/raw/math
      ratio: 0.06
      pattern: '*.txt'
    conversational:
      path: data/raw/conversational
      ratio: 0.07
      pattern: '*.txt'
    domain_business:
      path: data/raw/business
      ratio: 0.05
      pattern: '*.txt'
    multilingual:
      path: data/raw/multilingual
      ratio: 0.05
      pattern: '*.txt'

  # Output
  output_dir: data/processed
  shard_size: 100000000

  # Cleaning
  cleaning:
    remove_urls: true
    remove_emails: true
    normalize_whitespace: true
    normalize_unicode: true
    min_line_length: 10

  # Filtering
  filtering:
    min_doc_length: 100
    max_doc_length: 1000000
    max_special_char_ratio: 0.3
    min_alpha_ratio: 0.5

  # Deduplication
  deduplication:
    enabled: true
    method: sha256
"""


def gen_serving_config():
    return """# Serving Configuration

serving:
  host: '0.0.0.0'
  port: 8000
  model_name: yaya

  # Generation defaults
  generation:
    max_new_tokens: 256
    temperature: 0.7
    top_k: 50
    top_p: 0.9
    repetition_penalty: 1.1

  # Performance
  max_batch_size: 32
  max_concurrent_requests: 100
  timeout_seconds: 60

  # Quantization for serving
  quantization:
    enabled: false
    bits: 8
    group_size: 128
"""


def main():
    print("Generating configuration files...")
    print()

    write_file("configs/data/tokenizer.yaml", gen_tokenizer_config())
    write_file("configs/data/pipeline.yaml", gen_data_pipeline_config())
    write_file("configs/serving/serve.yaml", gen_serving_config())

    # Create data directory structure
    data_dirs = [
        "data/raw/web",
        "data/raw/code",
        "data/raw/books",
        "data/raw/academic",
        "data/raw/math",
        "data/raw/conversational",
        "data/raw/business",
        "data/raw/multilingual",
        "data/processed/train",
        "data/processed/eval",
        "data/tokenizer",
        "checkpoints",
        "logs",
    ]
    for d in data_dirs:
        full = os.path.join(PROJECT_ROOT, d)
        os.makedirs(full, exist_ok=True)
        # Add .gitkeep to track empty dirs
        gitkeep = os.path.join(full, ".gitkeep")
        if not os.path.exists(gitkeep):
            with open(gitkeep, "w") as f:
                pass

    print()
    print("  Created data directory structure")
    print()
    print("Setup complete! Next steps:")
    print("  1. Place training text files in data/raw/ subdirectories")
    print("  2. Run: python scripts/train_tokenizer.py --input_dir data/raw/web")
    print("  3. Run: python scripts/prepare_data.py --input_dir data/raw/web \\")
    print("          --output_dir data/processed/train \\")
    print("          --tokenizer_path data/tokenizer/yaya_tokenizer.model")
    print("  4. Run: python scripts/train.py \\")
    print("          --model_config configs/model/yaya_1_5b.yaml \\")
    print("          --train_config configs/training/train_1_5b.yaml")


if __name__ == "__main__":
    main()
