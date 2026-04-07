"""Securely set up API tokens for Yaya training.

Saves tokens to .env file (gitignored) so they're never committed.
These are used by the Kaggle notebook and monitoring scripts.

Usage:
    python scripts/setup_tokens.py
"""

import os
import getpass

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

def main():
    print("=" * 50)
    print("  YAYA AI — Token Setup")
    print("=" * 50)
    print(f"\nTokens will be saved to: {ENV_PATH}")
    print("This file is gitignored — your keys stay private.\n")

    # Load existing tokens
    existing = {}
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    existing[key.strip()] = val.strip()

    # HF Token
    current_hf = existing.get('HF_TOKEN', '')
    masked = current_hf[:6] + '...' + current_hf[-4:] if len(current_hf) > 10 else '(not set)'
    print(f"Current HF_TOKEN: {masked}")
    hf_token = getpass.getpass("Enter HF_TOKEN (paste, then press Enter — input is hidden): ")
    if hf_token.strip():
        existing['HF_TOKEN'] = hf_token.strip()
        print("  HF_TOKEN saved.")
    else:
        print("  Kept existing.")

    # W&B Key (optional)
    print()
    current_wb = existing.get('WANDB_API_KEY', '')
    masked_wb = current_wb[:6] + '...' + current_wb[-4:] if len(current_wb) > 10 else '(not set)'
    print(f"Current WANDB_API_KEY: {masked_wb}")
    wandb_key = getpass.getpass("Enter WANDB_API_KEY (optional, press Enter to skip): ")
    if wandb_key.strip():
        existing['WANDB_API_KEY'] = wandb_key.strip()
        print("  WANDB_API_KEY saved.")
    else:
        print("  Kept existing / skipped.")

    # Write .env
    with open(ENV_PATH, 'w') as f:
        for key, val in existing.items():
            f.write(f"{key}={val}\n")

    print(f"\nTokens saved to {ENV_PATH}")
    print("This file is in .gitignore — it will NOT be committed.\n")

    # Verify HF token works
    if existing.get('HF_TOKEN'):
        print("Verifying HF_TOKEN...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=existing['HF_TOKEN'])
            user = api.whoami()
            print(f"  Authenticated as: {user.get('name', user.get('fullname', 'OK'))}")
            print("  Token is valid!")
        except Exception as e:
            print(f"  WARNING: Token verification failed: {e}")
            print("  Check your token at https://huggingface.co/settings/tokens")

    print("\nDone! You can now run the monitor:")
    print("  python scripts/monitor_training.py")

if __name__ == "__main__":
    main()
