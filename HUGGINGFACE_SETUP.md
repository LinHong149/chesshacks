# Hugging Face Model Setup

Since your model files are too large for GitHub (>100MB), you need to host them on Hugging Face Hub.

## Step 1: Install Hugging Face CLI

```bash
pip install huggingface_hub
```

## Step 2: Login to Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens)

## Step 3: Create a Model Repository

Go to https://huggingface.co/new and create a new model repository (e.g., `your-username/chess-bot-v3`)

Or use the CLI:

```bash
huggingface-cli repo create chess-bot-v3 --type model
```

## Step 4: Upload Your Model

```bash
# Upload v3.pth to your Hugging Face repository
huggingface-cli upload your-username/chess-bot-v3 models/v3.pth v3.pth
```

Or use Python:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="models/v3.pth",
    path_in_repo="v3.pth",
    repo_id="your-username/chess-bot-v3",
    repo_type="model"
)
```

## Step 5: Configure Your Bot

Set the environment variable in your deployment:

```bash
export HF_MODEL_ID="your-username/chess-bot-v3"
```

Or add to your `.env.local`:

```
HF_MODEL_ID=your-username/chess-bot-v3
```

## Step 6: Update requirements.txt

Make sure `huggingface_hub` is in your requirements.txt (already added).

## For ChessHacks.com Deployment

When deploying to chesshacks.com, set the `HF_MODEL_ID` environment variable in your deployment settings. The bot will automatically download the model on first run.

## Alternative: Direct Download Script

If you prefer, you can also add a download script that runs on startup:

```python
# In src/main.py or a separate script
import os
from huggingface_hub import hf_hub_download

def ensure_model_downloaded():
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'v3.pth')
    if not os.path.exists(model_path):
        hf_model_id = os.getenv('HF_MODEL_ID', 'your-username/chess-bot-v3')
        print(f"Downloading model from Hugging Face: {hf_model_id}")
        hf_hub_download(
            repo_id=hf_model_id,
            filename="v3.pth",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
```

