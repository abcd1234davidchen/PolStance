import os
import torch
from transformers import AutoModel, BertTokenizerFast
from huggingface_hub import login
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from model import StanceClassifier
from dataset import StanceDataset, create_dataset
from Labeling.utils.HFManager import HFManager
from trainer import Trainer

if os.uname().nodename == "w61":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
else:
    pass

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train Stance Classifier")
    parser.add_argument("--mode", type=str, default="supcon", choices=["supcon", "classifier"], help="Training mode: supcon (Stage 1) or classifier (Stage 2)")
    parser.add_argument("--checkpoint", type=str, default="stance_embedder.pth", help="Path to checkpoint for Stage 2 loading")
    parser.add_argument("--epochs", type=int, default=32, help="Number of epochs")
    args = parser.parse_args()

    load_dotenv()

    # Configuration
    CHECKPOINT = "ckiplab/bert-base-chinese"
    MAX_LENGTH = 512
    BATCH_SIZE = 32
    NUM_CLASSES = 3
    NUM_EPOCHS = args.epochs
    WARM_UP_EPOCHS = 6
    PATIENCE = 4
    
    if args.mode == "supcon":
        SUPCON_MODE = True
        MODEL_SAVE_PATH = "stance_embedder.pth"
        print("=== Starting Stage 1: Supervised Contrastive Learning (SupCon) ===")
    else:
        SUPCON_MODE = False
        MODEL_SAVE_PATH = "stance_classifier.pth"
        print("=== Starting Stage 2: Classifier Training ===")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Data Preparation
    hf_manager = HFManager()
    hf_manager.download_db()
    df = create_dataset(hf_manager.db_path)
    print(df.head())

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    train_size = int(0.7 * len(df_shuffled))
    val_size = int(0.15 * len(df_shuffled))

    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size : train_size + val_size]
    test_df = df_shuffled[train_size + val_size :]

    print(
        f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
    )

    train_dataset = StanceDataset(
        texts=train_df["article"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    val_dataset = StanceDataset(
        texts=val_df["article"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    test_dataset = StanceDataset(
        texts=test_df["article"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Initialization
    base_model = AutoModel.from_pretrained(CHECKPOINT)
    base_model.resize_token_embeddings(len(tokenizer))

    model = StanceClassifier(base_model, num_classes=NUM_CLASSES)
    model.to(device)

    # Stage 2 Specific Setup
    if args.mode == "classifier":
        if os.path.exists(args.checkpoint):
            print(f"Loading SupCon weights from {args.checkpoint}...")
            # Load weights
            state_dict = torch.load(args.checkpoint, map_location=device)
            # We must load carefully. SupCon trained the transformer and projection head (if logic changed).
            # But in `model.py`, `classifier` is separate.
            # In Stage 1, we trained `transformer` and `layer_norm`.
            # We did NOT train `classifier`.
            
            model.load_state_dict(state_dict)
            
            print("Freezing transformer for linear evaluation...")
            model.freeze_transformer()
            
            # Important: Re-initialize the classifier head often helps if it was randomized or unused.
            # But since we just loaded state_dict, we loaded the random/unused weights from Stage 1.
            # It is safer to re-init them to ensure a fresh start for Stage 2.
            # However, `model.classifier` is a Sequential. We can re-init its modules.
            print("Re-initializing classifier layer...")
            for layer in model.classifier:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                elif isinstance(layer, torch.nn.Linear):
                     torch.nn.init.xavier_uniform_(layer.weight)
                     torch.nn.init.zeros_(layer.bias)
                     
        else:
            print(f"WARNING: Checkpoint {args.checkpoint} not found! Training from scratch (not recommended for Stage 2).")

    # Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        warmup_epochs= WARM_UP_EPOCHS if args.mode == "classifier" else 0, # No warmup needed for frozen encoder
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH,
        supcon_mode=SUPCON_MODE,
    )

    trainer.train()


if __name__ == "__main__":
    main()
