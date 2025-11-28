import os
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from src.model import StanceClassifier
from src.dataset import StanceDataset, create_dataset
from src.trainer import Trainer


def main():
    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_API_KEY"))

    # Configuration
    DB_PATH = "title.db"
    CHECKPOINT = "bert-base-chinese"
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    NUM_CLASSES = 3
    NUM_EPOCHS = 15
    MODEL_SAVE_PATH = "stance_classifier.pth"

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Data Preparation
    df = create_dataset(DB_PATH)
    print(df.head())

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.7 * len(df_shuffled))
    val_size = int(0.15 * len(df_shuffled))

    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size : train_size + val_size]
    test_df = df_shuffled[train_size + val_size :]

    print(
        f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
    )

    train_dataset = StanceDataset(
        texts=train_df["title"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=64,
    )
    val_dataset = StanceDataset(
        texts=val_df["title"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=64,
    )
    test_dataset = StanceDataset(
        texts=test_df["title"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=64,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Initialization
    base_model = AutoModel.from_pretrained(CHECKPOINT)
    base_model.resize_token_embeddings(len(tokenizer))

    model = StanceClassifier(base_model, num_classes=NUM_CLASSES)
    model.to(device)

    # Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_path=MODEL_SAVE_PATH,
    )

    trainer.train()


if __name__ == "__main__":
    main()
