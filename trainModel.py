import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sqlite3
from tqdm import tqdm

class StanceClassifier(nn.Module):
    def __init__(self,transformer_model, num_classes, dropout_rate=0.3):
        super(StanceClassifier, self).__init__()
        self.transformer = transformer_model
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(transformer_model.config.hidden_size)
        self.classifier = nn.Linear(transformer_model.config.hidden_size, num_classes)
        torch.nn.init.normal_(self.classifier.weight, std=0.01)
        torch.nn.init.zeros_(self.classifier.bias)
        self.freeze_transformer()
    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
    def forward(self, input_ids, attention_mask):
        if not any(p.requires_grad for p in self.transformer.parameters()):
            with torch.no_grad():
                outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0]
        
        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            print("⚠️ Transformer 輸出包含 NaN/Inf")
            pooled_output = torch.where(torch.isnan(pooled_output) | torch.isinf(pooled_output), 
                                      torch.zeros_like(pooled_output), pooled_output)
        
        pooled_output = self.layer_norm(pooled_output)
        # pooled_output = torch.clamp(pooled_output, min=-5, max=5)
        
        dropped_out = self.dropout(pooled_output)
        logits = self.classifier(dropped_out)
        # logits = torch.clamp(logits, min=-5, max=5)
        
        return logits

class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

load_dotenv()
login(token=os.getenv("HUGGINGFACE_API_KEY"))
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() else "cpu")

base_model = AutoModel.from_pretrained(checkpoint)
model = StanceClassifier(base_model, num_classes=3)
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(tokenizer))

def train_step(model, batch, optimizer, criterion, device):
    model.train()

    inputs = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()
    logits = model(inputs, attention_mask)

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("⚠️ logits 包含 NaN/Inf，跳過此 batch")
        return {'loss': 0.0, 'accuracy': 0.0}

    loss = criterion(logits, labels)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"⚠️ loss 是 NaN/Inf: {loss.item()}，跳過此 batch")
        return {'loss': 0.0, 'accuracy': 0.0}

    loss.backward()
    
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    if torch.isnan(total_norm):
        print("⚠️ 梯度包含 NaN，跳過此 batch")
        optimizer.zero_grad()
        return {'loss': 0.0, 'accuracy': 0.0}

    optimizer.step()
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    return {
        'loss': loss.item(),
        'accuracy': accuracy.item()
    }

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if torch.isnan(inputs).any() or torch.isnan(attention_mask).any():
                continue

            logits = model(inputs, attention_mask)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).float().sum()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct.item()
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }

def createDataset(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT title, label FROM titles", conn)

    print("檢查標籤分布:")
    print(df['label'].value_counts(normalize=True))

    df = df.dropna(subset=['title', 'label'])
    df = df[df['title'].str.len() > 0]
    df['label'] = df['label']-1

    conn.close()
    return df[['title', 'label']]

if __name__ == "__main__":
    df = createDataset('title.db')
    print(f"Device: {device}")
    
    print(df.head())
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df_shuffled))
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    train_dataset = StanceDataset(
        texts=train_df['title'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    val_dataset = StanceDataset(
        texts=val_df['title'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model.freeze_transformer()

    num_epochs = 8
    optimizer = AdamW([
        {'params': model.classifier.parameters(), 'lr': 1e-5},  # 進一步降低
        {'params': model.layer_norm.parameters(), 'lr': 1e-5},
        {'params': model.dropout.parameters(), 'lr': 1e-5}
    ], weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    for epoch in range(num_epochs):
        if epoch == 2:
            print("Unfreezing transformer for fine-tuning...")
            model.unfreeze_transformer()
            optimizer.add_param_group({'params': model.transformer.parameters(), 'lr': 2e-6})
        model.train()
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        valid_batches = 0
        for batch in tqdm(train_loader):
            train_metrics = train_step(model, batch, optimizer, criterion, device)
            if train_metrics['loss'] > 0:
                epoch_train_loss += train_metrics['loss']
                epoch_train_accuracy += train_metrics['accuracy']
                valid_batches += 1
                
                if valid_batches % 10 == 0:
                    avg_loss = epoch_train_loss / valid_batches
                    avg_acc = epoch_train_accuracy / valid_batches
                    print(f"  Valid batches: {valid_batches}, Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
        print("Evaluating on validation set...")
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss/valid_batches:.4f}, Train Acc: {epoch_train_accuracy/valid_batches:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    torch.save(model.state_dict(), 'stance_classifier.pth')