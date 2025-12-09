import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


from pytorch_metric_learning import losses

import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_epochs=16,
        warmup_epochs=6,
        patience=4,
        save_path="stance_classifier.pth",
        supcon_mode=False, 
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path
        self.warmup_epochs = warmup_epochs
        self.supcon_mode = supcon_mode
        
        self. n = []
        self.val_losses = []
        self.train_losses = []

        if self.supcon_mode:
            # SupConLoss with default temperature=0.1
            self.criterion = losses.SupConLoss(temperature=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = AdamW(
            [
                {"params": model.classifier_params(), "lr": 1e-4},
                {"params": model.layer_norm.parameters(), "lr": 1e-4},
                {"params": model.dropout.parameters(), "lr": 1e-4},
            ],
            weight_decay=1e-4,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=2, factor=0.5
        )

    def train_step(self, batch):
        self.model.train()

        inputs = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        self.optimizer.zero_grad()

        if self.supcon_mode:
            # Two views via dropout augmentation (passing same input twice)
            _, feature1 = self.model(inputs, attention_mask, return_embeddings=True)
            _, feature2 = self.model(inputs, attention_mask, return_embeddings=True)
            
            # Stack features and labels
            features = torch.cat([feature1, feature2], dim=0)
            targets = torch.cat([labels, labels], dim=0)

            if torch.isnan(features).any() or torch.isinf(features).any():
                print("WARNING: features include NaN/Inf")
                return {"loss": 0.0, "accuracy": 0.0}

            loss = self.criterion(features, targets)
            
            # Accuracy is not well-defined for SupCon in this step, return 0 or proxy
            accuracy = torch.tensor(0.0)
            
        else:
            logits = self.model(inputs, attention_mask)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("WARNING: logits include NaN/Inf")
                return {"loss": 0.0, "accuracy": 0.0}

            loss = self.criterion(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: loss is NaN/Inf: {loss.item()}，跳過此 batch")
            return {"loss": 0.0, "accuracy": 0.0}

        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=0.5
        )
        if torch.isnan(total_norm):
            print("WARNING: total_norm is NaN，跳過此 batch")
            self.optimizer.zero_grad()
            return {"loss": 0.0, "accuracy": 0.0}

        self.optimizer.step()
        
        return {"loss": loss.item(), "accuracy": accuracy.item()}

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if torch.isnan(inputs).any() or torch.isnan(attention_mask).any():
                    continue

                if self.supcon_mode:
                    # In eval mode, dropout is off, so two passes would be identical unless we force exp.
                    # Standard SupCon Eval: often just check loss on the batch or use KNN accuracy.
                    # For simplicity, let's just calc the SupCon loss on the batch (no augmentation or single view).
                    # Actually, calculating SupCon loss on single view is possible if there are positives in the batch.
                    _, features = self.model(inputs, attention_mask, return_embeddings=True)
                    logits = None # Not used
                    
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        continue

                    loss = self.criterion(features, labels)
                    # Accuracy placeholder
                    correct = torch.tensor(0.0) 
                else:
                    logits = self.model(inputs, attention_mask)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        continue

                    loss = self.criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)
                    correct = (predictions == labels).float().sum()

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Val loss NaN/Inf")
                    continue

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct.item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        mode_str = "SupCon" if self.supcon_mode else "Classifier"
        plt.title(f'Training and Validation Loss ({mode_str})')
        plt.legend()
        plt.grid(True)
        filename = f"loss_curve_{mode_str.lower()}.png"
        plt.savefig(filename)
        print(f"Loss plot saved to {filename}")
        plt.close()

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        print(f"Trainable LLMs params: {sum(p.numel() for p in list(self.model.transformer.parameters()))}")
        print(f"Trainable head params: {sum(p.numel() for p in self.model.classifier_params())}")
        for epoch in range(self.num_epochs):
            if epoch == self.warmup_epochs and not self.supcon_mode:
                print("Unfreezing transformer for fine-tuning...")
                self.model.unfreeze_transformer()
                self.optimizer.add_param_group(
                    {"params": self.model.transformer.parameters(), "lr": 2e-6}
                )

            epoch_train_loss = 0
            epoch_train_accuracy = 0
            valid_batches = 0

            pbar = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} Training"
            )
            for batch in pbar:
                train_metrics = self.train_step(batch)
                if train_metrics["loss"] > 0:
                    epoch_train_loss += train_metrics["loss"]
                    epoch_train_accuracy += train_metrics["accuracy"]
                    valid_batches += 1
                else:
                    print("Skipped a batch due to NaN/Inf issues.")

                if valid_batches > 0 :
                    if self.supcon_mode:
                        pbar.set_description(
                            f"Epoch: {epoch + 1}/{self.num_epochs} Loss: {epoch_train_loss / valid_batches:.4f}"
                        )
                    else:
                        pbar.set_description(
                            f"Epoch: {epoch + 1}/{self.num_epochs} Loss: {epoch_train_loss / valid_batches:.4f}, Acc: {epoch_train_accuracy / valid_batches:.4f}"
                        )

            print("Evaluating on validation set...")
            val_metrics = self.evaluate(self.val_loader)

            self.scheduler.step(val_metrics["loss"])

            avg_train_loss = (
                epoch_train_loss / valid_batches if valid_batches > 0 else 0
            )
            avg_train_acc = (
                epoch_train_accuracy / valid_batches if valid_batches > 0 else 0
            )

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_metrics["loss"])

            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            print(
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                print("New best model found, saving...")
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        self.plot_losses()

        print("Training complete. Evaluating on test set...")
        test_metrics = self.evaluate(self.test_loader)
        print(
            f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}"
        )
