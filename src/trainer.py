import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_epochs=16,
        patience=32,
        save_path="stance_classifier.pth",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_path = save_path

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
        logits = self.model(inputs, attention_mask)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: logits include NaN/Inf")
            return {"loss": 0.0, "accuracy": 0.0}

        loss = self.criterion(logits, labels)

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
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
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

                logits = self.model(inputs, attention_mask)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue

                loss = self.criterion(logits, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == labels).float().sum()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct.item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0
        print(f"Trainable LLMs params: {sum(p.numel() for p in list(self.model.transformer.parameters()))}")
        print(f"Trainable head params: {sum(p.numel() for p in self.model.classifier_params())}")
        for epoch in range(self.num_epochs):
            if epoch == 8:
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

                if valid_batches > 0:
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

        print("Training complete. Evaluating on test set...")
        test_metrics = self.evaluate(self.test_loader)
        print(
            f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}"
        )
