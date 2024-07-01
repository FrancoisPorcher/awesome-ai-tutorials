import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from evaluation import evaluate_model, format_metrics
from utils import load_model, get_device
from data_loader import get_train_and_test_data_loaders


class Trainer:
    def __init__(self, model, train_loader, test_loader, device, epochs=3, lr=2e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs,
        )

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(
                self.train_loader, desc=f"Training Epoch {epoch+1}/{self.epochs}"
            ):
                self.optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self):
        metrics = evaluate_model(self.model, self.test_loader, self.device)
        print(format_metrics(metrics))
        return metrics
