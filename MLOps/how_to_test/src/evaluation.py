import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import csv
import os
from src.utils import load_model, get_device


def evaluate_model(model, data_loader, device):
    """
    Evaluate a model on a DataLoader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): The DataLoader to evaluate the model on.
        device (torch.device): The device to perform the evaluation on.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="binary"
    )

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return metrics


def format_metrics(metrics):
    """
    Format the metrics into a nice text message.

    Args:
        metrics (dict): A dictionary containing the metrics.

    Returns:
        str: A formatted string containing the metrics.
    """
    message = "Model Evaluation Results\n"
    message += "========================\n"
    message += f"Accuracy:  {metrics['accuracy']:.4f}\n"
    message += f"Precision: {metrics['precision']:.4f}\n"
    message += f"Recall:    {metrics['recall']:.4f}\n"
    message += f"F1 Score:  {metrics['f1']:.4f}\n"
    message += "========================"
    return message


def save_metrics_to_csv(metrics, filepath):
    """
    Save the metrics to a CSV file.

    Args:
        metrics (dict): A dictionary containing the metrics.
        filepath (str): The path to the CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    from data_loader import get_test_data_loader

    try:
        model_path = "models/imdb_bert_finetuned.pth"
        model = load_model(model_path)
        device = get_device()
        data_loader = get_test_data_loader(batch_size=128, max_length=128)
        metrics = evaluate_model(model, data_loader, device)
        print(format_metrics(metrics))

        # Save metrics to CSV file
        csv_filepath = "results/evaluation_metrics.csv"
        save_metrics_to_csv(metrics, csv_filepath)
        print(f"Metrics saved to {csv_filepath}")

    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")
        print(
            "Please check your data loader and ensure it's returning the expected format."
        )
