# src/utils.py

import torch
from transformers import BertForSequenceClassification


def get_device():
    """
    Get the appropriate device (MPS if available, otherwise CUDA if available, otherwise CPU).

    Returns:
        torch.device: The device to use for computations.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(model_path=None, num_labels=2):
    """
    Load a BERT model for sequence classification.

    Args:
        model_path (str, optional): Path to a pre-trained model file (.pth). Defaults to None.
        num_labels (int, optional): Number of labels for the classification task. Defaults to 2.

    Returns:
        BertForSequenceClassification: The loaded BERT model.
    """
    device = get_device()
    print(f"Using device: {device}")

    if model_path:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        print("Loaded stock BERT model")

    model.to(device)
    print("BERT model loaded successfully on device {}".format(device))

    return model
