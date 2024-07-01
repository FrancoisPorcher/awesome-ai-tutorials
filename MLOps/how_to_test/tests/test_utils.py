import pytest
import torch
from src.utils import load_model, get_device

model_path = "models/imdb_bert_finetuned.pth"


def test_load_model():
    model = load_model(model_path)
    device = get_device()

    # Check that we have 2 classes (IMDB dataset has 2 classes: positive and negative reviews)
    assert (
        model.classifier.out_features == 2
    ), "Model has incorrect number of output features"

    # Debugging prints
    print(f"Expected device: {device}")
    print(f"Model parameter device: {next(model.parameters()).device}")

    # Check that the model is loaded on the correct device by comparing device types
    assert (
        next(model.parameters()).device.type == device.type
    ), f"Model is not on the correct device. Expected {device}, but got {next(model.parameters()).device}"


if __name__ == "__main__":
    pytest.main()
