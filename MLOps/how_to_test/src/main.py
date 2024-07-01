import argparse
from data_loader import get_train_and_test_data_loaders
from utils import get_device, load_model
from trainer import Trainer


def main(model_path=None):
    # Get data loaders
    train_loader, test_loader = get_train_and_test_data_loaders(
        batch_size=16, max_length=128
    )

    # Get device
    device = get_device()

    # Load the model
    if model_path:
        model = load_model(model_path)
    else:
        model = load_model()

    # Initialize the Trainer
    trainer = Trainer(model, train_loader, test_loader, device, epochs=3, lr=2e-5)

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument(
        "--model_path", type=str, help="Path to the model file to load."
    )

    args = parser.parse_args()
    main(model_path=args.model_path)
