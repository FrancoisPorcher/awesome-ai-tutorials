# src/data_loader.py

from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset
from transformers import BertTokenizer


def load_raw_data() -> Tuple[Dataset, Dataset]:
    """
    Load the raw IMDb dataset.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the raw training and test datasets.
    """
    dataset = hf_load_dataset("imdb")
    return dataset["train"], dataset["test"]


def clean_text(text: str) -> str:
    """
    Clean the input text by converting it to lowercase and stripping whitespace.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    return text.lower().strip()


def tokenize_text(
    text: str, tokenizer: BertTokenizer, max_length: int
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a single text using the BERT tokenizer.

    Args:
        text (str): The text to tokenize.
        tokenizer (BertTokenizer): The tokenizer to use.
        max_length (int): The maximum length of the tokenized sequence.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the tokenized data.
    """
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def process_data(
    dataset: Dataset, tokenizer: BertTokenizer, max_length: int = 128
) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    """
    Process the dataset: clean and tokenize the texts.

    Args:
        dataset (Dataset): The dataset to process.
        tokenizer (BertTokenizer): The tokenizer to use.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 128.

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], List[int]]: A tuple containing the processed data and the labels.
    """
    processed_data = []
    labels = []

    for item in dataset:
        cleaned_text = clean_text(item["text"])
        tokenized_data = tokenize_text(cleaned_text, tokenizer, max_length)
        processed_data.append(tokenized_data)
        labels.append(item["label"])

    return processed_data, labels


class IMDbDataset(Dataset):
    """
    A custom Dataset class for the IMDb dataset.
    """

    def __init__(
        self, processed_data: List[Dict[str, torch.Tensor]], labels: List[int]
    ):
        """
        Initialize the IMDbDataset.

        Args:
            processed_data (List[Dict[str, torch.Tensor]]): The processed input data.
            labels (List[int]): The corresponding labels.
        """
        self.processed_data = processed_data
        self.labels = labels

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the input_ids, attention_mask, and label for the specified index.
        """
        item = self.processed_data[idx]
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_imdb_dataset(
    processed_data: List[Dict[str, torch.Tensor]], labels: List[int]
) -> IMDbDataset:
    """
    Create an IMDbDataset instance from processed data and labels.

    Args:
        processed_data (List[Dict[str, torch.Tensor]]): The processed input data.
        labels (List[int]): The corresponding labels.

    Returns:
        IMDbDataset: An instance of the IMDbDataset.
    """
    return IMDbDataset(processed_data, labels)


def create_data_loader(
    dataset: IMDbDataset, batch_size: int, shuffle: bool
) -> DataLoader:
    """
    Create a DataLoader from an IMDbDataset.

    Args:
        dataset (IMDbDataset): The dataset to create a DataLoader for.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_train_data_loader(batch_size: int, max_length: int):
    # Load raw data
    train_raw, _ = load_raw_data()

    # Process data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_processed, train_labels = process_data(train_raw, tokenizer, max_length)

    # Load dataset
    train_dataset = create_imdb_dataset(train_processed, train_labels)

    # Create data loader
    train_loader = create_data_loader(train_dataset, batch_size, shuffle=True)

    print("train_data_loader loaded successfully!")

    return train_loader


def get_test_data_loader(batch_size: int, max_length: int):
    # Load raw data
    _, test_raw = load_raw_data()

    # Process data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_processed, test_labels = process_data(test_raw, tokenizer, max_length)

    # Load dataset
    test_dataset = create_imdb_dataset(test_processed, test_labels)

    # Create data loader
    test_loader = create_data_loader(test_dataset, batch_size, shuffle=False)

    print("test_data_loader loaded successfully!")

    return test_loader


def get_train_and_test_data_loaders(batch_size: int, max_length: int):
    train_loader = get_train_data_loader(batch_size, max_length)
    test_loader = get_test_data_loader(batch_size, max_length)

    return train_loader, test_loader
