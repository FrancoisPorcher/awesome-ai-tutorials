import pytest
import torch
from transformers import BertTokenizer
from src.data_loader import clean_text, tokenize_text


def test_clean_text():
    """
    Test the clean_text function to ensure it correctly cleans the text.
    """
    assert clean_text("Hello, World!") == "hello, world!", "Failed to clean basic text."
    assert clean_text("  Spaces  ") == "spaces", "Failed to trim and lowercase text."
    assert clean_text("") == "", "Failed to handle empty string."


def test_tokenize_text(bert_tokenizer):
    """
    Test the tokenize_text function to ensure it correctly tokenizes text using BERT tokenizer.
    """
    tokenizer = bert_tokenizer

    # Example input texts
    txt_1 = ["Hello, @! World!", "Spaces    "]

    # Tokenize the text
    res_1 = tokenize_text(text=txt_1, tokenizer=tokenizer, max_length=128)

    # Check that the result contains the expected keys
    assert "input_ids" in res_1, "Missing input_ids key."
    assert "attention_mask" in res_1, "Missing attention_mask key."
    assert "token_type_ids" in res_1, "Missing token_type_ids key."

    # Check that the values are torch tensors
    assert isinstance(
        res_1["input_ids"], torch.Tensor
    ), "input_ids is not a torch tensor."
    assert isinstance(
        res_1["attention_mask"], torch.Tensor
    ), "attention_mask is not a torch tensor."
    assert isinstance(
        res_1["token_type_ids"], torch.Tensor
    ), "token_type_ids is not a torch tensor."

    # Check that the shape of each tensor is correct
    assert (
        res_1["input_ids"].shape[1] == 128
    ), "input_ids length does not match max_length."
    assert (
        res_1["attention_mask"].shape[1] == 128
    ), "attention_mask length does not match max_length."
    assert (
        res_1["token_type_ids"].shape[1] == 128
    ), "token_type_ids length does not match max_length."

    # Check that the shapes are correct with the number of samples given
    assert res_1["input_ids"].shape[0] == len(
        txt_1
    ), "input_ids batch size does not match number of samples."
    assert res_1["attention_mask"].shape[0] == len(
        txt_1
    ), "attention_mask batch size does not match number of samples."
    assert res_1["token_type_ids"].shape[0] == len(
        txt_1
    ), "token_type_ids batch size does not match number of samples."

    # Verify the content of the tensors (if necessary)
    # Here, you may want to check specific token ids for given words, if applicable.
    # Example:
    # assert res_1["input_ids"][0][0] == tokenizer.cls_token_id, "First token should be [CLS]."
    # assert res_1["input_ids"][0][-1] == tokenizer.sep_token_id, "Last token should be [SEP]."

    # Test with single string input
    single_text = "Hello, World!"
    res_single = tokenize_text(text=single_text, tokenizer=tokenizer, max_length=128)

    assert isinstance(
        res_single["input_ids"], torch.Tensor
    ), "input_ids for single string is not a torch tensor."
    assert (
        res_single["input_ids"].shape[1] == 128
    ), "input_ids length for single string does not match max_length."
