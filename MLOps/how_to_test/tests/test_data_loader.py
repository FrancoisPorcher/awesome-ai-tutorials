import pytest
import torch
from transformers import BertTokenizer
from src.data_loader import clean_text, tokenize_text


def test_clean_text():
    """
    Test the clean_text function to ensure it correctly cleans the text.
    """
    assert clean_text("  Spaces  ") == "spaces", "Failed to trim and lowercase text."
    assert clean_text("Hello, World!") == "hello, world!", "Failed to clean basic text."
    assert clean_text("") == "", "Failed to handle empty string."



    
    
    

