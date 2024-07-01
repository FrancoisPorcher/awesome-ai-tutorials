import pytest
from transformers import BertTokenizer


@pytest.fixture()
def bert_tokenizer():
    """Fixture to initialize the BERT tokenizer."""
    return BertTokenizer.from_pretrained("bert-base-uncased")
