import sys
import os
import pytest
import pandas as pd

# Add the root directory of the project to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import get_test_data_loader
from src.utils import get_device, load_model
from src.evaluation import evaluate_model


def test_model_performance():
    evaluation_results_path = "results/evaluation_metrics.csv"
    # read the csv file
    df_evaluation_results = pd.read_csv(evaluation_results_path, index_col=0)

    # check that the results contain precision, recall and f1
    assert "precision" in df_evaluation_results.index
    assert "recall" in df_evaluation_results.index
    assert "f1" in df_evaluation_results.index

    # check that all values are above 0.8
    assert df_evaluation_results.loc["precision"].item() > 0.8
    assert df_evaluation_results.loc["recall"].item() > 0.8
    assert df_evaluation_results.loc["f1"].item() > 0.8
