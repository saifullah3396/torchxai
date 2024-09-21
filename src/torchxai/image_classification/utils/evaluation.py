from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.metrics import classification_report
from torchfusion.core.utilities.logging import get_logger

logger = get_logger(__name__)


def pretty_classification_report(y_true: List[int], y_pred: List[int]) -> pd.DataFrame:
    """
    Generate a pretty classification report.
    Parameters:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.
    Returns:
    - report_df (pandas DataFrame): A DataFrame containing the classification report with rounded values.
    Example:
    ```
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 1]
    >>> report_df = pretty_classification_report(y_true, y_pred)
    >>> print(report_df)
                    precision  recall  f1-score  support
    0                1.00    0.50      0.67     2.00
    1                0.67    1.00      0.80     2.00
    accuracy         0.75    0.75      0.75     4.00
    macro avg        0.83    0.75      0.73     4.00
    weighted avg     0.83    0.75      0.73     4.00
    ```
    """

    # Generate the classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Convert the dictionary report to a pandas DataFrame for pretty printing
    report_df = pd.DataFrame(report).transpose()

    # Display the DataFrame with rounded values for easier reading
    report_df = report_df.round(2)

    logger.info(f"Classification Report:\n{report_df}")
    return report_df
