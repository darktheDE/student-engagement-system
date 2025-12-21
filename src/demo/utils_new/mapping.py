"""
Label and prediction mapping utilities
"""
from src.demo.config import LABEL_MAP, BINARY_MAP


def map_prediction_to_binary(prediction):
    """
    Maps the multi-class prediction integer to a binary 0/1 score.
    
    Args:
        prediction: Predicted class (0-5)
        
    Returns:
        int: Binary value (0 = Not Engaged, 1 = Engaged)
    """
    return BINARY_MAP.get(prediction, 0)


def get_label_name(prediction):
    """
    Get the label name for a prediction
    
    Args:
        prediction: Predicted class (0-5)
        
    Returns:
        str: Label name
    """
    return LABEL_MAP.get(prediction, f"Unknown ({prediction})")
