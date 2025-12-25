"""
Metrics calculation utilities
"""


def calculate_engagement_rate(binary_history):
    """
    Calculate engagement rate from binary history
    
    Args:
        binary_history: List/deque of binary values (0 or 1)
        
    Returns:
        float: Engagement rate as percentage (0-100)
    """
    if not binary_history or len(binary_history) == 0:
        return 0.0
    
    return (sum(binary_history) / len(binary_history)) * 100


def calculate_state_breakdown(raw_history, label_map):
    """
    Calculate state breakdown from raw prediction history
    
    Args:
        raw_history: List/deque of raw predictions (0-5)
        label_map: Label mapping dict
        
    Returns:
        str: Formatted breakdown text
    """
    if not raw_history or len(raw_history) == 0:
        return "No Data (N/A)"
    
    counts = {}
    total = len(raw_history)
    
    for state in raw_history:
        label = label_map.get(state, "Unknown")
        counts[label] = counts.get(label, 0) + 1
    
    # Get top state
    top_state = max(counts, key=counts.get)
    top_pct = (counts[top_state] / total) * 100
    
    breakdown_text = f"Top: {top_state} ({top_pct:.0f}%)"
    
    return breakdown_text


def calculate_confidence(raw_history, window_size=10):
    """
    Calculate prediction confidence based on consistency
    
    Args:
        raw_history: List/deque of raw predictions
        window_size: Number of recent predictions to consider
        
    Returns:
        float: Confidence percentage (0-100)
    """
    if not raw_history or len(raw_history) == 0:
        return 0.0
    
    recent = list(raw_history)[-window_size:] if len(raw_history) >= window_size else list(raw_history)
    
    if not recent:
        return 0.0
    
    most_common = max(set(recent), key=recent.count)
    confidence = (recent.count(most_common) / len(recent)) * 100
    
    return confidence


def calculate_agreement_rate(history_left, history_right):
    """
    Calculate agreement rate between two model predictions
    
    Args:
        history_left: Raw prediction history for left model
        history_right: Raw prediction history for right model
        
    Returns:
        float: Agreement rate as percentage (0-100)
    """
    if not history_left or not history_right:
        return 0.0
    
    raw_left = list(history_left)
    raw_right = list(history_right)
    
    # Match lengths
    min_len = min(len(raw_left), len(raw_right))
    if min_len == 0:
        return 0.0
    
    raw_left = raw_left[-min_len:]
    raw_right = raw_right[-min_len:]
    
    agreements = sum(1 for l, r in zip(raw_left, raw_right) if l == r)
    agreement_rate = (agreements / min_len) * 100
    
    return agreement_rate
