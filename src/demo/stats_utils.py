def calculate_state_breakdown(history_raw, label_map):
    if not history_raw:
        return "No Data (N/A)"
    
    counts = {}
    total = len(history_raw)
    
    for state in history_raw:
        label = label_map.get(state, "Unknown")
        counts[label] = counts.get(label, 0) + 1
        
    # Get top state
    top_state = max(counts, key=counts.get)
    top_pct = (counts[top_state] / total) * 100
    
    breakdown_text = f"Top: {top_state} ({top_pct:.0f}%)"
    
    return breakdown_text
