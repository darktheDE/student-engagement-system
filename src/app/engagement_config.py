ENGAGEMENT_STATES = {
    # Nhóm Engaged
    'engaged': {
        'group': 'Engaged',
        'color_bgr': (0, 255, 0),       # Xanh lá (Green)
        'color_hex': '#10b981',
        'display_name': 'Engaged'
    },
    'confused': {
        'group': 'Engaged',
        'color_bgr': (255, 0, 0),       # Xanh dương (Blue)
        'color_hex': '#3b82f6',
        'display_name': 'Confused'
    },
    'frustrated': {
        'group': 'Engaged',
        'color_bgr': (0, 215, 255),     # Vàng đậm (Gold/Yellow)
        'color_hex': '#fbbf24',
        'display_name': 'Frustrated'
    },
    
    # Nhóm Not Engaged
    'bored': {
        'group': 'Not Engaged',
        'color_bgr': (128, 128, 128),   # Xám (Gray)
        'color_hex': '#6b7280',
        'display_name': 'Bored'
    },
    'drowsy': {
        'group': 'Not Engaged',
        'color_bgr': (0, 165, 255),     # Cam (Orange)
        'color_hex': '#f97316',
        'display_name': 'Drowsy'
    },
    'looking away': {
        'group': 'Not Engaged',
        'color_bgr': (0, 0, 255),       # Đỏ (Red)
        'color_hex': '#ef4444',
        'display_name': 'Looking Away'
    }
}


def get_state_info(state_key):
    return ENGAGEMENT_STATES.get(state_key.lower(), {
        'group': 'Unknown',
        'color_bgr': (200, 200, 200),
        'color_hex': '#95a5a6',
        'display_name': 'Unknown'
    })


def get_group_color(group):
    if group == 'Engaged':
        return (0, 255, 0), '#10b981'  # Xanh lá
    else:
        return (0, 0, 255), '#ef4444'  # Đỏ


def calculate_engagement_score(state_counts):

    weights = {
        'engaged': 10,
        'confused': 7,
        'frustrated': 5,
        'bored': 3,
        'drowsy': 2,
        'looking away': 1
    }
    
    total_count = sum(state_counts.values())
    if total_count == 0:
        return 0
    
    weighted_sum = sum(state_counts.get(state, 0) * weight 
                      for state, weight in weights.items())
    
    score = weighted_sum / total_count
    return round(score, 1)
