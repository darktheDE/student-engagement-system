def generate_timeline_chart_html(timeline_data, width=300, height=150):
    """
    Tạo Line Chart HTML/CSS đơn giản cho timeline engagement
    timeline_data: list [{'time': seconds, 'score': 0-10}, ...]
    """
    if not timeline_data or len(timeline_data) < 1:
        return """
        <div style='padding: 20px; text-align: center; color: #95a5a6; font-size: 11px;'>
            Chưa đủ dữ liệu để hiển thị biểu đồ
        </div>
        """
    
    # Normalize data
    max_time = max(point['time'] for point in timeline_data)
    max_score = 10
    
    # Tạo SVG points
    points = []
    for point in timeline_data:
        x = (point['time'] / max_time) * width if max_time > 0 else 0
        y = height - ((point['score'] / max_score) * height)
        points.append(f"{x},{y}")
    
    polyline_points = " ".join(points)
    
    # Xác định màu đường theo xu hướng
    if len(timeline_data) >= 2:
        trend = timeline_data[-1]['score'] - timeline_data[0]['score']
        line_color = "#10b981" if trend >= 0 else "#ef4444"
    else:
        line_color = "#60a5fa"
    
    svg = f"""
    <div style='background-color: #1a1f2e; border-radius: 8px; padding: 10px; margin-bottom: 10px;'>
        <div style='color: #60a5fa; font-weight: bold; margin-bottom: 8px; font-size: 11px; text-align: center;'>
            BIỂU ĐỒ BIẾN THIÊN HỨNG THÚ
        </div>
        <svg width="{width}" height="{height}" style="background-color: #0f1419; border-radius: 6px;">
            <!-- Grid lines -->
            <line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" stroke="#374151" stroke-width="1" stroke-dasharray="5,5"/>
            
            <!-- Đường biểu đồ -->
            <polyline points="{polyline_points}" 
                     fill="none" 
                     stroke="{line_color}" 
                     stroke-width="2"/>
            
            <!-- Điểm data -->
            {"".join([f'<circle cx="{(p["time"]/max_time)*width if max_time > 0 else 0}" cy="{height - (p["score"]/max_score)*height}" r="3" fill="{line_color}"/>' for p in timeline_data])}
            
            <!-- Y-axis labels -->
            <text x="5" y="15" fill="#95a5a6" font-size="10">10</text>
            <text x="5" y="{height/2 + 5}" fill="#95a5a6" font-size="10">5</text>
            <text x="5" y="{height - 5}" fill="#95a5a6" font-size="10">0</text>
            
            <!-- X-axis label -->
            <text x="{width - 40}" y="{height - 5}" fill="#95a5a6" font-size="10">{int(max_time)}s</text>
        </svg>
        <div style='margin-top: 6px; text-align: center; font-size: 10px;'>
            <span style='color: #95a5a6;'>Trục Y: Mức độ hứng thú (0-10) | Trục X: Thời gian (giây)</span>
        </div>
    </div>
    """
    
    return svg
