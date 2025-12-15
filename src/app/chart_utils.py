import base64


def generate_timeline_chart_html(timeline_data, width=300, height=150):
    """
    Generate a small SVG line chart and return HTML that embeds it as a base64 image.
    timeline_data: list of dicts [{'time': seconds, 'score': 0-10}, ...]
    """
    if not timeline_data or len(timeline_data) < 1:
        return """
        <div style='padding: 20px; text-align: center; color: #95a5a6; font-size: 11px;'>
            Chưa đủ dữ liệu để hiển thị biểu đồ
        </div>
        """

    # Normalize data
    max_time = max(point['time'] for point in timeline_data)
    max_score = 10.0

    # Build polyline and points
    pts = []
    circles = []
    for p in timeline_data:
        t = p.get('time', 0.0)
        s = p.get('score', 0.0)
        x = (t / max_time) * width if max_time > 0 else 0
        y = height - ((s / max_score) * height)
        pts.append(f"{x:.2f},{y:.2f}")
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#60a5fa"/>')

    polyline_points = " ".join(pts)

    # Trend color
    if len(timeline_data) >= 2:
        trend = timeline_data[-1]['score'] - timeline_data[0]['score']
        line_color = "#10b981" if trend >= 0 else "#ef4444"
    else:
        line_color = "#60a5fa"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <rect width="100%" height="100%" fill="#0f1419" rx="6"/>
        <line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" stroke="#374151" stroke-width="1" stroke-dasharray="5,5"/>
        <polyline points="{polyline_points}" fill="none" stroke="{line_color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
        {''.join(circles)}
        <text x="6" y="14" fill="#95a5a6" font-size="10">10</text>
        <text x="6" y="{height/2 + 6}" fill="#95a5a6" font-size="10">5</text>
        <text x="6" y="{height - 4}" fill="#95a5a6" font-size="10">0</text>
        <text x="{width - 40}" y="{height - 4}" fill="#95a5a6" font-size="10">{int(max_time)}s</text>
    </svg>'''

    # Embed SVG as base64 image so QTextEdit/QTextBrowser can render it reliably
    svg_bytes = svg.encode('utf-8')
    b64 = base64.b64encode(svg_bytes).decode('ascii')
    img_html = f"<div style='background-color: #1a1f2e; border-radius: 8px; padding: 8px; text-align:center;'>"
    img_html += f"<div style='color: #60a5fa; font-weight: bold; margin-bottom: 6px; font-size: 11px;'>BIỂU ĐỒ BIẾN THIÊN HỨNG THÚ</div>"
    img_html += f"<img src=\"data:image/svg+xml;base64,{b64}\" style=\"max-width:100%; height:{height}px; border-radius:6px;\"/>"
    img_html += "<div style='margin-top:6px; font-size:10px; color:#95a5a6;'>Trục Y: Mức độ hứng thú (0-10) | Trục X: Thời gian (giây)</div>"
    img_html += "</div>"

    return img_html
