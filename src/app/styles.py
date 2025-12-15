MAIN_WINDOW_STYLE = """
QMainWindow {
    background-color: #0f1419;
}
QGroupBox {
    font-weight: bold;
    font-size: 12px;
    color: #e0e6ed;
    border: 2px solid #4a9eff;
    border-radius: 8px;
    margin-top: 6px;
    padding: 10px 6px 6px 6px;
    background-color: #1a1f2e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #4a9eff;
}
#headerFrame {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #1e3a8a, stop:0.5 #1e40af, stop:1 #3b82f6);
    border-radius: 10px;
    padding: 8px 15px;
    margin-bottom: 3px;
    border: 2px solid #60a5fa;
}
#titleLabel {
    color: #ffffff;
    font-size: 22px;
    font-weight: bold;
    padding: 3px;
    letter-spacing: 0.5px;
}
#teamInfoLabel {
    color: #e0e7ff;
    font-size: 13px;
    padding: 2px;
    font-weight: 500;
}
#subtitleLabel {
    color: #e0e7ff;
    font-size: 13px;
    padding: 3px;
}
#videoLabel {
    background-color: #000000;
    color: #6b7280;
    border: 3px solid #374151;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 500;
}
#controlFrame {
    background-color: #1a1f2e;
    border-radius: 10px;
    border: 2px solid #374151;
}
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 24px;
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background-color: #2563eb;
    transform: translateY(-2px);
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #4b5563;
    color: #9ca3af;
}
#startBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #10b981, stop:1 #059669);
    border: 2px solid #34d399;
}
#startBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #059669, stop:1 #047857);
    border: 2px solid #6ee7b7;
}
#startBtn:pressed {
    background-color: #047857;
}
#stopBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ef4444, stop:1 #dc2626);
    border: 2px solid #f87171;
}
#stopBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #dc2626, stop:1 #b91c1c);
    border: 2px solid #fca5a5;
}
#stopBtn:pressed {
    background-color: #991b1b;
}
QComboBox {
    background-color: #1f2937;
    color: #e5e7eb;
    border: 2px solid #4a9eff;
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
}
QComboBox:hover {
    border-color: #60a5fa;
    background-color: #374151;
}
QComboBox::drop-down {
    border: none;
    width: 30px;
}
QTextEdit {
    background-color: #111827;
    color: #e5e7eb;
    border: 2px solid #374151;
    border-radius: 6px;
    padding: 8px;
    font-size: 12px;
    line-height: 1.4;
}
QLabel#faceCountLabel {
    color: #60a5fa;
    font-size: 14px;
    font-weight: bold;
    padding: 6px;
    background-color: #1a1f2e;
    border-radius: 5px;
}
QLabel#overallStatusLabel {
    color: #10b981;
    font-size: 20px;
    font-weight: bold;
    background-color: #1a1f2e;
    border-radius: 8px;
    padding: 10px;
}
QLabel#detailStatusLabel {
    color: #60a5fa;
    font-size: 16px;
    font-weight: 600;
    background-color: #1a1f2e;
    border-radius: 6px;
    padding: 8px;
}
QLabel#confidenceLabel {
    color: #e5e7eb;
    font-size: 12px;
    font-weight: 500;
    padding: 2px;
}
QTextEdit#confidenceBar {
    background-color: transparent;
    border: none;
    padding: 0px;
    margin: 0px;
}
QTextEdit#timelineChart {
    background-color: #0f1419;
    border: 2px solid #374151;
    border-radius: 8px;
    padding: 5px;
}
QLabel#engagementStatusLabel {
    color: #34d399;
    font-size: 12px;
    font-weight: 600;
    padding: 6px;
    background-color: #1a1f2e;
    border-radius: 5px;
}
QLabel#roiInfoLabel {
    color: #34d399;
    font-size: 13px;
    font-weight: 600;
    padding: 10px;
    background-color: #1a1f2e;
    border-radius: 8px;
}
"""
