"""Scandinavian Light theme for Glöd."""

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#F8F9FA"          # warm off-white window background
PANEL = "#EEF2F0"       # light sage panel / sidebar
CARD = "#FFFFFF"        # card / input background
BORDER = "#DEE2E6"      # subtle borders
TEXT = "#212529"        # charcoal text
TEXT_MUTED = "#6C757D"  # muted secondary text
ACCENT = "#5C8A6B"      # sage green primary accent
ACCENT_HOVER = "#4A7C59"
ACCENT_PRESSED = "#3D6B4C"
PROGRESS_FILL = "#68A07B"
ERROR = "#C0392B"
WARNING = "#E67E22"
SUCCESS = "#27AE60"


STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────────── */
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG};
}}

/* ── Panels / frames ────────────────────────────────────── */
QFrame#ControlPanel, QScrollArea#ControlScroll {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}

QGroupBox {{
    background-color: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 8px;
    padding: 4px 6px 4px 6px;
    font-weight: 600;
    color: {TEXT};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: -1px;
    padding: 0 4px;
    background-color: {PANEL};
    color: {ACCENT};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton {{
    background-color: {ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 5px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 13px;
    min-height: 26px;
}}

QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton:disabled {{
    background-color: #ADB5BD;
    color: #F8F9FA;
}}

/* Run Analysis — explicit name so style is never lost after setStyleSheet("") */
QPushButton#RunButton {{
    background-color: {ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 5px;
    padding: 4px 16px;
    font-weight: 700;
    font-size: 13px;
    min-height: 26px;
}}

QPushButton#RunButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton#RunButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton#RunButton:disabled {{
    background-color: #B8895A;
    color: #FDF3E8;
}}

QPushButton#SecondaryButton {{
    background-color: transparent;
    color: {ACCENT};
    border: 1.5px solid {ACCENT};
}}

QPushButton#SecondaryButton:hover {{
    background-color: {ACCENT};
    color: #FFFFFF;
}}

QPushButton#DangerButton {{
    background-color: transparent;
    color: {ERROR};
    border: 1.5px solid {ERROR};
}}

QPushButton#DangerButton:hover {{
    background-color: {ERROR};
    color: #FFFFFF;
}}

QPushButton#SmallButton {{
    padding: 3px 10px;
    font-size: 12px;
    min-height: 24px;
    border-radius: 4px;
}}

/* ── Inputs ──────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {CARD};
    border: 1.5px solid {BORDER};
    border-radius: 4px;
    padding: 2px 6px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    min-height: 22px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
    outline: none;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox::down-arrow {{
    image: none;
    width: 10px;
    height: 10px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_MUTED};
}}

QComboBox QAbstractItemView {{
    background-color: {CARD};
    border: 1.5px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #FFFFFF;
    outline: none;
}}

/* ── Labels ──────────────────────────────────────────────── */
QLabel {{
    color: {TEXT};
    background: transparent;
}}

QLabel#SectionLabel {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

QLabel#TitleLabel {{
    font-size: 20px;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.5px;
}}

QLabel#SubtitleLabel {{
    font-size: 12px;
    color: {TEXT_MUTED};
}}

QLabel#ErrorLabel {{
    color: {ERROR};
    font-size: 12px;
}}

/* ── Checkboxes ──────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {BORDER};
    border-radius: 3px;
    background-color: {CARD};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: none;
}}

QCheckBox::indicator:checked::after {{
    content: "";
}}

/* ── Radio buttons ───────────────────────────────────────── */
QRadioButton {{
    color: {TEXT};
    spacing: 6px;
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {BORDER};
    border-radius: 8px;
    background-color: {CARD};
}}

QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Progress bars ───────────────────────────────────────── */
QProgressBar {{
    background-color: {BORDER};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 11px;
    color: {TEXT};
}}

QProgressBar::chunk {{
    background-color: {PROGRESS_FILL};
    border-radius: 4px;
}}

/* ── Tabs ────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-top: none;
    background-color: {CARD};
}}

QTabBar::tab {{
    background-color: {PANEL};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 7px 18px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: {CARD};
    color: {ACCENT};
    font-weight: 700;
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {BG};
    color: {TEXT};
}}

/* ── Scroll bars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {PANEL};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: #C8D0CC;
    min-height: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #A8B4AE;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {PANEL};
    height: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background: #C8D0CC;
    min-width: 30px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #A8B4AE;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Splitter ────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
    width: 1px;
}}

/* ── Text edit (console log) ─────────────────────────────── */
QTextEdit {{
    background-color: #1E2124;
    color: #C8D0CC;
    border: none;
    border-radius: 4px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 8px;
}}

/* ── List widget ─────────────────────────────────────────── */
QListWidget {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 4px 8px;
    border-radius: 3px;
}}

QListWidget::item:selected {{
    background-color: {ACCENT};
    color: #FFFFFF;
}}

/* ── Tooltip ─────────────────────────────────────────────── */
QToolTip {{
    background-color: {TEXT};
    color: {BG};
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}}

/* ── Menu bar ────────────────────────────────────────────── */
QMenuBar {{
    background-color: {PANEL};
    border-bottom: 1px solid {BORDER};
    padding: 2px 4px;
}}

QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {ACCENT};
    color: #FFFFFF;
}}

QMenu {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 20px;
    border-radius: 3px;
}}

QMenu::item:selected {{
    background-color: {ACCENT};
    color: #FFFFFF;
}}

QMenu::separator {{
    height: 1px;
    background-color: {BORDER};
    margin: 4px 8px;
}}

/* ── Status bar ──────────────────────────────────────────── */
QStatusBar {{
    background-color: {PANEL};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-size: 12px;
    padding: 2px 8px;
}}

/* ── Record / Stop buttons (Live Capture) ───────────────── */
QPushButton#RecordButton {{
    background-color: {ACCENT};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-weight: 700;
    font-size: 15px;
    min-height: 46px;
    letter-spacing: 0.5px;
}}

QPushButton#RecordButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton#RecordButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton#RecordButton:disabled {{
    background-color: #ADB5BD;
    color: #F8F9FA;
}}

QPushButton#StopRecordButton {{
    background-color: {ERROR};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-weight: 700;
    font-size: 15px;
    min-height: 46px;
}}

QPushButton#StopRecordButton:hover {{
    background-color: #A93226;
}}

QPushButton#StopRecordButton:disabled {{
    background-color: #ADB5BD;
    color: #F8F9FA;
}}

/* ── Section divider label ──────────────────────────────── */
QLabel#DividerLabel {{
    color: {TEXT_MUTED};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}}
"""


def apply_theme(app):
    """Apply Scandinavian stylesheet to the QApplication."""
    app.setStyleSheet(STYLESHEET)
