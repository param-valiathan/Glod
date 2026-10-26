"""
MatplotlibCanvas — embeds a matplotlib Figure in a PyQt6 widget.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QSizePolicy, QTabWidget
from PyQt6.QtCore import Qt


class PlotCanvas(QWidget):
    """Single matplotlib figure embedded in Qt with a title bar and navigation toolbar."""

    def __init__(self, fig: Figure, title: str = "", parent=None):
        super().__init__(parent)
        self._fig = fig
        self._title = title

        self._canvas = FigureCanvasQTAgg(fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.setFixedHeight(32)

        # Compute minimum height from figure dimensions so the scroll area
        # cannot compress the plot below its natural render size.
        dpi = fig.get_dpi()
        _, fig_h = fig.get_size_inches()
        canvas_min_h = max(int(fig_h * dpi), 350)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(0)

        title_h = 0
        if title:
            lbl = QLabel(f"  {title}")
            lbl.setFixedHeight(28)
            lbl.setStyleSheet(
                "font-weight: 700; font-size: 12px; color: #5C8A6B;"
                "background: #F0F4F2; border-bottom: 1px solid #DEE2E6;"
                "border-top: 1px solid #DEE2E6; padding-left: 4px;"
            )
            layout.addWidget(lbl)
            title_h = 28

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # Enforce minimum height so QScrollArea cannot compress this widget.
        self.setMinimumHeight(canvas_min_h + 32 + title_h + 12)

        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    @property
    def figure(self) -> Figure:
        return self._fig

    @property
    def title(self) -> str:
        return self._title




class _CategoryPage(QScrollArea):
    """One scrollable column of PlotCanvas widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.setWidget(container)

    def add_canvas(self, canvas: PlotCanvas) -> None:
        self._layout.insertWidget(self._layout.count() - 1, canvas)


class PlotScrollPanel(QTabWidget):
    """
    Plots grouped into category tabs (Overview, Per-file, T_core, Statistics).
    API: add_figure(fig, title, category), figures(), clear().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pages: dict[str, _CategoryPage] = {}
        self._canvases: list[PlotCanvas] = []

    def add_figure(self, fig: Figure, title: str = "",
                   category: str = "Overview") -> None:
        """Append a new figure to the given category tab (created on demand)."""
        page = self._pages.get(category)
        if page is None:
            page = _CategoryPage()
            self._pages[category] = page
            self.addTab(page, category)
        canvas = PlotCanvas(fig, title)
        page.add_canvas(canvas)
        self._canvases.append(canvas)

    def figures(self) -> list[tuple[Figure, str]]:
        """Return all (figure, title) pairs."""
        return [(c.figure, c.title) for c in self._canvases]

    def clear(self) -> None:
        """Remove all figures and category tabs."""
        super().clear()
        import matplotlib.pyplot as plt
        for canvas in self._canvases:
            try:
                plt.close(canvas.figure)
            except Exception:
                pass
        for page in self._pages.values():
            page.deleteLater()
        self._pages.clear()
        self._canvases.clear()
