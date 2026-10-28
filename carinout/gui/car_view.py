"""车模俯视图：车头朝上，车外 8 个锚点（四角 + 十字），车内 2 个锚点。"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ..anchors import ANCHOR_SLOTS, AnchorSlot

CAR_RATIO = 0.46  # 车宽 / 车长

COLOR_BG_TOP = QColor("#f4f6fb")
COLOR_BG_BOTTOM = QColor("#e6eaf2")
COLOR_TEXT = QColor("#1f2937")
COLOR_MUTED = QColor("#6b7280")

STATE_COLORS = {
    "assigned": QColor("#2563eb"),
    "unassigned": QColor("#9ca3af"),
    "conflict": QColor("#dc2626"),
    "ok": QColor("#16a34a"),
    "missing": QColor("#f59e0b"),
}

LEGENDS = {
    False: (("assigned", "已分配 SLAVE"), ("unassigned", "未分配"), ("conflict", "映射冲突")),
    True: (("ok", "有有效数据"), ("missing", "缺失 / 无有效行"), ("unassigned", "未分配"), ("conflict", "映射冲突")),
}


@dataclass
class Coverage:
    """选中点位时某个锚点的数据情况。"""

    ok: bool
    text: str


@dataclass
class _Layout:
    marker: QPointF
    label: QRectF


class CarView(QWidget):
    anchorClicked = Signal(str)
    anchorDoubleClicked = Signal(str)

    MARKER_R = 11.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(560, 560)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._names: dict[str, str] = {s.key: s.default_name for s in ANCHOR_SLOTS}
        self._slaves: dict[str, list[str]] = {}
        self._coverage: dict[str, Coverage] | None = None
        self._selected: str | None = None
        self._title = ""
        self._layouts: dict[str, _Layout] = {}

    # ---- 外部接口 ----
    def set_names(self, names: dict[str, str]) -> None:
        self._names = dict(names)
        self.update()

    def set_assignment(self, slaves_by_key: dict[str, list[str]]) -> None:
        """锚点 key → 分配给它的 SLAVE 列表（多于 1 个即冲突）。"""
        self._slaves = {k: list(v) for k, v in slaves_by_key.items()}
        self.update()

    def set_coverage(self, coverage: dict[str, Coverage] | None) -> None:
        self._coverage = coverage
        self.update()

    def set_selected(self, key: str | None) -> None:
        self._selected = key
        self.update()

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    # ---- 状态 ----
    def _state(self, key: str) -> str:
        slaves = self._slaves.get(key, [])
        if len(slaves) > 1:
            return "conflict"
        if not slaves:
            return "unassigned"
        if self._coverage is not None:
            cov = self._coverage.get(key)
            return "ok" if cov and cov.ok else "missing"
        return "assigned"

    def _lines(self, key: str) -> list[str]:
        slaves = self._slaves.get(key, [])
        lines = [self._names.get(key, key), " / ".join(slaves) if slaves else "未分配"]
        if self._coverage is not None and len(slaves) == 1:
            cov = self._coverage.get(key)
            lines.append(cov.text if cov else "无此点位文件")
        return lines

    # ---- 几何 ----
    def _car_rect(self) -> QRectF:
        margin_x, margin_top, margin_bottom = 150.0, 120.0, 130.0
        avail_w = max(self.width() - 2 * margin_x, 80.0)
        avail_h = max(self.height() - margin_top - margin_bottom, 160.0)
        h = min(avail_h, avail_w / CAR_RATIO)
        w = h * CAR_RATIO
        top = margin_top + (avail_h - h) / 2
        return QRectF((self.width() - w) / 2, top, w, h)

    @staticmethod
    def _corner_radii(car: QRectF) -> tuple[float, float]:
        return car.width() * 0.30, car.width() * 0.20

    def _anchor_point(self, slot: AnchorSlot, car: QRectF) -> QPointF:
        x = car.left() + slot.x * car.width()
        y = car.top() + slot.y * car.height()
        if not slot.inside and slot.x != 0.5 and slot.y != 0.5:
            # 四角锚点落在圆角车身轮廓的 45° 处
            r_front, r_rear = self._corner_radii(car)
            k = (r_front if slot.y == 0 else r_rear) * (1 - 0.7071)
            x += k if slot.x == 0 else -k
            y += k if slot.y == 0 else -k
        return QPointF(x, y)

    def _label_rect(self, slot: AnchorSlot, marker: QPointF, lines: list[str]) -> QRectF:
        fm_bold = QFontMetricsF(self._font(12, bold=True))
        fm = QFontMetricsF(self._font(10))
        w = max([fm_bold.horizontalAdvance(lines[0])] + [fm.horizontalAdvance(t) for t in lines[1:]]) + 16
        h = fm_bold.height() + fm.height() * (len(lines) - 1) + 8
        gap = self.MARKER_R + 8
        if slot.inside:
            return QRectF(marker.x() - w / 2, marker.y() + gap - 2, w, h)
        if slot.x == 0:
            x = marker.x() - gap - w
        elif slot.x == 1:
            x = marker.x() + gap
        else:
            x = marker.x() - w / 2
        if slot.y == 0:
            y = marker.y() - gap - h
        elif slot.y == 1:
            y = marker.y() + gap
        else:
            y = marker.y() - h / 2
        return QRectF(x, y, w, h)

    def _compute_layouts(self) -> QRectF:
        car = self._car_rect()
        self._layouts = {}
        for slot in ANCHOR_SLOTS:
            marker = self._anchor_point(slot, car)
            self._layouts[slot.key] = _Layout(marker, self._label_rect(slot, marker, self._lines(slot.key)))
        return car

    def _hit(self, pos: QPointF) -> str | None:
        for key, lay in self._layouts.items():
            d = lay.marker - pos
            if d.x() ** 2 + d.y() ** 2 <= (self.MARKER_R + 6) ** 2 or lay.label.contains(pos):
                return key
        return None

    @staticmethod
    def _font(size: float, bold: bool = False) -> QFont:
        f = QFont()
        f.setPointSizeF(size)
        f.setBold(bold)
        return f

    # ---- 绘制 ----
    def paintEvent(self, _event) -> None:  # noqa: N802
        car = self._compute_layouts()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_background(p)
        self._paint_car(p, car)
        for slot in ANCHOR_SLOTS:
            self._paint_anchor(p, slot)
        self._paint_overlay_text(p)
        p.end()

    def _paint_background(self, p: QPainter) -> None:
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, COLOR_BG_TOP)
        g.setColorAt(1, COLOR_BG_BOTTOM)
        p.fillRect(self.rect(), g)

    def _paint_car(self, p: QPainter, car: QRectF) -> None:
        w, h = car.width(), car.height()

        def pt(fx: float, fy: float) -> QPointF:
            return QPointF(car.left() + fx * w, car.top() + fy * h)

        def rect(fx0: float, fy0: float, fx1: float, fy1: float) -> QRectF:
            return QRectF(pt(fx0, fy0), pt(fx1, fy1))

        # 车轮、后视镜（先画，被车身压住一部分）
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2b2f36"))
        for fy in (0.17, 0.81):
            for fx in (-0.045, 0.915):
                p.drawRoundedRect(rect(fx, fy - 0.065, fx + 0.13, fy + 0.065), w * 0.04, w * 0.04)
        p.setBrush(QColor("#4b5563"))
        for fx0, fx1 in ((-0.10, 0.04), (0.96, 1.10)):
            p.drawRoundedRect(rect(fx0, 0.262, fx1, 0.292), w * 0.03, w * 0.03)

        # 车身
        r_front, r_rear = self._corner_radii(car)
        body = _rounded_rect(car, r_front, r_rear)
        p.setPen(QPen(QColor("#5b6b82"), 2))
        p.setBrush(QColor("#ffffff"))
        p.drawPath(body)

        # 车头 / 车尾灯
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#fde68a"))
        for fx0, fx1 in ((0.10, 0.28), (0.72, 0.90)):
            p.drawRoundedRect(rect(fx0, 0.012, fx1, 0.032), 4, 4)
        p.setBrush(QColor("#fca5a5"))
        for fx0, fx1 in ((0.08, 0.26), (0.74, 0.92)):
            p.drawRoundedRect(rect(fx0, 0.972, fx1, 0.988), 3, 3)

        # 引擎盖线条
        p.setPen(QPen(QColor("#d6dce6"), 1.4))
        p.drawLine(pt(0.30, 0.05), pt(0.27, 0.20))
        p.drawLine(pt(0.70, 0.05), pt(0.73, 0.20))

        # 前挡、后挡、侧窗
        glass = QColor("#bfd4ee")
        p.setPen(QPen(QColor("#93acc9"), 1.2))
        p.setBrush(glass)
        p.drawPolygon(QPolygonF([pt(0.18, 0.215), pt(0.82, 0.215), pt(0.91, 0.305), pt(0.09, 0.305)]))
        p.drawPolygon(QPolygonF([pt(0.11, 0.865), pt(0.89, 0.865), pt(0.82, 0.925), pt(0.18, 0.925)]))
        p.drawRoundedRect(rect(0.035, 0.32, 0.075, 0.85), 3, 3)
        p.drawRoundedRect(rect(0.925, 0.32, 0.965, 0.85), 3, 3)

        # 座舱（掀顶视角）
        p.setPen(QPen(QColor("#c9d2df"), 1.2))
        p.setBrush(QColor("#f1f4f9"))
        p.drawRoundedRect(rect(0.09, 0.31, 0.91, 0.86), w * 0.05, w * 0.05)

        seat = QColor("#d5dce7")
        seat_edge = QPen(QColor("#b4bfce"), 1.2)
        p.setPen(seat_edge)
        p.setBrush(seat)
        for fy0, fy1 in ((0.345, 0.455), (0.525, 0.625)):  # 前排、中排独立座椅
            for fx0, fx1 in ((0.16, 0.42), (0.58, 0.84)):
                self._paint_seat(p, rect(fx0, fy0, fx1, fy1))
        self._paint_seat(p, rect(0.14, 0.715, 0.86, 0.825))  # 后排长座
        p.setBrush(QColor("#e2e7ef"))
        p.drawRoundedRect(rect(0.45, 0.35, 0.55, 0.50), 4, 4)  # 中控扶手

        # 方向盘
        p.setPen(QPen(QColor("#9aa6b8"), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(pt(0.20, 0.312), pt(0.38, 0.336)))

        # 车头提示
        p.setPen(COLOR_MUTED)
        p.setFont(self._font(10, bold=True))
        p.drawText(rect(0.0, 0.07, 1.0, 0.17), Qt.AlignmentFlag.AlignCenter, "▲ 车头")

    @staticmethod
    def _paint_seat(p: QPainter, r: QRectF) -> None:
        p.drawRoundedRect(r, 6, 6)
        back = QRectF(r.left() + 3, r.bottom() - r.height() * 0.28, r.width() - 6, r.height() * 0.22)
        p.drawRoundedRect(back, 4, 4)

    def _paint_anchor(self, p: QPainter, slot: AnchorSlot) -> None:
        lay = self._layouts[slot.key]
        state = self._state(slot.key)
        color = STATE_COLORS[state]
        selected = slot.key == self._selected

        # 标签
        lines = self._lines(slot.key)
        p.setPen(QPen(color, 2.4 if selected else 1.2))
        bg = QColor("#ffffff")
        bg.setAlpha(240)
        p.setBrush(bg)
        p.drawRoundedRect(lay.label, 6, 6)
        x = lay.label.left() + 8
        y = lay.label.top() + 4
        bold = self._font(12, bold=True)
        p.setFont(bold)
        p.setPen(COLOR_TEXT)
        fm_bold = QFontMetricsF(bold)
        p.drawText(QRectF(x, y, lay.label.width() - 16, fm_bold.height()), Qt.AlignmentFlag.AlignHCenter, lines[0])
        y += fm_bold.height()
        small = self._font(10)
        p.setFont(small)
        fm = QFontMetricsF(small)
        for i, text in enumerate(lines[1:]):
            p.setPen(color if i == 0 and state != "unassigned" else COLOR_MUTED)
            p.drawText(QRectF(x, y, lay.label.width() - 16, fm.height()), Qt.AlignmentFlag.AlignHCenter, text)
            y += fm.height()

        # 标记点
        r = self.MARKER_R
        if selected:
            ring = QColor(color)
            ring.setAlpha(70)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ring)
            p.drawEllipse(lay.marker, r + 7, r + 7)
        p.setPen(QPen(QColor("#ffffff"), 2.5))
        p.setBrush(color)
        p.drawEllipse(lay.marker, r, r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(lay.marker, r * 0.32, r * 0.32)

    def _paint_overlay_text(self, p: QPainter) -> None:
        if self._title:
            p.setPen(COLOR_TEXT)
            p.setFont(self._font(13, bold=True))
            p.drawText(QRectF(16, 12, self.width() - 32, 26), Qt.AlignmentFlag.AlignLeft, self._title)

        # 图例
        small = self._font(10)
        p.setFont(small)
        fm = QFontMetricsF(small)
        x, y = 16.0, self.height() - 16.0 - fm.height()
        for state, text in LEGENDS[self._coverage is not None]:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(STATE_COLORS[state])
            p.drawEllipse(QPointF(x + 6, y + fm.height() / 2), 6, 6)
            p.setPen(COLOR_MUTED)
            p.drawText(QPointF(x + 18, y + fm.ascent()), text)
            x += 18 + fm.horizontalAdvance(text) + 18

        hint = "单击锚点：定位映射  ·  双击锚点：重命名"
        p.setPen(COLOR_MUTED)
        p.drawText(
            QRectF(16, self.height() - 16 - fm.height(), self.width() - 32, fm.height()),
            Qt.AlignmentFlag.AlignRight,
            hint,
        )

    # ---- 交互 ----
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        key = self._hit(event.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if key else Qt.CursorShape.ArrowCursor)
        if key:
            slot = next(s for s in ANCHOR_SLOTS if s.key == key)
            QToolTip.showText(event.globalPosition().toPoint(), f"{self._names[key]}：{slot.description}", self)
        else:
            QToolTip.hideText()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            key = self._hit(event.position())
            if key:
                self.anchorClicked.emit(key)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            key = self._hit(event.position())
            if key:
                self.anchorDoubleClicked.emit(key)


def _rounded_rect(r: QRectF, r_top: float, r_bottom: float) -> QPainterPath:
    """上下圆角半径不同的矩形（车头更圆润）。"""
    path = QPainterPath()
    path.moveTo(r.left() + r_top, r.top())
    path.lineTo(r.right() - r_top, r.top())
    path.arcTo(QRectF(r.right() - 2 * r_top, r.top(), 2 * r_top, 2 * r_top), 90, -90)
    path.lineTo(r.right(), r.bottom() - r_bottom)
    path.arcTo(QRectF(r.right() - 2 * r_bottom, r.bottom() - 2 * r_bottom, 2 * r_bottom, 2 * r_bottom), 0, -90)
    path.lineTo(r.left() + r_bottom, r.bottom())
    path.arcTo(QRectF(r.left(), r.bottom() - 2 * r_bottom, 2 * r_bottom, 2 * r_bottom), 270, -90)
    path.lineTo(r.left(), r.top() + r_top)
    path.arcTo(QRectF(r.left(), r.top(), 2 * r_top, 2 * r_top), 180, -90)
    path.closeSubpath()
    return path
