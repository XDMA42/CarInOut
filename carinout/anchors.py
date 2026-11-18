"""锚点定义与配置持久化。

12 个锚点的位置固定（车外 10 个：四角 + 十字 + 左右后视镜；车内 2 个），
名称可在界面中修改，修改结果和 SLAVE→锚点 的映射一起保存到 config.json。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .parser import DEFAULT_PATTERN, LEGACY_DEFAULT_PATTERNS

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass(frozen=True)
class AnchorSlot:
    """一个锚点位置。x、y 为相对车身的归一化坐标：(0,0) 车身左上角，(1,1) 右下角，车头朝上；
    x < 0 或 x > 1 表示伸出车身之外（后视镜）。"""

    key: str
    default_name: str
    description: str
    x: float
    y: float
    inside: bool


# 顺序即合并后输出列的顺序，也是“按编号分配”的顺序；新锚点追加在末尾，已有锚点的顺序和编号保持不变
ANCHOR_SLOTS: tuple[AnchorSlot, ...] = (
    AnchorSlot("front_left", "FL", "左前角", 0.0, 0.0, False),
    AnchorSlot("front", "F", "车头正中", 0.5, 0.0, False),
    AnchorSlot("front_right", "FR", "右前角", 1.0, 0.0, False),
    AnchorSlot("right", "R", "右侧正中", 1.0, 0.5, False),
    AnchorSlot("rear_right", "RR", "右后角", 1.0, 1.0, False),
    AnchorSlot("rear", "B", "车尾正中", 0.5, 1.0, False),
    AnchorSlot("rear_left", "RL", "左后角", 0.0, 1.0, False),
    AnchorSlot("left", "L", "左侧正中", 0.0, 0.5, False),
    AnchorSlot("inside_front", "IF", "车内前部", 0.5, 0.36, True),
    AnchorSlot("inside_rear", "IR", "车内后部", 0.5, 0.72, True),
    AnchorSlot("mirror_left", "LM", "左后视镜", -0.06, 0.277, False),
    AnchorSlot("mirror_right", "RM", "右后视镜", 1.06, 0.277, False),
)

SLOT_KEYS = tuple(s.key for s in ANCHOR_SLOTS)


def default_names() -> dict[str, str]:
    return {s.key: s.default_name for s in ANCHOR_SLOTS}


def validate_name(name: str, names: dict[str, str], key: str) -> str | None:
    """返回错误信息；名称合法时返回 None。名称会作为列名后缀和文件内容，需唯一且不含分隔符。"""
    name = name.strip()
    if not name:
        return "名称不能为空"
    if any(ch in name for ch in ',"\r\n'):
        return "名称不能包含逗号、引号或换行"
    if any(v == name for k, v in names.items() if k != key):
        return f"名称“{name}”已被其他锚点使用"
    return None


def _input_dirs(data: dict) -> list[str]:
    """上次的原始数据文件夹列表；兼容旧版本只保存一个文件夹的 last_input_dir。"""
    dirs = data.get("last_input_dirs")
    if isinstance(dirs, list):
        return [d for d in dirs if isinstance(d, str) and d]
    old = data.get("last_input_dir")
    return [old] if isinstance(old, str) and old else []


@dataclass
class AppConfig:
    anchor_names: dict[str, str]
    # SLAVE 标识 → 锚点 key；未出现在字典中的 SLAVE 视为未分配
    slave_mapping: dict[str, str]
    filename_pattern: str
    last_input_dirs: list[str] = field(default_factory=list)
    last_output_dir: str = ""
    # 场景 → 标签（0 车外 / 1 车内）；未出现的场景视为未标注
    scene_labels: dict[str, int] = field(default_factory=dict)
    # 单独设置标签的点位：场景 → {点位: 标签}，优先于场景标签。按场景名 + 点位号保存，所有原始数据文件夹共用
    point_labels: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        path = path or CONFIG_PATH
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        names = default_names()
        for key, value in (data.get("anchor_names") or {}).items():
            if key in names and isinstance(value, str) and value.strip():
                names[key] = value.strip()
        if len(set(names.values())) != len(names):
            names = default_names()
        mapping = {
            slave: key
            for slave, key in (data.get("slave_mapping") or {}).items()
            if key in names
        }
        pattern = data.get("filename_pattern") or DEFAULT_PATTERN
        if pattern in LEGACY_DEFAULT_PATTERNS:
            pattern = DEFAULT_PATTERN
        return cls(
            anchor_names=names,
            slave_mapping=mapping,
            filename_pattern=pattern,
            last_input_dirs=_input_dirs(data),
            last_output_dir=data.get("last_output_dir", ""),
            scene_labels={
                scene: label
                for scene, label in (data.get("scene_labels") or {}).items()
                if type(label) is int and label in (0, 1)
            },
            point_labels={
                scene: valid
                for scene, points in (data.get("point_labels") or {}).items()
                if isinstance(points, dict)
                and (valid := {str(p): v for p, v in points.items() if type(v) is int and v in (0, 1)})
            },
        )

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        payload = {
            "anchor_names": self.anchor_names,
            "slave_mapping": self.slave_mapping,
            # 使用默认规则时不写入，以后默认规则更新能直接生效
            "filename_pattern": "" if self.filename_pattern == DEFAULT_PATTERN else self.filename_pattern,
            "last_input_dirs": self.last_input_dirs,
            "last_output_dir": self.last_output_dir,
            "scene_labels": self.scene_labels,
            "point_labels": self.point_labels,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 标签 ----
    def label_for(self, scene: str, point: str) -> int | None:
        """点位的实际标签：单独设置过的用它，否则跟随场景；都没有时为 None（未标注）。"""
        override = self.point_labels.get(scene, {}).get(point)
        return override if override is not None else self.scene_labels.get(scene)

    def is_point_overridden(self, scene: str, point: str) -> bool:
        return point in self.point_labels.get(scene, {})

    def set_point_label(self, scene: str, point: str, label: int | None) -> None:
        """单独设置点位标签；label 为 None 表示恢复跟随场景。"""
        points = self.point_labels.setdefault(scene, {})
        if label is None:
            points.pop(point, None)
        else:
            points[point] = label
        if not points:
            del self.point_labels[scene]
