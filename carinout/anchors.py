"""锚点定义与配置持久化。

10 个锚点的位置固定（车外 8 个：四角 + 十字；车内 2 个），
名称可在界面中修改，修改结果和 SLAVE→锚点 的映射一起保存到 config.json。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .parser import DEFAULT_PATTERN

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass(frozen=True)
class AnchorSlot:
    """一个锚点位置。x、y 为相对车身的归一化坐标：(0,0) 车身左上角，(1,1) 右下角，车头朝上。"""

    key: str
    default_name: str
    description: str
    x: float
    y: float
    inside: bool


# 顺序即合并后输出列的顺序
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


@dataclass
class AppConfig:
    anchor_names: dict[str, str]
    # SLAVE 标识 → 锚点 key；未出现在字典中的 SLAVE 视为未分配
    slave_mapping: dict[str, str]
    filename_pattern: str
    last_input_dir: str = ""
    last_output_dir: str = ""

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
        return cls(
            anchor_names=names,
            slave_mapping=mapping,
            filename_pattern=data.get("filename_pattern") or DEFAULT_PATTERN,
            last_input_dir=data.get("last_input_dir", ""),
            last_output_dir=data.get("last_output_dir", ""),
        )

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        payload = {
            "anchor_names": self.anchor_names,
            "slave_mapping": self.slave_mapping,
            "filename_pattern": self.filename_pattern,
            "last_input_dir": self.last_input_dir,
            "last_output_dir": self.last_output_dir,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
