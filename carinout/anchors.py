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


@dataclass
class AppConfig:
    anchor_names: dict[str, str]
    # SLAVE 标识 → 锚点 key；未出现在字典中的 SLAVE 视为未分配
    slave_mapping: dict[str, str]
    filename_pattern: str
    last_input_dir: str = ""
    last_output_dir: str = ""
    # 场景 → 标签（0 车外 / 1 车内）；未出现的场景视为未标注
    scene_labels: dict[str, int] = field(default_factory=dict)
    # 小场景类别：编号 → 名称。编号创建后固定，删除类别不会改变其他类别的编号
    sub_categories: dict[int, str] = field(default_factory=dict)
    # 场景 → 小场景类别编号；未出现的场景视为未标注
    scene_subs: dict[str, int] = field(default_factory=dict)
    # 下一个新类别的编号；删除过的编号不再复用，避免同一编号前后代表不同类别
    next_sub_id: int = 0

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
        categories: dict[int, str] = {}
        for cid, name in (data.get("sub_categories") or {}).items():
            if str(cid).isdigit() and isinstance(name, str) and name.strip():
                categories[int(cid)] = name.strip()
        pattern = data.get("filename_pattern") or DEFAULT_PATTERN
        if pattern in LEGACY_DEFAULT_PATTERNS:
            pattern = DEFAULT_PATTERN
        return cls(
            anchor_names=names,
            slave_mapping=mapping,
            filename_pattern=pattern,
            last_input_dir=data.get("last_input_dir", ""),
            last_output_dir=data.get("last_output_dir", ""),
            scene_labels={
                scene: label
                for scene, label in (data.get("scene_labels") or {}).items()
                if type(label) is int and label in (0, 1)
            },
            sub_categories=dict(sorted(categories.items())),
            next_sub_id=max(
                data["next_sub_id"] if type(data.get("next_sub_id")) is int else 0,
                max(categories, default=-1) + 1,
            ),
            scene_subs={
                scene: cid
                for scene, cid in (data.get("scene_subs") or {}).items()
                if type(cid) is int and cid in categories
            },
        )

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        payload = {
            "anchor_names": self.anchor_names,
            "slave_mapping": self.slave_mapping,
            # 使用默认规则时不写入，以后默认规则更新能直接生效
            "filename_pattern": "" if self.filename_pattern == DEFAULT_PATTERN else self.filename_pattern,
            "last_input_dir": self.last_input_dir,
            "last_output_dir": self.last_output_dir,
            "scene_labels": self.scene_labels,
            "sub_categories": {str(cid): name for cid, name in self.sub_categories.items()},
            "scene_subs": self.scene_subs,
            "next_sub_id": self.next_sub_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 小场景类别 ----
    def add_sub_category(self, name: str) -> int:
        """新建小场景类别并返回编号（从 0 开始递增，删除过的编号不复用）。"""
        cid = max(self.next_sub_id, max(self.sub_categories, default=-1) + 1)
        self.sub_categories[cid] = name.strip()
        self.next_sub_id = cid + 1
        return cid

    def remove_sub_category(self, cid: int) -> list[str]:
        """删除类别，并清除使用它的场景标注；返回受影响的场景。"""
        self.sub_categories.pop(cid, None)
        scenes = [scene for scene, c in self.scene_subs.items() if c == cid]
        for scene in scenes:
            del self.scene_subs[scene]
        return scenes

    def validate_sub_category_name(self, name: str, cid: int | None = None) -> str | None:
        """返回错误信息；名称合法时返回 None。cid 为正在重命名的类别。"""
        name = name.strip()
        if not name:
            return "名称不能为空"
        if any(ch in name for ch in ',"\r\n'):
            return "名称不能包含逗号、引号或换行"
        if any(n == name for c, n in self.sub_categories.items() if c != cid):
            return f"类别“{name}”已存在"
        return None
