"""CSV 文件名解析：场景.*点位.*锚点.*.csv，例如 “中排座椅_53_SLAVE2.csv”。"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# 场景允许包含分隔符（非贪婪 + 回溯），点位为不含分隔符的一段，锚点为 SLAVE+数字，其后允许任意后缀
DEFAULT_PATTERN = r"^(?P<scene>.+?)[_\-\s]+(?P<point>[^_\-\s]+)[_\-\s]+(?P<anchor>SLAVE\d+).*\.csv$"

REQUIRED_GROUPS = ("scene", "point", "anchor")


@dataclass(frozen=True)
class ParsedFile:
    path: Path
    scene: str
    point: str
    slave: str


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """编译文件名正则；缺少必需的命名分组时抛出 ValueError。"""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"正则表达式无效：{exc}") from exc
    missing = [g for g in REQUIRED_GROUPS if g not in regex.groupindex]
    if missing:
        raise ValueError("正则表达式缺少命名分组：" + "、".join(f"(?P<{g}>...)" for g in missing))
    return regex


def parse_filename(name: str, regex: re.Pattern[str]) -> tuple[str, str, str] | None:
    m = regex.match(name)
    if not m:
        return None
    scene, point, anchor = (m.group(g).strip() for g in REQUIRED_GROUPS)
    if not (scene and point and anchor):
        return None
    return scene, point, anchor.upper()


def scan_folder(folder: Path, pattern: str = DEFAULT_PATTERN) -> tuple[list[ParsedFile], list[Path]]:
    """扫描文件夹（不递归）中的 csv，返回 (解析成功的文件, 无法解析的 csv)。"""
    regex = compile_pattern(pattern)
    parsed: list[ParsedFile] = []
    unmatched: list[Path] = []
    for path in sorted(folder.iterdir(), key=lambda p: natural_key(p.name)):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        result = parse_filename(path.name, regex)
        if result is None:
            unmatched.append(path)
        else:
            parsed.append(ParsedFile(path, *result))
    return parsed, unmatched


def group_by_point(files: list[ParsedFile]) -> dict[tuple[str, str], dict[str, list[Path]]]:
    """(场景, 点位) → {SLAVE: [文件...]}，按自然顺序排序。"""
    groups: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        groups[(f.scene, f.point)][f.slave].append(f.path)
    return {
        key: {slave: groups[key][slave] for slave in sorted(groups[key], key=natural_key)}
        for key in sorted(groups, key=lambda k: (natural_key(k[0]), natural_key(k[1])))
    }


def natural_key(text: str) -> list:
    """让 “SLAVE10” 排在 “SLAVE2” 之后、点位 “100” 排在 “53” 之后。"""
    return [(0, int(tok), "") if tok.isdigit() else (1, 0, tok.lower()) for tok in re.split(r"(\d+)", text) if tok]
