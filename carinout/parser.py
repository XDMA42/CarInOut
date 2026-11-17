"""CSV 文件名解析：场景.*点位.*锚点.*.csv。

支持两种写法：
- “中间座椅+脚坑-68-[COM261][SLAVE4] 2026-10-23 15-44-12.743.csv”：锚点取方括号内容，跳过 [COMxxx]
- “中排座椅_53_SLAVE2.csv”：不带方括号时锚点为 SLAVE+数字
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# 场景允许包含分隔符（非贪婪 + 回溯），点位为不含分隔符的一段，锚点之后允许任意后缀（如时间戳）
DEFAULT_PATTERN = (
    r"^(?P<scene>.+?)[_\-\s]+(?P<point>[^_\-\s\[\]]+)[_\-\s]+"
    r"(?:\[COM\d+\]\s*)?(?P<anchor>\[[^\]]+\]|SLAVE\d+).*\.csv$"
)
# 旧版本的默认规则，config.json 中保存的若是这些，加载时升级为新的默认规则
LEGACY_DEFAULT_PATTERNS = (
    r"^(?P<scene>.+?)[_\-\s]+(?P<point>[^_\-\s]+)[_\-\s]+(?P<anchor>SLAVE\d+).*\.csv$",
)

REQUIRED_GROUPS = ("scene", "point", "anchor")


@dataclass(frozen=True)
class ParsedFile:
    path: Path
    scene: str
    point: str
    slave: str
    source: str = ""  # 来自哪个原始数据文件夹（文件夹名，多个同名时加以区分）


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
    anchor = anchor.strip("[]").strip()
    if not (scene and point and anchor):
        return None
    return scene, point, anchor.upper()


def unique_source_names(folders: list[Path]) -> list[str]:
    """各原始数据文件夹的简称：默认取文件夹名；重名时加上上级文件夹名，仍重名再加序号。"""
    paths = [Path(f).resolve() for f in folders]
    names = [p.name or "dataset" for p in paths]
    counts = Counter(names)
    names = [f"{p.parent.name}_{n}" if counts[n] > 1 and p.parent.name else n for p, n in zip(paths, names)]
    seen: Counter[str] = Counter()
    result = []
    for n in names:
        seen[n] += 1
        result.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return result


def scan_folder(
    folder: Path, pattern: str = DEFAULT_PATTERN, source: str | None = None
) -> tuple[list[ParsedFile], list[Path]]:
    """扫描文件夹（不递归）中的 csv，返回 (解析成功的文件, 无法解析的 csv)。

    source 为这些文件的来源简称，默认取文件夹名。
    """
    regex = compile_pattern(pattern)
    source = source if source is not None else unique_source_names([folder])[0]
    parsed: list[ParsedFile] = []
    unmatched: list[Path] = []
    for path in sorted(folder.iterdir(), key=lambda p: natural_key(p.name)):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        result = parse_filename(path.name, regex)
        if result is None:
            unmatched.append(path)
        else:
            parsed.append(ParsedFile(path, *result, source=source))
    return parsed, unmatched


def scan_folders(folders: list[Path], pattern: str = DEFAULT_PATTERN) -> tuple[list[ParsedFile], list[Path]]:
    """依次扫描多个原始数据文件夹，来源简称见 unique_source_names。"""
    parsed: list[ParsedFile] = []
    unmatched: list[Path] = []
    for folder, source in zip(folders, unique_source_names(folders)):
        p, u = scan_folder(folder, pattern, source)
        parsed += p
        unmatched += u
    return parsed, unmatched


def group_by_point(files: list[ParsedFile]) -> dict[tuple[str, str, str], dict[str, list[Path]]]:
    """(来源, 场景, 点位) → {SLAVE: [文件...]}，按自然顺序排序。

    不同文件夹中场景、点位相同的数据是不同批次的采集，分开合并。
    """
    groups: dict[tuple[str, str, str], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        groups[(f.source, f.scene, f.point)][f.slave].append(f.path)
    return {
        key: {slave: groups[key][slave] for slave in sorted(groups[key], key=natural_key)}
        for key in sorted(groups, key=lambda k: tuple(natural_key(part) for part in k))
    }


def natural_key(text: str) -> list:
    """让 “SLAVE10” 排在 “SLAVE2” 之后、点位 “100” 排在 “53” 之后。"""
    return [(0, int(tok), "") if tok.isdigit() else (1, 0, tok.lower()) for tok in re.split(r"(\d+)", text) if tok]
