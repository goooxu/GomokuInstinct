"""技术报告（docs/）的一致性检查。

文档腐烂是静默的：代码改了名字、挪了文件，文档照样"看起来对"，只有读者照着做
不通时才会发现。这里把几条能机器验的约束固定下来：

* 相对链接与图片目标必须存在；
* SVG 必须能解析（手写 SVG 最常见的错就是标签没闭合），且必须有铺满 viewBox 的
  背景矩形 —— 否则 GitHub 暗色模式下深色文字配透明背景，整张图不可读；
* 文档里提到的仓库内路径必须真实存在；
* 每章末尾「代码索引」表里的符号必须真的在那个文件里；
* 不得泄漏开发机信息（CLAUDE.md 的硬约束）。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"

# markdown 的相对链接与图片：[文字](目标) / ![文字](目标)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# 行内代码里的仓库内路径，例如 `gomoku_instinct/model/net.py`
REPO_PATH_RE = re.compile(
    r"`((?:gomoku_instinct|csrc|scripts|tests|configs)/[\w./{},-]+)`"
)
SVG_NS = "{http://www.w3.org/2000/svg}"


def md_files() -> list[Path]:
    return sorted(DOCS.rglob("*.md")) if DOCS.is_dir() else []


def svg_files() -> list[Path]:
    return sorted(DOCS.rglob("*.svg")) if DOCS.is_dir() else []


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO)) for p in paths]


def test_docs_directory_exists():
    assert DOCS.is_dir(), "docs/ 不存在"
    assert (DOCS / "README.md").is_file(), "docs/README.md（索引）缺失"


# ── 链接 ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc", md_files(), ids=_ids(md_files()))
def test_relative_links_resolve(doc: Path):
    broken = []
    for target in LINK_RE.findall(doc.read_text()):
        target = target.split()[0].strip()  # 去掉 ![](x.svg "标题") 里的标题
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = (doc.parent / target.split("#")[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert not broken, f"{doc.name} 里有失效链接: {broken}"


def test_index_links_every_document():
    """索引必须链到每一篇，不能有孤儿文档 —— 写了却没人能找到等于没写。"""
    index = (DOCS / "README.md").read_text()
    orphans = [
        p.name for p in md_files()
        if p.name != "README.md" and p.name not in index
    ]
    assert not orphans, f"docs/README.md 没有链接到: {orphans}"


# ── SVG ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("svg", svg_files(), ids=_ids(svg_files()))
def test_svg_parses_and_is_dark_mode_safe(svg: Path):
    try:
        root = ET.parse(svg).getroot()
    except ET.ParseError as exc:
        pytest.fail(f"{svg.name} 不是合法 XML: {exc}")

    view_box = root.get("viewBox")
    assert view_box, f"{svg.name} 缺少 viewBox"
    _, _, vw, vh = (float(v) for v in view_box.replace(",", " ").split())

    # 铺满画布的背景矩形：GitHub 暗色模式下透明背景 + 深色文字会整张不可读
    covering = [
        r for r in root.iter(f"{SVG_NS}rect")
        if float(r.get("width", 0)) >= vw and float(r.get("height", 0)) >= vh
    ]
    assert covering, f"{svg.name} 没有铺满 viewBox 的背景矩形（暗色模式下会不可读）"
    fill = (covering[0].get("fill") or "").lower()
    assert fill and fill != "none", f"{svg.name} 的背景矩形没有填充色"

    # GitHub 会 sanitize 掉 <style> 与脚本，用了等于白用
    for tag in ("style", "script"):
        assert not list(root.iter(f"{SVG_NS}{tag}")), \
            f"{svg.name} 用了 <{tag}>，GitHub 渲染时会被剥掉"


# ── 代码引用 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc", md_files(), ids=_ids(md_files()))
def test_referenced_repo_paths_exist(doc: Path):
    """文档里提到的仓库内路径必须真实存在。

    支持 `csrc/renju.{h,cpp}` 这种花括号简写。
    """
    missing = []
    for ref in REPO_PATH_RE.findall(doc.read_text()):
        for path in _expand_braces(ref):
            if not (REPO / path).exists():
                missing.append(path)
    assert not missing, f"{doc.name} 引用了不存在的路径: {sorted(set(missing))}"


def _expand_braces(ref: str) -> list[str]:
    match = re.search(r"\{([^}]*)\}", ref)
    if not match:
        return [ref]
    return [
        ref[: match.start()] + part.strip() + ref[match.end():]
        for part in match.group(1).split(",")
    ]


@pytest.mark.parametrize("doc", md_files(), ids=_ids(md_files()))
def test_code_index_symbols_exist(doc: Path):
    """「代码索引」表里的符号必须真的在那个文件里。

    这是防文档腐烂最有效的一条：改了类名而没改文档，这里会红。
    """
    missing = []
    for line in doc.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 4:
            continue
        path_cell, symbol_cell = cells[1], cells[2]
        path_match = re.fullmatch(r"`([\w./-]+\.(?:py|cpp|h|yaml|sh|html))`", path_cell)
        symbol_match = re.fullmatch(r"`([\w.]+)`", symbol_cell)
        if not (path_match and symbol_match):
            continue
        target = REPO / path_match.group(1)
        if not target.is_file():
            missing.append(f"{path_match.group(1)}（文件不存在）")
            continue
        symbol = symbol_match.group(1).split(".")[-1]
        if symbol not in target.read_text():
            missing.append(f"{path_match.group(1)}::{symbol}")
    assert not missing, f"{doc.name} 的代码索引对不上: {missing}"


# ── 保密 ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc", md_files(), ids=_ids(md_files()))
def test_no_dev_machine_details(doc: Path):
    """文档里不得出现开发机信息。

    模式写成通用形式（任意 IPv4、任意 /home/ 绝对路径），这样这条测试本身
    也不会把要保护的东西写进仓库。
    """
    text = doc.read_text()
    leaks = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
    leaks += re.findall(r"/home/[\w.-]+", text)
    assert not leaks, f"{doc.name} 疑似泄漏开发机信息: {sorted(set(leaks))}"
