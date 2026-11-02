"""The report's text, tables and figures, section by section. build_report.py places them in the template.

Blocks:  P(text)   Bullets([...])   Numbered([...])   H3(text)   Fig(file, number, caption)
         Table(number, title, header, rows, widths)      **double stars** make text bold.
"""
from dataclasses import dataclass, field


@dataclass
class P:
    text: str


@dataclass
class Bullets:
    items: list


@dataclass
class Numbered:
    items: list


@dataclass
class H3:
    text: str


@dataclass
class Fig:
    file: str
    number: str
    caption: str
    width: float = 1.0                     # share of the template's figure width


@dataclass
class Table:
    number: str
    title: str
    header: list
    rows: list
    widths: list = field(default_factory=list)
    size: int = 19                         # half-points


# Filled in step by step. Each entry: {"heading": start of the template's heading, "blocks": [...]}.
CHAPTERS = []
