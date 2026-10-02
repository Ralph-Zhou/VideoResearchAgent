"""ANSI color utilities for readable terminal output."""

import sys

_NO_COLOR = not sys.stdout.isatty()

# ── ANSI codes ──
RESET   = "" if _NO_COLOR else "\033[0m"
BOLD    = "" if _NO_COLOR else "\033[1m"
DIM     = "" if _NO_COLOR else "\033[2m"

RED     = "" if _NO_COLOR else "\033[31m"
GREEN   = "" if _NO_COLOR else "\033[32m"
YELLOW  = "" if _NO_COLOR else "\033[33m"
BLUE    = "" if _NO_COLOR else "\033[34m"
MAGENTA = "" if _NO_COLOR else "\033[35m"
CYAN    = "" if _NO_COLOR else "\033[36m"
WHITE   = "" if _NO_COLOR else "\033[37m"

BG_BLUE    = "" if _NO_COLOR else "\033[44m"
BG_GREEN   = "" if _NO_COLOR else "\033[42m"
BG_RED     = "" if _NO_COLOR else "\033[41m"
BG_YELLOW  = "" if _NO_COLOR else "\033[43m"
BG_MAGENTA = "" if _NO_COLOR else "\033[45m"
BG_CYAN    = "" if _NO_COLOR else "\033[46m"

# ── Node color mapping ──
NODE_COLORS = {
    "planner":  CYAN,
    "searcher": BLUE,
    "selector": MAGENTA,
    "watcher":  YELLOW,
    "checker":  WHITE,
    "analyst":  GREEN,
}

NODE_ICONS = {
    "planner":  "🧠",
    "searcher": "🔍",
    "selector": "📋",
    "watcher":  "👁️ ",
    "checker":  "🔄",
    "analyst":  "📊",
}


def node_tag(node: str) -> str:
    color = NODE_COLORS.get(node, WHITE)
    icon = NODE_ICONS.get(node, "▶")
    return f"{color}{BOLD}{icon} {node.upper()}{RESET}"


def header(text: str) -> str:
    return f"\n{BOLD}{CYAN}{'━' * 60}{RESET}\n{BOLD}{CYAN}  {text}{RESET}\n{BOLD}{CYAN}{'━' * 60}{RESET}"


def subheader(text: str) -> str:
    return f"{BOLD}{BLUE}── {text} ──{RESET}"


def success(text: str) -> str:
    return f"{GREEN}{BOLD}{text}{RESET}"


def error(text: str) -> str:
    return f"{RED}{BOLD}{text}{RESET}"


def warning(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def key_value(key: str, value: str, key_color: str = CYAN) -> str:
    return f"  {key_color}{BOLD}{key:15s}{RESET} {value}"


def result_box(lines: list[str], color: str = GREEN) -> str:
    width = 60
    bar = f"{color}{BOLD}{'━' * width}{RESET}"
    padded = [f"{color}┃{RESET} {line.ljust(width - 4)} {color}┃{RESET}" for line in lines]
    top = f"{color}┏{'━' * (width - 2)}┓{RESET}"
    bot = f"{color}┗{'━' * (width - 2)}┛{RESET}"
    return "\n".join([top] + padded + [bot])
