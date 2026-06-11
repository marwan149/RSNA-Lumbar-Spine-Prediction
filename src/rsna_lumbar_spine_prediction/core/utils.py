import re
from typing import Any


def atoi(text: str) -> Any:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> list[Any]:
    return [atoi(c) for c in re.split(r"(\d+)", text)]
