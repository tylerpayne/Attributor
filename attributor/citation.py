from dataclasses import dataclass


@dataclass
class Citation:
    source: str
    output: str
    attribution: float


class Citer:
    pass
