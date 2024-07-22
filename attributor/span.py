from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    start: int | None = field(default=None)
    end: int | None = field(default=None)
    step: int = field(default=1)
    window_size: int = field(default=1)

    def __post_init__(self):
        assert self.start is None or self.start >= 0
        assert self.end is None or self.end > 0
        assert self.step > 0
        assert self.window_size > 0
