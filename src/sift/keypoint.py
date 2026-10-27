from dataclasses import dataclass


@dataclass
class KeyPoint:
    x: float
    y: float
    size: float
    octave: int
    class_id: int
    response: float