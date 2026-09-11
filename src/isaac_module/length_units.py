"""Length conversion between the Viam wire's millimeters and the sim's meters."""

MM_PER_M = 1000.0


def to_millimeters(meters: float) -> float:
    return meters * MM_PER_M


def to_meters(millimeters: float) -> float:
    return millimeters / MM_PER_M
