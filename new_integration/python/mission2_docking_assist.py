"""Operator-only guidance for Mission 2.2 chamber docking.

The manual requires physical control and accuracy. This advisor reports what
the camera sees but deliberately returns words, never motor commands.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DockingGuidance:
    message: str
    aligned: bool
    close: bool


class DockingAdvisor:
    def __init__(self, centre_tolerance: float = 0.12,
                 close_area_fraction: float = 0.18):
        self.centre_tolerance = centre_tolerance
        self.close_area_fraction = close_area_fraction

    def advise(self, visible_tag_id: int | None, target_tag_id: int,
               horizontal_error: float | None = None,
               area_fraction: float = 0.0) -> DockingGuidance:
        if visible_tag_id is None:
            return DockingGuidance("Target chamber not visible - search manually", False, False)
        if visible_tag_id != target_tag_id:
            return DockingGuidance(
                f"Wrong chamber: tag {visible_tag_id}; target is {target_tag_id}", False, False)
        if horizontal_error is None:
            return DockingGuidance("Target tag found - centre it in the camera", False, False)

        close = area_fraction >= self.close_area_fraction
        if horizontal_error < -self.centre_tolerance:
            return DockingGuidance("Target left - steer left slowly", False, close)
        if horizontal_error > self.centre_tolerance:
            return DockingGuidance("Target right - steer right slowly", False, close)
        if close:
            return DockingGuidance("Centred and close - dock slowly under manual control", True, True)
        return DockingGuidance("Centred - move forward slowly under manual control", True, False)

