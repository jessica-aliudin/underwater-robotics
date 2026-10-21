"""Visual-only black-line guidance for manually operated Mission 3."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LineObservation:
    found: bool
    horizontal_error: float = 0.0
    angle_degrees: float = 0.0
    area_fraction: float = 0.0

    @property
    def guidance(self) -> str:
        if not self.found:
            return "Black line not visible"
        if self.horizontal_error < -0.15:
            return "Line left - steer left manually"
        if self.horizontal_error > 0.15:
            return "Line right - steer right manually"
        if self.angle_degrees < -12:
            return "Line angles left - correct left manually"
        if self.angle_degrees > 12:
            return "Line angles right - correct right manually"
        return "Line centred - continue forward manually"


def analyse_black_line(frame: np.ndarray, threshold: int = 70,
                       minimum_area_fraction: float = 0.003) -> tuple[LineObservation, np.ndarray]:
    """Find the largest dark floor line in the lower two-thirds of a frame.

    The returned overlay is for the operator. No motion request is produced.
    """

    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    height, width = frame.shape[:2]
    y0 = height // 3
    roi = frame[y0:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.inRange(gray, 0, threshold)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = frame.copy()
    cv2.line(overlay, (width // 2, y0), (width // 2, height), (255, 255, 0), 1)
    if not contours:
        return LineObservation(False), overlay

    contour = max(contours, key=cv2.contourArea)
    area_fraction = cv2.contourArea(contour) / float(width * height)
    if area_fraction < minimum_area_fraction:
        return LineObservation(False), overlay

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return LineObservation(False), overlay
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"]) + y0
    horizontal_error = (cx - width / 2) / (width / 2)

    rect = cv2.minAreaRect(contour)
    raw_angle = rect[2]
    # OpenCV's rectangle angle changes convention when width/height swap.
    rw, rh = rect[1]
    angle = raw_angle if rw < rh else raw_angle + 90.0
    if angle > 90:
        angle -= 180

    shifted = contour.copy()
    shifted[:, 0, 1] += y0
    cv2.drawContours(overlay, [shifted], -1, (0, 255, 0), 2)
    cv2.circle(overlay, (cx, cy), 7, (0, 0, 255), -1)
    observation = LineObservation(True, float(horizontal_error), float(angle), float(area_fraction))
    cv2.putText(overlay, observation.guidance, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return observation, overlay

