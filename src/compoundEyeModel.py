# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:23:34 2024

@author: usama
"""


"""Compound-eye projection and event-polarity model.

The gnomonic projection uses the complete equations, including user-defined
projection centers. A fixed image-to-ommatidium map is built once per image
geometry. Temporal state is updated separately from spatial projection so the
video can be sampled at either the source FPS or the flicker-fusion frequency.
"""

import numpy as np


class CompoundEyeModel:
    """Project a grayscale scene onto a compound-eye grid."""

    threshIntensity = 0.5  # Threshold in log-intensity difference units

    def __init__(
        self,
        species,
        angle_x,
        angle_y,
        phi_x_0=0.0,
        phi_y_0=0.0,
        flicker_fusion_fps=125.0,
    ):
        # 1. Anatomical parameters (Manduca sexta).
        if species.lower() != "ms":
            raise ValueError(f"Unknown species: {species!r}")

        self.radEye = 0.0041 / 2.0  # Eye radius, meters
        self.d_phi_x = 0.96         # Horizontal interommatidial angle, degrees
        self.d_phi_y = 0.96         # Vertical interommatidial angle, degrees

        # FFF controls the optional output sampling rate, not spatial geometry.
        if not np.isfinite(flicker_fusion_fps) or flicker_fusion_fps <= 0:
            raise ValueError("flicker_fusion_fps must be positive and finite")
        self.fpsVision = float(flicker_fusion_fps)

        # 2. Angular visual field and user-defined gnomonic centers.
        self.n_phi_x = int(angle_x / self.d_phi_x)
        self.n_phi_y = int(angle_y / self.d_phi_y)
        if self.n_phi_x < 2 or self.n_phi_y < 2:
            raise ValueError("Both angular fields must contain at least 2 points")

        self.phi_x_0 = float(phi_x_0)
        self.phi_y_0 = float(phi_y_0)

        # 3. State for detecting events between consecutive *output* samples.
        self.visionMatrix_0 = None  # Previous temporally sampled visual field
        self.visionMatrix_1 = None  # Current temporally sampled visual field

        # 4. Mapping is lazily computed on the first input frame.
        self._map_key = None
        self._generateGnomonicProjection()

    def _generateAngularGrid(self, angle_0, d_angle, n_angles):
        """Return n equally spaced angles centered on angle_0 (degrees)."""
        return angle_0 + (np.arange(n_angles) - (n_angles - 1) / 2.0) * d_angle

    def _generateGnomonicProjection(self):
        """Use the *full* gnomonic equations and arbitrary center angles."""
        # Angles are converted from degrees to radians for NumPy trigonometry.
        self.phi_x_all = self._generateAngularGrid(
            self.phi_x_0, self.d_phi_x, self.n_phi_x
        )
        self.phi_y_all = self._generateAngularGrid(
            self.phi_y_0, self.d_phi_y, self.n_phi_y
        )
        self.Phi_x, self.Phi_y = np.meshgrid(self.phi_x_all, self.phi_y_all)

        phi_x = np.deg2rad(self.Phi_x)
        phi_y = np.deg2rad(self.Phi_y)
        phi_x_0 = np.deg2rad(self.phi_x_0)
        phi_y_0 = np.deg2rad(self.phi_y_0)

        # cos(theta): denominator of the complete gnomonic projection.
        cos_theta = (
            np.sin(phi_x) * np.sin(phi_x_0)
            + np.cos(phi_x) * np.cos(phi_x_0) * np.cos(phi_y_0 - phi_y)
        )

        # Gnomonic projection exists only within 90 degrees of its center.
        self.visible = cos_theta > 1e-8
        denominator = np.where(self.visible, cos_theta, np.nan)

        # Full projection equations (same sign and angle conventions as original).
        self.X = -self.radEye * (
            np.cos(phi_x) * np.sin(phi_x_0)
            - np.sin(phi_x) * np.cos(phi_x_0) * np.cos(phi_y_0 - phi_y)
        ) / denominator
        self.Y = self.radEye * (
            np.cos(phi_x_0) * np.sin(phi_y_0 - phi_y)
        ) / denominator

    def set_projection_center(self, phi_x_0, phi_y_0):
        """Move the projection center and invalidate geometry/temporal caches."""
        self.phi_x_0 = float(phi_x_0)
        self.phi_y_0 = float(phi_y_0)
        self._generateGnomonicProjection()
        self._map_key = None
        self.reset_temporal_state()

    def reset_temporal_state(self):
        """Clear the event history (e.g., before starting a new video)."""
        self.visionMatrix_0 = None
        self.visionMatrix_1 = None

    def choose_output_fps(self, input_fps, fps_mode):
        """Select input FPS or the model's flicker-fusion FPS."""
        if not np.isfinite(input_fps) or input_fps <= 0:
            raise ValueError("Input FPS must be positive and finite")
        if fps_mode == "input":
            return float(input_fps)
        if fps_mode == "flicker_fusion":
            return self.fpsVision
        raise ValueError("fps_mode must be 'input' or 'flicker_fusion'")

    def _build_pixel_map(self, image_shape, mpx, mpy):
        """Cache source-pixel -> facet assignment using polygon ray casting.

        Each facet's integration area is the polygon joining four neighboring
        projected ommatidial centers. For overlaps, the first cell wins.
        """
        h, w = image_shape
        ny, nx = self.X.shape
        cx, cy = np.floor(w / 2.0), np.floor(h / 2.0)

        # Project coordinates (meters) to image coordinates (pixels).
        px = self.X / mpx + cx
        py = -self.Y / mpy + cy

        # Negative values mark image pixels outside the valid visual field.
        lookup = np.full((h, w), -1, dtype=np.int32)

        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # Corners in the same order as the original quadrilateral code.
                corners = (
                    (iy, ix), (iy, ix + 1),
                    (iy + 1, ix + 1), (iy + 1, ix),
                )
                if not all(self.visible[r, c] for r, c in corners):
                    continue

                vx = np.array([px[r, c] for r, c in corners])
                vy = np.array([py[r, c] for r, c in corners])
                if not (np.isfinite(vx).all() and np.isfinite(vy).all()):
                    continue

                # Limit ray casting to the polygon's image-space bounding box.
                x0 = max(0, int(np.ceil(vx.min())))
                x1 = min(w, int(np.floor(vx.max())) + 1)
                y0 = max(0, int(np.ceil(vy.min())))
                y1 = min(h, int(np.floor(vy.max())) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue

                xx = np.arange(x0, x1, dtype=np.float64)[None, :]
                yy = np.arange(y0, y1, dtype=np.float64)[:, None]
                inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)

                # Vectorized version of the original four-edge ray-casting test.
                for k in range(4):
                    a = (k + 1) % 4
                    if vy[k] == vy[a]:
                        continue
                    x_intersection = (
                        (vx[a] - vx[k]) * (yy - vy[k]) / (vy[a] - vy[k])
                        + vx[k]
                    )
                    inside ^= (
                        ((vy[k] > yy) != (vy[a] > yy))
                        & (xx < x_intersection)
                    )

                patch = lookup[y0:y1, x0:x1]
                patch[inside & (patch == -1)] = iy * nx + ix

        # Cache only the assigned pixels and count contributors per facet.
        flat = lookup.ravel()
        self._src_idx = np.flatnonzero(flat >= 0)
        self._dst_idx = flat[self._src_idx]
        self._counts = np.bincount(self._dst_idx, minlength=ny * nx)
        self._map_key = (h, w, float(mpx), float(mpy))

    def project_frame(self, frameImage, mpx, mpy):
        """Return an unquantized (float32) visual field for one source frame.

        This method intentionally does NOT update temporal event state.
        Spatial projection is computed once per *source* frame, regardless
        of how many output frames are generated by temporal interpolation.
        """
        if frameImage.ndim != 2:
            raise ValueError("project_frame requires a 2D grayscale image")
        if not (np.isfinite(mpx) and np.isfinite(mpy) and mpx > 0 and mpy > 0):
            raise ValueError("mpx and mpy must be positive and finite")

        h, w = frameImage.shape
        key = (h, w, float(mpx), float(mpy))
        if self._map_key != key:
            self._build_pixel_map((h, w), mpx, mpy)

        ny, nx = self.X.shape
        sums = np.bincount(
            self._dst_idx,
            weights=frameImage.ravel()[self._src_idx],
            minlength=ny * nx,
        )
        mean_intensity = np.divide(
            sums,
            self._counts,
            out=np.zeros_like(sums),
            where=self._counts > 0,
        )
        return mean_intensity.reshape(ny, nx).astype(np.float32)

    def set_visual_field(self, visual_field):
        """Advance temporal history by exactly one *output* video frame."""
        vf = np.asarray(visual_field, dtype=np.float32)
        if vf.shape != self.X.shape:
            raise ValueError(f"Expected visual field of shape {self.X.shape}")
        self.visionMatrix_0 = self.visionMatrix_1
        self.visionMatrix_1 = vf
        return np.clip(vf, 0, 255).astype(np.uint8)

    def giveVisualFieldMatrix(self, frameImage, mpx, mpy):
        """Backward-compatible spatial projection + one temporal step.

        Use project_frame()/set_visual_field() for FPS conversion so temporal
        history advances at the *output* frame rate rather than source rate.
        """
        vf = self.project_frame(frameImage, mpx, mpy)
        return self.set_visual_field(vf)

    def giveEventPolarityMatrix(self, makeBipolar=True):
        """Find log-intensity events between consecutive OUTPUT samples."""
        if self.visionMatrix_0 is None or self.visionMatrix_1 is None:
            raise RuntimeError("At least two output samples are required")

        delta = np.log1p(self.visionMatrix_1) - np.log1p(self.visionMatrix_0)

        if makeBipolar:
            # 0 = negative event, 128 = no event, 255 = positive event.
            return np.where(
                delta > self.threshIntensity, 255,
                np.where(delta < -self.threshIntensity, 0, 128),
            ).astype(np.uint8)

        # Unipolar: any sufficiently large magnitude of change becomes 255.
        return (np.abs(delta) > self.threshIntensity).astype(np.uint8) * 255


    