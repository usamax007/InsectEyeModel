# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:22:45 2024

@author: Usama
"""


"""Convert an input video to compound-eye visual and event-polarity videos.

FPS_MODE = "input" uses the video's original FPS.
FPS_MODE = "flicker_fusion" samples at the model's flicker-fusion FPS.

Spatial projection uses the full user-centered gnomonic equations. When the
output FPS differs, input visual fields are temporally linearly interpolated.
This does NOT recover visual information absent from the source recording.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from compoundEyeModel import CompoundEyeModel


# ============================================================================
# 1. USER SETTINGS
# ============================================================================

# Source-video rate or the moth's flicker-fusion rate (Hz).
FPS_MODE = "input"       # "input" or "flicker_fusion"
FLICKER_FUSION_FPS = 125.0       # Sampling FPS used by flicker_fusion mode

# Output dimensions (width, height) in video pixels.
OUTPUT_SIZE = (186, 186)

# Angular field of view (degrees) and projection-center coordinates (degrees).
SPECIES = "Ms"
ANGLE_X, ANGLE_Y = 180.0, 180.0
PHI_X_0, PHI_Y_0 = 0.0, 0.0

# Physical scene sampling pitch (meters per source-video pixel).
MPX = 8.26e-5 / 12.0
MPY = 8.26e-5 / 12.0

# True: negative / neutral / positive event levels = 0 / 128 / 255.
# False: no event / event = 0 / 255.
MAKE_BIPOLAR = True

# Stop after this many OUTPUT frames, or None to process the entire video.
MAX_OUTPUT_FRAMES = None

# Print an instantaneous processing-rate estimate after this many output frames.
PRINT_INTERVAL = 30

# Project layout: this script lives in a directory under the project root.
# In a Jupyter notebook, replace this with Path.cwd().parent as needed.
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "input" / "flower_scn.mp4"
OUTPUT_DIR = PROJECT_DIR / "output"
VISION_PATH = OUTPUT_DIR / "output_vis.mp4"
EVENT_PATH = OUTPUT_DIR / "output_epm.mp4"


# ============================================================================
# 2. SOURCE FRAME -> SPATIAL VISUAL FIELD
# ============================================================================

def read_projected_frame(capture, model):
    """Decode one source video frame and integrate its intensity across facets.

    Returns None at end of file. This function is called once per SOURCE frame,
    not once per emitted output frame, which avoids redundant spatial mapping.
    """
    ok, frame_bgr = capture.read()
    if not ok:
        return None

    grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return model.project_frame(grayscale, MPX, MPY)


# ============================================================================
# 3. STREAMING TEMPORAL RESAMPLING
# ============================================================================

def iter_output_visual_fields(capture, model, source_fps, output_fps):
    """Yield each temporally sampled visual field without buffering the video.

    Source frame i represents time i/source_fps. Output frame j represents
    time j/output_fps. Fields at intermediate times use LINEAR interpolation.

    Each source frame is projected exactly once. Between the last observed
    source frame and the video end, hold the last field for one source-frame
    interval to preserve the source video's nominal playback duration.
    """
    prev_field = read_projected_frame(capture, model)
    if prev_field is None:
        return  # Empty video

    source_index = 0
    output_index = 0

    while True:
        # Look ahead by one source frame to interpolate within this interval.
        next_field = read_projected_frame(capture, model)

        # Emit all output time points in [source_index, source_index + 1),
        # expressed in units of source-frame intervals.
        while output_index * source_fps < (source_index + 1) * output_fps:
            source_position = output_index * source_fps / output_fps
            alpha = np.clip(source_position - source_index, 0.0, 1.0)

            if next_field is None or alpha == 0.0:
                # Exact source timestamp, or final source-frame hold.
                output_field = prev_field
            else:
                # Interpolate the small projected fields, not full-size frames.
                output_field = (
                    (1.0 - alpha) * prev_field + alpha * next_field
                )

            yield output_field
            output_index += 1

        if next_field is None:
            break  # End of video, final interval has been emitted

        prev_field = next_field
        source_index += 1


# ============================================================================
# 4. OUTPUT FRAME ENCODING
# ============================================================================

def resize_visual_field(image):
    """Resize smooth grayscale intensity for the output video."""
    h, w = image.shape
    interpolation = (
        cv2.INTER_AREA
        if (OUTPUT_SIZE[0] < w or OUTPUT_SIZE[1] < h)
        else cv2.INTER_LINEAR
    )
    return cv2.resize(image, OUTPUT_SIZE, interpolation=interpolation)


def resize_event_field(image):
    """Resize discrete event levels without introducing intermediate values."""
    return cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)


# ============================================================================
# 5. MAIN VIDEO PROCESSING
# ============================================================================

def main():
    """Run model, generate aligned videos, and release resources reliably."""

    if PRINT_INTERVAL <= 0:
        raise ValueError("PRINT_INTERVAL must be positive")
    if MAX_OUTPUT_FRAMES is not None and MAX_OUTPUT_FRAMES < 0:
        raise ValueError("MAX_OUTPUT_FRAMES must be None or nonnegative")

    # A. Validate paths, open source file and read nominal source metadata.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(INPUT_PATH))
    if not capture.isOpened():
        raise IOError(f"Could not open input video: {INPUT_PATH}")

    vision_writer = None
    event_writer = None

    try:
        input_fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(input_fps) or input_fps <= 0:
            raise ValueError("Source FPS unavailable: cannot resample accurately")

        # B. Initialize geometry, then choose the actual output sampling rate.
        model = CompoundEyeModel(
            species=SPECIES,
            angle_x=ANGLE_X,
            angle_y=ANGLE_Y,
            phi_x_0=PHI_X_0,
            phi_y_0=PHI_Y_0,
            flicker_fusion_fps=FLICKER_FUSION_FPS,
        )
        output_fps = model.choose_output_fps(input_fps, FPS_MODE)
        model.reset_temporal_state()

        print(f"Input:  {INPUT_PATH} ({width} x {height}, {input_fps:g} FPS)")
        print(f"Mode:   {FPS_MODE}")
        print(f"Output: {output_fps:g} FPS")
        print(f"Source frame count (reported): {n_source_frames}")
        if n_source_frames > 0:
            duration = n_source_frames / input_fps
            expected_frames = int(np.ceil(duration * output_fps - 1e-10))
            print(f"Nominal duration: {duration:.3f} s")
            print(f"Expected output frames: ~{expected_frames}")

        # C. Open two grayscale MP4 videos at the SELECTED output FPS.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vision_writer = cv2.VideoWriter(
            str(VISION_PATH), fourcc, output_fps, OUTPUT_SIZE, isColor=False
        )
        event_writer = cv2.VideoWriter(
            str(EVENT_PATH), fourcc, output_fps, OUTPUT_SIZE, isColor=False
        )
        if not vision_writer.isOpened() or not event_writer.isOpened():
            raise IOError("Could not open one or both MP4 output writers")

        # D. Emit one visual frame and one event frame per output time sample.
        #    Events always compare consecutive frames at output_fps intervals.
        start = perf_counter()
        checkpoint = start
        output_count = 0

        for visual_field in iter_output_visual_fields(
            capture, model, input_fps, output_fps
        ):
            if MAX_OUTPUT_FRAMES is not None and output_count >= MAX_OUTPUT_FRAMES:
                break

            # Advance temporal memory only after temporal resampling.
            visual_u8 = model.set_visual_field(visual_field)
            vision_writer.write(resize_visual_field(visual_u8))

            if model.visionMatrix_0 is None:
                # Neutral event frame for t = 0; keeps output videos aligned.
                neutral = 128 if MAKE_BIPOLAR else 0
                event_u8 = np.full(visual_u8.shape, neutral, dtype=np.uint8)
            else:
                event_u8 = model.giveEventPolarityMatrix(MAKE_BIPOLAR)
            event_writer.write(resize_event_field(event_u8))

            output_count += 1
            if output_count % PRINT_INTERVAL == 0:
                now = perf_counter()
                recent_fps = PRINT_INTERVAL / (now - checkpoint)
                print(f"Output frames: {output_count} | Processing: {recent_fps:.1f} FPS")
                checkpoint = now

        # E. Display total wall-clock processing throughput and output paths.
        elapsed = perf_counter() - start
        speed = output_count / elapsed if elapsed > 0 else 0.0
        print(f"\nFinished: {output_count} output frames in {elapsed:.2f} s")
        print(f"Average processing speed: {speed:.1f} output frames/s")
        print(f"Video FPS: {output_fps:g} | Duration: {output_count / output_fps:.3f} s")
        print(f"Visual field:   {VISION_PATH}")
        print(f"Event polarity: {EVENT_PATH}")

    finally:
        # F. Release files even if video decoding or projection raises an error.
        capture.release()
        if vision_writer is not None:
            vision_writer.release()
        if event_writer is not None:
            event_writer.release()


if __name__ == "__main__":
    main()

