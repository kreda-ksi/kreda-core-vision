# kreda-core-vision

Chalkboard lecture capture engine.
Watches a camera stream pointed at sliding chalkboards, detects when board content changes, and saves each distinct board state as an image.
The output is then passed to a pipeline that generates [Typst](https://github.com/typst/typst) lecture notes (see [kreda-orchestrator](https://github.com/kreda-ksi/kreda-orchestrator)).

KREDA consists of three main parts:
- **core-vision** (this repo, C++/OpenCV) captures footage,
- **[kreda-orchestrator](https://github.com/kreda-ksi/kreda-orchestrator)** (Python) dedupes, transcribes via VLM, and synthesizes notes,
- **[kreda-datasets](https://github.com/kreda-ksi/kreda-datasets)** holds test footage.

## How it works

One capture thread reads the stream (RTSP or file), one processing thread consumes frames through a depth-1 queue (latest wins live, lossless for files).
Each frame is dewarped per board column at two resolutions, a high-res content path for saved images and a motion path for detection.
Motion is measured against an exponentially-smoothed reference, a state machine decides saves:

- **periodic** - every `n` seconds if chalk content changed,
- **still** - after a motion episode ends,
- **slide** - board movement detected (full-width motion), saves a pre-slide frame from a ring buffer,
- **final** - end-of-run flush.

A body-masking filter (connected-component analysis) distinguishes lecturer-sized change from chalk-stroke sized change, so saves are gated on chalk, not motion.
Occluded frames are also saved, to ensure no data loss over precision. 
The orchestrator's dedup picks clean frames.

## Setup and calibration

The default setup assumes a ~1080p input from the camera.
Preferably, the camera is hung from the ceiling without anything occluding the view from close (it triggers slide detection).

To set up the engine, first run in a new room opens a calibration window.
Click 4 corners per board column (top-left, top-right, bottom-right, bottom-left).
Clicks are stored (default: `calibration.xml`) with a reference frame (default: `calibration_ref.png`), warp matrices are derived at load.
Subsequent startups self-correct small camera drift (ORB + RANSAC homography against the reference) and refuse loudly on large changes (in that case, re-run with `-rc/--recalibrate`).
