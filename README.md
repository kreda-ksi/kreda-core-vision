# kreda-core-vision

Chalkboard lecture capture engine.
Watches a camera stream pointed at sliding chalkboards, detects when board content changes, and saves each distinct board state as an image.
The output is then passed to a pipeline that generates [Typst](https://github.com/typst/typst) lecture notes (see [kreda-orchestrator](https://github.com/kreda-ksi/kreda-orchestrator)).

KREDA consists of three main parts:
- **kreda-core-vision** (this repo, C++/OpenCV) captures footage,
- **[kreda-orchestrator](https://github.com/kreda-ksi/kreda-orchestrator)** (Python) dedupes, transcribes via VLM, and synthesizes notes,
- **[kreda-datasets](https://github.com/kreda-ksi/kreda-datasets)** holds test footage.

## Dependencies

- [C++20](https://gcc.gnu.org/projects/cxx-status.html#cxx20)
- [OpenCV 4.6.0](https://github.com/opencv/opencv)
- [CMake 3.20](https://github.com/kitware/cmake)
- [doctest](https://github.com/doctest/doctest)

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

## Preview

### Raw footage

![Before preview](docs/before.gif)

### GUI preview

![After preview](docs/after.gif)

## Setup and calibration

The default setup assumes a ~1080p input from the camera.
Preferably, the camera is hung from the ceiling without anything occluding the view from close (it triggers slide detection).

To set up the engine, first run in a new room opens a calibration window.
Click 4 corners per board column (top-left, top-right, bottom-right, bottom-left).
Clicks are stored (default: `calibration.xml`) with a reference frame (default: `calibration_ref.png`), warp matrices are derived at load.
Subsequent startups self-correct small camera drift (ORB + RANSAC homography against the reference) and refuse loudly on large changes (in that case, re-run with `-rc/--recalibrate`).

## Usage

```bash
./kreda_vision_engine [OPTIONS] <rtsp-url | video-file>
```

| Flag                 | Abbreviation | Effect                                   |
| :---                 | :---:        | :---:                                    |
| `--headless`         | `-h`         | no GUI, requires existing calibration    |
| `--no-raw`           | `-nr`        | suppress raw debug windows               |
| `--no-log`           | `-nl`        | disable CSV/sidecar telemetry            |
| `--calib <path>`     | `-c <path>`  | override calibration file path           |
| `--out <path>`       | `-o <path>`  | override output directory path           |
| `--log-file <path>`  | `-lf <path>` | override CSV file path                   |
| `--ref-file <path>`  | `-rf <path>` | override calibration reference file path |
| `--grid-file <path>` | `-gf <path>` | override motion grid JSON file path      |
| `--recalibrate`      | `-rc`        | force manual calibration                 |
| `--duration <min>`   | `-d <min>`   | auto-stop after `min` minutes            |

File input replays deterministically. Same file with the same options end up with the same saved PNGs, same CSV (modulo `RUN_START`).

For headless deployment `SIGTERM`/`SIGINT` trigger clean shutdown (with final flush).

## Output

All artifacts land in `staging/`, under `run_{time_of_run}` directory.
PNG frames are named `track_{col_id}_{stream_ms}_{reason}.png`.
It also stores a CSV event log, as well as a JSON sidecar with recency-weighted motion-occupancy grids per save.

### CSV log structure

Columns represent as follows:

- `t_ms` - relative time of the row data,
- `track` - column ID of the row data,
- `type`:
    - `EVENT` on image save, slide, start/end of file,
    - `FRAME` otherwise,
- `changed_or_what`:
    - if `type == EVENT` stores event description, 
    - if `type == FRAME` stores changed pixels relative to last saved frame,
- `detail` - stores:
    - epoch time on `RUN_START`,
    - chalk pixels on `SAVE_{reason}_{raw_pxs}`/`SKIP_{reason}_{raw_pxs}`,
    - changed pixels on `SLIDE_DETECTED`,
    - wall clock duration on `FILE_EOF`.
    - `detects movement|is chalkboard sliding|is in slide recovery` (e.g. `100` means movement, no slide) if `type == FRAME`.

### JSON sidecar structure

The sidecar consists of two objects - a header and frames.

Header stores only the grid dimensions (calculated in [`config.hpp`](src/config.hpp)).

The `frames` object stores the state of the motion grid on each frame where `type == EVENT` (from [CSV log structure](#csv-log-structure)).

Each frame stores: 

- `timestamp_ms` - its timestamp (time relative to run start),
- `event_type` - as the name suggests - the type of the event that occured on that frame,
- `occupancy_grid` - the actual grid data, where each cell is a value from 0-9 (quantized from 0-255).

## Development

```bash
cmake -B build && cmake --build build # build
ctest --test-dir build -j             # unit tests
run-clang-tidy -p build               # lint
find src -name '*.cpp' -o -name '*.hpp' \
    | xargs clang-format --dry-run -Werror # format
```

CI enforces format (via `clang-format`), lint (via `clang-tidy`), and build+test on Ubuntu.

## Design decisions

- **No ML-based human detection** - given the constraints, results obtained by a motion-based model end up being on par/better than people detection with a HOG algorithm and linear SVM.
- **No background subtraction** - MOG2-like models get poisoned by stationary lecturers (ghosts, recovery deadlocks). Motion is frame differencing against an EMA reference, presence is never inferred.
- **Recall-first** - the LLM donwstream tolerates occluded and duplicate frames, but it can't recover a missed board state.
- **Two time bases** - all telemetry and filenames use stream time (`POS_MSEC` for files, monotonic-since-start for live). Wall clock appears only in the `RUN_START` bridge event.
- **Constants are dependent** - thresholds are area-fractions, grid size is motion resolution-based.
