# kreda-core-vision

Chalkboard lecture capture engine.
Watches a camera stream pointed at sliding chalkboards, detects when board content changes, and saves each distinct board state as an image.
The output is then passed to a pipeline that generates [Typst](https://github.com/typst/typst) lecture notes (see [kreda-orchestrator](https://github.com/kreda-ksi/kreda-orchestrator)).

KREDA consists of three main parts:
- core-vision (this repo, C++/OpenCV) captures footage,
- [kreda-orchestrator](https://github.com/kreda-ksi/kreda-orchestrator) (Python) dedupes, transcribes via VLM, and synthesizes notes,
- [kreda-datasets](https://github.com/kreda-ksi/kreda-datasets) holds test footage.
