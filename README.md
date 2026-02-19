# Security Camera + Face Triggered Realtime Voice

Local Python app that:
- runs webcam face detection/recognition on CPU,
- matches against profiles in `people/`,
- starts a GPT Realtime voice session when a known person is recognized,
- returns to camera mode after the conversation ends.

## Requirements

- Python 3.13
- `uv`
- Webcam + microphone + speakers
- `OPENAI_API_KEY` set in your environment

## Setup (uv only)

From the project root:

```bash
uv init --python 3.13
uv add numpy opencv-python insightface onnxruntime sounddevice openai
```

If the project is already initialized (this repo already is), only run:

```bash
uv sync
```

## Prepare People Profiles (Modular)

Recommended profile structure:

```text
people/
  alice/
    pictures/
      1.jpg
      2.jpg
    prompt.txt
  bob/
    pictures/
      face.png
    instructions.txt
```

Rules:
- each person is a folder under `people/`
- add one or more images under `people/<person>/pictures/`
- optional per-person prompt file in `people/<person>/`:
  - `prompt.txt`
  - `prompt.md`
  - `instructions.txt`
- if a person has only one image, that also works as `people/<person>/single.jpg`

When recognized, the person prompt (if present) is appended to the base realtime instructions.

First run builds `.cache/face_db.npz`; cache refreshes when face images change.

## Run

```bash
# copy template once and set your key
cp .env.example .env
# load env vars from .env (or export OPENAI_API_KEY directly)
set -a; source .env; set +a
uv run python -m src.main
```


Controls:
- In camera mode: press `q` to quit.
- In conversation mode: press `q` to end the conversation and return to camera mode.

## Architecture

```
src/
  main.py
  config.py
  vision/
    camera.py
    overlay.py
    recorder.py
  recognition/
    face_db.py
    embedder.py
    matcher.py
    trigger.py
  realtime/
    client.py
    audio_io.py
    session.py
    prompts.py
```

Notes:
- `main.py` owns the state machine and all cross-module orchestration.
- `vision/`, `recognition/`, and `realtime/` are separated with no circular imports.
- CPU execution is default (`InsightFace` via `onnxruntime` CPU provider).
