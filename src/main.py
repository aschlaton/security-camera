from __future__ import annotations

import os
import enum
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
from loguru import logger

from src.config import AppConfig, load_config
from src.realtime import RealtimeSessionRunner
from src.recognition import FaceDatabase, FaceEmbedder, FaceMatcher, RecognitionTrigger
from src.vision import CameraError, CameraStream, FaceOverlay, draw_overlays_and_status


class AppState(enum.StrEnum):
    IDLE_CAMERA = "IDLE_CAMERA"
    STARTING_SESSION = "STARTING_SESSION"
    IN_CONVERSATION = "IN_CONVERSATION"
    ENDING_SESSION = "ENDING_SESSION"


def run() -> int:
    try:
        config = load_config()
    except RuntimeError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    logs_dir = Path("logs") / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        logs_dir / "replies.log",
        filter=lambda r: "User said" in r["message"] or r["message"].startswith("Response"),
        level="INFO",
    )

    if not config.recognition.people_dir.exists():
        config.recognition.people_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Created {config.recognition.people_dir}. "
            "Add person profiles under people/<name>/pictures and optional prompt file.",
            file=sys.stderr,
        )

    embedder = FaceEmbedder()
    people_dir = config.recognition.people_dir.resolve()
    instructions_by_person = load_person_instructions(people_dir)
    face_db = FaceDatabase.load_or_build(
        people_dir=config.recognition.people_dir,
        cache_path=config.recognition.cache_path,
        embedder=embedder,
    )
    if face_db.is_empty:
        print(
            "No valid face embeddings found in people/. "
            "Recognition will run, but no one can match until faces are added.",
            file=sys.stderr,
        )
    else:
        for name in set(face_db.names):
            prompt = (instructions_by_person.get(str(name)) or "").strip()
            if not prompt:
                print(
                    f"Missing or empty people/{name}/prompt.md. Every person must have instructions.",
                    file=sys.stderr,
                )
                return 1

    matcher = FaceMatcher(
        min(
            config.recognition.initial_similarity_threshold,
            config.recognition.session_similarity_threshold,
        )
    )
    trigger = RecognitionTrigger(
        initial_similarity_threshold=config.recognition.initial_similarity_threshold,
        initial_required_frames=config.recognition.initial_required_frames,
        session_similarity_threshold=config.recognition.session_similarity_threshold,
        session_missed_frames=config.recognition.session_missed_frames,
        cooldown_seconds=config.recognition.cooldown_seconds,
    )

    camera = CameraStream(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height,
    )
    try:
        camera.open()
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    session_runner = RealtimeSessionRunner(config.realtime, config.audio)
    state = AppState.IDLE_CAMERA
    active_person: str | None = None
    frame_idx = 0
    overlays: list[FaceOverlay] = []
    should_quit = False
    recog_max_w = config.camera.recognition_max_width
    recog_every = config.camera.recognition_every_n_frames
    window_name = config.camera.window_name
    recog_future = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    with ThreadPoolExecutor(max_workers=1) as recog_executor:
        try:
            while not should_quit:
                frame = camera.read()
                frame_idx += 1
                run_recognition = frame_idx % recog_every == 0

                if recog_future is not None and recog_future.done():
                    try:
                        overlays, maybe_trigger_name, maybe_trigger_similarity = recog_future.result()
                        if state == AppState.IDLE_CAMERA:
                            fired_name = trigger.update_idle(maybe_trigger_name, maybe_trigger_similarity)
                            if fired_name is not None:
                                logger.info("Identified: {}", fired_name)
                                active_person = fired_name
                                trigger.reset_session_tracking()
                                state = AppState.STARTING_SESSION
                        elif state == AppState.IN_CONVERSATION:
                            if trigger.update_session(active_person, maybe_trigger_name, maybe_trigger_similarity):
                                state = AppState.ENDING_SESSION
                    except Exception:
                        pass
                    recog_future = None

                if run_recognition and recog_future is None and state in (AppState.IDLE_CAMERA, AppState.IN_CONVERSATION):
                    recog_future = recog_executor.submit(
                        process_frame, frame.copy(), embedder, face_db, matcher, recog_max_w
                    )

                if state == AppState.STARTING_SESSION:
                    if config.realtime.disabled:
                        if active_person is not None:
                            trigger.start_cooldown(active_person)
                        active_person = None
                        state = AppState.IDLE_CAMERA
                    else:
                        try:
                            person_name = active_person or "unknown"
                            logger.info("Voice session started for {}", person_name)
                            person_prompt = instructions_by_person[person_name]
                            session_runner.start(person_name, person_prompt)
                            state = AppState.IN_CONVERSATION
                        except Exception as exc:
                            print(f"Failed to start realtime session: {exc}", file=sys.stderr)
                            state = AppState.IDLE_CAMERA
                            active_person = None

                if state == AppState.IN_CONVERSATION:
                    if not session_runner.is_running:
                        state = AppState.ENDING_SESSION

                if state == AppState.ENDING_SESSION:
                    logger.info("Voice session ended for {}", active_person or "unknown")
                    session_runner.stop()
                    if active_person is not None:
                        trigger.start_cooldown(active_person)
                    trigger.reset_session_tracking()
                    active_person = None
                    state = AppState.IDLE_CAMERA

                status_text = build_status_text(state, active_person, config)
                output = draw_overlays_and_status(frame, overlays, status_text)
                cv2.imshow(window_name, output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    if state == AppState.IN_CONVERSATION:
                        state = AppState.ENDING_SESSION
                    else:
                        should_quit = True
        except KeyboardInterrupt:
            should_quit = True
        finally:
            session_runner.stop()
            camera.close()
            cv2.destroyAllWindows()
    return 0


def build_status_text(state: AppState, active_person: str | None, config: AppConfig) -> str:
    if state == AppState.IDLE_CAMERA:
        return (
            f"{state.value} | start>={config.recognition.initial_similarity_threshold:.2f}"
            f" x{config.recognition.initial_required_frames} | q:quit"
        )
    if state == AppState.STARTING_SESSION:
        return f"{state.value} | person={active_person or 'unknown'}"
    if state == AppState.IN_CONVERSATION:
        return f"{state.value} | person={active_person or 'unknown'} | q:end conversation"
    return f"{state.value}"


def load_person_instructions(people_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for person_path in people_dir.iterdir():
        if not person_path.is_dir():
            continue
        prompt_path = person_path / "prompt.md"
        if not prompt_path.exists():
            continue
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            out[person_path.name] = text
    return out


def process_frame(
    frame: cv2.typing.MatLike,
    embedder: FaceEmbedder,
    face_db: FaceDatabase,
    matcher: FaceMatcher,
    recognition_max_width: int = 0,
) -> tuple[list[FaceOverlay], str | None, float | None]:
    h, w = frame.shape[:2]
    if recognition_max_width > 0 and w > recognition_max_width:
        scale = recognition_max_width / w
        small_w = recognition_max_width
        small_h = int(h * scale)
        recog_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        detections = embedder.detect_faces(recog_frame)
        sx, sy = w / small_w, h / small_h

        def scale_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            x1, y1, x2, y2 = bbox
            return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
    else:
        detections = embedder.detect_faces(frame)

        def scale_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            return bbox

    if not detections:
        return [], None, None

    embeddings = [d.embedding for d in detections]
    results = matcher.match_batch(embeddings, face_db)

    overlays = []
    best_name = None
    best_score = -1.0
    for detection, result in zip(detections, results, strict=True):
        label = result.name if result.is_match and result.name else "Unknown"
        overlays.append(
            FaceOverlay(
                bbox=scale_bbox(detection.bbox),
                label=label,
                similarity=result.similarity if result.name else None,
                is_match=result.is_match,
            )
        )
        if result.is_match and result.name and result.similarity > best_score:
            best_score = result.similarity
            best_name = result.name
    return overlays, best_name, best_score if best_name is not None else None


if __name__ == "__main__":
    raise SystemExit(run())
