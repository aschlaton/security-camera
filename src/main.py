from __future__ import annotations

import enum
import sys
import time

import cv2

from src.config import AppConfig, load_config
from src.realtime import RealtimeSessionRunner
from src.recognition import FaceDatabase, FaceEmbedder, FaceMatcher, RecognitionTrigger
from src.vision import CameraError, CameraStream, FaceOverlay, draw_overlays, draw_status


class AppState(enum.StrEnum):
    IDLE_CAMERA = "IDLE_CAMERA"
    STARTING_SESSION = "STARTING_SESSION"
    IN_CONVERSATION = "IN_CONVERSATION"
    ENDING_SESSION = "ENDING_SESSION"


def run() -> int:
    config = load_config()

    if not config.recognition.people_dir.exists():
        config.recognition.people_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Created {config.recognition.people_dir}. "
            "Add person profiles under people/<name>/pictures and optional prompt file.",
            file=sys.stderr,
        )

    embedder = FaceEmbedder()
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

    matcher = FaceMatcher(config.recognition.similarity_threshold)
    trigger = RecognitionTrigger(
        required_consecutive_matches=config.recognition.required_consecutive_matches,
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
    last_seen_active_person = time.monotonic()
    frame_idx = 0
    overlays: list[FaceOverlay] = []
    should_quit = False
    status_text = "IDLE_CAMERA (press q to quit)"

    cv2.namedWindow(config.camera.window_name, cv2.WINDOW_NORMAL)
    try:
        while not should_quit:
            frame = camera.read()
            frame_idx += 1
            run_recognition = frame_idx % config.camera.recognition_every_n_frames == 0

            if state == AppState.IDLE_CAMERA and run_recognition:
                overlays, maybe_trigger_name = process_frame(frame, embedder, face_db, matcher)
                fired_name = trigger.update(maybe_trigger_name)
                if fired_name is not None:
                    active_person = fired_name
                    state = AppState.STARTING_SESSION
            elif state == AppState.IN_CONVERSATION and run_recognition:
                overlays, maybe_trigger_name = process_frame(frame, embedder, face_db, matcher)
                if active_person is not None and maybe_trigger_name == active_person:
                    last_seen_active_person = time.monotonic()
            elif run_recognition:
                overlays, _ = process_frame(frame, embedder, face_db, matcher)

            if state == AppState.STARTING_SESSION:
                if not config.realtime.openai_api_key:
                    print("OPENAI_API_KEY is missing; cannot start conversation.", file=sys.stderr)
                    state = AppState.IDLE_CAMERA
                    active_person = None
                else:
                    try:
                        person_name = active_person or "Guest"
                        session_runner.start(
                            person_name,
                            face_db.prompt_for_person(person_name),
                        )
                        last_seen_active_person = time.monotonic()
                        state = AppState.IN_CONVERSATION
                    except Exception as exc:
                        print(f"Failed to start realtime session: {exc}", file=sys.stderr)
                        state = AppState.IDLE_CAMERA
                        active_person = None

            if state == AppState.IN_CONVERSATION:
                if not session_runner.is_running:
                    state = AppState.ENDING_SESSION
                elif active_person is not None:
                    elapsed_absent = time.monotonic() - last_seen_active_person
                    if elapsed_absent > config.recognition.person_absent_timeout_seconds:
                        state = AppState.ENDING_SESSION

            if state == AppState.ENDING_SESSION:
                session_runner.stop()
                if active_person is not None:
                    trigger.start_cooldown(active_person)
                active_person = None
                state = AppState.IDLE_CAMERA

            status_text = build_status_text(state, active_person, config)
            output = draw_status(draw_overlays(frame, overlays), status_text)
            cv2.imshow(config.camera.window_name, output)

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
        return f"{state.value} | threshold={config.recognition.similarity_threshold:.2f} | q:quit"
    if state == AppState.STARTING_SESSION:
        return f"{state.value} | person={active_person or 'unknown'}"
    if state == AppState.IN_CONVERSATION:
        return f"{state.value} | person={active_person or 'unknown'} | q:end conversation"
    return f"{state.value}"


def process_frame(
    frame,
    embedder: FaceEmbedder,
    face_db: FaceDatabase,
    matcher: FaceMatcher,
) -> tuple[list[FaceOverlay], str | None]:
    detections = embedder.detect_faces(frame)
    overlays: list[FaceOverlay] = []
    best_name: str | None = None
    best_score = -1.0
    for detection in detections:
        result = matcher.match(detection.embedding, face_db)
        label = result.name if result.is_match and result.name else "Unknown"
        overlays.append(
            FaceOverlay(
                bbox=detection.bbox,
                label=label,
                similarity=result.similarity if result.name else None,
                is_match=result.is_match,
            )
        )
        if result.is_match and result.name and result.similarity > best_score:
            best_score = result.similarity
            best_name = result.name
    return overlays, best_name


if __name__ == "__main__":
    raise SystemExit(run())
