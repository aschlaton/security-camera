from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import FaceEmbedder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT_FILENAMES = ("prompt.txt", "prompt.md", "instructions.txt")


@dataclass(slots=True)
class FaceDBEntry:
    name: str
    embedding: np.ndarray
    source: Path


class FaceDatabase:
    def __init__(self, entries: list[FaceDBEntry], person_prompts: dict[str, str] | None = None) -> None:
        self.entries = entries
        self.person_prompts = person_prompts or {}
        self.names = np.asarray([entry.name for entry in entries], dtype=np.str_)
        self.embeddings = np.asarray([entry.embedding for entry in entries], dtype=np.float32)

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def prompt_for_person(self, name: str) -> str | None:
        return self.person_prompts.get(name)

    @classmethod
    def load_or_build(
        cls,
        people_dir: Path,
        cache_path: Path,
        embedder: FaceEmbedder,
    ) -> FaceDatabase:
        people_dir.mkdir(parents=True, exist_ok=True)
        image_paths, prompts = _collect_face_assets(people_dir)
        if image_paths and cache_path.exists() and _cache_is_fresh(cache_path, image_paths):
            loaded = _load_cache(cache_path)
            if loaded is not None:
                return cls(loaded, prompts)

        entries: list[FaceDBEntry] = []
        for image_path in image_paths:
            embedding = embedder.embedding_from_image_path(str(image_path))
            if embedding is None:
                continue
            entries.append(
                FaceDBEntry(
                    name=_name_for_image_path(people_dir, image_path),
                    embedding=embedding,
                    source=image_path,
                )
            )
        _save_cache(cache_path, entries)
        return cls(entries, prompts)


def _cache_is_fresh(cache_path: Path, image_paths: list[Path]) -> bool:
    cache_mtime = cache_path.stat().st_mtime
    latest_face_mtime = max(path.stat().st_mtime for path in image_paths)
    return cache_mtime >= latest_face_mtime


def _load_cache(cache_path: Path) -> list[FaceDBEntry] | None:
    try:
        data = np.load(cache_path, allow_pickle=False)
        names = data["names"]
        embeddings = data["embeddings"]
        sources = data["sources"]
        entries: list[FaceDBEntry] = []
        for name, embedding, source in zip(names, embeddings, sources, strict=True):
            entries.append(
                FaceDBEntry(
                    name=str(name),
                    embedding=np.asarray(embedding, dtype=np.float32),
                    source=Path(str(source)),
                )
            )
        return entries
    except Exception:
        return None


def _save_cache(cache_path: Path, entries: list[FaceDBEntry]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        np.savez(
            cache_path,
            names=np.array([], dtype=np.str_),
            embeddings=np.empty((0, 0), dtype=np.float32),
            sources=np.array([], dtype=np.str_),
        )
        return
    names = np.asarray([entry.name for entry in entries], dtype=np.str_)
    embeddings = np.asarray([entry.embedding for entry in entries], dtype=np.float32)
    sources = np.asarray([str(entry.source) for entry in entries], dtype=np.str_)
    np.savez(cache_path, names=names, embeddings=embeddings, sources=sources)


def _collect_face_assets(people_dir: Path) -> tuple[list[Path], dict[str, str]]:
    image_paths: list[Path] = []
    person_prompts: dict[str, str] = {}
    for person_path in people_dir.iterdir():
        if person_path.is_file() and person_path.suffix.lower() in IMAGE_EXTS:
            image_paths.append(person_path)
            continue
        if not person_path.is_dir():
            continue
        pictures_dir = person_path / "pictures"
        if pictures_dir.exists() and pictures_dir.is_dir():
            image_paths.extend(
                p
                for p in pictures_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        image_paths.extend(
            p
            for p in person_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        for filename in PROMPT_FILENAMES:
            prompt_path = person_path / filename
            if not prompt_path.exists():
                continue
            text = prompt_path.read_text(encoding="utf-8").strip()
            if text:
                person_prompts[person_path.name] = text
            break
    return sorted(image_paths), person_prompts


def _name_for_image_path(people_dir: Path, image_path: Path) -> str:
    parent = image_path.parent
    if parent == people_dir:
        return image_path.stem
    if parent.name == "pictures" and parent.parent != people_dir:
        return parent.parent.name
    if parent != people_dir:
        return parent.name
    return image_path.stem
