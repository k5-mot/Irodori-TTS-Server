from __future__ import annotations

import argparse
import fnmatch
import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from .config import get_settings
from .voices import VOICE_EXTENSIONS, VoiceRegistry

logger = logging.getLogger(__name__)
DEFAULT_VOICE_PATTERNS = ("samples/clone_ref*.wav",)
ALL_SAMPLE_PATTERNS = ("samples/*.wav",)


@dataclass(frozen=True)
class FetchedVoice:
    voice_id: str
    source: str
    path: Path
    existed: bool


def fetch_voice_samples(
    *,
    repo_id: str,
    voices_dir: Path,
    patterns: Iterable[str] = DEFAULT_VOICE_PATTERNS,
    revision: str | None = None,
    replace: bool = False,
    dry_run: bool = False,
) -> list[FetchedVoice]:
    files = list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
    matched = _matching_voice_files(files, patterns)
    if not matched:
        pattern_text = ", ".join(patterns)
        raise ValueError(f"No voice sample files matched {pattern_text!r} in {repo_id!r}.")

    voices_dir = voices_dir.expanduser()
    if not dry_run:
        voices_dir.mkdir(parents=True, exist_ok=True)

    fetched = []
    for repo_path in matched:
        filename = Path(repo_path).name
        voice_id = Path(filename).stem
        VoiceRegistry.validate_voice_id(voice_id)
        target = voices_dir / filename
        existed = target.exists()
        if dry_run:
            fetched.append(FetchedVoice(voice_id, repo_path, target, existed))
            continue
        if existed and not replace:
            fetched.append(FetchedVoice(voice_id, repo_path, target, existed=True))
            continue
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            revision=revision,
            repo_type="model",
        )
        shutil.copyfile(cached_path, target)
        fetched.append(FetchedVoice(voice_id, repo_path, target, existed=existed))
    return fetched


def _matching_voice_files(files: Iterable[str], patterns: Iterable[str]) -> list[str]:
    pattern_list = tuple(patterns)
    out = []
    for file in files:
        path = Path(file)
        if path.suffix.lower() not in VOICE_EXTENSIONS:
            continue
        if any(fnmatch.fnmatch(file, pattern) for pattern in pattern_list):
            out.append(file)
    return sorted(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Fetch Irodori-TTS sample reference voices into the voices directory."
    )
    parser.add_argument(
        "--repo",
        default=settings.voice_samples_repo or settings.hf_checkpoint,
        help="Hugging Face model repo that contains sample voices.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--voices-dir", type=Path, default=settings.voices_dir)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Repository file glob to download. Can be repeated.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Download every WAV under samples/ instead of only clone_ref*.wav.",
    )
    parser.add_argument("--replace", action="store_true", help="Overwrite existing voice files.")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading.")
    args = parser.parse_args()

    patterns = tuple(args.patterns or (ALL_SAMPLE_PATTERNS if args.all_samples else DEFAULT_VOICE_PATTERNS))
    fetched = fetch_voice_samples(
        repo_id=str(args.repo),
        voices_dir=args.voices_dir,
        patterns=patterns,
        revision=args.revision,
        replace=bool(args.replace),
        dry_run=bool(args.dry_run),
    )
    for item in fetched:
        state = "exists" if item.existed and not args.replace else "fetched"
        if args.dry_run:
            state = "would-fetch" if not item.existed or args.replace else "exists"
        logger.info("%s voice=%s source=%s target=%s", state, item.voice_id, item.source, item.path)
    logger.info("voice fetch complete: count=%d voices_dir=%s", len(fetched), args.voices_dir)


if __name__ == "__main__":
    main()
