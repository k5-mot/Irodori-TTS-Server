from __future__ import annotations

from pathlib import Path

import pytest

from irodori_openai_tts import fetch_voices


def test_fetch_voice_samples_downloads_default_clone_refs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "clone_ref1.wav").write_bytes(b"ref1")
    (cache / "clone_ref2.wav").write_bytes(b"ref2")

    monkeypatch.setattr(
        fetch_voices,
        "list_repo_files",
        lambda **kwargs: [
            "README.md",
            "samples/clone_ref1.wav",
            "samples/clone_ref2.wav",
            "samples/standard_sample1.wav",
        ],
    )
    monkeypatch.setattr(
        fetch_voices,
        "hf_hub_download",
        lambda *, filename, **kwargs: str(cache / Path(filename).name),
    )

    fetched = fetch_voices.fetch_voice_samples(
        repo_id="Aratako/Irodori-TTS-500M-v3",
        voices_dir=tmp_path / "voices",
    )

    assert [item.voice_id for item in fetched] == ["clone_ref1", "clone_ref2"]
    assert (tmp_path / "voices" / "clone_ref1.wav").read_bytes() == b"ref1"
    assert (tmp_path / "voices" / "clone_ref2.wav").read_bytes() == b"ref2"


def test_fetch_voice_samples_skips_existing_without_replace(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "clone_ref1.wav").write_bytes(b"existing")
    cache = tmp_path / "cache.wav"
    cache.write_bytes(b"new")

    monkeypatch.setattr(
        fetch_voices,
        "list_repo_files",
        lambda **kwargs: ["samples/clone_ref1.wav"],
    )

    def fail_download(**kwargs):
        raise AssertionError("download should not be called for existing voice")

    monkeypatch.setattr(fetch_voices, "hf_hub_download", fail_download)

    fetched = fetch_voices.fetch_voice_samples(
        repo_id="repo",
        voices_dir=voices_dir,
    )

    assert fetched[0].existed is True
    assert (voices_dir / "clone_ref1.wav").read_bytes() == b"existing"


def test_fetch_voice_samples_replaces_existing_when_requested(tmp_path, monkeypatch):
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "clone_ref1.wav").write_bytes(b"existing")
    cache = tmp_path / "cache.wav"
    cache.write_bytes(b"new")

    monkeypatch.setattr(
        fetch_voices,
        "list_repo_files",
        lambda **kwargs: ["samples/clone_ref1.wav"],
    )
    monkeypatch.setattr(fetch_voices, "hf_hub_download", lambda **kwargs: str(cache))

    fetched = fetch_voices.fetch_voice_samples(
        repo_id="repo",
        voices_dir=voices_dir,
        replace=True,
    )

    assert fetched[0].existed is True
    assert (voices_dir / "clone_ref1.wav").read_bytes() == b"new"


def test_fetch_voice_samples_can_select_all_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_voices,
        "list_repo_files",
        lambda **kwargs: [
            "samples/clone_ref1.wav",
            "samples/standard_sample1.wav",
            "samples/readme.txt",
        ],
    )

    fetched = fetch_voices.fetch_voice_samples(
        repo_id="repo",
        voices_dir=tmp_path / "voices",
        patterns=fetch_voices.ALL_SAMPLE_PATTERNS,
        dry_run=True,
    )

    assert [item.voice_id for item in fetched] == ["clone_ref1", "standard_sample1"]
    assert not (tmp_path / "voices").exists()


def test_fetch_voice_samples_raises_when_no_files_match(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_voices, "list_repo_files", lambda **kwargs: ["README.md"])

    with pytest.raises(ValueError, match="No voice sample files matched"):
        fetch_voices.fetch_voice_samples(repo_id="repo", voices_dir=tmp_path / "voices")
