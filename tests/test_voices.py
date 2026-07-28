from __future__ import annotations

import json

import pytest

from irodori_openai_tts.config import Settings
from irodori_openai_tts.voices import LATENT_EXTENSIONS, VOICE_EXTENSIONS, VoiceRegistry


def make_registry(tmp_path, **overrides) -> VoiceRegistry:
    settings = Settings(
        voices_dir=tmp_path,
        allow_no_ref_voice=True,
        _env_file=None,
        **overrides,
    )
    return VoiceRegistry(settings)


def test_resolve_scanned_audio_file(tmp_path):
    (tmp_path / "speaker.wav").write_bytes(b"wav")
    registry = make_registry(tmp_path)

    voice = registry.resolve("speaker")

    assert voice.voice_id == "speaker"
    assert voice.ref_wav == str(tmp_path / "speaker.wav")
    assert voice.ref_latent is None


@pytest.mark.parametrize("extension", sorted(VOICE_EXTENSIONS))
def test_resolve_all_supported_audio_extensions(tmp_path, extension):
    path = tmp_path / f"speaker{extension}"
    path.write_bytes(b"audio")
    registry = make_registry(tmp_path)

    voice = registry.resolve("speaker")

    assert voice.ref_wav == str(path)


def test_resolve_scanned_latent_file(tmp_path):
    (tmp_path / "speaker.pt").write_bytes(b"latent")
    registry = make_registry(tmp_path)

    voice = registry.resolve({"id": "speaker"})

    assert voice.voice_id == "speaker"
    assert voice.ref_latent == str(tmp_path / "speaker.pt")


@pytest.mark.parametrize("extension", sorted(LATENT_EXTENSIONS))
def test_resolve_all_supported_latent_extensions(tmp_path, extension):
    path = tmp_path / f"speaker{extension}"
    path.write_bytes(b"latent")
    registry = make_registry(tmp_path)

    voice = registry.resolve("speaker")

    assert voice.ref_latent == str(path)


def test_resolve_alias_file_with_ref_wav(tmp_path):
    (tmp_path / "narrator.wav").write_bytes(b"wav")
    (tmp_path / "voices.json").write_text(
        json.dumps(
            {
                "calm": {
                    "ref_wav": "narrator.wav",
                }
            }
        ),
        encoding="utf-8",
    )
    registry = make_registry(tmp_path)

    voice = registry.resolve("calm")

    assert voice.ref_wav == str(tmp_path / "narrator.wav")


def test_resolve_default_voice(tmp_path):
    (tmp_path / "default.wav").write_bytes(b"wav")
    registry = make_registry(tmp_path, default_voice="default")

    voice = registry.resolve(None)

    assert voice.voice_id == "default"


def test_resolve_no_ref_voice(tmp_path):
    registry = make_registry(tmp_path)

    voice = registry.resolve("none")

    assert voice.no_ref is True


def test_resolve_unknown_voice_raises_key_error(tmp_path):
    registry = make_registry(tmp_path)

    with pytest.raises(KeyError, match="Unknown voice"):
        registry.resolve("missing")


def test_write_file_create_replace_and_delete(tmp_path):
    registry = make_registry(tmp_path)

    created = registry.write_file(filename="speaker.wav", data=b"old")
    assert created.path.read_bytes() == b"old"
    assert registry.get_file("speaker") is not None

    with pytest.raises(FileExistsError):
        registry.write_file(filename="speaker.wav", data=b"new")

    replaced = registry.write_file(filename="speaker.flac", data=b"new", voice_id="speaker", replace=True)
    assert replaced.path.name == "speaker.flac"
    assert not (tmp_path / "speaker.wav").exists()
    assert replaced.path.read_bytes() == b"new"

    assert registry.delete_file("speaker") is True
    assert registry.delete_file("speaker") is False


@pytest.mark.parametrize("extension", sorted(VOICE_EXTENSIONS))
def test_write_file_accepts_all_supported_audio_extensions(tmp_path, extension):
    registry = make_registry(tmp_path)

    created = registry.write_file(filename=f"speaker{extension}", data=b"audio")

    assert created.path == tmp_path / f"speaker{extension}"
    assert created.path.read_bytes() == b"audio"


def test_write_file_rejects_bad_voice_id_extension_and_empty_data(tmp_path):
    registry = make_registry(tmp_path)

    with pytest.raises(ValueError, match="voice_id"):
        registry.write_file(filename="speaker.wav", data=b"wav", voice_id="../bad")
    with pytest.raises(ValueError, match="Unsupported voice file extension"):
        registry.write_file(filename="speaker.txt", data=b"text")
    with pytest.raises(ValueError, match="must not be empty"):
        registry.write_file(filename="speaker.wav", data=b"")


def test_seed_bundled_voices_copies_supported_files_without_replacing(tmp_path):
    target = tmp_path / "voices"
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    target.mkdir()
    (bundled / "clone_ref1.wav").write_bytes(b"bundled")
    (bundled / "cached.pt").write_bytes(b"latent")
    (bundled / "speaker.speaker.safetensors").write_bytes(b"embed")
    (bundled / "notes.txt").write_bytes(b"text")
    (target / "clone_ref1.wav").write_bytes(b"existing")
    registry = make_registry(target, bundled_voices_dir=bundled)

    copied = registry.seed_bundled_voices()

    assert sorted(path.name for path in copied) == ["cached.pt", "speaker.speaker.safetensors"]
    assert (target / "clone_ref1.wav").read_bytes() == b"existing"
    assert (target / "cached.pt").read_bytes() == b"latent"
    assert (target / "speaker.speaker.safetensors").read_bytes() == b"embed"
    assert not (target / "notes.txt").exists()


def test_seed_bundled_voices_ignores_missing_source(tmp_path):
    registry = make_registry(tmp_path / "voices", bundled_voices_dir=tmp_path / "missing")

    assert registry.seed_bundled_voices() == []
