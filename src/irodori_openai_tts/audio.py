from __future__ import annotations

import shutil
import subprocess
import threading
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from typing import Any

import soundfile as sf
import torch
import torchaudio

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def normalize_response_format(value: str | None, *, default: str) -> str:
    fmt = (default if value is None else str(value)).strip().lower()
    if fmt not in CONTENT_TYPES:
        allowed = ", ".join(sorted(CONTENT_TYPES))
        raise ValueError(f"Unsupported response_format={value!r}. Expected one of: {allowed}.")
    return fmt


def encode_audio(audio: torch.Tensor, sample_rate: int, response_format: str) -> bytes:
    fmt = normalize_response_format(response_format, default="mp3")
    wav = audio.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"Expected audio shape (channels, samples), got {tuple(wav.shape)}")
    wav = wav.clamp(-1.0, 1.0).contiguous()

    if fmt == "pcm":
        pcm = (wav.squeeze(0).numpy() * 32767.0).astype("<i2", copy=False)
        return pcm.tobytes()

    if fmt in {"wav", "flac", "mp3", "opus"}:
        try:
            return _encode_with_soundfile(wav, int(sample_rate), fmt)
        except Exception as exc:
            if fmt in {"wav", "flac"}:
                raise
            soundfile_exc = exc
    else:
        soundfile_exc = None

    try:
        return _encode_with_torchaudio(wav, int(sample_rate), fmt)
    except Exception as torchaudio_exc:
        try:
            return _encode_with_ffmpeg(wav, int(sample_rate), fmt)
        except Exception as ffmpeg_exc:
            details = [f"torchaudio: {torchaudio_exc}"]
            if soundfile_exc is not None:
                details.insert(0, f"soundfile: {soundfile_exc}")
            details.append(f"ffmpeg: {ffmpeg_exc}")
            raise RuntimeError(
                f"Failed to encode audio as {fmt}. Install soundfile with MP3/Opus support, "
                "FFmpeg-enabled torchaudio, or ffmpeg in PATH; or request "
                "response_format='wav'/'flac'/'pcm'. "
                f"Encoder errors: {'; '.join(details)}"
            ) from ffmpeg_exc


class StreamingAudioEncoder:
    def __init__(self, response_format: str, sample_rate: int) -> None:
        self.fmt = normalize_response_format(response_format, default="mp3")
        self.sample_rate = int(sample_rate)
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: Queue[bytes] = Queue()
        self._stderr_queue: Queue[bytes] = Queue()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._channels: int | None = None
        self._closed = False

    def write(self, audio: torch.Tensor) -> bytes:
        if self._closed:
            raise RuntimeError("Streaming audio encoder is already closed.")
        wav = _normalize_audio_tensor(audio)
        if self.fmt == "pcm":
            return _pcm16_bytes(wav)
        if self.fmt == "wav":
            payload = b""
            if self._channels is None:
                self._channels = int(wav.shape[0])
                payload += _streaming_wav_header(
                    sample_rate=self.sample_rate,
                    channels=self._channels,
                )
            elif self._channels != wav.shape[0]:
                raise ValueError("Streaming audio chunks must have the same channel count.")
            return payload + _pcm16_bytes(wav)

        self._ensure_process(channels=wav.shape[0])
        assert self._process is not None
        assert self._process.stdin is not None
        if self._process.poll() is not None:
            self._raise_process_error()
        self._process.stdin.write(_float32_interleaved_bytes(wav))
        self._process.stdin.flush()
        return self._read_available_output()

    def close(self) -> bytes:
        if self._closed:
            return b""
        self._closed = True
        if self.fmt in {"pcm", "wav"}:
            return b""
        if self._process is None:
            return b""

        process = self._process
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise RuntimeError("Timed out while finalizing streaming audio encoder.") from exc

        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        output = self._read_available_output()
        if process.returncode:
            self._raise_process_error()
        return output

    def _ensure_process(self, *, channels: int) -> None:
        if self._process is not None:
            if self._channels != channels:
                raise ValueError("Streaming audio chunks must have the same channel count.")
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg executable was not found in PATH; binary streaming for "
                f"response_format={self.fmt!r} is unavailable. Use response_format='pcm' "
                "or install ffmpeg."
            )

        self._channels = int(channels)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-codec:a",
            _streaming_ffmpeg_codec(self.fmt),
            "-f",
            _ffmpeg_format(self.fmt),
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = _start_pipe_reader(self._process.stdout, self._stdout_queue)
        self._stderr_thread = _start_pipe_reader(self._process.stderr, self._stderr_queue)

    def _read_available_output(self) -> bytes:
        chunks = []
        while True:
            try:
                chunks.append(self._stdout_queue.get_nowait())
            except Empty:
                break
        return b"".join(chunks)

    def _raise_process_error(self) -> None:
        stderr = []
        while True:
            try:
                stderr.append(self._stderr_queue.get_nowait())
            except Empty:
                break
        message = b"".join(stderr).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg streaming encoder failed: {message or 'unknown error'}")


def _start_pipe_reader(pipe: Any, queue: Queue[bytes]) -> threading.Thread:
    def read_pipe() -> None:
        try:
            read = getattr(pipe, "read1", pipe.read)
            while True:
                data = read(65536)
                if not data:
                    break
                queue.put(data)
        finally:
            pipe.close()

    thread = threading.Thread(target=read_pipe, daemon=True)
    thread.start()
    return thread


def _normalize_audio_tensor(audio: torch.Tensor) -> torch.Tensor:
    wav = audio.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"Expected audio shape (channels, samples), got {tuple(wav.shape)}")
    return wav.clamp(-1.0, 1.0).contiguous()


def _pcm16_bytes(wav: torch.Tensor) -> bytes:
    pcm = (wav.transpose(0, 1).numpy() * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes()


def _streaming_wav_header(*, sample_rate: int, channels: int) -> bytes:
    bits_per_sample = 16
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    sentinel_size = 0xFFFFFFFF
    return b"".join(
        [
            b"RIFF",
            sentinel_size.to_bytes(4, "little"),
            b"WAVE",
            b"fmt ",
            (16).to_bytes(4, "little"),
            (1).to_bytes(2, "little"),
            channels.to_bytes(2, "little"),
            sample_rate.to_bytes(4, "little"),
            byte_rate.to_bytes(4, "little"),
            block_align.to_bytes(2, "little"),
            bits_per_sample.to_bytes(2, "little"),
            b"data",
            sentinel_size.to_bytes(4, "little"),
        ]
    )


def _float32_interleaved_bytes(wav: torch.Tensor) -> bytes:
    return wav.transpose(0, 1).contiguous().numpy().astype("<f4", copy=False).tobytes()


def _soundfile_format(fmt: str) -> tuple[str, str | None]:
    if fmt == "mp3":
        return "MP3", "MPEG_LAYER_III"
    if fmt == "opus":
        return "OGG", "OPUS"
    return fmt.upper(), None


def _encode_with_soundfile(wav: torch.Tensor, sample_rate: int, fmt: str) -> bytes:
    audio = wav.transpose(0, 1).numpy()
    sf_format, subtype = _soundfile_format(fmt)
    buffer = BytesIO()
    sf.write(
        buffer,
        audio,
        sample_rate,
        format=sf_format,
        subtype=subtype,
    )
    return buffer.getvalue()


def _torchaudio_format(fmt: str) -> str:
    if fmt == "opus":
        return "ogg"
    if fmt == "aac":
        return "adts"
    return fmt


def _torchaudio_suffix(fmt: str) -> str:
    if fmt == "opus":
        return ".opus"
    if fmt == "aac":
        return ".aac"
    return f".{fmt}"


def _encode_with_torchaudio(wav: torch.Tensor, sample_rate: int, fmt: str) -> bytes:
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"speech{_torchaudio_suffix(fmt)}"
        torchaudio.save(
            str(path),
            wav,
            sample_rate,
            format=_torchaudio_format(fmt),
        )
        return path.read_bytes()


def _ffmpeg_format(fmt: str) -> str:
    if fmt == "aac":
        return "adts"
    if fmt == "opus":
        return "ogg"
    return fmt


def _ffmpeg_codec(fmt: str) -> str:
    if fmt == "mp3":
        return "libmp3lame"
    if fmt == "opus":
        return "libopus"
    if fmt == "aac":
        return "aac"
    return fmt


def _streaming_ffmpeg_codec(fmt: str) -> str:
    if fmt == "wav":
        return "pcm_s16le"
    return _ffmpeg_codec(fmt)


def _encode_with_ffmpeg(wav: torch.Tensor, sample_rate: int, fmt: str) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable was not found in PATH")

    with TemporaryDirectory() as directory:
        source = Path(directory) / "speech.wav"
        target = Path(directory) / f"speech{_torchaudio_suffix(fmt)}"
        sf.write(
            source,
            wav.transpose(0, 1).numpy(),
            sample_rate,
            format="WAV",
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-codec:a",
            _ffmpeg_codec(fmt),
            "-f",
            _ffmpeg_format(fmt),
            str(target),
        ]
        subprocess.run(command, check=True, capture_output=True)
        return target.read_bytes()
