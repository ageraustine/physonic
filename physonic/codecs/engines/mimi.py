"""Mimi neural audio codec implementation (Kyutai)

Chunking strategy: context priming for a causal codec.

Mimi is fully causal (causal convs + causally-masked bottleneck transformers,
see the Moshi paper, Sec. 3.3.1). Therefore only the *left* edge of an
independently processed chunk is degraded; the right edge is exact. Instead of
symmetric overlap-add, each chunk is processed with CONTEXT_FRAMES of real
left context prepended, and the output for that context region is discarded.
As context grows, the kept region converges to the unchunked result. A short
linear crossfade (~10 ms) is applied at decode seams as insurance against
residual discontinuities.

Processing modes:
    - Sequential (default): workers=1, encode=6.0s, decode=2.0s chunks
    - Parallel: workers=25% of CPUs (2-4), encode=4.0s, decode=4.0s chunks
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typing import Any
from physonic.codecs.base import BaseCodecEngine
from physonic.codecs.core.exceptions import CodecError

# Mimi's fixed downsampling factor: 24kHz audio @ 12.5 codes/sec
MIMI_DOWNSAMPLE_FACTOR = 1920

# Left context prepended to every chunk (encode: audio frames, decode: code frames).
# 8 frames = 0.64 s. This is the quality knob for chunked processing: Mimi's
# bottleneck transformer attends over far more history than this, so the first
# kept frames of each chunk see less context than in unchunked processing.
# Increasing CONTEXT_FRAMES shrinks that gap if boundary artifacts remain
# audible, at proportional compute cost.
CONTEXT_FRAMES = 8

# Linear crossfade at decode seams: 240 samples = 10 ms @ 24 kHz.
# Linear (equal-voltage) rather than equal-power is deliberate: both sides of
# the seam are near-identical, highly correlated signals.
CROSSFADE_SAMPLES = 240

# Default chunk lengths (based on benchmarks)
# Sequential mode: longer encode chunks, shorter decode chunks
SEQUENTIAL_ENCODE_CHUNK = 6.0
SEQUENTIAL_DECODE_CHUNK = 2.0
# Parallel mode: balanced chunks work better with multiple workers
PARALLEL_ENCODE_CHUNK = 4.0
PARALLEL_DECODE_CHUNK = 4.0


def get_parallel_workers(cpu_percent: int = 25) -> int:
    """Calculate optimal worker count based on CPU percentage.

    Args:
        cpu_percent: Percentage of CPUs to use (default: 25%)

    Returns:
        Number of workers, minimum 2, maximum 4

    Examples:
        - 4-core @ 25%: 1 -> clamped to 2 workers
        - 8-core @ 25%: 2 workers
        - 16-core @ 25%: 4 workers (capped)
    """
    cpu_count = os.cpu_count() or 4
    workers = max(1, int(cpu_count * cpu_percent / 100))
    return max(2, min(4, workers))  # Clamp between 2-4


class MimiAudioEngine(BaseCodecEngine):
    """Kyutai Mimi neural audio codec - 1.1kbps at 12.5Hz (lazy-loaded)

    Paper: http://kyutai.org/Moshi.pdf
    Model: kyutai/mimi on Hugging Face

    Features:
        - Context-primed chunking for quality
        - Sequential processing by default (reliable)
        - Optional parallel processing for speedup
    """

    def __init__(
        self,
        parallel: bool = False,
        cpu_percent: int = 25,
        encode_chunk_length: float | None = None,
        decode_chunk_length: float | None = None,
        workers: int | None = None,
    ):
        """Initialize Mimi engine with optimal CPU-friendly settings

        Args:
            parallel: Enable parallel processing (default: False = sequential)
            cpu_percent: Percentage of CPUs for parallel mode (default: 25%)
                        - 25% is optimal, avoids system throttling
                        - Ignored when parallel=False or workers is set
            encode_chunk_length: Override chunk duration for encoding
                                 - None: Use mode-specific default (6.0s seq, 4.0s parallel)
                                 - 0 or negative: Disable chunking
            decode_chunk_length: Override chunk duration for decoding
                                 - None: Use mode-specific default (2.0s seq, 4.0s parallel)
                                 - 0 or negative: Disable chunking
            workers: Override worker count (ignores parallel/cpu_percent)
                    - None: Auto (1 for sequential, calculated for parallel)
                    - 1: Force sequential
                    - >1: Force parallel with specific worker count

        Performance (30s audio):
            - Sequential: ~7.1s (reliable, no system impact)
            - Parallel (25% CPU): ~5.7s (~20% faster)

        Note:
            Stereo input is downmixed to mono on encode (Mimi is a mono codec).
        """
        self._parallel = parallel
        self._cpu_percent = cpu_percent
        self._encode_chunk_length = encode_chunk_length
        self._decode_chunk_length = decode_chunk_length

        # Determine workers
        if workers is not None:
            self._workers = workers
        elif parallel:
            self._workers = get_parallel_workers(cpu_percent)
        else:
            self._workers = 1
        self._model = None
        self._feature_extractor = None
        self._device = None
        self._sampling_rate = None

    def _get_encode_chunk_length(self) -> float | None:
        """Get chunk length for encoding."""
        if self._encode_chunk_length is not None:
            if self._encode_chunk_length <= 0:
                return None
            return self._encode_chunk_length
        # Use mode-specific default
        return PARALLEL_ENCODE_CHUNK if self._workers > 1 else SEQUENTIAL_ENCODE_CHUNK

    def _get_decode_chunk_length(self) -> float | None:
        """Get chunk length for decoding."""
        if self._decode_chunk_length is not None:
            if self._decode_chunk_length <= 0:
                return None
            return self._decode_chunk_length
        # Use mode-specific default
        return PARALLEL_DECODE_CHUNK if self._workers > 1 else SEQUENTIAL_DECODE_CHUNK

    def _ensure_model_loaded(self):
        """Lazy load Mimi model and feature extractor (CPU only)"""
        if self._model is None:
            try:
                import warnings
                import logging

                # Suppress torch and transformers warnings
                logging.getLogger("transformers").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", category=UserWarning)

                import torch
                from transformers import MimiModel, AutoFeatureExtractor

                # Use public model
                model_id = "kyutai/mimi"

                self._device = torch.device("cpu")

                # Load model and feature extractor
                self._model = MimiModel.from_pretrained(model_id)
                self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

                self._model.to(self._device)
                self._model.eval()
                self._sampling_rate = self._feature_extractor.sampling_rate
            except ImportError as e:
                raise CodecError(
                    "Mimi dependencies not installed. Run: pip install transformers torch"
                ) from e
            except Exception as e:
                raise CodecError(f"Failed to load Mimi model: {e}") from e

    def encode(self, audio_data: np.ndarray, sample_rate: int) -> dict[str, Any]:
        """Encode audio using Mimi with context-primed chunking

        Args:
            audio_data: Shape (channels, samples) or (samples,) for mono.
                        Stereo is downmixed to mono.
            sample_rate: Input sample rate

        Returns:
            Dictionary with encoded data and metadata
        """
        self._ensure_model_loaded()

        if audio_data.size == 0:
            raise CodecError("Mimi encoding failed: input audio is empty")

        try:
            import torch

            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=0)
            channels = 1

            if sample_rate != self._sampling_rate:
                import torchaudio

                audio_torch = torch.from_numpy(audio_data).float()
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self._sampling_rate,
                    resampling_method="sinc_interp_kaiser",
                    lowpass_filter_width=64
                )
                audio_data = resampler(audio_torch).numpy()
                sample_rate = self._sampling_rate

            inputs = self._feature_extractor(
                raw_audio=audio_data,
                sampling_rate=self._sampling_rate,
                return_tensors="pt"
            )
            input_values = inputs["input_values"].to(self._device)

            chunk_length = self._get_encode_chunk_length()

            if chunk_length is not None:
                # Pad the whole signal to a frame multiple once, so every
                # chunk boundary is frame-aligned and per-chunk padding /
                # code trimming logic is unnecessary.
                n_samples = input_values.shape[-1]
                total_frames = math.ceil(n_samples / MIMI_DOWNSAMPLE_FACTOR)
                pad = total_frames * MIMI_DOWNSAMPLE_FACTOR - n_samples
                if pad:
                    input_values = torch.nn.functional.pad(input_values, (0, pad))

                chunk_frames = max(
                    1, int(chunk_length * self._sampling_rate / MIMI_DOWNSAMPLE_FACTOR)
                )

                # Build chunk parameters
                chunk_params = []
                for start_f in range(0, total_frames, chunk_frames):
                    end_f = min(start_f + chunk_frames, total_frames)
                    ctx_f = min(CONTEXT_FRAMES, start_f)
                    chunk_params.append((start_f, end_f, ctx_f))

                def encode_chunk(params):
                    """Encode a single chunk (called in parallel)."""
                    start_f, end_f, ctx_f = params
                    seg = input_values[
                        :, :,
                        (start_f - ctx_f) * MIMI_DOWNSAMPLE_FACTOR:
                        end_f * MIMI_DOWNSAMPLE_FACTOR
                    ]

                    with torch.no_grad():
                        codes = self._model.encode(seg).audio_codes[0]

                    expected = ctx_f + (end_f - start_f)
                    if codes.shape[-1] != expected:
                        raise CodecError(
                            f"Unexpected code frame count from Mimi encoder: "
                            f"got {codes.shape[-1]}, expected {expected}"
                        )

                    # Discard codes belonging to the context prefix
                    return codes[:, ctx_f:]

                # Parallel encoding (or sequential if workers=1)
                if self._workers > 1 and len(chunk_params) > 1:
                    with ThreadPoolExecutor(max_workers=self._workers) as executor:
                        chunk_codes = list(executor.map(encode_chunk, chunk_params))
                else:
                    chunk_codes = [encode_chunk(p) for p in chunk_params]

                audio_codes = torch.cat(chunk_codes, dim=-1).cpu().numpy()
            else:
                with torch.no_grad():
                    encoder_outputs = self._model.encode(input_values)
                audio_codes = encoder_outputs.audio_codes[0].cpu().numpy()

            return {
                "codes": [audio_codes],
                "shape": [1] + list(audio_codes.shape),
                "sample_rate": self._sampling_rate,
                "channels": channels,
            }
        except CodecError:
            raise
        except Exception as e:
            raise CodecError(f"Mimi encoding failed: {e}") from e

    def decode(self, encoded_data: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Decode Mimi codes back to audio with chunked decoding

        Args:
            encoded_data: Dictionary from encode() method

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._ensure_model_loaded()

        try:
            import torch

            audio_codes_np = encoded_data["codes"][0]
            audio_codes_np = np.round(audio_codes_np).astype(np.int64)
            audio_codes_tensor = torch.from_numpy(audio_codes_np).to(self._device)

            if audio_codes_tensor.shape[-1] == 0:
                raise CodecError("Mimi decoding failed: no code frames to decode")

            chunk_length = self._get_decode_chunk_length()

            if chunk_length is not None:
                chunk_frames = max(
                    1, int(chunk_length * self._sampling_rate / MIMI_DOWNSAMPLE_FACTOR)
                )
                total_frames = audio_codes_tensor.shape[-1]

                # Build chunk params
                chunk_params = []
                for start_f in range(0, total_frames, chunk_frames):
                    end_f = min(start_f + chunk_frames, total_frames)
                    ctx_f = min(CONTEXT_FRAMES, start_f)
                    chunk_params.append((start_f, end_f, ctx_f))

                def decode_chunk(params):
                    """Decode a single chunk (called in parallel)."""
                    start_f, end_f, ctx_f = params
                    seg_codes = audio_codes_tensor[:, start_f - ctx_f:end_f]
                    with torch.no_grad():
                        return self._model.decode(seg_codes.unsqueeze(0)).audio_values

                # Parallel decoding (or sequential if workers=1)
                if self._workers > 1 and len(chunk_params) > 1:
                    with ThreadPoolExecutor(max_workers=self._workers) as executor:
                        raw_chunks = list(executor.map(decode_chunk, chunk_params))
                else:
                    raw_chunks = [decode_chunk(p) for p in chunk_params]

                # Sequential crossfade (must be in order)
                decoded_chunks: list[torch.Tensor] = []
                for decoded, params in zip(raw_chunks, chunk_params):
                    _, _, ctx_f = params

                    if ctx_f == 0:
                        decoded_chunks.append(decoded)
                        continue

                    # Discard context prefix, keep CROSSFADE_SAMPLES for blending
                    cut = ctx_f * MIMI_DOWNSAMPLE_FACTOR - CROSSFADE_SAMPLES
                    decoded = decoded[..., cut:]

                    fade_in = torch.linspace(
                        0.0, 1.0, CROSSFADE_SAMPLES,
                        device=decoded.device, dtype=decoded.dtype,
                    )
                    prev = decoded_chunks[-1]
                    prev[..., -CROSSFADE_SAMPLES:] = (
                        prev[..., -CROSSFADE_SAMPLES:] * (1.0 - fade_in)
                        + decoded[..., :CROSSFADE_SAMPLES] * fade_in
                    )
                    decoded_chunks.append(decoded[..., CROSSFADE_SAMPLES:])

                # Note: trimming any encode-time padding back to the original
                # duration is handled by caller if needed
                audio_values = torch.cat(decoded_chunks, dim=-1)
            else:
                audio_codes_batched = audio_codes_tensor.unsqueeze(0)
                with torch.no_grad():
                    audio_values = self._model.decode(audio_codes_batched).audio_values

            if audio_values.dim() == 3:
                audio_array = audio_values.squeeze(0).cpu().numpy()
            else:
                audio_array = audio_values.cpu().numpy()

            if audio_array.ndim == 1:
                audio_array = audio_array[np.newaxis, :]

            return audio_array, encoded_data["sample_rate"]
        except CodecError:
            raise
        except Exception as e:
            raise CodecError(f"Mimi decoding failed: {e}") from e

    @property
    def name(self) -> str:
        return "mimi"
