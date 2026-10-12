"""Room acoustics: image-source method for early reflections + Schroeder reverberator.

This module implements physically-based room acoustics for indoor robot training environments.
Early reflections use the image-source method (mirror-box reflections), and late diffuse
reverb uses a Schroeder-Moorer reverberator with parallel comb filters and serial allpass filters.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RoomSpec:
    """Room acoustic properties for indoor environments."""
    dimensions: Tuple[float, float, float]  # (length, width, height) in meters
    absorption: float  # Wall absorption coefficient (0-1), 0=perfectly reflective, 1=fully absorptive
    rt60: float  # Reverberation time: time for 60dB decay (seconds)
    max_reflection_order: int = 2  # Maximum reflection order for image-source method
    temperature: float = 20.0  # Celsius (affects speed of sound)
    humidity: float = 50.0  # Percent relative humidity (affects air absorption)

    @property
    def volume(self) -> float:
        """Room volume in cubic meters."""
        return float(np.prod(self.dimensions))

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound in m/s, temperature-corrected."""
        # c = 331.3 + 0.606 * T (°C)
        return 331.3 + 0.606 * self.temperature


def compute_image_sources(
    source_pos: np.ndarray,
    listener_pos: np.ndarray,
    room: RoomSpec,
    max_order: int = None
) -> List[Dict]:
    """Compute image sources using mirror-box method for early reflections.

    The image-source method models specular reflections by mirroring the source
    position across room boundaries. This generates discrete early reflections
    that are perceptually distinct from the direct sound.

    Args:
        source_pos: Source 3D position [x, y, z] in meters
        listener_pos: Listener 3D position [x, y, z] in meters
        room: Room specification with dimensions and absorption
        max_order: Maximum reflection order (default: room.max_reflection_order)
                  Order 1 = first-order reflections (6 images, one per wall)
                  Order 2 = second-order reflections (26 images total)

    Returns:
        List of image source dictionaries, each containing:
        - position: [x, y, z] of mirrored source
        - delay: delay in samples at 44.1 kHz
        - attenuation: amplitude factor (includes 1/r and absorption)
        - order: reflection order (number of wall bounces)

    Example:
        >>> room = RoomSpec((10, 8, 3), absorption=0.2, rt60=1.0)
        >>> source = np.array([2, 2, 1.5])
        >>> listener = np.array([5, 5, 1.5])
        >>> images = compute_image_sources(source, listener, room, max_order=1)
        >>> len(images)
        6  # One reflection per wall
    """
    if max_order is None:
        max_order = room.max_reflection_order

    c = room.speed_of_sound
    sr = 44100  # Fixed sample rate

    images = []
    L, W, H = room.dimensions

    # Generate all combinations of reflections up to max_order
    # nx, ny, nz indicate number of reflections in each dimension
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                # Reflection order = total number of wall bounces
                order = abs(nx) + abs(ny) + abs(nz)

                # Skip direct sound (order 0) and beyond max order
                if order == 0 or order > max_order:
                    continue

                # Mirror source position using image-source formula:
                # image_coord = 2 * n * room_size - (-1)^n * source_coord
                # This creates mirror reflections across room boundaries
                image_x = 2 * nx * L - ((-1) ** nx) * source_pos[0]
                image_y = 2 * ny * W - ((-1) ** ny) * source_pos[1]
                image_z = 2 * nz * H - ((-1) ** nz) * source_pos[2]
                image_pos = np.array([image_x, image_y, image_z])

                # Compute propagation delay from image source to listener
                distance = np.linalg.norm(image_pos - listener_pos)
                delay_sec = distance / c
                delay_samples = int(delay_sec * sr)

                # Compute attenuation from two factors:
                # 1. Geometric spreading (1/r law)
                distance_atten = 1.0 / max(distance, 0.1)

                # 2. Wall absorption (compounds with each reflection)
                # absorption_atten = (1 - α)^order, where α is absorption coefficient
                absorption_atten = (1.0 - room.absorption) ** order

                total_atten = distance_atten * absorption_atten

                images.append({
                    'position': image_pos,
                    'delay': delay_samples,
                    'attenuation': float(total_atten),
                    'order': order
                })

    return images


def apply_image_source_reverb(
    mono: np.ndarray,
    image_sources: List[Dict]
) -> np.ndarray:
    """Apply early reflections using precomputed image sources.

    This function creates discrete echoes by time-shifting and scaling copies
    of the direct sound according to the image-source delays and attenuations.

    Args:
        mono: Mono audio signal, shape (N,)
        image_sources: List of image source dicts from compute_image_sources

    Returns:
        Mono audio with early reflections added, shape (N + max_delay,)
        Output is longer than input due to reflection tail

    Example:
        >>> mono = np.random.randn(44100)  # 1 second
        >>> images = compute_image_sources(source, listener, room)
        >>> reverb_mono = apply_image_source_reverb(mono, images)
        >>> len(reverb_mono) > len(mono)  # Longer due to reflection tail
        True
    """
    if len(image_sources) == 0:
        return mono

    # Find maximum delay to size output buffer
    max_delay = max(img['delay'] for img in image_sources)
    out_len = len(mono) + max_delay
    out = np.zeros(out_len, dtype=np.float32)

    # Copy direct sound (no attenuation)
    out[:len(mono)] = mono

    # Add each reflection as delayed, attenuated copy
    for img in image_sources:
        delay = img['delay']
        atten = img['attenuation']

        # Ensure delay doesn't exceed buffer
        if delay + len(mono) <= out_len:
            out[delay:delay + len(mono)] += mono * atten

    return out


class SchroederReverberator:
    """Schroeder-Moorer reverberator for late diffuse reverb.

    Implements a classic algorithmic reverb with:
    - 4 parallel comb filters (FIR feedback delays)
    - 2 serial allpass filters (smoothing)

    The comb filters create the echo density, while allpass filters smooth
    the frequency response to avoid metallic coloration. Delay times and
    feedback gains are automatically tuned based on RT60 and room volume.

    Reference:
        Schroeder, M. R. (1962). "Natural sounding artificial reverberation."
        Journal of the Audio Engineering Society.

    Example:
        >>> reverb = SchroederReverberator(rt60=1.5, room_volume=200)
        >>> mono = np.random.randn(44100)  # 1 second
        >>> wet = reverb.process(mono)
        >>> len(wet) == len(mono)  # Same length
        True
    """

    def __init__(self, rt60: float, room_volume: float, sample_rate: int = 44100):
        """Initialize Schroeder reverberator.

        Args:
            rt60: 60dB decay time in seconds
            room_volume: Room volume in m³ (affects delay line lengths)
            sample_rate: Audio sample rate (default: 44100 Hz)
        """
        self.sr = sample_rate
        self.rt60 = rt60

        # Comb filter delays scale with room size
        # Use cube root of volume to get characteristic length
        base_delay_sec = 0.001 * np.cbrt(room_volume)

        # Mutually prime delay times (in milliseconds) to avoid periodic artifacts
        comb_delays_ms = np.array([29.7, 37.1, 41.1, 43.7]) * base_delay_sec * 1000
        self.comb_delays = (comb_delays_ms * sample_rate / 1000).astype(int)

        # Compute feedback gains for desired RT60
        # Sabine formula: g = 10^(-3 * delay / RT60)
        # This ensures each comb filter decays to -60dB in RT60 seconds
        self.comb_gains = []
        for delay in self.comb_delays:
            delay_sec = delay / sample_rate
            g = 10 ** (-3 * delay_sec / rt60)
            self.comb_gains.append(min(g, 0.999))  # Clamp to ensure stability

        # Allpass filter delays (smaller, for smoothing frequency response)
        allpass_delays_ms = [5.0, 1.7]
        self.allpass_delays = [
            (int(d * sample_rate / 1000), 0.7) for d in allpass_delays_ms
        ]

        # Initialize circular buffers
        self.comb_buffers = [np.zeros(d, dtype=np.float32) for d in self.comb_delays]
        self.allpass_buffers = [
            np.zeros(d, dtype=np.float32) for d, _ in self.allpass_delays
        ]
        self.comb_indices = [0] * len(self.comb_delays)
        self.allpass_indices = [0] * len(self.allpass_delays)

    def process(self, mono: np.ndarray) -> np.ndarray:
        """Apply Schroeder reverb to mono signal.

        Processes input sample-by-sample through parallel comb filters
        followed by serial allpass filters.

        Args:
            mono: Input mono audio, shape (N,)

        Returns:
            Wet reverb signal, shape (N,)
        """
        out = np.zeros(len(mono), dtype=np.float32)

        for i in range(len(mono)):
            # Parallel comb filters
            comb_sum = 0.0
            for j, (delay, gain) in enumerate(zip(self.comb_delays, self.comb_gains)):
                # Read from circular buffer
                idx = self.comb_indices[j]
                delayed = self.comb_buffers[j][idx]

                # Comb filter: y[n] = x[n] + g * y[n - delay]
                output = mono[i] + gain * delayed

                # Write back to buffer
                self.comb_buffers[j][idx] = output
                self.comb_indices[j] = (idx + 1) % delay

                comb_sum += output

            # Average parallel outputs
            signal = comb_sum / len(self.comb_delays)

            # Serial allpass filters (smooth frequency response)
            for j, (delay, gain) in enumerate(self.allpass_delays):
                idx = self.allpass_indices[j]
                delayed = self.allpass_buffers[j][idx]

                # Allpass: y[n] = -g * x[n] + x[n - delay] + g * y[n - delay]
                output = -gain * signal + delayed
                self.allpass_buffers[j][idx] = signal + gain * output
                self.allpass_indices[j] = (idx + 1) % delay

                signal = output

            out[i] = signal

        return out


def apply_room_acoustics(
    mono: np.ndarray,
    source_pos: np.ndarray,
    listener_pos: np.ndarray,
    room: RoomSpec,
    enable_early: bool = True,
    enable_late: bool = True,
    wet_mix: float = 0.3
) -> np.ndarray:
    """Apply room acoustics (early reflections + late reverb) to mono audio.

    This is the main entry point for adding room acoustics. It combines:
    1. Direct sound (unchanged)
    2. Early reflections (image-source method)
    3. Late diffuse reverb (Schroeder reverberator)

    Args:
        mono: Mono audio signal, shape (N,)
        source_pos: Source 3D position [x, y, z] in meters
        listener_pos: Listener 3D position [x, y, z] in meters
        room: Room specification (dimensions, RT60, absorption)
        enable_early: Enable early reflections (default: True)
        enable_late: Enable late diffuse reverb (default: True)
        wet_mix: Wet/dry mix for late reverb (0-1), default 0.3

    Returns:
        Mono audio with room acoustics, shape (N + tail,)
        Output may be longer due to reflection tail

    Example:
        >>> room = RoomSpec((20, 15, 3), absorption=0.3, rt60=1.0)
        >>> mono = np.random.randn(44100)
        >>> source = np.array([5, 5, 1.5])
        >>> listener = np.array([10, 10, 1.5])
        >>> reverb_audio = apply_room_acoustics(mono, source, listener, room)
        >>> len(reverb_audio) >= len(mono)
        True
    """
    out = mono.copy()

    # Early reflections (discrete echoes from image-source method)
    if enable_early:
        images = compute_image_sources(source_pos, listener_pos, room)
        out = apply_image_source_reverb(out, images)

    # Late diffuse reverb (Schroeder reverberator)
    if enable_late:
        reverb = SchroederReverberator(room.rt60, room.volume, 44100)

        # Process original mono (not the early-reflection-enhanced version)
        # to avoid double-reverbing early reflections
        reverb_signal = reverb.process(mono)

        # Mix dry and wet signals
        dry_mix = 1.0 - wet_mix

        # Ensure same length by padding shorter signal
        max_len = max(len(out), len(reverb_signal))
        if len(out) < max_len:
            out = np.pad(out, (0, max_len - len(out)), mode='constant')
        if len(reverb_signal) < max_len:
            reverb_signal = np.pad(reverb_signal, (0, max_len - len(reverb_signal)), mode='constant')

        # Combine: dry (direct + early reflections) + wet (late reverb)
        out = dry_mix * out + wet_mix * reverb_signal

    return out.astype(np.float32)
