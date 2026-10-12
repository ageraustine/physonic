"""Visualization utilities for audio and RL metrics.

Provides plotting functions for:
- Audio waveforms
- Spectrograms and mel-spectrograms
- RL metrics (rewards, confusion matrices, etc.)
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int = 44100,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot audio waveform.

    Args:
        audio: Audio array (mono or stereo)
        sample_rate: Sample rate in Hz
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle stereo
    if audio.ndim == 2:
        # Plot both channels
        time = np.arange(len(audio)) / sample_rate
        ax.plot(time, audio[:, 0], label="Left", alpha=0.7)
        ax.plot(time, audio[:, 1], label="Right", alpha=0.7)
        ax.legend()
    else:
        # Mono
        time = np.arange(len(audio)) / sample_rate
        ax.plot(time, audio, color='blue', alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot audio spectrogram.

    Args:
        audio: Audio array (mono)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length for STFT
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from scipy import signal

    # Compute STFT
    if audio.ndim == 2:
        audio = audio[:, 0]  # Use first channel if stereo

    f, t, Sxx = signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
    )

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Power (dB)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mel-spectrogram (from RL environment observations).

    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_steps) or (n_mels, time_steps, channels)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Handle stereo mel-specs (n_mels, time, 2)
    if mel_spec.ndim == 3:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        for i, ax in enumerate(axes):
            im = ax.imshow(
                mel_spec[:, :, i],
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest'
            )
            ax.set_ylabel(f'Mel Band ({"Left" if i == 0 else "Right"})')
            fig.colorbar(im, ax=ax, label='Power (dB)')

        axes[1].set_xlabel('Time Frame')
        axes[0].set_title(title)
    else:
        # Mono
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('Mel Band')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Power (dB)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics(
    rewards: List[float],
    title: str = "Episode Rewards",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot RL training metrics.

    Args:
        rewards: List of rewards per episode
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.6, label='Raw')

    # Add rolling average
    if len(rewards) > 10:
        window = min(50, len(rewards) // 10)
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], rolling_mean, 'r-', linewidth=2, label=f'Rolling Mean ({window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot confusion matrix for classification tasks.

    Args:
        cm: Confusion matrix (n_classes, n_classes)
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            percentage = cm_normalized[i, j] * 100
            text = f'{count}\n({percentage:.1f}%)'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_episode_summary(
    recording_path: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comprehensive episode summary.

    Creates a multi-panel figure showing:
    - Mel-spectrogram observation
    - Rewards over time
    - Actions taken
    - Key metrics

    Args:
        recording_path: Path to episode recording
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    from physonic.visualization.recorder import EpisodeRecorder

    # Load episode
    recorder = EpisodeRecorder()
    recording = recorder.load_episode(recording_path)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Mel-spectrogram (if available)
    if recording.observations:
        ax1 = fig.add_subplot(gs[0, :])
        obs = recording.observations[0]
        if obs.ndim >= 2:
            # Assume first observation is representative
            if obs.ndim == 3:
                obs_plot = obs[:, :, 0]  # First channel
            else:
                obs_plot = obs
            im = ax1.imshow(obs_plot, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title("First Observation (Mel-Spectrogram)")
            ax1.set_ylabel("Mel Band")
            ax1.set_xlabel("Time Frame")
            fig.colorbar(im, ax=ax1)

    # 2. Rewards over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(recording.rewards, 'o-', markersize=4)
    ax2.set_title("Rewards per Step")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)

    # 3. Actions over time
    ax3 = fig.add_subplot(gs[1, 1])
    actions_array = np.array(recording.actions)
    if actions_array.ndim == 1:
        ax3.plot(actions_array, 'o-', markersize=4)
    else:
        # Multi-dimensional actions
        ax3.plot(actions_array, 'o-', markersize=3, alpha=0.7)
    ax3.set_title("Actions per Step")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Action")
    ax3.grid(True, alpha=0.3)

    # 4. Metrics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    metrics_data = [
        ["Episode ID", recording.episode_id],
        ["Total Steps", len(recording.actions)],
        ["Total Reward", f"{sum(recording.rewards):.3f}"],
        ["Mean Reward", f"{np.mean(recording.rewards):.3f}"],
        ["Min Reward", f"{min(recording.rewards):.3f}"],
        ["Max Reward", f"{max(recording.rewards):.3f}"],
    ]

    # Add environment-specific metrics from last info
    if recording.infos:
        last_info = recording.infos[-1]
        for key, value in last_info.items():
            if key not in ["audio", "observation"]:  # Skip large arrays
                if isinstance(value, float):
                    metrics_data.append([key, f"{value:.3f}"])
                else:
                    metrics_data.append([key, str(value)])

    table = ax4.table(cellText=metrics_data, colLabels=["Metric", "Value"],
                     cellLoc='left', loc='center', colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title("Episode Metrics", pad=20)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    print("Visualization Utilities Examples")
    print("=" * 60)

    # Example 1: Plot waveform
    print("\nExample 1: Waveform plotting")
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    fig = plot_waveform(audio, sample_rate, title="440 Hz Sine Wave",
                       save_path="outputs/test_waveform.png")
    print("  Saved waveform plot to outputs/test_waveform.png")
    plt.close(fig)

    # Example 2: Plot spectrogram
    print("\nExample 2: Spectrogram plotting")
    # Chirp signal
    chirp = np.sin(2 * np.pi * (100 + 200 * t) * t)
    fig = plot_spectrogram(chirp, sample_rate, title="Chirp Signal",
                          save_path="outputs/test_spectrogram.png")
    print("  Saved spectrogram plot to outputs/test_spectrogram.png")
    plt.close(fig)

    # Example 3: Plot mel-spectrogram
    print("\nExample 3: Mel-spectrogram plotting")
    # Simulated mel-spec
    mel_spec = np.random.randn(64, 87, 2) * 10 + 40  # Random mel-spec
    fig = plot_mel_spectrogram(mel_spec, title="Random Mel-Spectrogram",
                              save_path="outputs/test_mel_spec.png")
    print("  Saved mel-spectrogram plot to outputs/test_mel_spec.png")
    plt.close(fig)

    # Example 4: Plot training metrics
    print("\nExample 4: Training metrics plotting")
    rewards = -np.abs(np.random.randn(200)) + np.linspace(-1, -0.2, 200)  # Improving rewards
    fig = plot_metrics(rewards, title="Training Progress",
                      save_path="outputs/test_metrics.png")
    print("  Saved metrics plot to outputs/test_metrics.png")
    plt.close(fig)

    # Example 5: Plot confusion matrix
    print("\nExample 5: Confusion matrix plotting")
    cm = np.array([[45, 3, 1, 1], [2, 48, 0, 0], [1, 0, 47, 2], [0, 1, 2, 47]])
    labels = ["steel", "glass", "wood", "rubber"]
    fig = plot_confusion_matrix(cm, labels, title="Material Classification",
                                save_path="outputs/test_confusion_matrix.png")
    print("  Saved confusion matrix to outputs/test_confusion_matrix.png")
    plt.close(fig)

    print("\nDone! Check outputs/ directory for generated plots.")
