"""
Standalone CLI tool for rendering videos from saved Physonic episodes.

Usage:
    # Single camera
    python3 -m physonic.render_video \\
        --input outputs/mono/drop_steel_on_glass \\
        --camera front \\
        --output outputs/mono/drop_steel_on_glass_front.mp4

    # Multi-camera
    python3 -m physonic.render_video \\
        --input outputs/mono/drop_steel_on_glass \\
        --cameras front,side,top \\
        --output outputs/mono/videos/

    # Tracking mode
    python3 -m physonic.render_video \\
        --input outputs/multi/cascade \\
        --camera tracking \\
        --track-object sphere1 \\
        --output outputs/multi/cascade_tracking.mp4
"""

import argparse
import sys
from pathlib import Path
import mujoco

from physonic.export.episode import load_episode
from physonic.contact.rig import (
    build_xml,
    build_multi_xml,
    reconstruct_scene_spec,
    reconstruct_multi_scene_spec
)
from physonic.render import (
    CameraPreset,
    RenderConfig,
    VideoRenderer,
    encode_video,
    get_camera_mode
)


def render_episode_video(
    episode_dir: Path,
    camera_mode: str,
    output_path: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    track_object: str = None,
    scene_center: tuple = None,
    scenario_name: str = None
):
    """Render video from episode telemetry.

    Args:
        episode_dir: Path to episode outputs directory
        camera_mode: Camera mode ("front", "multi", "tracking", "egocentric")
        output_path: Output video path or directory (for multi-camera)
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        track_object: Object name to track (for tracking/egocentric modes)
        scene_center: Optional scene center for fixed cameras (x, y, z)
        scenario_name: Optional scenario name (for flat directory format)

    Returns:
        List of output video paths created
    """
    print(f"Loading episode from {episode_dir}...")
    episode = load_episode(episode_dir, scenario_name=scenario_name)

    print(f"Episode: {episode.name}")
    print(f"  Duration: {episode.duration:.2f}s")
    print(f"  Spatial: {episode.is_spatial}")
    print(f"  Multi-object: {episode.is_multi}")

    # Reconstruct MuJoCo model
    print("Reconstructing MuJoCo model...")
    if episode.is_multi:
        # Multi-object scenario
        spec = reconstruct_multi_scene_spec(
            episode.name,
            episode.registry,
            episode.duration
        )
        xml, registry = build_multi_xml(spec)

        # Use first object as default tracking target
        if track_object is None and camera_mode in ("tracking", "egocentric"):
            track_object = list(registry.keys())[0]
            print(f"  Using default tracking object: {track_object}")
    else:
        # Single object scenario
        spec = reconstruct_scene_spec(episode.name, episode.duration)
        xml = build_xml(spec)
        registry = None

        # For single-object, track "obj"
        if track_object is None and camera_mode in ("tracking", "egocentric"):
            track_object = "obj"

    model = mujoco.MjModel.from_xml_string(xml)
    print(f"  Model created with {model.nbody} bodies, {model.ngeom} geoms")

    # Configure cameras
    if scene_center is None:
        scene_center = (0.0, 0.0, 0.25)  # Default scene center

    cameras = get_camera_mode(
        camera_mode,
        scene_center=scene_center,
        track_object=track_object
    )

    config = RenderConfig(
        width=width,
        height=height,
        fps=fps,
        cameras=cameras
    )

    print(f"Rendering with {len(cameras)} camera(s)...")
    for cam in cameras:
        print(f"  - {cam.name}: {cam.type} camera")

    # Render frames from telemetry
    with VideoRenderer(model, config) as renderer:
        frames_dict = renderer.render_from_telemetry(
            episode.telemetry,
            registry=registry,
            duration=episode.duration
        )

    # Encode videos
    output_paths = []

    if len(cameras) == 1:
        # Single camera - save to specified output path
        cam_name = cameras[0].name
        frames = frames_dict[cam_name]

        print(f"Encoding {len(frames)} frames to {output_path}...")
        encode_video(frames, output_path, fps=fps)
        output_paths.append(output_path)

        print(f"✓ Video saved: {output_path}")

    else:
        # Multi-camera - save to separate files in output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = episode.name

        for cam_name, frames in frames_dict.items():
            video_path = output_dir / f"{base_name}_{cam_name}.mp4"

            print(f"Encoding {len(frames)} frames to {video_path}...")
            encode_video(frames, video_path, fps=fps)
            output_paths.append(video_path)

            print(f"✓ Video saved: {video_path}")

    return output_paths


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render videos from Physonic episode telemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single front camera
  python3 -m physonic.render_video \\
      --input outputs/mono/drop_steel_on_glass \\
      --camera front \\
      --output video.mp4

  # Multi-camera (front, side, top)
  python3 -m physonic.render_video \\
      --input outputs/mono/drop_steel_on_glass \\
      --camera multi \\
      --output outputs/mono/videos/

  # Tracking camera following object
  python3 -m physonic.render_video \\
      --input outputs/multi/cascade \\
      --camera tracking \\
      --track-object sphere1 \\
      --output cascade_tracking.mp4

  # Custom resolution and frame rate
  python3 -m physonic.render_video \\
      --input outputs/mono/drop_steel_on_glass \\
      --camera front \\
      --width 1280 --height 720 --fps 60 \\
      --output video_hd.mp4
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to episode directory (e.g., outputs/mono/drop_steel_on_glass)"
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=["front", "multi", "tracking", "egocentric"],
        help="Camera mode (default: front)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output video path (.mp4) or directory (for multi-camera)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width in pixels (default: 640)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height in pixels (default: 480)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )

    parser.add_argument(
        "--track-object",
        type=str,
        help="Object name to track (for tracking/egocentric modes)"
    )

    parser.add_argument(
        "--scene-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Scene center coordinates for fixed cameras (default: 0 0 0.25)"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name (for flat directory format, e.g., 'drop_steel_on_glass')"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Parse scene center
    scene_center = None
    if args.scene_center:
        scene_center = tuple(args.scene_center)

    # Render video
    try:
        output_paths = render_episode_video(
            episode_dir=args.input,
            camera_mode=args.camera,
            output_path=args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            track_object=args.track_object,
            scene_center=scene_center,
            scenario_name=args.scenario
        )

        print(f"\n✓ Successfully rendered {len(output_paths)} video(s)")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
