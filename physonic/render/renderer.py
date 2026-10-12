"""
MuJoCo video renderer with multi-camera support.

This module provides the VideoRenderer class for capturing RGB frames from
MuJoCo simulations. Supports both real-time capture during simulation and
post-hoc rendering from saved telemetry.
"""

from typing import Dict, List, Optional
import numpy as np
import mujoco

from physonic.render.camera import CameraConfig, RenderConfig


class VideoRenderer:
    """Renders MuJoCo simulation to video frames.

    Manages multiple camera viewpoints and provides both real-time capture
    and post-hoc rendering from telemetry replay.

    Example:
        # Real-time capture
        config = RenderConfig(width=640, height=480, fps=30, cameras=[...])
        with VideoRenderer(model, config) as renderer:
            for _ in range(100):
                mujoco.mj_step(model, data)
                frames = renderer.capture(data)  # {camera_name: RGB frame}

        # Post-hoc rendering from telemetry
        with VideoRenderer(model, config) as renderer:
            frames_dict = renderer.render_from_telemetry(
                telemetry={"t": [...], "xpos": [...]},
                duration=2.5
            )
    """

    def __init__(self, model: mujoco.MjModel, config: RenderConfig):
        """Initialize renderer with MuJoCo model and configuration.

        Args:
            model: MuJoCo model to render
            config: Rendering configuration with cameras and resolution

        Raises:
            ValueError: If configuration is invalid
        """
        self.model = model
        self.config = config

        # Create one renderer per camera for offscreen rendering
        self.renderers: Dict[str, mujoco.Renderer] = {}
        for cam_config in config.cameras:
            renderer = mujoco.Renderer(model, config.height, config.width)
            self.renderers[cam_config.name] = renderer

        # Store camera configurations
        self.camera_configs = {cam.name: cam for cam in config.cameras}

        # Camera for each viewpoint
        self.cameras: Dict[str, mujoco.MjvCamera] = {}

        for cam_config in config.cameras:
            # Create and configure camera
            camera = mujoco.MjvCamera()
            self._update_camera(camera, cam_config)
            self.cameras[cam_config.name] = camera

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()

    def close(self):
        """Release rendering resources."""
        # MuJoCo Renderer handles cleanup automatically
        pass

    def _update_camera(self, camera: mujoco.MjvCamera, config: CameraConfig):
        """Update MjvCamera from CameraConfig.

        Converts our CameraConfig (lookat, distance, azimuth, elevation)
        to MuJoCo's camera representation.

        Args:
            camera: MuJoCo camera to update
            config: Camera configuration
        """
        # Set camera type to free (not tracking a body)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Set lookat point
        camera.lookat[:] = config.lookat

        # Set distance from lookat
        camera.distance = config.distance

        # Set azimuth and elevation (in degrees)
        camera.azimuth = config.azimuth
        camera.elevation = config.elevation

    def _update_tracking_cameras(self, data: mujoco.MjData):
        """Update lookat positions for tracking cameras.

        For tracking cameras, update the lookat point to follow the
        tracked object's current position.

        Args:
            data: Current MuJoCo simulation state
        """
        for cam_name, cam_config in self.camera_configs.items():
            if cam_config.type == "tracking" and cam_config.track_object:
                # Get object body ID
                try:
                    body_id = self.model.body(cam_config.track_object).id
                    # Update lookat to object position
                    cam_config.lookat[:] = data.xpos[body_id]
                    # Update MuJoCo camera
                    self._update_camera(self.cameras[cam_name], cam_config)
                except KeyError:
                    # Object not found - keep previous lookat
                    pass

    def capture(self, data: mujoco.MjData) -> Dict[str, np.ndarray]:
        """Capture frame(s) from current simulation state.

        Args:
            data: Current MuJoCo simulation state

        Returns:
            Dictionary mapping camera name to RGB frame (H, W, 3) uint8 array

        Example:
            frames = renderer.capture(data)
            front_frame = frames["front"]  # (480, 640, 3) uint8
        """
        # Update tracking cameras to current object positions
        self._update_tracking_cameras(data)

        frames = {}
        for cam_name in self.renderers:
            # Update scene with current data and render
            # The renderer.update_scene() will call mjv_updateScene internally
            self.renderers[cam_name].update_scene(
                data,
                self.cameras[cam_name]
            )

            # Get RGB pixels (H, W, 3) uint8
            pixels = self.renderers[cam_name].render()
            frames[cam_name] = pixels

        return frames

    def render_from_telemetry(
        self,
        telemetry: Dict,
        registry: Optional[Dict] = None,
        duration: Optional[float] = None
    ) -> Dict[str, List[np.ndarray]]:
        """Post-hoc rendering: replay telemetry and capture frames.

        Reconstructs simulation state from saved telemetry and renders
        frames at target fps. Useful for generating videos after simulation
        is complete.

        Args:
            telemetry: Dictionary with physics telemetry
                Required keys:
                - "t": Time array (s)
                - "xpos": Object positions array (for single object) or
                         "xpos_obj1", "xpos_obj2", ... (for multi-object)
            registry: Optional multi-object registry with object info
            duration: Optional duration override (uses max time if None)

        Returns:
            Dictionary mapping camera name to list of RGB frames
            {camera_name: [frame0, frame1, ...]}

        Example:
            frames = renderer.render_from_telemetry(
                telemetry={"t": times, "xpos": positions},
                duration=2.5
            )
            # Save frames to video
            for cam_name, frame_list in frames.items():
                encode_video(frame_list, f"{cam_name}.mp4")
        """
        # Extract time array
        physics_times = np.array(telemetry["t"])

        # Determine duration
        if duration is None:
            duration = float(physics_times[-1])

        # Generate frame times at target fps
        frame_times = np.arange(0, duration, 1.0 / self.config.fps)

        # Create MjData for rendering (will be updated per frame)
        data = mujoco.MjData(self.model)

        # Storage for frames from each camera
        all_frames: Dict[str, List[np.ndarray]] = {
            name: [] for name in self.renderers
        }

        # Render each frame
        for frame_t in frame_times:
            # Find nearest physics timestep
            idx = np.argmin(np.abs(physics_times - frame_t))

            # Reconstruct object position(s) in data.xpos
            self._reconstruct_state(data, telemetry, idx, registry)

            # Forward kinematics to update full state
            mujoco.mj_forward(self.model, data)

            # Capture frame from all cameras
            frames = self.capture(data)

            # Store frames
            for cam_name, frame in frames.items():
                all_frames[cam_name].append(frame)

        return all_frames

    def _reconstruct_state(
        self,
        data: mujoco.MjData,
        telemetry: Dict,
        idx: int,
        registry: Optional[Dict]
    ):
        """Reconstruct simulation state at given telemetry index.

        Updates data.xpos with object positions from telemetry.

        Args:
            data: MuJoCo data to update
            telemetry: Telemetry dictionary
            idx: Time index to reconstruct
            registry: Optional multi-object registry
        """
        if registry is not None:
            # Multi-object scenario
            for obj_name, obj_info in registry.items():
                key = f"xpos_{obj_name}"
                if key in telemetry:
                    xpos_array = np.array(telemetry[key])
                    # Get body ID for this object
                    try:
                        body_id = self.model.body(obj_name).id
                        # Update position
                        data.xpos[body_id] = xpos_array[idx]
                    except KeyError:
                        # Object not in model
                        pass
        else:
            # Single object scenario
            if "xpos" in telemetry:
                xpos_array = np.array(telemetry["xpos"])
                # Assume primary object is named "obj"
                try:
                    body_id = self.model.body("obj").id
                    data.xpos[body_id] = xpos_array[idx]
                except (KeyError, IndexError):
                    # Fallback: just set position if possible
                    if len(xpos_array[idx]) == 3:
                        # Set first movable body
                        data.xpos[1] = xpos_array[idx]
