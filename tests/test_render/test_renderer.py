"""Tests for MuJoCo video renderer."""

import pytest
import numpy as np
import mujoco

from physonic.render import (
    CameraConfig,
    CameraPreset,
    RenderConfig,
    VideoRenderer
)
from physonic.contact.rig import ObjectSpec, MultiSceneSpec, build_multi_xml


@pytest.fixture
def simple_model():
    """Create a simple MuJoCo model for testing."""
    # Create a scene with a single sphere
    spec = MultiSceneSpec(
        kind="test",
        objects=[
            ObjectSpec("sphere1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5))
        ],
        surface="steel"
    )

    xml, registry = build_multi_xml(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    return model, data, registry


class TestVideoRenderer:
    """Test VideoRenderer class."""

    def test_create_renderer(self, simple_model):
        """Test creating video renderer."""
        model, _, _ = simple_model

        config = RenderConfig(
            width=320,
            height=240,
            fps=30,
            cameras=[CameraPreset.fixed_front()]
        )

        with VideoRenderer(model, config) as renderer:
            assert len(renderer.renderers) == 1
            assert "front" in renderer.renderers
            assert "front" in renderer.cameras

    def test_create_multi_camera_renderer(self, simple_model):
        """Test creating renderer with multiple cameras."""
        model, _, _ = simple_model

        config = RenderConfig(
            width=320,
            height=240,
            fps=30,
            cameras=[
                CameraPreset.fixed_front(),
                CameraPreset.fixed_side(),
                CameraPreset.fixed_top()
            ]
        )

        with VideoRenderer(model, config) as renderer:
            assert len(renderer.renderers) == 3
            assert "front" in renderer.renderers
            assert "side" in renderer.renderers
            assert "top" in renderer.renderers

    def test_capture_frame(self, simple_model):
        """Test capturing a single frame."""
        model, data, _ = simple_model

        config = RenderConfig(
            width=320,
            height=240,
            fps=30,
            cameras=[CameraPreset.fixed_front()]
        )

        with VideoRenderer(model, config) as renderer:
            # Step simulation
            mujoco.mj_step(model, data)

            # Capture frame
            frames = renderer.capture(data)

            assert "front" in frames
            assert frames["front"].shape == (240, 320, 3)
            assert frames["front"].dtype == np.uint8

    def test_capture_multi_camera(self, simple_model):
        """Test capturing from multiple cameras."""
        model, data, _ = simple_model

        config = RenderConfig(
            width=160,
            height=120,
            fps=30,
            cameras=[
                CameraPreset.fixed_front(),
                CameraPreset.fixed_side()
            ]
        )

        with VideoRenderer(model, config) as renderer:
            mujoco.mj_step(model, data)
            frames = renderer.capture(data)

            assert len(frames) == 2
            assert "front" in frames
            assert "side" in frames
            assert frames["front"].shape == (120, 160, 3)
            assert frames["side"].shape == (120, 160, 3)

    def test_tracking_camera(self, simple_model):
        """Test tracking camera follows object."""
        model, data, _ = simple_model

        # Create tracking camera for sphere1
        config = RenderConfig(
            width=160,
            height=120,
            fps=30,
            cameras=[CameraPreset.tracking("sphere1")]
        )

        with VideoRenderer(model, config) as renderer:
            # Get initial object position
            body_id = model.body("sphere1").id
            initial_pos = data.xpos[body_id].copy()

            # Simulate to move object (sphere falls)
            for _ in range(100):
                mujoco.mj_step(model, data)

            # Capture frame
            frames = renderer.capture(data)

            # Object should have moved
            final_pos = data.xpos[body_id]
            assert not np.allclose(initial_pos, final_pos)

            # Camera should have updated lookat
            camera_config = renderer.camera_configs["tracking_sphere1"]
            np.testing.assert_array_almost_equal(camera_config.lookat, final_pos, decimal=5)

    def test_render_from_telemetry(self, simple_model):
        """Test post-hoc rendering from telemetry."""
        model, data, registry = simple_model

        # Simulate and collect telemetry
        physics_times = []
        sphere_positions = []

        for i in range(50):  # 50 steps at 5kHz = 0.01s
            mujoco.mj_step(model, data)
            physics_times.append(data.time)
            body_id = model.body("sphere1").id
            sphere_positions.append(data.xpos[body_id].copy())

        telemetry = {
            "t": np.array(physics_times),
            "xpos_sphere1": np.array(sphere_positions)
        }

        # Render from telemetry
        config = RenderConfig(
            width=160,
            height=120,
            fps=30,  # Should get 1 frame for ~0.01s duration
            cameras=[CameraPreset.fixed_front()]
        )

        with VideoRenderer(model, config) as renderer:
            frames_dict = renderer.render_from_telemetry(
                telemetry,
                registry=registry,
                duration=0.01
            )

        assert "front" in frames_dict
        assert len(frames_dict["front"]) == 1  # 1 frame for 0.01s at 30fps
        assert frames_dict["front"][0].shape == (120, 160, 3)

    def test_render_from_telemetry_multi_frame(self, simple_model):
        """Test rendering multiple frames from telemetry."""
        model, data, registry = simple_model

        # Simulate longer duration
        physics_times = []
        sphere_positions = []

        # Simulate 0.1s at 5kHz = 500 steps
        for i in range(500):
            mujoco.mj_step(model, data)
            physics_times.append(data.time)
            body_id = model.body("sphere1").id
            sphere_positions.append(data.xpos[body_id].copy())

        telemetry = {
            "t": np.array(physics_times),
            "xpos_sphere1": np.array(sphere_positions)
        }

        # Render at 30 fps
        config = RenderConfig(
            width=160,
            height=120,
            fps=30,
            cameras=[CameraPreset.fixed_front()]
        )

        with VideoRenderer(model, config) as renderer:
            frames_dict = renderer.render_from_telemetry(
                telemetry,
                registry=registry,
                duration=0.1
            )

        # 0.1s at 30fps = 3 frames (t=0, 1/30, 2/30)
        assert len(frames_dict["front"]) == 3

    def test_render_multi_camera_from_telemetry(self, simple_model):
        """Test rendering multiple cameras from telemetry."""
        model, data, registry = simple_model

        # Quick simulation
        physics_times = []
        sphere_positions = []

        for i in range(100):
            mujoco.mj_step(model, data)
            physics_times.append(data.time)
            body_id = model.body("sphere1").id
            sphere_positions.append(data.xpos[body_id].copy())

        telemetry = {
            "t": np.array(physics_times),
            "xpos_sphere1": np.array(sphere_positions)
        }

        # Multi-camera config
        config = RenderConfig(
            width=160,
            height=120,
            fps=30,
            cameras=[
                CameraPreset.fixed_front(),
                CameraPreset.fixed_side(),
                CameraPreset.fixed_top()
            ]
        )

        with VideoRenderer(model, config) as renderer:
            frames_dict = renderer.render_from_telemetry(
                telemetry,
                registry=registry,
                duration=0.02
            )

        # Should have frames from all cameras
        assert "front" in frames_dict
        assert "side" in frames_dict
        assert "top" in frames_dict

        # 0.02s at 30fps = 1 frame
        assert len(frames_dict["front"]) == 1
        assert len(frames_dict["side"]) == 1
        assert len(frames_dict["top"]) == 1


@pytest.mark.integration
class TestRendererIntegration:
    """Integration tests for renderer with real simulations."""

    def test_render_falling_sphere(self, simple_model):
        """Test rendering a falling sphere simulation."""
        model, data, registry = simple_model

        # Simulate falling sphere
        physics_times = []
        sphere_positions = []

        # Simulate 0.5 seconds at 5kHz
        for i in range(2500):
            mujoco.mj_step(model, data)
            if i % 10 == 0:  # Sample every 10th step
                physics_times.append(data.time)
                body_id = model.body("sphere1").id
                sphere_positions.append(data.xpos[body_id].copy())

        telemetry = {
            "t": np.array(physics_times),
            "xpos_sphere1": np.array(sphere_positions)
        }

        # Render at 30 fps
        config = RenderConfig(
            width=320,
            height=240,
            fps=30,
            cameras=[
                CameraPreset.fixed_front(),
                CameraPreset.tracking("sphere1")
            ]
        )

        with VideoRenderer(model, config) as renderer:
            frames_dict = renderer.render_from_telemetry(
                telemetry,
                registry=registry,
                duration=0.5
            )

        # 0.5s at 30fps = 15 frames
        assert len(frames_dict["front"]) == 15
        assert len(frames_dict["tracking_sphere1"]) == 15

        # All frames should be valid
        for frame in frames_dict["front"]:
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8
            # Check frame is not all black or all white
            assert np.mean(frame) > 0
            assert np.mean(frame) < 255

    def test_different_resolutions(self, simple_model):
        """Test rendering at different resolutions."""
        model, data, _ = simple_model

        resolutions = [
            (640, 480),
            (320, 240),
            (160, 120),
            (80, 60)
        ]

        for width, height in resolutions:
            config = RenderConfig(
                width=width,
                height=height,
                fps=30,
                cameras=[CameraPreset.fixed_front()]
            )

            with VideoRenderer(model, config) as renderer:
                mujoco.mj_step(model, data)
                frames = renderer.capture(data)

                assert frames["front"].shape == (height, width, 3)
