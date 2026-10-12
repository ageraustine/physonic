"""Tests for camera configuration and presets."""

import pytest
import numpy as np

from physonic.render.camera import (
    CameraConfig,
    RenderConfig,
    CameraPreset,
    get_camera_mode
)


class TestCameraConfig:
    """Test CameraConfig dataclass."""

    def test_create_fixed_camera(self):
        """Test creating a fixed camera configuration."""
        cam = CameraConfig(
            name="front",
            type="fixed",
            lookat=np.array([0.0, 0.0, 0.25]),
            distance=2.0,
            azimuth=0.0,
            elevation=-15.0,
            fov=45.0
        )

        assert cam.name == "front"
        assert cam.type == "fixed"
        np.testing.assert_array_equal(cam.lookat, [0.0, 0.0, 0.25])
        assert cam.distance == 2.0
        assert cam.azimuth == 0.0
        assert cam.elevation == -15.0
        assert cam.fov == 45.0
        assert cam.track_object is None

    def test_create_tracking_camera(self):
        """Test creating a tracking camera configuration."""
        cam = CameraConfig(
            name="tracking_obj",
            type="tracking",
            lookat=np.array([0.0, 0.0, 0.25]),
            distance=1.5,
            azimuth=45.0,
            elevation=-20.0,
            fov=45.0,
            track_object="sphere1"
        )

        assert cam.name == "tracking_obj"
        assert cam.type == "tracking"
        assert cam.track_object == "sphere1"

    def test_invalid_camera_type(self):
        """Test that invalid camera type raises error."""
        with pytest.raises(ValueError, match="Camera type must be"):
            CameraConfig(
                name="bad",
                type="invalid",
                lookat=np.array([0.0, 0.0, 0.0]),
                distance=1.0,
                azimuth=0.0,
                elevation=0.0
            )

    def test_tracking_without_object(self):
        """Test that tracking camera without object raises error."""
        with pytest.raises(ValueError, match="track_object"):
            CameraConfig(
                name="tracking",
                type="tracking",
                lookat=np.array([0.0, 0.0, 0.0]),
                distance=1.0,
                azimuth=0.0,
                elevation=0.0
            )

    def test_invalid_lookat_shape(self):
        """Test that invalid lookat shape raises error."""
        with pytest.raises(ValueError, match="lookat must be 3D"):
            CameraConfig(
                name="bad",
                type="fixed",
                lookat=np.array([0.0, 0.0]),  # Only 2D
                distance=1.0,
                azimuth=0.0,
                elevation=0.0
            )

    def test_negative_distance(self):
        """Test that negative distance raises error."""
        with pytest.raises(ValueError, match="distance must be positive"):
            CameraConfig(
                name="bad",
                type="fixed",
                lookat=np.array([0.0, 0.0, 0.0]),
                distance=-1.0,
                azimuth=0.0,
                elevation=0.0
            )

    def test_invalid_fov(self):
        """Test that invalid FOV raises error."""
        with pytest.raises(ValueError, match="fov must be in"):
            CameraConfig(
                name="bad",
                type="fixed",
                lookat=np.array([0.0, 0.0, 0.0]),
                distance=1.0,
                azimuth=0.0,
                elevation=0.0,
                fov=200.0  # Too large
            )

    def test_lookat_list_conversion(self):
        """Test that lookat as list is converted to numpy array."""
        cam = CameraConfig(
            name="test",
            type="fixed",
            lookat=[1.0, 2.0, 3.0],  # List, not array
            distance=1.0,
            azimuth=0.0,
            elevation=0.0
        )

        assert isinstance(cam.lookat, np.ndarray)
        np.testing.assert_array_equal(cam.lookat, [1.0, 2.0, 3.0])


class TestRenderConfig:
    """Test RenderConfig dataclass."""

    def test_create_config(self):
        """Test creating a render configuration."""
        cam = CameraPreset.fixed_front()
        config = RenderConfig(
            width=640,
            height=480,
            fps=30,
            cameras=[cam]
        )

        assert config.width == 640
        assert config.height == 480
        assert config.fps == 30
        assert len(config.cameras) == 1

    def test_default_values(self):
        """Test default values."""
        cam = CameraPreset.fixed_front()
        config = RenderConfig(cameras=[cam])

        assert config.width == 640
        assert config.height == 480
        assert config.fps == 30

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        cam = CameraPreset.fixed_front()
        with pytest.raises(ValueError, match="width and height must be positive"):
            RenderConfig(width=-1, height=480, cameras=[cam])

    def test_invalid_fps(self):
        """Test that invalid FPS raises error."""
        cam = CameraPreset.fixed_front()
        with pytest.raises(ValueError, match="fps must be positive"):
            RenderConfig(fps=-1, cameras=[cam])

    def test_no_cameras(self):
        """Test that config without cameras raises error."""
        with pytest.raises(ValueError, match="Must specify at least one camera"):
            RenderConfig(cameras=[])


class TestCameraPreset:
    """Test CameraPreset factory methods."""

    def test_fixed_front(self):
        """Test fixed front camera preset."""
        cam = CameraPreset.fixed_front()

        assert cam.name == "front"
        assert cam.type == "fixed"
        assert cam.azimuth == 0.0
        assert cam.elevation == -15.0
        assert cam.distance == 2.0
        assert cam.fov == 45.0
        np.testing.assert_array_equal(cam.lookat, [0.0, 0.0, 0.25])

    def test_fixed_front_custom_center(self):
        """Test fixed front with custom scene center."""
        center = np.array([1.0, 2.0, 3.0])
        cam = CameraPreset.fixed_front(center)

        np.testing.assert_array_equal(cam.lookat, center)

    def test_fixed_side(self):
        """Test fixed side camera preset."""
        cam = CameraPreset.fixed_side()

        assert cam.name == "side"
        assert cam.type == "fixed"
        assert cam.azimuth == 90.0
        assert cam.elevation == -15.0

    def test_fixed_top(self):
        """Test fixed top camera preset."""
        cam = CameraPreset.fixed_top()

        assert cam.name == "top"
        assert cam.type == "fixed"
        assert cam.elevation == 89.0  # Nearly straight down
        assert cam.distance == 2.5

    def test_tracking(self):
        """Test tracking camera preset."""
        cam = CameraPreset.tracking("sphere1")

        assert cam.name == "tracking_sphere1"
        assert cam.type == "tracking"
        assert cam.track_object == "sphere1"
        assert cam.distance == 1.5
        assert cam.azimuth == 45.0
        assert cam.elevation == -20.0

    def test_tracking_custom_params(self):
        """Test tracking camera with custom parameters."""
        cam = CameraPreset.tracking(
            "obj",
            distance=3.0,
            azimuth=90.0,
            elevation=-30.0
        )

        assert cam.distance == 3.0
        assert cam.azimuth == 90.0
        assert cam.elevation == -30.0

    def test_egocentric(self):
        """Test egocentric camera preset."""
        cam = CameraPreset.egocentric("robot")

        assert cam.name == "egocentric_robot"
        assert cam.type == "tracking"
        assert cam.track_object == "robot"
        assert cam.distance == 0.01  # Very close
        assert cam.fov == 90.0  # Wide FOV


class TestGetCameraMode:
    """Test get_camera_mode function."""

    def test_front_mode(self):
        """Test 'front' camera mode."""
        cameras = get_camera_mode("front")

        assert len(cameras) == 1
        assert cameras[0].name == "front"
        assert cameras[0].type == "fixed"

    def test_multi_mode(self):
        """Test 'multi' camera mode."""
        cameras = get_camera_mode("multi")

        assert len(cameras) == 3
        names = [cam.name for cam in cameras]
        assert "front" in names
        assert "side" in names
        assert "top" in names

    def test_tracking_mode(self):
        """Test 'tracking' camera mode."""
        cameras = get_camera_mode("tracking", track_object="sphere1")

        assert len(cameras) == 1
        assert cameras[0].type == "tracking"
        assert cameras[0].track_object == "sphere1"

    def test_tracking_without_object(self):
        """Test that tracking mode without object raises error."""
        with pytest.raises(ValueError, match="track_object"):
            get_camera_mode("tracking")

    def test_egocentric_mode(self):
        """Test 'egocentric' camera mode."""
        cameras = get_camera_mode("egocentric", track_object="robot")

        assert len(cameras) == 1
        assert cameras[0].type == "tracking"
        assert cameras[0].track_object == "robot"
        assert cameras[0].fov == 90.0

    def test_egocentric_without_object(self):
        """Test that egocentric mode without object raises error."""
        with pytest.raises(ValueError, match="track_object"):
            get_camera_mode("egocentric")

    def test_unknown_mode(self):
        """Test that unknown mode raises error."""
        with pytest.raises(ValueError, match="Unknown camera mode"):
            get_camera_mode("invalid")

    def test_custom_scene_center(self):
        """Test modes with custom scene center."""
        center = np.array([1.0, 0.0, 0.5])
        cameras = get_camera_mode("multi", scene_center=center)

        for cam in cameras:
            np.testing.assert_array_equal(cam.lookat, center)
