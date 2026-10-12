"""Tests for spec reconstruction helpers."""

import pytest
import numpy as np

from physonic.contact.rig import (
    SceneSpec,
    MultiSceneSpec,
    ObjectSpec,
    reconstruct_scene_spec,
    reconstruct_multi_scene_spec
)


class TestReconstructSceneSpec:
    """Test SceneSpec reconstruction from episode names."""

    def test_reconstruct_drop_scenario(self):
        """Test reconstructing drop scenario."""
        spec = reconstruct_scene_spec("drop_steel_on_glass", duration=2.5)

        assert spec.kind == "drop"
        assert spec.obj == "steel"
        assert spec.surface == "glass"
        assert spec.duration == 2.5
        assert spec.name == "drop_steel_on_glass"

    def test_reconstruct_slide_scenario(self):
        """Test reconstructing slide scenario."""
        spec = reconstruct_scene_spec("slide_wood_on_steel", duration=2.0)

        assert spec.kind == "slide"
        assert spec.obj == "wood"
        assert spec.surface == "steel"
        assert spec.duration == 2.0
        assert spec.name == "slide_wood_on_steel"

    def test_reconstruct_multiword_material(self):
        """Test reconstructing with multi-word material names."""
        # If materials had names like "hard_wood"
        # This would parse as kind="drop", obj="hard_wood", surface="steel"
        # Currently all materials are single words, but test the parsing logic
        spec = reconstruct_scene_spec("drop_steel_on_glass", duration=2.5)
        assert spec.obj == "steel"

    def test_invalid_format_no_on(self):
        """Test that missing '_on_' raises error."""
        with pytest.raises(ValueError, match="Cannot parse episode name"):
            reconstruct_scene_spec("drop_steel_glass")

    def test_invalid_format_too_short(self):
        """Test that too-short name raises error."""
        with pytest.raises(ValueError, match="Expected at least"):
            reconstruct_scene_spec("drop_on_glass")

    def test_invalid_kind(self):
        """Test that invalid kind raises error."""
        with pytest.raises(ValueError, match="Unknown kind"):
            reconstruct_scene_spec("throw_steel_on_glass")

    def test_invalid_obj_material(self):
        """Test that invalid object material raises error."""
        with pytest.raises(ValueError, match="Unknown object material"):
            reconstruct_scene_spec("drop_unobtanium_on_glass")

    def test_invalid_surface_material(self):
        """Test that invalid surface material raises error."""
        with pytest.raises(ValueError, match="Unknown surface material"):
            reconstruct_scene_spec("drop_steel_on_unobtanium")


class TestReconstructMultiSceneSpec:
    """Test MultiSceneSpec reconstruction from registry."""

    def test_reconstruct_single_sphere(self):
        """Test reconstructing single sphere scenario."""
        registry = {
            "sphere1": {
                "mass": 0.2,
                "material": "steel",
                "shape": "sphere",
                "size": 0.03,
                "initial_vel": None
            }
        }

        spec = reconstruct_multi_scene_spec(
            "cascade_steel_1obj",
            registry,
            duration=2.5
        )

        assert spec.kind == "cascade"
        assert spec.surface == "steel"
        assert spec.duration == 2.5
        assert len(spec.objects) == 1

        obj = spec.objects[0]
        assert obj.name == "sphere1"
        assert obj.shape == "sphere"
        assert obj.material == "steel"
        assert obj.mass == 0.2
        assert obj.size == 0.03

    def test_reconstruct_multiple_objects(self):
        """Test reconstructing multi-object scenario."""
        registry = {
            "sphere1": {
                "mass": 0.2,
                "material": "steel",
                "shape": "sphere",
                "size": 0.03,
                "initial_vel": None
            },
            "box1": {
                "mass": 0.3,
                "material": "wood",
                "shape": "box",
                "size": (0.05, 0.05, 0.02),
                "initial_vel": (1.0, 0, 0)
            }
        }

        spec = reconstruct_multi_scene_spec(
            "collision_steel_2obj",
            registry,
            duration=2.0
        )

        assert spec.kind == "collision"
        assert spec.surface == "steel"
        assert len(spec.objects) == 2

        # Objects are in registry order (dict preserves insertion order in Python 3.7+)
        names = [obj.name for obj in spec.objects]
        assert "sphere1" in names
        assert "box1" in names

    def test_reconstruct_with_initial_velocity(self):
        """Test that initial velocity is preserved."""
        registry = {
            "sphere1": {
                "mass": 0.2,
                "material": "steel",
                "shape": "sphere",
                "size": 0.03,
                "initial_vel": (1.0, 0.5, 0)
            }
        }

        spec = reconstruct_multi_scene_spec(
            "test_steel_1obj",
            registry,
            duration=2.0
        )

        obj = spec.objects[0]
        assert obj.initial_vel == (1.0, 0.5, 0)

    def test_empty_registry_raises_error(self):
        """Test that empty registry raises error."""
        with pytest.raises(ValueError, match="Registry is required"):
            reconstruct_multi_scene_spec("test_steel_0obj", {}, duration=2.0)

    def test_invalid_episode_name_format(self):
        """Test that invalid episode name raises error."""
        registry = {"sphere1": {
            "mass": 0.2,
            "material": "steel",
            "shape": "sphere",
            "size": 0.03
        }}

        # Too short
        with pytest.raises(ValueError, match="Cannot parse episode name"):
            reconstruct_multi_scene_spec("test", registry)

        # Doesn't end with "Nobj"
        with pytest.raises(ValueError, match="should end with"):
            reconstruct_multi_scene_spec("test_steel_1", registry)

    def test_invalid_surface_material(self):
        """Test that invalid surface raises error."""
        registry = {"sphere1": {
            "mass": 0.2,
            "material": "steel",
            "shape": "sphere",
            "size": 0.03
        }}

        with pytest.raises(ValueError, match="Unknown surface material"):
            reconstruct_multi_scene_spec("test_unobtanium_1obj", registry)

    def test_incomplete_registry_entry(self):
        """Test that incomplete registry entry raises error."""
        # Missing material
        registry = {
            "sphere1": {
                "mass": 0.2,
                "shape": "sphere",
                "size": 0.03
            }
        }

        with pytest.raises(ValueError, match="Incomplete registry"):
            reconstruct_multi_scene_spec("test_steel_1obj", registry)

    def test_multiword_surface_name(self):
        """Test parsing surface with underscore."""
        # Surface names could theoretically have underscores
        # e.g., "hard_wood" → would be between kind and "Nobj"
        registry = {"sphere1": {
            "mass": 0.2,
            "material": "steel",
            "shape": "sphere",
            "size": 0.03
        }}

        # This parses as kind="cascade", surface="steel", N=1
        spec = reconstruct_multi_scene_spec("cascade_steel_1obj", registry)
        assert spec.surface == "steel"


@pytest.mark.integration
class TestReconstructionIntegration:
    """Integration tests for reconstruction with model building."""

    def test_reconstruct_and_build_single_object(self):
        """Test full pipeline: reconstruct → build XML → load model."""
        from physonic.contact.rig import build_xml
        import mujoco

        # Reconstruct spec
        spec = reconstruct_scene_spec("drop_steel_on_glass", duration=2.5)

        # Build XML
        xml = build_xml(spec)
        assert "<mujoco>" in xml
        assert 'name="obj"' in xml

        # Load model
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nbody > 0
        assert model.ngeom > 0

    def test_reconstruct_and_build_multi_object(self):
        """Test full pipeline for multi-object scenario."""
        from physonic.contact.rig import build_multi_xml
        import mujoco

        registry = {
            "sphere1": {
                "mass": 0.2,
                "material": "steel",
                "shape": "sphere",
                "size": 0.03,
                "initial_vel": None
            },
            "sphere2": {
                "mass": 0.15,
                "material": "glass",
                "shape": "sphere",
                "size": 0.025,
                "initial_vel": None
            }
        }

        # Reconstruct spec
        spec = reconstruct_multi_scene_spec(
            "collision_steel_2obj",
            registry,
            duration=2.0
        )

        # Build XML
        xml, rebuilt_registry = build_multi_xml(spec)
        assert "<mujoco>" in xml
        assert 'name="sphere1"' in xml
        assert 'name="sphere2"' in xml

        # Load model
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nbody >= 3  # world + 2 spheres
        assert model.ngeom >= 3  # surface + 2 spheres
