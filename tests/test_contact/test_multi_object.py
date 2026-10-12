"""Unit tests for multi-object physics scenarios."""
import numpy as np
import pytest

from physonic.contact.rig import (
    ObjectSpec, MultiSceneSpec, build_multi_xml, simulate_multi, extract_events_multi
)
from physonic.contact.scenarios import (
    scenario_two_sphere_collision, scenario_stack_collapse, scenario_sphere_cascade
)


class TestObjectSpec:
    """Tests for ObjectSpec dataclass."""

    def test_object_spec_creation(self):
        """Test creating an ObjectSpec."""
        obj = ObjectSpec(
            name="test_sphere",
            shape="sphere",
            material="steel",
            mass=0.2,
            size=0.03,
            initial_pos=(0, 0, 0.5)
        )

        assert obj.name == "test_sphere"
        assert obj.shape == "sphere"
        assert obj.material == "steel"
        assert obj.mass == 0.2
        assert obj.size == 0.03
        assert obj.initial_pos == (0, 0, 0.5)
        assert obj.initial_vel is None

    def test_object_spec_with_velocity(self):
        """Test ObjectSpec with initial velocity."""
        obj = ObjectSpec(
            "box1", "box", "wood", 0.5,
            (0.05, 0.05, 0.02),
            (0, 0, 0.3),
            initial_vel=(1.0, 0, 0)
        )

        assert obj.initial_vel == (1.0, 0, 0)


class TestMultiSceneSpec:
    """Tests for MultiSceneSpec dataclass."""

    def test_multi_scene_spec_creation(self):
        """Test creating a MultiSceneSpec."""
        objects = [
            ObjectSpec("sphere1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5)),
            ObjectSpec("box1", "box", "wood", 0.3, (0.05, 0.05, 0.02), (0.1, 0, 0.2))
        ]

        spec = MultiSceneSpec(
            kind="test",
            objects=objects,
            surface="steel",
            duration=2.0,
            seed=42
        )

        assert spec.kind == "test"
        assert len(spec.objects) == 2
        assert spec.surface == "steel"
        assert spec.duration == 2.0
        assert spec.seed == 42

    def test_multi_scene_name_property(self):
        """Test that scene name is generated correctly."""
        spec = MultiSceneSpec(
            kind="collision",
            objects=[ObjectSpec("s1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5))],
            surface="wood"
        )

        assert spec.name == "collision_wood_1obj"


class TestBuildMultiXML:
    """Tests for build_multi_xml function."""

    def test_build_xml_single_sphere(self):
        """Test XML generation for a single sphere."""
        spec = MultiSceneSpec(
            kind="test",
            objects=[
                ObjectSpec("sphere1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5))
            ],
            surface="steel"
        )

        xml, registry = build_multi_xml(spec)

        # Check XML contains expected elements
        assert '<mujoco>' in xml
        assert 'name="sphere1"' in xml
        assert 'type="sphere"' in xml
        assert 'size="0.03"' in xml
        assert 'name="surface"' in xml

        # Check registry
        assert "sphere1" in registry
        assert registry["sphere1"]["mass"] == 0.2
        assert registry["sphere1"]["material"] == "steel"
        assert registry["sphere1"]["shape"] == "sphere"

    def test_build_xml_multiple_objects(self):
        """Test XML generation for multiple objects."""
        spec = MultiSceneSpec(
            kind="test",
            objects=[
                ObjectSpec("box1", "box", "wood", 0.5, (0.05, 0.05, 0.02), (0, 0, 0.1)),
                ObjectSpec("sphere1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5)),
            ],
            surface="rubber"
        )

        xml, registry = build_multi_xml(spec)

        # Check both objects in XML
        assert 'name="box1"' in xml
        assert 'name="sphere1"' in xml
        assert 'type="box"' in xml
        assert 'type="sphere"' in xml

        # Check both in registry
        assert "box1" in registry
        assert "sphere1" in registry
        assert len(registry) == 2

    def test_build_xml_different_shapes(self):
        """Test XML generation for different geometric shapes."""
        shapes_to_test = [
            ("sphere", "sphere", 0.03),
            ("box", "box", (0.05, 0.05, 0.02)),
            ("capsule", "capsule", (0.02, 0.04)),
            ("cylinder", "cylinder", (0.02, 0.04)),
        ]

        for i, (name, shape, size) in enumerate(shapes_to_test):
            spec = MultiSceneSpec(
                kind="test",
                objects=[
                    ObjectSpec(f"{name}_obj", shape, "steel", 0.2, size, (0, 0, 0.5))
                ],
                surface="steel"
            )

            xml, registry = build_multi_xml(spec)

            assert f'type="{shape}"' in xml
            assert f'name="{name}_obj"' in xml
            assert f"{name}_obj" in registry


class TestSimulateMulti:
    """Tests for simulate_multi function."""

    def test_simulate_single_sphere_drop(self):
        """Test simulation with a single sphere dropping."""
        spec = MultiSceneSpec(
            kind="drop",
            objects=[
                ObjectSpec("sphere1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5))
            ],
            surface="steel",
            duration=1.0
        )

        telemetry, registry = simulate_multi(spec)

        # Should have sphere-surface contact
        assert ("sphere1", "surface") in telemetry or ("surface", "sphere1") in telemetry

        # Get the contact pair
        pair_key = ("sphere1", "surface") if ("sphere1", "surface") in telemetry else ("surface", "sphere1")
        tele = telemetry[pair_key]

        # Check telemetry structure
        assert "t" in tele
        assert "fn" in tele
        assert "ft" in tele
        assert "pos" in tele

        # Should have contact data
        assert len(tele["t"]) > 0
        assert len(tele["fn"]) > 0

    def test_simulate_two_spheres(self):
        """Test simulation with two spheres that should collide."""
        spec = MultiSceneSpec(
            kind="collision",
            objects=[
                ObjectSpec("s1", "sphere", "steel", 0.2, 0.03, (-0.2, 0, 0.5), (2, 0, 0)),
                ObjectSpec("s2", "sphere", "glass", 0.2, 0.03, (0.2, 0, 0.5), (-2, 0, 0)),
            ],
            surface="steel",
            duration=1.5
        )

        telemetry, registry = simulate_multi(spec)

        # Should have at least surface contacts
        pair_keys = set(telemetry.keys())

        # Normalize keys (order doesn't matter)
        normalized_keys = {tuple(sorted(pair)) for pair in pair_keys}

        # Check registry
        assert "s1" in registry
        assert "s2" in registry
        assert "geom_id" in registry["s1"]


    def test_simulate_stack(self):
        """Test simulation with stacked boxes."""
        spec = scenario_stack_collapse(surface="wood", seed=42)

        telemetry, registry = simulate_multi(spec)

        # Should have 3 boxes
        assert len(registry) == 3
        assert all(f"box{i+1}" in registry for i in range(3))

        # Should have some contacts (at minimum with surface)
        assert len(telemetry) > 0


class TestExtractEventsMulti:
    """Tests for extract_events_multi function."""

    def test_extract_events_single_drop(self):
        """Test event extraction for a single sphere drop."""
        spec = MultiSceneSpec(
            kind="drop",
            objects=[
                ObjectSpec("sphere1", "sphere", "steel", 0.2, 0.03, (0, 0, 0.5))
            ],
            surface="steel",
            duration=1.5
        )

        telemetry, registry = simulate_multi(spec)
        events_by_pair = extract_events_multi(telemetry, registry)

        # Should have events
        assert len(events_by_pair) > 0

        # Find sphere-surface pair
        sphere_surface_pair = None
        for pair in events_by_pair:
            if "sphere1" in pair and "surface" in pair:
                sphere_surface_pair = pair
                break

        assert sphere_surface_pair is not None
        events = events_by_pair[sphere_surface_pair]

        # Should have at least one impact
        assert len(events) >= 1

        # Check event structure
        ev = events[0]
        assert "t" in ev
        assert "J" in ev
        assert "obj1" in ev
        assert "obj2" in ev
        assert "contact_pos" in ev
        assert "sphere1" in [ev["obj1"], ev["obj2"]]
        assert "surface" in [ev["obj1"], ev["obj2"]]

    def test_extract_events_two_spheres(self):
        """Test event extraction for two-sphere collision."""
        spec = scenario_two_sphere_collision(seed=42)

        telemetry, registry = simulate_multi(spec)
        events_by_pair = extract_events_multi(telemetry, registry)

        # Aggregate all events
        all_events = []
        for events in events_by_pair.values():
            all_events.extend(events)

        # Should have multiple events
        assert len(all_events) > 0

        # Check all events have required fields
        for ev in all_events:
            assert "obj1" in ev
            assert "obj2" in ev
            assert "t" in ev
            assert "J" in ev
            assert "contact_pos" in ev


class TestScenarios:
    """Tests for predefined scenarios."""

    def test_scenario_two_sphere_collision(self):
        """Test two-sphere collision scenario."""
        spec = scenario_two_sphere_collision(surface="steel", seed=42)

        assert spec.kind == "collision"
        assert len(spec.objects) == 2
        assert spec.surface == "steel"
        assert spec.duration == 1.5

    def test_scenario_stack_collapse(self):
        """Test stack collapse scenario."""
        spec = scenario_stack_collapse(surface="wood", seed=0)

        assert spec.kind == "stack_collapse"
        assert len(spec.objects) == 3
        assert spec.surface == "wood"

    def test_scenario_sphere_cascade(self):
        """Test sphere cascade with custom count."""
        spec = scenario_sphere_cascade(n_spheres=7, seed=42)

        assert spec.kind == "sphere_cascade"
        assert len(spec.objects) == 7

        # Check alternating materials
        for i, obj in enumerate(spec.objects):
            expected_mat = "steel" if i % 2 == 0 else "glass"
            assert obj.material == expected_mat


@pytest.mark.integration
class TestMultiObjectIntegration:
    """Integration tests for complete multi-object pipeline."""

    def test_full_pipeline_two_spheres(self):
        """Test complete pipeline from spec to events."""
        # Create spec
        spec = scenario_two_sphere_collision(seed=42)

        # Simulate
        telemetry, registry = simulate_multi(spec)

        # Extract events
        events_by_pair = extract_events_multi(telemetry, registry)

        # Validate
        assert len(registry) == 2
        assert len(telemetry) > 0
        assert len(events_by_pair) > 0

        # Check all events are labeled
        for events in events_by_pair.values():
            for ev in events:
                assert ev["obj1"] in list(registry.keys()) + ["surface"]
                assert ev["obj2"] in list(registry.keys()) + ["surface"]

    def test_full_pipeline_cascade(self):
        """Test complete pipeline with sphere cascade."""
        spec = scenario_sphere_cascade(n_spheres=5, surface="steel", seed=0)

        telemetry, registry = simulate_multi(spec)
        events_by_pair = extract_events_multi(telemetry, registry)

        # Should have 5 spheres
        assert len(registry) == 5

        # Aggregate events
        total_events = sum(len(evs) for evs in events_by_pair.values())

        # Should have multiple impacts (each sphere bounces several times)
        assert total_events >= 5  # At least one impact per sphere
