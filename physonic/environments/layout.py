"""Grid-based layout generation for procedural environments.

This module provides utilities for placing objects in environments with:
- Collision detection
- Placement constraints (spacing, clearance, density)
- Random position finding
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class LayoutConstraints:
    """Constraints for procedural layout generation.

    Attributes:
        min_spacing: Minimum space between objects (meters)
        wall_clearance: Minimum distance from walls (meters)
        aisle_width: Minimum aisle width for navigation (meters)
        max_density: Maximum fraction of floor space that can be occupied (0.0-1.0)
    """
    min_spacing: float = 0.5
    wall_clearance: float = 0.3
    aisle_width: float = 1.5
    max_density: float = 0.4


class LayoutGrid:
    """Grid-based layout for placing objects with collision avoidance.

    The grid divides the floor into cells and tracks which cells are occupied.
    This enables fast collision detection and placement constraints.

    Attributes:
        room_size: (width, depth) in meters
        grid_resolution: Size of each grid cell in meters
        occupied: 2D boolean array indicating occupied cells
        constraints: Layout constraints for placement
    """

    def __init__(
        self,
        room_size: Tuple[float, float],
        grid_resolution: float = 0.5,
        constraints: Optional[LayoutConstraints] = None,
    ):
        """Initialize layout grid.

        Args:
            room_size: (width, depth) of room in meters
            grid_resolution: Size of each grid cell in meters (default: 0.5m)
            constraints: Layout constraints (default: LayoutConstraints())
        """
        self.room_width, self.room_depth = room_size
        self.grid_resolution = grid_resolution
        self.constraints = constraints or LayoutConstraints()

        # Create grid
        self.grid_width = int(np.ceil(self.room_width / grid_resolution))
        self.grid_depth = int(np.ceil(self.room_depth / grid_resolution))
        self.occupied = np.zeros((self.grid_width, self.grid_depth), dtype=bool)

        # Apply wall clearance
        self._apply_wall_clearance()

    def _apply_wall_clearance(self):
        """Mark grid cells near walls as occupied."""
        clearance_cells = int(np.ceil(self.constraints.wall_clearance / self.grid_resolution))

        # Mark perimeter cells as occupied
        if clearance_cells > 0:
            self.occupied[:clearance_cells, :] = True  # Left wall
            self.occupied[-clearance_cells:, :] = True  # Right wall
            self.occupied[:, :clearance_cells] = True  # Front wall
            self.occupied[:, -clearance_cells:] = True  # Back wall

    def _world_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates.

        Args:
            position: (x, y) in meters (world coordinates)

        Returns:
            (grid_x, grid_y) grid cell indices
        """
        grid_x = int(position[0] / self.grid_resolution)
        grid_y = int(position[1] / self.grid_resolution)
        return grid_x, grid_y

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (cell center).

        Args:
            grid_pos: (grid_x, grid_y) grid cell indices

        Returns:
            (x, y) in meters (world coordinates, cell center)
        """
        x = (grid_pos[0] + 0.5) * self.grid_resolution
        y = (grid_pos[1] + 0.5) * self.grid_resolution
        return x, y

    def can_place(
        self,
        position: Tuple[float, float],
        size: Tuple[float, float],
        clearance: Optional[float] = None,
    ) -> bool:
        """Check if object can be placed at position without collision.

        Args:
            position: (x, y) world position (center of object)
            size: (width, depth) of object in meters
            clearance: Additional spacing around object (default: min_spacing from constraints)

        Returns:
            True if object can be placed, False if collision
        """
        if clearance is None:
            clearance = self.constraints.min_spacing

        width, depth = size
        # Include clearance in size check
        total_width = width + 2 * clearance
        total_depth = depth + 2 * clearance

        # Get object bounds in world coordinates
        x, y = position
        x_min = x - total_width / 2
        x_max = x + total_width / 2
        y_min = y - total_depth / 2
        y_max = y + total_depth / 2

        # Check room bounds
        if x_min < 0 or x_max > self.room_width or y_min < 0 or y_max > self.room_depth:
            return False

        # Convert to grid coordinates
        grid_x_min, grid_y_min = self._world_to_grid((x_min, y_min))
        grid_x_max, grid_y_max = self._world_to_grid((x_max, y_max))

        # Clamp to grid bounds
        grid_x_min = max(0, grid_x_min)
        grid_x_max = min(self.grid_width - 1, grid_x_max)
        grid_y_min = max(0, grid_y_min)
        grid_y_max = min(self.grid_depth - 1, grid_y_max)

        # Check if any cells are occupied
        region = self.occupied[grid_x_min:grid_x_max+1, grid_y_min:grid_y_max+1]
        return not np.any(region)

    def place_object(
        self,
        position: Tuple[float, float],
        size: Tuple[float, float],
    ):
        """Mark grid cells as occupied by object.

        Args:
            position: (x, y) world position (center of object)
            size: (width, depth) of object in meters
        """
        width, depth = size
        x, y = position

        # Get object bounds
        x_min = x - width / 2
        x_max = x + width / 2
        y_min = y - depth / 2
        y_max = y + depth / 2

        # Convert to grid coordinates
        grid_x_min, grid_y_min = self._world_to_grid((x_min, y_min))
        grid_x_max, grid_y_max = self._world_to_grid((x_max, y_max))

        # Clamp to grid bounds
        grid_x_min = max(0, grid_x_min)
        grid_x_max = min(self.grid_width - 1, grid_x_max)
        grid_y_min = max(0, grid_y_min)
        grid_y_max = min(self.grid_depth - 1, grid_y_max)

        # Mark cells as occupied
        self.occupied[grid_x_min:grid_x_max+1, grid_y_min:grid_y_max+1] = True

    def find_free_position(
        self,
        size: Tuple[float, float],
        max_attempts: int = 100,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[float, float]]:
        """Find random free position for object of given size.

        Uses random search with collision checking.

        Args:
            size: (width, depth) of object in meters
            max_attempts: Maximum number of random attempts
            rng: Random number generator (default: np.random.default_rng())

        Returns:
            (x, y) position if found, None if no free position after max_attempts
        """
        if rng is None:
            rng = np.random.default_rng()

        width, depth = size

        for _ in range(max_attempts):
            # Random position in room
            x = rng.uniform(width/2, self.room_width - width/2)
            y = rng.uniform(depth/2, self.room_depth - depth/2)
            position = (x, y)

            if self.can_place(position, size):
                return position

        return None  # No free position found

    def get_occupancy(self) -> float:
        """Get current occupancy as fraction of total grid.

        Returns:
            Fraction of grid cells that are occupied (0.0-1.0)
        """
        total_cells = self.grid_width * self.grid_depth
        occupied_cells = np.sum(self.occupied)
        return occupied_cells / total_cells

    def is_full(self) -> bool:
        """Check if grid has reached maximum density.

        Returns:
            True if occupancy >= max_density constraint
        """
        return self.get_occupancy() >= self.constraints.max_density

    def place_row(
        self,
        start_position: Tuple[float, float],
        object_size: Tuple[float, float],
        num_objects: int,
        direction: Tuple[float, float] = (1, 0),
        spacing: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Place objects in a row with uniform spacing.

        Args:
            start_position: (x, y) starting position for first object
            object_size: (width, depth) of each object
            num_objects: Number of objects to place in row
            direction: (dx, dy) unit vector for row direction (default: (1, 0) = along X)
            spacing: Distance between object centers (default: size + aisle_width)

        Returns:
            List of (x, y) positions successfully placed (may be < num_objects if collisions)
        """
        if spacing is None:
            # Default spacing: object size + aisle width
            spacing = max(object_size[0], object_size[1]) + self.constraints.aisle_width

        # Normalize direction
        dir_length = np.sqrt(direction[0]**2 + direction[1]**2)
        dir_x = direction[0] / dir_length
        dir_y = direction[1] / dir_length

        placed_positions = []
        for i in range(num_objects):
            # Calculate position
            x = start_position[0] + i * spacing * dir_x
            y = start_position[1] + i * spacing * dir_y
            position = (x, y)

            # Check and place
            if self.can_place(position, object_size):
                self.place_object(position, object_size)
                placed_positions.append(position)
            else:
                break  # Stop if collision

        return placed_positions

    def place_grid_pattern(
        self,
        start_position: Tuple[float, float],
        object_size: Tuple[float, float],
        num_rows: int,
        num_cols: int,
        row_spacing: Optional[float] = None,
        col_spacing: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Place objects in a grid pattern.

        Args:
            start_position: (x, y) starting position for first object
            object_size: (width, depth) of each object
            num_rows: Number of rows
            num_cols: Number of columns
            row_spacing: Distance between rows (default: depth + aisle_width)
            col_spacing: Distance between columns (default: width + min_spacing)

        Returns:
            List of (x, y) positions successfully placed
        """
        if row_spacing is None:
            row_spacing = object_size[1] + self.constraints.aisle_width
        if col_spacing is None:
            col_spacing = object_size[0] + self.constraints.min_spacing

        placed_positions = []

        for row in range(num_rows):
            for col in range(num_cols):
                x = start_position[0] + col * col_spacing
                y = start_position[1] + row * row_spacing
                position = (x, y)

                if self.can_place(position, object_size):
                    self.place_object(position, object_size)
                    placed_positions.append(position)

        return placed_positions


# Example usage
if __name__ == "__main__":
    print("Layout Grid Examples")
    print("=" * 60)

    # Example 1: Basic placement
    print("\nExample 1: Basic object placement")
    grid = LayoutGrid(room_size=(10, 10), grid_resolution=0.5)

    # Try to place a 1x1 object at center
    position = (5, 5)
    size = (1.0, 1.0)

    if grid.can_place(position, size):
        grid.place_object(position, size)
        print(f"  Placed object at {position}, size {size}")
        print(f"  Occupancy: {grid.get_occupancy():.1%}")
    else:
        print(f"  Cannot place object at {position}")

    # Example 2: Find random positions
    print("\nExample 2: Random placement")
    grid2 = LayoutGrid(room_size=(20, 15))
    rng = np.random.default_rng(42)

    for i in range(10):
        pos = grid2.find_free_position((2.0, 1.0), rng=rng)
        if pos:
            grid2.place_object(pos, (2.0, 1.0))
            print(f"  Placed object {i+1} at ({pos[0]:.2f}, {pos[1]:.2f})")
        else:
            print(f"  No free position for object {i+1}")

    print(f"  Final occupancy: {grid2.get_occupancy():.1%}")

    # Example 3: Row placement
    print("\nExample 3: Row of server racks")
    grid3 = LayoutGrid(room_size=(20, 15))

    rack_size = (0.6, 1.0)  # Server rack dimensions
    start_pos = (2.0, 3.0)

    positions = grid3.place_row(
        start_position=start_pos,
        object_size=rack_size,
        num_objects=8,
        direction=(1, 0),  # Along X axis
        spacing=1.5,
    )

    print(f"  Placed {len(positions)} racks in row")
    for i, pos in enumerate(positions):
        print(f"    Rack {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")

    # Example 4: Grid pattern
    print("\nExample 4: Grid pattern (warehouse shelves)")
    grid4 = LayoutGrid(room_size=(30, 20))

    shelf_size = (1.2, 0.5)
    positions = grid4.place_grid_pattern(
        start_position=(2.0, 2.0),
        object_size=shelf_size,
        num_rows=4,
        num_cols=6,
        row_spacing=3.0,  # 3m between rows (wide aisles)
        col_spacing=2.0,  # 2m between columns
    )

    print(f"  Placed {len(positions)} shelves in grid (target: 24)")
    print(f"  Final occupancy: {grid4.get_occupancy():.1%}")

    print("\nDone!")
