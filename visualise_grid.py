import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# Import your classes here if this file is separate, 
# or paste this class into your main script.
# Assuming the classes (Warehouse, Robot, CellType, etc.) are available or imported.
# For this standalone example, I will assume the classes exist in the namespace 
# or I will mock the necessary imports from the provided code for the demo to run.

class WarehouseVisualizer:
    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.fig, self.ax = None, None
        
        # Define colors for the grid elements
        # 0: Empty (White)
        # 1: Shelf (Tan/Brown)
        # 2: Charging Station (Green)
        # 3: Loading Dock (Dark Gray)
        # 4: Obstacle (Black)
        self.colors = ['#FFFFFF', '#D2691E', '#98FB98', '#708090', '#000000']
        self.cmap = ListedColormap(self.colors)
        self.norm = BoundaryNorm([0, 1, 2, 3, 4, 5], self.cmap.N)

    def render(self, step_number=None, show=True):
        """
        Renders the current state of the warehouse.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.ax.clear()
        
        # 1. Plot the Grid Layout
        # We assume warehouse.grid is (height, width)
        # imshow renders (row, col), which matches (y, x) if origin is upper
        self.ax.imshow(self.warehouse.grid, cmap=self.cmap, norm=self.norm, origin='upper')

        # 2. Plot the Robots
        for robot in self.warehouse.robots:
            self._plot_robot(robot)

        # 3. Aesthetics
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)
        self.ax.set_xticks(np.arange(-0.5, self.warehouse.width, 1))
        self.ax.set_yticks(np.arange(-0.5, self.warehouse.height, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis='both', which='both', length=0)
        
        title = "Warehouse Simulation"
        if step_number is not None:
            title += f" - Step {step_number}"
        self.ax.set_title(title)

        # 4. Create Legend
        legend_patches = [
            mpatches.Patch(color=self.colors[0], label='Empty'),
            mpatches.Patch(color=self.colors[1], label='Shelf'),
            mpatches.Patch(color=self.colors[2], label='Charging Stn'),
            mpatches.Patch(color=self.colors[3], label='Loading Dock'),
            mpatches.Patch(color='blue', label='Robot (Idle/Mov)'),
            mpatches.Patch(color='red', label='Robot (Carrying)'),
            mpatches.Patch(color='gold', label='Robot (Charging)'),
        ]
        self.ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)

    def _plot_robot(self, robot):
        """Helper to plot a single robot with state-based styling"""
        
        # Determine Color based on state
        if robot.state == "charging":
            color = 'gold'
        elif robot.carrying_item:
            color = 'red'
        else:
            color = 'dodgerblue'
            
        # Draw Robot Circle
        # Note: In matplotlib imshow, x is column index, y is row index.
        # Ensure robot.position.x maps to horizontal and y to vertical.
        circle = plt.Circle((robot.position.x, robot.position.y), 0.4, color=color, zorder=10)
        self.ax.add_patch(circle)
        
        # Add ID Text
        self.ax.text(robot.position.x, robot.position.y, str(robot.id),
                     color='white', ha='center', va='center', 
                     fontweight='bold', fontsize=8, zorder=11)
        
        # Optional: Draw target line if moving
        if robot.target and robot.path:
            # Draw a faint line showing the path
            path_x = [p.x for p in robot.path]
            path_y = [p.y for p in robot.path]
            # Add current pos to start of line
            path_x.insert(0, robot.position.x)
            path_y.insert(0, robot.position.y)
            
            self.ax.plot(path_x, path_y, color=color, linestyle=':', linewidth=1, alpha=0.5)


# ==========================================
# DEMONSTRATION CODE
# This block mimics your starter code to make this file runnable 
# ==========================================

if __name__ == "__main__":
    import random
    from enum import Enum
    from dataclasses import dataclass
    from typing import List, Optional

    # Configuration
    NUM_ROBOTS = 10
    WAREHOUSE_W = 30
    WAREHOUSE_H = 30

    # --- Minimal mock of your classes to make the visualizer run standalone ---
    class CellType(Enum):
        EMPTY = 0
        SHELF = 1
        CHARGING_STATION = 2
        LOADING_DOCK = 3
        OBSTACLE = 4

    @dataclass
    class Position:
        x: int
        y: int
        def __eq__(self, other): return self.x == other.x and self.y == other.y

    @dataclass
    class Robot:
        id: int
        position: Position
        state: str = "idle"
        carrying_item: bool = False
        target: Optional[Position] = None
        path: List[Position] = None
        battery: float = 100.0

    class Warehouse:
        def __init__(self, width=20, height=20):
            self.width = width
            self.height = height
            self.grid = np.zeros((height, width), dtype=int)
            self.robots = []
            self._create_layout()

        def _create_layout(self):
            # Same layout logic as your starter code
            for i in range(2, self.height - 2, 3):
                for j in range(2, self.width - 2, 3):
                    self.grid[i, j] = CellType.SHELF.value
            
            # Extra shelves
            for i in [3, 6, 9]:
                for j in [4, 7, 10]:
                    if i < self.height and j < self.width:
                        self.grid[i, j] = CellType.SHELF.value

            # Loading docks: top middle, right middle, bottom middle, left middle
            dock_positions = [
                (0, self.width // 2),              # Top middle
                (self.height // 2, self.width - 1), # Right middle
                (self.height - 1, self.width // 2), # Bottom middle
                (self.height // 2, 0),              # Left middle
            ]
            for y, x in dock_positions:
                if y < self.height and x < self.width:
                    self.grid[y, x] = CellType.LOADING_DOCK.value

            # Charging stations in all 4 corners
            charging_positions = [
                (0, 0),                          # Top-left corner
                (0, self.width - 1),             # Top-right corner
                (self.height - 1, 0),            # Bottom-left corner
                (self.height - 1, self.width - 1), # Bottom-right corner
            ]
            for y, x in charging_positions:
                if y < self.height and x < self.width:
                    self.grid[y, x] = CellType.CHARGING_STATION.value

    # --- Setup and Run Visualization ---
    
    # 1. Create Environment
    wh = Warehouse(width=WAREHOUSE_W, height=WAREHOUSE_H)
    
    # 2. Add robots based on config
    print(f"Initializing {NUM_ROBOTS} robots...")
    for i in range(NUM_ROBOTS):
        # Find a random valid spot (not a shelf)
        while True:
            rx = random.randint(0, WAREHOUSE_W - 1)
            ry = random.randint(0, WAREHOUSE_H - 1)
            
            if wh.grid[ry, rx] != CellType.SHELF.value:
                # Check if this position is a charging station
                is_charging_station = (wh.grid[ry, rx] == CellType.CHARGING_STATION.value)
                
                # Determine allowed states based on location
                allowed_states = ["idle", "moving"]
                if is_charging_station:
                    allowed_states.append("charging")
                
                # Randomize state from allowed options
                state = random.choice(allowed_states)
                
                # If charging, they can't be carrying items (usually)
                carrying = False
                if state != "charging":
                    carrying = random.choice([True, False])
                
                # Mock path if moving
                path = []
                target = None
                if state == "moving":
                    target = Position(rx, ry) # dummy target
                    path = [Position(rx, ry)] 

                robot = Robot(
                    id=i, 
                    position=Position(rx, ry), 
                    state=state, 
                    carrying_item=carrying,
                    target=target,
                    path=path
                )
                wh.robots.append(robot)
                break

    # 3. Visualize
    print(f"Generating Warehouse Visual ({WAREHOUSE_W}x{WAREHOUSE_H})...")
    viz = WarehouseVisualizer(wh)
    viz.render()