import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import os
import tempfile
import io
from typing import List, Dict
from collections import defaultdict
import imageio
HAS_IMAGEIO = True

from src.warehouse_simulator import Warehouse, WarehouseSimulator, Position as PositionBaseline
from src.warehouse_simulator_ML import Warehouse as WarehouseML, WarehouseSimulator as WarehouseSimulatorML, Position as PositionML
import src.warehouse_simulator as ws_baseline
import src.warehouse_simulator_ML as ws_ml
import src.config as config

# Configuration
NUM_ROBOTS = config.NUM_ROBOTS
WAREHOUSE_W = config.WAREHOUSE_WIDTH
WAREHOUSE_H = config.WAREHOUSE_HEIGHT
SEED = 3
STEPS_DAY = 24 * 60  # 24 hours * 60 steps/hour
STEPS_WEEK = 7 * 24 * 60  # 7 days * 24 hours/day * 60 steps/hour
MOTION_BLUR_FRAMES = 0  # Number of intermediate frames for motion blur (disabled)
PEAK_HOURS = config.PEAK_HOURS  # Peak hours from config

# Visualization warmup: Number of steps to run simulation before starting to capture frames
# Set to 0 to visualize from the start, or specify steps (e.g., 9 * STEPS_DAY for day 10)
WARMUP_STEPS = 0 * STEPS_DAY  # Default: Second week Wednesday (Day 10)

# Real-time display settings
SHOW_REALTIME = False  # Set to True to display animation in real-time as it's generated
SAVE_TO_GIF = True    # Set to True to save animation to GIF file
REALTIME_FPS = 60     # FPS for real-time display (lower = slower, easier to see)

# FPS settings
FPS_DAY = 30  # FPS for day animation
FPS_WEEK = 30  # FPS for week animation

# Calculate actual durations with 30 fps
# Day: 1440 steps → 1440/30 = 48 seconds
# Week: 10080 steps → 10080/30 = 336 seconds
ACTUAL_DURATION_DAY = STEPS_DAY / FPS_DAY
ACTUAL_DURATION_WEEK = STEPS_WEEK / FPS_WEEK

# Color mappings
ROBOT_COLORS = {
    "carrying": "red",
    "picking": "orange",
    "delivering": "pink",
    "charging": "green",
    "idle": "blue",
    "moving": "blue"  # Moving robots also blue (idle color)
}

# Cell type values (matching CellType enum)
CELL_TYPE_VALUES = {
    0: "white",      # EMPTY
    1: "brown",      # SHELF
    2: "lime",       # CHARGING_STATION
    3: "grey",       # LOADING_DOCK
    4: "black"       # OBSTACLE
}

def get_robot_color(robot) -> str:
    """Determine robot color based on state and carrying status"""
    if robot.state == "charging":
        return ROBOT_COLORS["charging"]
    elif robot.state == "picking":
        return ROBOT_COLORS["picking"]
    elif robot.state == "delivering":
        return ROBOT_COLORS["delivering"]
    elif robot.carrying_item:
        return ROBOT_COLORS["carrying"]
    else:
        return ROBOT_COLORS["idle"]

def get_time_info(step: int, is_week: bool = False, start_step: int = 0):
    """Calculate time of day, day number, and peak hour status from step number"""
    steps_per_hour = 60
    hours_per_day = 24
    total_steps_per_day = hours_per_day * steps_per_hour
    
    # Calculate absolute step from start of simulation
    absolute_step = start_step + step
    
    # Always calculate day from absolute step
    day = (absolute_step // total_steps_per_day) + 1
    step_in_day = absolute_step % total_steps_per_day
    
    hour = step_in_day // steps_per_hour
    minute = step_in_day % steps_per_hour
    
    # Format time as HH:MM
    time_str = f"{hour:02d}:{minute:02d}"
    
    # Check if peak hour
    is_peak = hour in PEAK_HOURS
    
    return day, hour, minute, time_str, is_peak

def visualize_frame(warehouse, step: int, title: str, ax, is_week=False, start_step: int = 0):
    """Draw a single frame of the warehouse visualization with optional motion blur"""
    ax.clear()
    
    # Get time information
    day, hour, minute, time_str, is_peak = get_time_info(step, is_week, start_step)
    
    # Draw grid cells
    for y in range(warehouse.height):
        for x in range(warehouse.width):
            cell_type = int(warehouse.grid[y, x])
            color = CELL_TYPE_VALUES.get(cell_type, "white")
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                               facecolor=color, edgecolor='lightgray', linewidth=0.5)
            ax.add_patch(rect)
    
    # Draw robots
    for robot in warehouse.robots:
        color = get_robot_color(robot)
        circle = Circle((robot.position.x, robot.position.y), 0.3, 
                       color=color, ec='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        # Add robot ID text
        ax.text(robot.position.x, robot.position.y, str(robot.id), 
               ha='center', va='center', fontsize=8, fontweight='bold', zorder=11)
    
    ax.set_xlim(-0.5, warehouse.width - 0.5)
    ax.set_ylim(-0.5, warehouse.height - 0.5)
    ax.set_aspect('equal')
    
    # Calculate average battery level
    battery_levels = [robot.battery for robot in warehouse.robots]
    avg_battery = np.mean(battery_levels) if battery_levels else 0.0
    
    # Create title with time, day, peak hour info, and battery level
    peak_indicator = " [PEAK HOUR]" if is_peak else ""
    battery_text = f" | Avg Battery: {avg_battery:.1f}%"
    if is_week:
        title_text = f"{title} - Day {day}, {time_str}{peak_indicator}{battery_text}"
    else:
        title_text = f"{title} - {time_str}{peak_indicator}{battery_text}"
    
    ax.set_title(title_text, fontsize=12, fontweight='bold', 
                color='red' if is_peak else 'black')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()  # Invert so (0,0) is top-left
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ROBOT_COLORS["idle"], edgecolor='black', label='Idle'),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROBOT_COLORS["carrying"], edgecolor='black', label='Carrying'),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROBOT_COLORS["picking"], edgecolor='black', label='Picking'),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROBOT_COLORS["delivering"], edgecolor='black', label='Delivering'),
        plt.Rectangle((0, 0), 1, 1, facecolor=ROBOT_COLORS["charging"], edgecolor='black', label='Charging'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

def create_animation(warehouse, simulator, steps: int, title: str, filename: str, is_week=False, start_step: int = 0):
    """Run simulation and create GIF animation with optional real-time display"""
    if SAVE_TO_GIF and not HAS_IMAGEIO:
        print(f"Warning: imageio not available, cannot save GIF. Showing real-time only.")
    
    # Determine FPS and calculate actual duration
    fps = FPS_WEEK if is_week else FPS_DAY
    total_frames = steps  # No motion blur, so one frame per step
    actual_duration = total_frames / fps
    
    print(f"Creating animation for {title} ({steps} steps, capturing every step)...")
    if start_step > 0:
        print(f"  Starting from step {start_step} (warmup phase)...")
    print(f"  GIF FPS: {fps}, Total frames: {total_frames}, Duration: ~{actual_duration:.1f}s")
    if SHOW_REALTIME:
        print(f"  Real-time display: Enabled @ {REALTIME_FPS} fps")
    if SAVE_TO_GIF:
        print(f"  Saving to GIF: Enabled")
    
    frames = [] if SAVE_TO_GIF else None
    
    # Enable interactive mode for real-time display
    if SHOW_REALTIME:
        plt.ion()
    
    # Create figure once and reuse it (major performance improvement)
    fig, ax = plt.subplots(figsize=(10, 10))
    if SHOW_REALTIME:
        plt.show(block=False)
    
    try:
        # Warmup: Run simulation up to start_step without capturing frames
        if start_step > 0:
            print(f"  Running warmup: {start_step} steps...")
            for warmup_step in range(start_step):
                simulator.step(dt=1.0)
                if warmup_step % 1000 == 0 and warmup_step > 0:
                    print(f"    Warmup step {warmup_step}/{start_step} ({100*warmup_step/start_step:.1f}%)...")
            print(f"  Warmup complete. Starting frame capture...")
        
        # Capture initial frame (step 0 relative to start_step)
        visualize_frame(warehouse, 0, title, ax, is_week=is_week, start_step=start_step)
        
        if SHOW_REALTIME:
            plt.draw()
            plt.pause(1.0 / REALTIME_FPS)
        
        if SAVE_TO_GIF and HAS_IMAGEIO:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            buf.close()
        
        for step in range(1, steps):
            # Run simulation step
            simulator.step(dt=1.0)
            
            # Capture frame for every step (reuse same figure/axis)
            visualize_frame(warehouse, step, title, ax, is_week=is_week, start_step=start_step)
            
            # Update real-time display
            if SHOW_REALTIME:
                plt.draw()
                plt.pause(1.0 / REALTIME_FPS)
            
            # Save frame for GIF if enabled
            if SAVE_TO_GIF and HAS_IMAGEIO:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                frame = imageio.imread(buf)
                frames.append(frame)
                buf.close()
            
            # Progress updates
            if step % 100 == 0:
                progress_msg = f"  Step {step}/{steps} ({100*step/steps:.1f}%)"
                if SAVE_TO_GIF:
                    progress_msg += f" - Captured {len(frames)} frames"
                print(progress_msg)
        
        # Save as GIF with specified fps
        if SAVE_TO_GIF and frames and HAS_IMAGEIO:
            duration = len(frames) / fps
            print(f"  Saving {len(frames)} frames to {filename}...")
            os.makedirs('figs', exist_ok=True)
            imageio.mimsave(f'figs/{filename}', frames, fps=fps, loop=0)
            print(f"  Animation saved to figs/{filename} ({len(frames)} frames @ {fps} fps, ~{duration:.1f}s duration)")
        elif SAVE_TO_GIF and not frames:
            print(f"  No frames captured for {title}")
        
        # Keep window open if showing real-time
        if SHOW_REALTIME:
            print(f"  Animation complete. Close the window to continue...")
            plt.ioff()  # Turn off interactive mode
            plt.show(block=True)  # Block until window is closed
            
    finally:
        # Clean up figure
        plt.close(fig)

def run_baseline_animation(steps: int, seed: int, start_step: int = 0):
    """Run baseline simulation and create animation"""
    np.random.seed(seed)
    ws_baseline.SEED = seed
    config.SEED = seed
    
    warehouse = Warehouse(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionBaseline(i, 0))
    sim = WarehouseSimulator(warehouse)
    
    scope = "day" if steps == STEPS_DAY else "week"
    is_week = (steps == STEPS_WEEK)
    create_animation(warehouse, sim, steps, "Baseline", f"baseline_animation_{scope}.gif", is_week=is_week, start_step=start_step)

def run_ml_animation(steps: int, seed: int, start_step: int = 0):
    """Run ML simulation and create animation"""
    np.random.seed(seed)
    ws_ml.SEED = seed
    config.SEED = seed
    
    warehouse = WarehouseML(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionML(i, 0))
    sim = WarehouseSimulatorML(warehouse)
    
    scope = "day" if steps == STEPS_DAY else "week"
    is_week = (steps == STEPS_WEEK)
    create_animation(warehouse, sim, steps, "ML", f"ml_animation_{scope}.gif", is_week=is_week, start_step=start_step)

def create_order_heatmap(warehouse, simulator, steps: int, title: str, filename: str):
    """
    Run simulation and create a heatmap PNG showing which shelves had the most orders.
    
    Args:
        warehouse: Warehouse instance
        simulator: WarehouseSimulator instance
        steps: Number of simulation steps to run
        title: Title for the visualization
        filename: Output filename (will be saved to figs/ directory)
    """
    print(f"Running simulation for {title} heatmap ({steps} steps)...")
    
    # Run simulation
    for step in range(steps):
        simulator.step(dt=1.0)
        if step % 100 == 0:
            print(f"  Step {step}/{steps} ({100*step/steps:.1f}%)...")
    
    # Count orders per shelf position
    order_counts = defaultdict(int)
    total_orders = 0
    
    # Count from completed orders
    for order in warehouse.completed_orders:
        pos_key = (order.item_location.x, order.item_location.y)
        order_counts[pos_key] += 1
        total_orders += 1
    
    # Also count from pending orders (if any)
    for order in warehouse.orders:
        pos_key = (order.item_location.x, order.item_location.y)
        order_counts[pos_key] += 1
        total_orders += 1
    
    print(f"  Total orders processed: {total_orders}")
    print(f"  Unique shelf locations with orders: {len(order_counts)}")
    
    # Create heatmap data array (only for shelf cells)
    heatmap_data = np.zeros((warehouse.height, warehouse.width), dtype=float)
    
    # Fill in order counts for shelf positions
    for (x, y), count in order_counts.items():
        if 0 <= y < warehouse.height and 0 <= x < warehouse.width:
            heatmap_data[y, x] = count
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw warehouse grid (base layer)
    for y in range(warehouse.height):
        for x in range(warehouse.width):
            cell_type = int(warehouse.grid[y, x])
            color = CELL_TYPE_VALUES.get(cell_type, "white")
            
            # Only draw heatmap on shelf cells, use base color for others
            if cell_type == 1:  # SHELF
                # Use a light version of the shelf color as base
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                               facecolor="lightgray", edgecolor='gray', linewidth=0.5, zorder=1)
            else:
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                               facecolor=color, edgecolor='lightgray', linewidth=0.5, zorder=1)
            ax.add_patch(rect)
    
    # Overlay heatmap only on shelf cells
    # Create a mask for shelf cells
    shelf_mask = (warehouse.grid == 1)  # SHELF cells
    
    # Normalize heatmap data (0 to max orders)
    max_orders = np.max(heatmap_data) if np.max(heatmap_data) > 0 else 1
    normalized_data = heatmap_data / max_orders if max_orders > 0 else heatmap_data
    
    # Create custom colormap (blue to red, with transparency)
    # Use a colormap that goes from cool (low) to hot (high)
    try:
        cmap = plt.colormaps['YlOrRd']  # Yellow-Orange-Red colormap (matplotlib 3.5+)
    except (AttributeError, KeyError):
        cmap = plt.cm.YlOrRd  # Fallback for older matplotlib versions
    
    # Draw heatmap overlay only on shelf cells
    for y in range(warehouse.height):
        for x in range(warehouse.width):
            if shelf_mask[y, x] and heatmap_data[y, x] > 0:
                # Get color from colormap based on normalized value
                color_val = normalized_data[y, x]
                rgba = cmap(color_val)
                
                # Draw semi-transparent rectangle for heatmap
                rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                               facecolor=rgba, edgecolor='black', linewidth=1.0, 
                               alpha=0.7, zorder=2)
                ax.add_patch(rect)
                
                # Add order count text on shelves with orders
                if heatmap_data[y, x] > 0:
                    ax.text(x, y, str(int(heatmap_data[y, x])), 
                           ha='center', va='center', fontsize=8, 
                           fontweight='bold', zorder=3, color='black')
    
    # Set up axes
    ax.set_xlim(-0.5, warehouse.width - 0.5)
    ax.set_ylim(-0.5, warehouse.height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert so (0,0) is top-left
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    
    # Create title with statistics
    max_orders_at_shelf = int(max_orders) if max_orders > 0 else 0
    title_text = f"{title} - Order Frequency Heatmap\n"
    title_text += f"Total Orders: {total_orders} | Max Orders per Shelf: {max_orders_at_shelf}"
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_orders))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Number of Orders', rotation=270, labelpad=20, fontsize=11)
    
    # Add legend for cell types
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor="brown", edgecolor='black', label='Shelf'),
        Rectangle((0, 0), 1, 1, facecolor="lime", edgecolor='black', label='Charging Station'),
        Rectangle((0, 0), 1, 1, facecolor="grey", edgecolor='black', label='Loading Dock'),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor='black', label='Empty'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Save figure
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Heatmap saved to figs/{filename}")

def run_baseline_heatmap(steps: int, seed: int):
    """Run baseline simulation and create order frequency heatmap"""
    np.random.seed(seed)
    ws_baseline.SEED = seed
    config.SEED = seed
    
    warehouse = Warehouse(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionBaseline(i, 0))
    sim = WarehouseSimulator(warehouse)
    
    scope = "day" if steps == STEPS_DAY else "week"
    create_order_heatmap(warehouse, sim, steps, "Baseline", f"baseline_heatmap_{scope}.png")

def run_ml_heatmap(steps: int, seed: int):
    """Run ML simulation and create order frequency heatmap"""
    np.random.seed(seed)
    ws_ml.SEED = seed
    config.SEED = seed
    
    warehouse = WarehouseML(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionML(i, 0))
    sim = WarehouseSimulatorML(warehouse)
    
    scope = "day" if steps == STEPS_DAY else "week"
    create_order_heatmap(warehouse, sim, steps, "ML", f"ml_heatmap_{scope}.png")

def main():
    # Calculate which day we're visualizing based on warmup steps
    start_day = (WARMUP_STEPS // STEPS_DAY) + 1
    days_before = WARMUP_STEPS // STEPS_DAY
    
    print(f"Creating animations with seed {SEED}")
    print(f"Motion blur: Disabled")
    print(f"Capturing: Every step (no sampling)")
    print(f"Day animation: {FPS_DAY} fps (~{ACTUAL_DURATION_DAY:.1f}s, {STEPS_DAY} frames)")
    print(f"Week animation: {FPS_WEEK} fps (~{ACTUAL_DURATION_WEEK:.1f}s, {STEPS_WEEK} frames)")
    if WARMUP_STEPS > 0:
        print(f"Warmup: {WARMUP_STEPS} steps ({days_before} days) - Visualizing from Day {start_day}")
    else:
        print(f"Warmup: None - Visualizing from Day 1 (start of simulation)")
    print()
    
    # First: Full day animations (24 hours = 1440 steps)
    print("=" * 60)
    if WARMUP_STEPS > 0:
        print(f"PART 1: DAY {start_day} ANIMATIONS (24 hours)")
    else:
        print("PART 1: FULL DAY ANIMATIONS (24 hours)")
    print("=" * 60)
    print()
    
    run_baseline_animation(STEPS_DAY, SEED, start_step=WARMUP_STEPS)
    print()
    run_ml_animation(STEPS_DAY, SEED, start_step=WARMUP_STEPS)
    print()
    
    # Create heatmaps
    print("=" * 60)
    print("PART 2: ORDER FREQUENCY HEATMAPS")
    print("=" * 60)
    print()
    
    print("Creating baseline heatmap...")
    run_baseline_heatmap(STEPS_DAY, SEED)
    print()
    
    print("Creating ML heatmap...")
    run_ml_heatmap(STEPS_DAY, SEED)
    print()
    
    # # Second: Full week animations (7 days = 10080 steps)
    # print("=" * 60)
    # print("PART 2: FULL WEEK ANIMATIONS (7 days)")
    # print("=" * 60)
    # print()
    
    # run_baseline_animation(STEPS_WEEK, SEED)
    # print()
    # run_ml_animation(STEPS_WEEK, SEED)
    # print()
    
    print("=" * 60)
    print("All animations and heatmaps saved to figs/ directory")
    print("=" * 60)

if __name__ == "__main__":
    main()

