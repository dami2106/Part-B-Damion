import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from scipy.signal import find_peaks
from warehouse_simulator import Warehouse, WarehouseSimulator, Position as PositionBaseline
from warehouse_simulator_coop_astar import Warehouse as WarehouseCoop, WarehouseSimulator as WarehouseSimulatorCoop, Position as PositionCoop
import warehouse_simulator as ws_baseline
import warehouse_simulator_coop_astar as ws_coop

# ...existing code...
# Benchmark configuration
NUM_ROBOTS = 10
WAREHOUSE_W = 30
WAREHOUSE_H = 30

# Order generation configuration
PEAK_HOURS = [8, 17]  # Hours of day when orders peak
BASE_RATE = 8
PEAK_MULTIPLIER = 2.0

# 24 hours * 60 steps/hour
STEPS_DAY = 24 * 60
# 7 days * 24 hours/day * 60 steps/hour
STEPS_WEEK = 7 * 24 * 60

# Collect these metrics from get_metrics()
METRIC_KEYS = [
    "total_completed",
    "pending_orders",
    "avg_time_all",
    "coord_overhead",
    "total_wait_steps",
    "total_move_steps",
    "avg_time_P1",
    "avg_time_P2",
    "avg_time_P3",
    "count_P1",
    "count_P2",
    "count_P3",
    "orders_received",
    "avg_battery",
    "min_battery",
    "charging_robots",
]

def run_simulation_baseline(steps: int, seed: int) -> List[Dict]:
    # Ensure same randomness as schedule generation
    np.random.seed(seed)
    ws_baseline.SEED = seed  # ensure SyntheticDataGenerator and any module-level uses match

    warehouse = Warehouse(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionBaseline(i, 0))
    sim = WarehouseSimulator(warehouse)

    metrics_time_series = []
    prev_total_orders = 0
    for step in range(steps):
        sim.step(dt=1.0)
        metrics = sim.get_metrics()
        # Track incoming orders (new orders received this step)
        current_total = metrics['total_completed'] + metrics['pending_orders']
        metrics['orders_received'] = current_total - prev_total_orders
        prev_total_orders = current_total
        metrics_time_series.append(metrics)
    return metrics_time_series

def run_simulation_coop(steps: int, seed: int) -> List[Dict]:
    np.random.seed(seed)
    ws_coop.SEED = seed

    warehouse = WarehouseCoop(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionCoop(i, 0))
    sim = WarehouseSimulatorCoop(warehouse)

    metrics_time_series = []
    prev_total_orders = 0
    for step in range(steps):
        sim.step(dt=1.0)
        metrics = sim.get_metrics()
        # Track incoming orders (new orders received this step)
        current_total = metrics['total_completed'] + metrics['pending_orders']
        metrics['orders_received'] = current_total - prev_total_orders
        prev_total_orders = current_total
        metrics_time_series.append(metrics)
    return metrics_time_series

def aggregate_runs(runs: List[List[Dict]]) -> Dict[str, np.ndarray]:
    """
    Convert list of runs (each is list of per-step metrics dicts) into arrays:
    returns dict of key -> (mean, std) arrays over time.
    """
    # Align lengths (should be same)
    steps = len(runs[0])
    # Stack per metric
    agg = {}
    for key in METRIC_KEYS:
        # shape: (num_seeds, steps)
        data = np.array([[run[t].get(key, 0) for t in range(steps)] for run in runs], dtype=float)
        agg[key] = {
            "mean": np.nanmean(data, axis=0),
            "std": np.nanstd(data, axis=0),
        }
    return agg

def plot_metric(time_axis: np.ndarray, baseline_stats: Dict[str, np.ndarray], coop_stats: Dict[str, np.ndarray], title: str, ylabel: str, filename: str, scope: str, peak_hours_to_mark: List[int] = None):
    plt.figure(figsize=(10, 6))
    # Baseline
    b_mean = baseline_stats["mean"]
    b_std = baseline_stats["std"]
    plt.plot(time_axis, b_mean, label="Baseline", color="#1f77b4")
    plt.fill_between(time_axis, b_mean - b_std, b_mean + b_std, color="#1f77b4", alpha=0.2)

    # Cooperative
    c_mean = coop_stats["mean"]
    c_std = coop_stats["std"]
    plt.plot(time_axis, c_mean, label="Cooperative A*", color="#ff7f0e")
    plt.fill_between(time_axis, c_mean - c_std, c_mean + c_std, color="#ff7f0e", alpha=0.2)

    # Add markers based on scope
    if scope == "week":
        # Add red dashed lines for each day (every 24 hours)
        for day in range(1, 7):
            plt.axvline(x=day * 24, color='red', linestyle='--', alpha=0.5, linewidth=1)
    elif scope == "day" and peak_hours_to_mark:
        # Add dashed lines at detected order arrival peaks
        for i, peak_hour in enumerate(peak_hours_to_mark):
            label = f'Peak {i+1} ({peak_hour}h)' if i == 0 else f'Peak {i+1} ({peak_hour}h)'
            plt.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.5, linewidth=1, label=label)

    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_orders_rate_hourly(time_axis: np.ndarray, baseline_stats: Dict[str, np.ndarray], coop_stats: Dict[str, np.ndarray], title: str, filename: str, scope: str):
    """Plot hourly order arrival rate (smoothed over 60 steps = 1 hour)"""
    plt.figure(figsize=(10, 6))
    
    # Smooth orders_received over 60-step windows (1 hour)
    window_size = 60
    b_mean = baseline_stats["mean"]
    c_mean = coop_stats["mean"]
    
    # Compute hourly rate by summing over 60-step windows
    n_hours = len(b_mean) // window_size
    hourly_time = np.arange(n_hours)
    b_hourly = np.array([np.sum(b_mean[i*window_size:(i+1)*window_size]) for i in range(n_hours)])
    c_hourly = np.array([np.sum(c_mean[i*window_size:(i+1)*window_size]) for i in range(n_hours)])
    
    # Average the two to get consensus arrival rate
    avg_hourly = (b_hourly + c_hourly) / 2
    
    plt.plot(hourly_time, b_hourly, label="Baseline", color="#1f77b4", marker='o', markersize=3, alpha=0.7)
    plt.plot(hourly_time, c_hourly, label="Cooperative A*", color="#ff7f0e", marker='s', markersize=3, alpha=0.7)
    
    # Detect actual peaks from the data (use first 24 hours for daily pattern)
    detected_peaks = []
    if scope == "day":
        # Find top N peaks in first 24 hours (N = number of PEAK_HOURS)
        num_expected_peaks = len(PEAK_HOURS)
        daily_data = avg_hourly[:24]
        # Find peaks that are local maxima
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(daily_data, height=np.mean(daily_data))
        if len(peaks) >= num_expected_peaks:
            # Get top N peaks
            peak_heights = daily_data[peaks]
            top_peak_indices = np.argsort(peak_heights)[-num_expected_peaks:]
            detected_peaks = sorted(peaks[top_peak_indices].tolist())
    
    # Add markers based on scope
    if scope == "week":
        for day in range(1, 7):
            plt.axvline(x=day * 24, color='red', linestyle='--', alpha=0.5, linewidth=1)
        # Mark detected peaks within each day
        if detected_peaks:
            for day in range(7):
                for peak in detected_peaks:
                    plt.axvline(x=day * 24 + peak, color='green', linestyle=':', alpha=0.3, linewidth=1)
    elif scope == "day" and detected_peaks:
        for i, peak in enumerate(detected_peaks):
            plt.axvline(x=peak, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Peak {i+1} (hour {peak})')
    
    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel("Orders per Hour")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    return detected_peaks  # Return for use in other plots

def plot_all(time_axis_hours: np.ndarray, agg_baseline: Dict[str, Dict[str, np.ndarray]], agg_coop: Dict[str, Dict[str, np.ndarray]], scope_title: str, scope: str):
    # First, detect actual peaks from order arrival data
    detected_peaks = plot_orders_rate_hourly(time_axis_hours, agg_baseline["orders_received"], agg_coop["orders_received"], f"{scope_title}: Order Arrival Rate (Hourly)", f"{scope}_orders_arrival_rate.png", scope)
    
    # Create multiple figures for key metrics
    plot_metric(time_axis_hours, agg_baseline["total_completed"], agg_coop["total_completed"], f"{scope_title}: Total Completed Orders", "Orders", f"{scope}_total_completed.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["pending_orders"], agg_coop["pending_orders"], f"{scope_title}: Pending Orders", "Orders", f"{scope}_pending_orders.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["orders_received"], agg_coop["orders_received"], f"{scope_title}: Orders Received (per Step)", "Orders", f"{scope}_orders_received.png", scope, detected_peaks)
    
    plot_metric(time_axis_hours, agg_baseline["avg_time_all"], agg_coop["avg_time_all"], f"{scope_title}: Avg Completion Time (All)", "Steps", f"{scope}_avg_time_all.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["coord_overhead"], agg_coop["coord_overhead"], f"{scope_title}: Coordination Overhead", "Wait / (Wait + Move)", f"{scope}_coord_overhead.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["total_wait_steps"], agg_coop["total_wait_steps"], f"{scope_title}: Total Wait Steps", "Steps", f"{scope}_total_wait_steps.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["total_move_steps"], agg_coop["total_move_steps"], f"{scope_title}: Total Move Steps", "Steps", f"{scope}_total_move_steps.png", scope, detected_peaks)

    # Priority-specific completion times
    plot_metric(time_axis_hours, agg_baseline["avg_time_P1"], agg_coop["avg_time_P1"], f"{scope_title}: Avg Completion Time (Normal Priority)", "Steps", f"{scope}_avg_time_normal.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["avg_time_P2"], agg_coop["avg_time_P2"], f"{scope_title}: Avg Completion Time (High Priority)", "Steps", f"{scope}_avg_time_high.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["avg_time_P3"], agg_coop["avg_time_P3"], f"{scope_title}: Avg Completion Time (Urgent Priority)", "Steps", f"{scope}_avg_time_urgent.png", scope, detected_peaks)

    # Priority-specific counts
    plot_metric(time_axis_hours, agg_baseline["count_P1"], agg_coop["count_P1"], f"{scope_title}: Normal Priority Orders Completed", "Orders", f"{scope}_count_normal.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["count_P2"], agg_coop["count_P2"], f"{scope_title}: High Priority Orders Completed", "Orders", f"{scope}_count_high.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["count_P3"], agg_coop["count_P3"], f"{scope_title}: Urgent Priority Orders Completed", "Orders", f"{scope}_count_urgent.png", scope, detected_peaks)

    # Battery metrics
    plot_metric(time_axis_hours, agg_baseline["avg_battery"], agg_coop["avg_battery"], f"{scope_title}: Average Robot Battery Level", "Battery %", f"{scope}_avg_battery.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["min_battery"], agg_coop["min_battery"], f"{scope_title}: Minimum Robot Battery Level", "Battery %", f"{scope}_min_battery.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["charging_robots"], agg_coop["charging_robots"], f"{scope_title}: Number of Robots Charging", "Robots", f"{scope}_charging_robots.png", scope, detected_peaks)

def run_benchmark(seeds: List[int], steps: int, scope_title: str, scope: str):
    # Run both simulators across seeds
    runs_baseline = []
    runs_coop = []
    for seed in seeds:
        runs_baseline.append(run_simulation_baseline(steps=steps, seed=seed))
        runs_coop.append(run_simulation_coop(steps=steps, seed=seed))

    # Aggregate means/stds over seeds
    agg_b = aggregate_runs(runs_baseline)
    agg_c = aggregate_runs(runs_coop)

    # Prepare time axis in hours
    time_axis_hours = np.arange(steps) / 60.0
    # Plot
    plot_all(time_axis_hours, agg_b, agg_c, scope_title, scope)

def main():
    # Standard set of seeds in benchmarking: 5 seeds
    seeds = [0, 1, 2, 3, 4]

    # Day benchmark
    print("Running day benchmark...")
    run_benchmark(seeds=seeds, steps=STEPS_DAY, scope_title="Day (24h)", scope="day")
    # Week benchmark
    print("Running week benchmark...")
    run_benchmark(seeds=seeds, steps=STEPS_WEEK, scope_title="Week (7d)", scope="week")

    print("\nAll figures saved to figs/ directory")

if __name__ == "__main__":
    main()
