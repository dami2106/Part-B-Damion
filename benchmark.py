import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from scipy.signal import find_peaks
from warehouse_simulator import Warehouse, WarehouseSimulator, Position as PositionBaseline
from warehouse_simulator_coop_astar import Warehouse as WarehouseCoop, WarehouseSimulator as WarehouseSimulatorCoop, Position as PositionCoop
from warehouse_simulator_ML import Warehouse as WarehouseML, WarehouseSimulator as WarehouseSimulatorML, Position as PositionML
import warehouse_simulator as ws_baseline
import warehouse_simulator_coop_astar as ws_coop
import warehouse_simulator_ML as ws_ml
import config


# ...existing code...
# Benchmark configuration
NUM_ROBOTS = config.NUM_ROBOTS
WAREHOUSE_W = config.WAREHOUSE_WIDTH
WAREHOUSE_H = config.WAREHOUSE_HEIGHT

# Order generation configuration
PEAK_HOURS = config.PEAK_HOURS  # Hours of day when orders peak
BASE_RATE = config.BASE_RATE  # Base orders per hour
PEAK_MULTIPLIER = config.PEAK_MULTIPLIER

# 24 hours * 60 steps/hour
STEPS_DAY = 24 * 60
# 7 days * 24 hours/day * 60 steps/hour
STEPS_WEEK = 7 * 24 * 60

# 30 days * 24 hours/day * 60 steps/hour
STEPS_MONTH = 30 * 24 * 60

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
    config.SEED = seed
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
    ws_coop.SEED = seed  # ensure SyntheticDataGenerator and any module-level uses match
    config.SEED = seed

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

def run_simulation_ml(steps: int, seed: int) -> List[Dict]:
    np.random.seed(seed)
    ws_ml.SEED = seed  # ensure SyntheticDataGenerator and any module-level uses match
    config.SEED = seed

    warehouse = WarehouseML(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionML(i, 0))
    sim = WarehouseSimulatorML(warehouse)

    metrics_time_series = []
    prev_total_orders = 0
    for step in range(steps):
        sim.step(dt=1.0)
        metrics = sim.get_metrics()
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

def compute_rolling_avg_time(runs: List[List[Dict]], window_steps: int = 60) -> Dict[str, np.ndarray]:
    """
    Compute rolling window average completion times.
    Only shows values when there's been recent completion activity.
    
    Uses weighted average formula to compute average for orders completed in the window:
    If at t-window: N orders, avg = A
    If at t: M orders, avg = B
    Then average for (M-N) orders in window = (M*B - N*A) / (M-N)
    """
    steps = len(runs[0])
    num_seeds = len(runs)
    
    # Track rolling averages for each priority level
    rolling_metrics = {
        "avg_time_all": [],
        "avg_time_P1": [],
        "avg_time_P2": [],
        "avg_time_P3": [],
    }
    
    for seed_idx in range(num_seeds):
        run = runs[seed_idx]
        
        rolling_all = []
        rolling_p1 = []
        rolling_p2 = []
        rolling_p3 = []
        
        for t in range(steps):
            # Get current metrics
            current = run[t]
            prev_t = max(0, t - window_steps)
            prev = run[prev_t]
            
            # Calculate rolling average using weighted average formula
            # For all orders
            n_prev = prev['total_completed']
            n_curr = current['total_completed']
            avg_prev = prev['avg_time_all'] if n_prev > 0 else 0
            avg_curr = current['avg_time_all'] if n_curr > 0 else 0
            
            recent_completed = n_curr - n_prev
            if recent_completed > 0:
                # Calculate average for orders in window: (total_curr - total_prev) / count
                # total_curr = n_curr * avg_curr, total_prev = n_prev * avg_prev
                total_time_curr = n_curr * avg_curr
                total_time_prev = n_prev * avg_prev
                window_avg = (total_time_curr - total_time_prev) / recent_completed
                rolling_all.append(window_avg)
            else:
                # No recent activity - show NaN
                rolling_all.append(np.nan)
            
            # For P1
            n_prev_p1 = prev['count_P1']
            n_curr_p1 = current['count_P1']
            avg_prev_p1 = prev['avg_time_P1'] if n_prev_p1 > 0 else 0
            avg_curr_p1 = current['avg_time_P1'] if n_curr_p1 > 0 else 0
            recent_p1 = n_curr_p1 - n_prev_p1
            if recent_p1 > 0:
                total_time_curr_p1 = n_curr_p1 * avg_curr_p1
                total_time_prev_p1 = n_prev_p1 * avg_prev_p1
                window_avg_p1 = (total_time_curr_p1 - total_time_prev_p1) / recent_p1
                rolling_p1.append(window_avg_p1)
            else:
                rolling_p1.append(np.nan)
            
            # For P2
            n_prev_p2 = prev['count_P2']
            n_curr_p2 = current['count_P2']
            avg_prev_p2 = prev['avg_time_P2'] if n_prev_p2 > 0 else 0
            avg_curr_p2 = current['avg_time_P2'] if n_curr_p2 > 0 else 0
            recent_p2 = n_curr_p2 - n_prev_p2
            if recent_p2 > 0:
                total_time_curr_p2 = n_curr_p2 * avg_curr_p2
                total_time_prev_p2 = n_prev_p2 * avg_prev_p2
                window_avg_p2 = (total_time_curr_p2 - total_time_prev_p2) / recent_p2
                rolling_p2.append(window_avg_p2)
            else:
                rolling_p2.append(np.nan)
            
            # For P3
            n_prev_p3 = prev['count_P3']
            n_curr_p3 = current['count_P3']
            avg_prev_p3 = prev['avg_time_P3'] if n_prev_p3 > 0 else 0
            avg_curr_p3 = current['avg_time_P3'] if n_curr_p3 > 0 else 0
            recent_p3 = n_curr_p3 - n_prev_p3
            if recent_p3 > 0:
                total_time_curr_p3 = n_curr_p3 * avg_curr_p3
                total_time_prev_p3 = n_prev_p3 * avg_prev_p3
                window_avg_p3 = (total_time_curr_p3 - total_time_prev_p3) / recent_p3
                rolling_p3.append(window_avg_p3)
            else:
                rolling_p3.append(np.nan)
        
        rolling_metrics["avg_time_all"].append(rolling_all)
        rolling_metrics["avg_time_P1"].append(rolling_p1)
        rolling_metrics["avg_time_P2"].append(rolling_p2)
        rolling_metrics["avg_time_P3"].append(rolling_p3)
    
    # Aggregate across seeds
    result = {}
    for key in rolling_metrics:
        data = np.array(rolling_metrics[key], dtype=float)
        result[key] = {
            "mean": np.nanmean(data, axis=0),
            "std": np.nanstd(data, axis=0),
        }
    
    return result

def plot_metric(time_axis: np.ndarray,
                baseline_stats: Dict[str, np.ndarray],
                coop_stats: Dict[str, np.ndarray],
                ml_stats: Dict[str, np.ndarray],
                title: str, ylabel: str, filename: str, scope: str,
                peak_hours_to_mark: List[int] = None):
    plt.figure(figsize=(10, 6))
    # Baseline
    b_mean = baseline_stats["mean"]
    b_std = baseline_stats["std"]
    # Handle NaN values - matplotlib will automatically create gaps
    plt.plot(time_axis, b_mean, label="Baseline", color="#1f77b4")
    # Only fill between where we have valid data
    valid_mask_b = ~np.isnan(b_mean)
    if np.any(valid_mask_b):
        plt.fill_between(time_axis[valid_mask_b], 
                         (b_mean - b_std)[valid_mask_b], 
                         (b_mean + b_std)[valid_mask_b], 
                         color="#1f77b4", alpha=0.2)

    # Cooperative
    c_mean = coop_stats["mean"]
    c_std = coop_stats["std"]
    plt.plot(time_axis, c_mean, label="Cooperative A*", color="#ff7f0e")
    valid_mask_c = ~np.isnan(c_mean)
    if np.any(valid_mask_c):
        plt.fill_between(time_axis[valid_mask_c], 
                         (c_mean - c_std)[valid_mask_c], 
                         (c_mean + c_std)[valid_mask_c], 
                         color="#ff7f0e", alpha=0.2)

    # ML
    m_mean = ml_stats["mean"]
    m_std = ml_stats["std"]
    plt.plot(time_axis, m_mean, label="ML", color="#2ca02c")
    valid_mask_m = ~np.isnan(m_mean)
    if np.any(valid_mask_m):
        plt.fill_between(time_axis[valid_mask_m], 
                         (m_mean - m_std)[valid_mask_m], 
                         (m_mean + m_std)[valid_mask_m], 
                         color="#2ca02c", alpha=0.2)

    # Add markers based on scope
    if scope == "week":
        # Add red dashed lines for each day (every 24 hours)
        for day in range(1, 7):
            plt.axvline(x=day * 24, color='red', linestyle='--', alpha=0.5, linewidth=1)
    elif scope == "month":
        # Add weekly separators at days 7, 14, 21, 28
        for week_day in [7, 14, 21, 28]:
            plt.axvline(x=week_day * 24, color='red', linestyle='--', alpha=0.5, linewidth=1)
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

def plot_orders_rate_hourly(time_axis: np.ndarray,
                            baseline_stats: Dict[str, np.ndarray],
                            coop_stats: Dict[str, np.ndarray],
                            ml_stats: Dict[str, np.ndarray],
                            title: str, filename: str, scope: str):
    """Plot hourly order arrival rate (smoothed over 60 steps = 1 hour)"""
    plt.figure(figsize=(10, 6))
    
    # Smooth orders_received over 60-step windows (1 hour)
    window_size = 60
    b_mean = baseline_stats["mean"]
    c_mean = coop_stats["mean"]
    m_mean = ml_stats["mean"]
    
    # Compute hourly rate by summing over 60-step windows
    n_hours = len(b_mean) // window_size
    hourly_time = np.arange(n_hours)
    b_hourly = np.array([np.sum(b_mean[i*window_size:(i+1)*window_size]) for i in range(n_hours)])
    c_hourly = np.array([np.sum(c_mean[i*window_size:(i+1)*window_size]) for i in range(n_hours)])
    m_hourly = np.array([np.sum(m_mean[i*window_size:(i+1)*window_size]) for i in range(n_hours)])
    
    # Average the two to get consensus arrival rate
    avg_hourly = (b_hourly + c_hourly + m_hourly) / 3
    
    plt.plot(hourly_time, b_hourly, label="Baseline", color="#1f77b4", marker='o', markersize=3, alpha=0.7)
    plt.plot(hourly_time, c_hourly, label="Cooperative A*", color="#ff7f0e", marker='s', markersize=3, alpha=0.7)
    plt.plot(hourly_time, m_hourly, label="ML", color="#2ca02c", marker='^', markersize=3, alpha=0.7)
    
    # Detect actual peaks from the data (use first 24 hours for daily pattern)
    detected_peaks = []
    if scope == "day":
        # Find top N peaks in first 24 hours (N = number of PEAK_HOURS)
        num_expected_peaks = len(PEAK_HOURS)
        daily_data = avg_hourly[:24]
        # Find peaks that are local maxima
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
    elif scope == "month":
        # Weekly separators at days 7, 14, 21, 28
        for week_day in [7, 14, 21, 28]:
            plt.axvline(x=week_day * 24, color='red', linestyle='--', alpha=0.5, linewidth=1)
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

def extract_final_avg_times(runs: List[List[Dict]]) -> Dict[str, Dict[str, float]]:
    """
    Extract final average completion times per priority from runs.
    Returns dict with priority -> {mean, std} across seeds.
    """
    final_times = {
        "P1": [],
        "P2": [],
        "P3": []
    }
    
    for run in runs:
        # Get the final metrics (last step)
        final_metrics = run[-1]
        final_times["P1"].append(final_metrics.get("avg_time_P1", 0))
        final_times["P2"].append(final_metrics.get("avg_time_P2", 0))
        final_times["P3"].append(final_metrics.get("avg_time_P3", 0))
    
    result = {}
    for priority in ["P1", "P2", "P3"]:
        times_array = np.array(final_times[priority])
        # Filter out zeros (no orders of that priority completed)
        valid_times = times_array[times_array > 0]
        if len(valid_times) > 0:
            result[priority] = {
                "mean": np.mean(valid_times),
                "std": np.std(valid_times)
            }
        else:
            result[priority] = {
                "mean": 0.0,
                "std": 0.0
            }
    
    return result

def plot_priority_completion_times(
    baseline_stats: Dict[str, Dict[str, float]],
    coop_stats: Dict[str, Dict[str, float]],
    ml_stats: Dict[str, Dict[str, float]],
    title: str,
    filename: str
):
    """
    Create bar chart comparing average completion times by priority.
    Each priority has 3 bars: Baseline, Cooperative A*, ML
    """
    priorities = ["P1", "P2", "P3"]
    priority_labels = ["Normal", "High", "Urgent"]
    
    x = np.arange(len(priorities))
    width = 0.25  # Width of bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract means and stds
    baseline_means = [baseline_stats[p]["mean"] for p in priorities]
    baseline_stds = [baseline_stats[p]["std"] for p in priorities]
    
    coop_means = [coop_stats[p]["mean"] for p in priorities]
    coop_stds = [coop_stats[p]["std"] for p in priorities]
    
    ml_means = [ml_stats[p]["mean"] for p in priorities]
    ml_stds = [ml_stats[p]["std"] for p in priorities]
    
    # Create bars
    bars1 = ax.bar(x - width, baseline_means, width, label="Baseline", 
                   color="#1f77b4", yerr=baseline_stds, capsize=5)
    bars2 = ax.bar(x, coop_means, width, label="Cooperative A*", 
                   color="#ff7f0e", yerr=coop_stds, capsize=5)
    bars3 = ax.bar(x + width, ml_means, width, label="ML", 
                   color="#2ca02c", yerr=ml_stds, capsize=5)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_xlabel("Order Priority", fontsize=12)
    ax.set_ylabel("Average Completion Time (Steps)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(priority_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all(time_axis_hours: np.ndarray,
             agg_baseline: Dict[str, Dict[str, np.ndarray]],
             agg_coop: Dict[str, Dict[str, np.ndarray]],
             agg_ml: Dict[str, Dict[str, np.ndarray]],
             scope_title: str, scope: str):
    # First, detect actual peaks from order arrival data
    detected_peaks = plot_orders_rate_hourly(
        time_axis_hours,
        agg_baseline["orders_received"],
        agg_coop["orders_received"],
        agg_ml["orders_received"],
        f"{scope_title}: Order Arrival Rate (Hourly)",
        f"{scope}_orders_arrival_rate.png",
        scope
    )
    
    # Create multiple figures for key metrics
    plot_metric(time_axis_hours, agg_baseline["total_completed"], agg_coop["total_completed"], agg_ml["total_completed"], f"{scope_title}: Total Completed Orders", "Orders", f"{scope}_total_completed.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["pending_orders"], agg_coop["pending_orders"], agg_ml["pending_orders"], f"{scope_title}: Pending Orders", "Orders", f"{scope}_pending_orders.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["orders_received"], agg_coop["orders_received"], agg_ml["orders_received"], f"{scope_title}: Orders Received (per Step)", "Orders", f"{scope}_orders_received.png", scope, detected_peaks)
    
    plot_metric(time_axis_hours, agg_baseline["avg_time_all"], agg_coop["avg_time_all"], agg_ml["avg_time_all"], f"{scope_title}: Avg Completion Time (All) - Rolling 1h Window", "Steps", f"{scope}_avg_time_all.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["coord_overhead"], agg_coop["coord_overhead"], agg_ml["coord_overhead"], f"{scope_title}: Coordination Overhead", "Wait / (Wait + Move)", f"{scope}_coord_overhead.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["total_wait_steps"], agg_coop["total_wait_steps"], agg_ml["total_wait_steps"], f"{scope_title}: Total Wait Steps", "Steps", f"{scope}_total_wait_steps.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["total_move_steps"], agg_coop["total_move_steps"], agg_ml["total_move_steps"], f"{scope_title}: Total Move Steps", "Steps", f"{scope}_total_move_steps.png", scope, detected_peaks)

    # Priority-specific completion times
    plot_metric(time_axis_hours, agg_baseline["avg_time_P1"], agg_coop["avg_time_P1"], agg_ml["avg_time_P1"], f"{scope_title}: Avg Completion Time (Normal Priority) - Rolling 1h Window", "Steps", f"{scope}_avg_time_normal.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["avg_time_P2"], agg_coop["avg_time_P2"], agg_ml["avg_time_P2"], f"{scope_title}: Avg Completion Time (High Priority) - Rolling 1h Window", "Steps", f"{scope}_avg_time_high.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["avg_time_P3"], agg_coop["avg_time_P3"], agg_ml["avg_time_P3"], f"{scope_title}: Avg Completion Time (Urgent Priority) - Rolling 1h Window", "Steps", f"{scope}_avg_time_urgent.png", scope, detected_peaks)

    # Priority-specific counts
    plot_metric(time_axis_hours, agg_baseline["count_P1"], agg_coop["count_P1"], agg_ml["count_P1"], f"{scope_title}: Normal Priority Orders Completed", "Orders", f"{scope}_count_normal.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["count_P2"], agg_coop["count_P2"], agg_ml["count_P2"], f"{scope_title}: High Priority Orders Completed", "Orders", f"{scope}_count_high.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["count_P3"], agg_coop["count_P3"], agg_ml["count_P3"], f"{scope_title}: Urgent Priority Orders Completed", "Orders", f"{scope}_count_urgent.png", scope, detected_peaks)

    # Battery metrics
    plot_metric(time_axis_hours, agg_baseline["avg_battery"], agg_coop["avg_battery"], agg_ml["avg_battery"], f"{scope_title}: Average Robot Battery Level", "Battery %", f"{scope}_avg_battery.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["min_battery"], agg_coop["min_battery"], agg_ml["min_battery"], f"{scope_title}: Minimum Robot Battery Level", "Battery %", f"{scope}_min_battery.png", scope, detected_peaks)
    plot_metric(time_axis_hours, agg_baseline["charging_robots"], agg_coop["charging_robots"], agg_ml["charging_robots"], f"{scope_title}: Number of Robots Charging", "Robots", f"{scope}_charging_robots.png", scope, detected_peaks)

def run_benchmark(seeds: List[int], steps: int, scope_title: str, scope: str):
    # Run both simulators across seeds
    runs_baseline = []
    runs_coop = []
    runs_ml = []
    for seed in seeds:
        runs_baseline.append(run_simulation_baseline(steps=steps, seed=seed))
        runs_coop.append(run_simulation_coop(steps=steps, seed=seed))
        runs_ml.append(run_simulation_ml(steps=steps, seed=seed))

    # Aggregate means/stds over seeds
    agg_b = aggregate_runs(runs_baseline)
    agg_c = aggregate_runs(runs_coop)
    agg_m = aggregate_runs(runs_ml)
    
    # Compute rolling window averages for completion times (1 hour window = 60 steps)
    rolling_b = compute_rolling_avg_time(runs_baseline, window_steps=60)
    rolling_c = compute_rolling_avg_time(runs_coop, window_steps=60)
    rolling_m = compute_rolling_avg_time(runs_ml, window_steps=60)
    
    # Replace cumulative averages with rolling window averages
    agg_b["avg_time_all"] = rolling_b["avg_time_all"]
    agg_b["avg_time_P1"] = rolling_b["avg_time_P1"]
    agg_b["avg_time_P2"] = rolling_b["avg_time_P2"]
    agg_b["avg_time_P3"] = rolling_b["avg_time_P3"]
    
    agg_c["avg_time_all"] = rolling_c["avg_time_all"]
    agg_c["avg_time_P1"] = rolling_c["avg_time_P1"]
    agg_c["avg_time_P2"] = rolling_c["avg_time_P2"]
    agg_c["avg_time_P3"] = rolling_c["avg_time_P3"]
    
    agg_m["avg_time_all"] = rolling_m["avg_time_all"]
    agg_m["avg_time_P1"] = rolling_m["avg_time_P1"]
    agg_m["avg_time_P2"] = rolling_m["avg_time_P2"]
    agg_m["avg_time_P3"] = rolling_m["avg_time_P3"]

    # Prepare time axis in hours
    time_axis_hours = np.arange(steps) / 60.0
    # Plot
    plot_all(time_axis_hours, agg_b, agg_c, agg_m, scope_title, scope)
    
    # Extract final average completion times per priority and create bar charts
    final_baseline = extract_final_avg_times(runs_baseline)
    final_coop = extract_final_avg_times(runs_coop)
    final_ml = extract_final_avg_times(runs_ml)
    
    plot_priority_completion_times(
        final_baseline,
        final_coop,
        final_ml,
        f"{scope_title}: Average Completion Time by Priority",
        f"{scope}_priority_completion_times.png"
    )

def main():
    # Standard set of seeds in benchmarking: 5 seeds
    seeds = [0, 1, 2, 3, 4]

    # Day benchmark
    print("Running day benchmark...")
    run_benchmark(seeds=seeds, steps=STEPS_DAY, scope_title="Day (24h)", scope="day")
    # Week benchmark
    print("Running week benchmark...")
    run_benchmark(seeds=seeds, steps=STEPS_WEEK, scope_title="Week (7d)", scope="week")

    # Month benchmark
    print("Running month benchmark...")
    # run_benchmark(seeds=seeds, steps=STEPS_MONTH, scope_title="Month (30d)", scope="month")

    print("\nAll figures saved to figs/ directory")

if __name__ == "__main__":
    main()
