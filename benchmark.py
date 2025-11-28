import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from warehouse_simulator import Warehouse, WarehouseSimulator, Position as PositionBaseline
from warehouse_simulator_coop_astar import Warehouse as WarehouseCoop, WarehouseSimulator as WarehouseSimulatorCoop, Position as PositionCoop
import warehouse_simulator as ws_baseline
import warehouse_simulator_coop_astar as ws_coop

# ...existing code...
# Benchmark configuration
NUM_ROBOTS = 10
WAREHOUSE_W = 50
WAREHOUSE_H = 50

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
    for step in range(steps):
        sim.step(dt=1.0)
        metrics_time_series.append(sim.get_metrics())
    return metrics_time_series

def run_simulation_coop(steps: int, seed: int) -> List[Dict]:
    np.random.seed(seed)
    ws_coop.SEED = seed

    warehouse = WarehouseCoop(width=WAREHOUSE_W, height=WAREHOUSE_H)
    for i in range(NUM_ROBOTS):
        warehouse.add_robot(PositionCoop(i, 0))
    sim = WarehouseSimulatorCoop(warehouse)

    metrics_time_series = []
    for step in range(steps):
        sim.step(dt=1.0)
        metrics_time_series.append(sim.get_metrics())
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

def plot_metric(time_axis: np.ndarray, baseline_stats: Dict[str, np.ndarray], coop_stats: Dict[str, np.ndarray], title: str, ylabel: str):
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

    plt.title(title)
    plt.xlabel("Time (hours)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_all(time_axis_hours: np.ndarray, agg_baseline: Dict[str, Dict[str, np.ndarray]], agg_coop: Dict[str, Dict[str, np.ndarray]], scope_title: str):
    # Create multiple figures for key metrics
    plot_metric(time_axis_hours, agg_baseline["total_completed"], agg_coop["total_completed"], f"{scope_title}: Total Completed Orders", "Orders")
    plot_metric(time_axis_hours, agg_baseline["pending_orders"], agg_coop["pending_orders"], f"{scope_title}: Pending Orders", "Orders")
    plot_metric(time_axis_hours, agg_baseline["avg_time_all"], agg_coop["avg_time_all"], f"{scope_title}: Avg Completion Time (All)", "Steps")
    plot_metric(time_axis_hours, agg_baseline["coord_overhead"], agg_coop["coord_overhead"], f"{scope_title}: Coordination Overhead", "Wait / (Wait + Move)")
    plot_metric(time_axis_hours, agg_baseline["total_wait_steps"], agg_coop["total_wait_steps"], f"{scope_title}: Total Wait Steps", "Steps")
    plot_metric(time_axis_hours, agg_baseline["total_move_steps"], agg_coop["total_move_steps"], f"{scope_title}: Total Move Steps", "Steps")

    # Priority-specific completion times
    plot_metric(time_axis_hours, agg_baseline["avg_time_P1"], agg_coop["avg_time_P1"], f"{scope_title}: Avg Completion Time P1", "Steps")
    plot_metric(time_axis_hours, agg_baseline["avg_time_P2"], agg_coop["avg_time_P2"], f"{scope_title}: Avg Completion Time P2", "Steps")
    plot_metric(time_axis_hours, agg_baseline["avg_time_P3"], agg_coop["avg_time_P3"], f"{scope_title}: Avg Completion Time P3", "Steps")

    # Priority-specific counts
    plot_metric(time_axis_hours, agg_baseline["count_P1"], agg_coop["count_P1"], f"{scope_title}: Count P1 Completed", "Orders")
    plot_metric(time_axis_hours, agg_baseline["count_P2"], agg_coop["count_P2"], f"{scope_title}: Count P2 Completed", "Orders")
    plot_metric(time_axis_hours, agg_baseline["count_P3"], agg_coop["count_P3"], f"{scope_title}: Count P3 Completed", "Orders")

def run_benchmark(seeds: List[int], steps: int, scope_title: str):
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
    plot_all(time_axis_hours, agg_b, agg_c, scope_title)

def main():
    # Standard set of seeds in benchmarking: 5 seeds
    seeds = [0, 1, 2, 3, 4]

    # Day benchmark
    run_benchmark(seeds=seeds, steps=STEPS_DAY, scope_title="Day (24h)")
    # Week benchmark
    run_benchmark(seeds=seeds, steps=STEPS_WEEK, scope_title="Week (7d)")

    # Show all figures
    plt.show()

if __name__ == "__main__":
    main()
