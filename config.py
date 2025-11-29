"""
Configuration file for warehouse simulator
"""

# Battery settings
CHARGE_RATE = 5.0
DRAIN_RATE = 0.5

# Simulation settings
STEPS_PER_HOUR = 60  # Simulation steps per hour (granularity)

# Order generation settings
N_DAYS = 7
BASE_RATE = 8  # Base orders per hour
PEAK_HOURS = [8, 17]  # Peak hours during the day
PEAK_MULTIPLIER = 2.0  # How much busier at peak times

# Simulation parameters
WAREHOUSE_WIDTH = 30
WAREHOUSE_HEIGHT = 30
NUM_ROBOTS = 20
SIMULATION_STEPS = None  # Will be calculated as N_DAYS * 24 * STEPS_PER_HOUR
PRINT_INTERVAL = 20  # Print metrics every N steps

# Random seed for reproducibility
SEED = 42
