"""
Configuration file for warehouse simulator
"""

# Battery settings
CHARGE_RATE = 20.0
DRAIN_RATE = 0.2

# Simulation settings
STEPS_PER_HOUR = 60  # Simulation steps per hour (granularity) we modeled it as minutes so 60 per hour

# Order generation settings
N_DAYS = 7 * 5 #how many days to generate data for (7x5 is a month and a bit)
BASE_RATE = 8  # Base orders per hour
PEAK_HOURS = [8, 17]  # Peak hours during the day
PEAK_MULTIPLIER = 2.0  # How much busier at peak times

# Simulation parameters
WAREHOUSE_WIDTH = 20
WAREHOUSE_HEIGHT = 20
NUM_ROBOTS = 10
SIMULATION_STEPS = None  # Will be calculated as N_DAYS * 24 * STEPS_PER_HOUR
PRINT_INTERVAL = 20  # Print metrics every N steps

# Random seed for reproducibility (overwritten if using main.py for averaging)
SEED = 42
