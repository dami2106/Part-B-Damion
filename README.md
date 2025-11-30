# Problem 3: Warehouse Robot Coordination 🤖

## Setup & Run the Code
### Setting up the environment 
An `environment.yml` is provided with a full breakdown of required packages. This can be installed using Mamba or Conda:  
```bash
conda env create -f environment.yml #conda 
mamba env create -f environment.yml #mamba
```
If neither Mamba nor Conda is available, a `requirements.txt` is also provided which can be installed via *pip*:  
```bash
pip install -r requirements.txt
```

### Benchmark all Implementations
To run all benchmarks for the daily, weekly and monthly setting, you can make use of the `main.py` file which will run all of the implemented benchmarks, average them over 5 seeds, and plot all the corresponding results with error bars shaded (figs are saved in `/figs`). 
```bash
python main.py
```

### Run Individual Implementations
Each implementation can be run on its own, using the commands:
```bash
python -m src.warehouse_simulator #runs the basic benchmark 
python -m src.warehouse_simulator_coop_astar #runs the improved benchmark 
python -m src.warehouse_simulator_ML #runs the ML benchmark 
```
### Customising runs
For both of the above configurations, the problem setting parameters can be modified in the `src/config.py` file, here you can specify things like warehouse size, num robots, etc. All benchmark files will use this config. 

## The Approaches Used 
### Baseline (A* and Greedy Assigning)
### Improvement (Cooperative A* and Hungarian Matching)
### ML Approach (Random Forest, Exponential Smoothing)

## Summary of Results



## Trade offs and Limitations



