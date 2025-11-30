# Problem 3: Warehouse Robot Coordination 🤖

## Env Setup 
An `environment.yml` is provided with a full breakdown of required packages. This can be installed using Mamba or Conda:  
```bash
conda env create -f environment.yml #conda 
mamba env create -f environment.yml #mamba
```
If neither Mamba nor Conda is available, a `requirements.txt` is also provided which can be installed via *pip*:  
```bash
pip install -r requirements.txt
```


## How to Run
### Benchmark all implementations
To run all benchmarks for the daily, weekly and monthly setting, you can make use of the `main.py` file which will run all of the implemented benchmarks, average them over 5 seeds, and plot all the corresponding results with error bars shaded (figs are saved in `/figs`). 
```bash
python main.py
```

### Run individual implementations
Each implementation can be run on its own, using the commands:
```bash
python src/warehouse_simulator.py #runs the basic benchmark 
python src/warehouse_simulator_coop_astar.py #runs the improved benchmark 
python src.warehouse_simulator_ML.py #runs the ML benchmark 
```
### Customising runs
For both of the above configurations, the problem setting parameters can be modified in the `config.py` file, here you can specify things like warehouse size, num robots, etc. All benchmark files will use this config. 

## Packages / Libraries used:
 - `python>=3.12`
 - `matplotlib==3.10.6`
 - `numpy==2.3.5`
 - `pandas==2.3.3`
 - `scikit-learn==1.7.2`
 - `scipy==1.16.3`
 - `PyQt6==6.9.1`



