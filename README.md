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

## Environment and Data Setup Used
 - We use the given starter code to implement the environment but make several changes. 
 - We first update the code to make use of the Poisson distribution provided in `src/synthetic_data.py` to generate more realistic order data scenarios compared to the given random 10% orders. Here we use the config outlined in `src/config.py` (such as having a peak at 8, and 5), and then having a peak multiplier of 2. 
 - Our warehouse is 20x20, with chargers in each corner, and a dock on each side middle portion (so 4 chargers and 4 docks total.)
 - We deploy 10 robots in the warehouse 
 - Shelves are placed in a grid like pattern with some extra shelves added to add obstacles and congestion. 
 - Certain shelves are favoured depending on our sampling seed (they are not randomly sampled when creating an order). This is static and doesnt change. This means certain shelves are used more than others
 - Order priority is assigned with priorities : 70% for normal, 20% for high, and 10% for urgent.


## The Approaches Used 
### Baseline (A* and Greedy Assigning)
 - This is the baseline (control group) setup (naive and greedy). 
 - **Task Assignment:** Here we use the given first come first serve greedy approach (included in the code example). Orders are simply assigned to the nearest un-occupied robot. 
 - **Pathfinding:** We use a standard A* approach on a static grid, where our heuristic uses the Manhattan distance formulation. 
 - **Collision Handling:** We use a reactive collision setting, robots are moving blindly since they do not look into the future, so if they encounter another robot infront of them they simply wait until they can move forward. 
 - **Order Priority:** Before assigning orders in the greedy setting, orders are sorted by priority (urgent first) and then they are assigned. 
 - **Battery Management:** Simple state checking, if a robot is idle with less battery than the given threshold, it goes to charge.  

### Improvement (Cooperative A* and Hungarian Matching)
 - Improvement over the baseline using an improved A* and better task assignment 
 - **Task Assignment:** Here instead of assigning orders one by one, we batch the pending orders and idle robots. We then construct a cost matrix based on each robots Manhattan distance to the order (where each cost C_ij is the distance from the robot i to order j). We then make use of *scipy.optimize.linear_sum_assignment* to implement the Hungarian matching (it finds the global minimum matching between orders and robots all at once using this cost matrix we construct). 
 - **Pathfinding:** Here we implement Prioritised Planning using space time A*: we implement time as a dimension when planning by making use of a reservation table. When a robot plans (we plan robot one by one), it reserves its path in time (for example it will reserve cell (1, 2) at time t=2, which is 2 steps in the future). Robots planning after this robot will then treat these reserved cells as dynamic obstacles. This means that if Robot A reserves cell  (1, 2) at time t=2, robot B knows that cell will be free at t=3. 
 - **Collision Handling:** In this setting due to the space time reservations on paths, robots should not collide. If a path cant be found, the robot is left idle (in a wait state) and tries to replan after 1 step. 
 - **Order Priority:** When constructing the cost matrix in the *task assignment* function, we first get the distance from robot to order but while giving urgent priority orders a smaller or negative cost to prioritise them. This is done by subtracting 100 for urgent, 50 for high and 0 for normal. This causes robots to be assigned to the urgent and high orders before the normal orders as they have a much smaller cost. 
 - **Battery Management:** When a robots battery hits 30% or less, its goal is set to a charger no matter what. Its path to the charger is then also added to the space time queue just as an order would be. 


### ML Approach (Random Forest, Exponential Smoothing)
 - Our sustainable appraoch designed to minimise energy cost per order, improving total throughput. 
 - Here, the path finding and task assignment is mainly the same as in `Improvement (Cooperative A* and Hungarian Matching)` from above. However, we modify it to accomodate new techniques:
 - **Firstly, for the Temporal Model (Demand Volume):**
   -  Here we use a Random Forest Regressor from SKLearn. We train this on live data as we recieve it in the simulation. 
   - The features we use for prediction are : `Hour of the Day`, `Day of the Week`, `Previous Hour's Order Volume` (which is our lag feature). 
   - This model is trained to predict the next hours order volume given the current history. We note that the model starts training as it starts running the simulation, so it is fully online and adaptive, and requires no pre training. 
- **Secondly, the Spatial Model (Hotspot Tracking):**
   - Due to our environment setup, where some shelves are busier than others, we implement hotspot tracking
   - We keep a frequency map of each position in the map, and track how many times it is visited. We apply an exponential decay incase different groupins of shelves are favoured on different times / days to adapt to shifting demand trends (ie if some shelves are used more on Monday vs Wednesday). 

- **How are these models implemented and used:** 
   - Every 60 minutes in the environment, we feed the current times and constructed features into our temporal model. That model will tell us how busy we will be in the next hour. We use this info to first check, is the predicted volume greater than a pre-defined threshold? If yes, then we can trigger a pre certain events: 
      1. Here we force the battery threshold down to 30% (we dont want robots charging when we get an influx of orders) in order to meet demand. 
      2. We also move robots to hotspot regions (these are shelves used most often based on the current frequency map). Here we look at idle robots with battery that is atleast 10% above the threshold. We then spread robots evenly across the top 5 shelf hotspots (we scan a 7x7 area around the active shelf to find the closest valid cell). If no hotspots or available spots exist, the robot goes to the middle of the warehouse. 
   - If the temporal model predicts a load below the threshold then we can perform other logic:
     1. In this case, we set our robots battery threshold to 60% which makes them all recharge to be ready for the next high load period. 
     2. We still make sure orders are completed using the original logic from the improved appraoach. 

## Summary of Results



## Future work and Ideas
Given the time constraint, there were several avenues we did not explore that we would have liked to, such as:
 - Hotspot shelves changing depending on the day of the week (our implementation generates hotspot shelves once and keeps them like that throughout)
 - Multi agent reinforcement learning as a benchmark against our ML solution as here the agent would be solely responsible for learning the best techniques to be efficient. Reward function design could be : positive rewards for each order (scaled by the priority), vs negative rewards for batteries going flat. 
 - Trying different models out for the temporal aspect, random forest was a known safe bet. 
 - We woul have liked to use an LSTM, but that requires a lot more data than a random forest and more training time on a CPU only machine. 


