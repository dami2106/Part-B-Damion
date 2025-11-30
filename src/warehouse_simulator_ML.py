"""
Warehouse Robot Fleet Coordination - Starter Code
Problem 3: Multi-Robot Task Assignment and Pathfinding

Improved ML + optimised implementation
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
import heapq
import random
from enum import Enum
from src.synthetic_data import SyntheticDataGenerator   

from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestRegressor
import src.config as config

CHARGE_RATE = config.CHARGE_RATE
DRAIN_RATE = config.DRAIN_RATE
SEED = config.SEED

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
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    

@dataclass
class Robot:
    """Represents a warehouse robot"""
    id: int
    position: Position
    target: Optional[Position] = None
    path: List[Position] = None
    carrying_item: bool = False
    state: str = "idle"  # idle, moving, picking, delivering, charging
    order_id: Optional[int] = None  # easier to track robot current order (so we can mark complete later)
    battery: float = 100.0
    battery_thresh: float = 30.0

    move_steps: int = 0 #steps moved
    wait_steps: int = 0 #steps spent waiiting for a cell 
    

@dataclass
class Order:
    """Represents a customer order"""
    id: int
    item_location: Position  # which shelf has the item
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    arrival_time: float = 0.0
    completion_time: Optional[float] = None
    

class Warehouse:
    """Warehouse environment"""
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.robots: List[Robot] = []
        self.orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.time = 0.0
        self.total_collisions = 0  #See how many times we wait for block to be open 
        
        # Initialize warehouse layout
        self._create_layout()
        
    def _create_layout(self):
        mid_y = self.height // 2
        mid_x = self.width // 2
        
        center_clear_radius = 1 
        for i in range(2, self.height - 2, 3):
            for j in range(2, self.width - 2, 3):
                if abs(i - mid_y) <= center_clear_radius and abs(j - mid_x) <= center_clear_radius:
                    continue
                self.grid[i, j] = CellType.SHELF.value # Place shelves in a grid pattern
         
        # Extra shelves for density
        for i in [3, 6, 9]:
            for j in [4, 7, 10]:
                # Skip if in center area becase of batteries
                if abs(i - mid_y) <= center_clear_radius and abs(j - mid_x) <= center_clear_radius:
                    continue
                if self.grid[i, j] == CellType.EMPTY.value:
                    self.grid[i, j] = CellType.SHELF.value

        dock_positions = [
            (1, mid_x),
            (mid_y, self.width - 2),
            (self.height - 2, mid_x),
            (mid_y, 1), #Docks on the perimeter cetnres
        ]
        for y, x in dock_positions:
            self.grid[y, x] = CellType.LOADING_DOCK.value
        
        #Chargers in the middle 
        charging_positions = [
            (mid_y - 1, mid_x - 1), (mid_y - 1, mid_x),
            (mid_y, mid_x - 1), (mid_y, mid_x),
        ]
        
        for y, x in charging_positions:
            self.grid[y, x] = CellType.CHARGING_STATION.value
        
        for y, x in charging_positions:
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]: 
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if (self.grid[ny, nx] != CellType.LOADING_DOCK.value and 
                        self.grid[ny, nx] != CellType.CHARGING_STATION.value):
                        self.grid[ny, nx] = CellType.EMPTY.value #clear adjacent cells for access to charger
        
    def add_robot(self, position: Position):
        """Add a robot to the warehouse"""
        robot = Robot(
            id=len(self.robots),
            position=position,
            path=[]
        )
        self.robots.append(robot)
        
    def add_order(self, order: Order):
        """Add new order to queue"""
        order.arrival_time = self.time
        self.orders.append(order)
        
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is valid and not occupied"""
        if pos.x < 0 or pos.x >= self.width:
            return False
        if pos.y < 0 or pos.y >= self.height:
            return False
        if self.grid[pos.y, pos.x] == CellType.OBSTACLE.value:
            return False
        # if self.grid[pos.y, pos.x] == CellType.SHELF.value:
        #     return False
        return True
    
    def get_neighbors(self, pos: Position) -> List[Position]:
        """Get valid neighboring positions (4-directional)"""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = Position(pos.x + dx, pos.y + dy)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        return neighbors


class PathPlanner:
    """Pathfinding algorithms for robots"""
    

    @staticmethod
    def robot_priority(robot: Robot) -> int:
        # carrying > pick > idle / other
        if robot.carrying_item or robot.state == "delivering":
            return 0
        elif robot.state in ["picking", "moving"] and not robot.carrying_item:
            return 1
        return 2

    @staticmethod
    def a_star(warehouse: Warehouse,  #map of warehouse 2d arr 
               start: Position,  #start pos of calling robot 
               goal: Position,   #goal pos of calling robot 
               reserved_positions: Set[Tuple[int, int, int]] = None, #include time now so using x, y, t
               start_time: int = 0
               ) -> List[Position]:
        
        if reserved_positions is None:
            reserved_positions = set()

        frontier = [] #priority queue for A* 
        count = 0 # to break ties in priority queue python limitation (incremement for each push)
        heapq.heappush(frontier, (0, count, start_time, start)) #(f score, count, start time, position)


        came_from   = {(start, start_time): None} #map node, time to parent for getting path at end 
        cost_so_far = {(start, start_time): 0} #g score for each node and time  (cost from start to node)

        curr_node = None 
        final_state = None
        max_time = start_time + 200  #had an issue where we got into inf loop, so limit search time to 200 steps 


        while frontier:
            _, _, curr_time, curr_node = heapq.heappop(frontier)

            if curr_node == goal:
                final_state = (curr_node, curr_time)
                break 
            
            if curr_time >= max_time: 
                continue

            potential_moves = warehouse.get_neighbors(curr_node)
            potential_moves.append(curr_node) #can also wait in place
            
            
            for next_node in potential_moves:
                
                next_time = curr_time + 1

                if (next_node.x, next_node.y, next_time) in reserved_positions:
                    continue  # Position reserved at given time 

                new_cost = cost_so_far[(curr_node, curr_time)] + 1

                #Check if this path to next node at next time is better
                if (next_node, next_time) not in cost_so_far or new_cost < cost_so_far[(next_node, next_time)]:
                    cost_so_far[(next_node, next_time)] = new_cost
                    priority = new_cost + PathPlanner.manhattan(next_node, goal)
                    count += 1
                    heapq.heappush(frontier, (priority, count, next_time, next_node))
                    came_from[(next_node, next_time)] = (curr_node, curr_time)

        if final_state is None:
            return []

        #reconstruct path as normal for the robot to follow 
        path = []
        curr = final_state
        while curr is not None:
            path_node, path_time = curr
            path.append(path_node)
            curr = came_from.get(curr)

        path.reverse()

        if len(path) > 0:
            path.pop(0)  #remove start pos cant plan to same cell 

        return path

    
    @staticmethod
    def prioritised_planning(warehouse: Warehouse,
                             robots: List[Robot],
                             goals: List[Position],
                             existing_reservations: Set[Tuple[int, int, int]] = None) -> Dict[int, List[Position]]:
        """
        Multi-robot pathfinding with collision avoidance
        """

        if not robots or not goals:
            return {} # no roobts to plan or no goals to plan to 
        
        #Create a list of robot goal pairs (the robot and their coresponding goal)
        robot_goal_pairs = list(zip(robots, goals))
        robot_goal_pairs.sort(key=lambda rg: PathPlanner.robot_priority(rg[0]))  #Sort based on importance of task  

        all_paths = {} #Store path map robot -> path 
        reserved_states = set() #remember to store x, y, t have time now 
        
        # Add existing reservations (from robots with paths that aren't being replanned)
        if existing_reservations:
            reserved_states.update(existing_reservations)

        #Reserve the robots starting pos (start at 0 time)
        for r in robots:
            reserved_states.add((r.position.x, r.position.y, 0))


        for rob, goal in robot_goal_pairs:
            path = PathPlanner.a_star(warehouse, rob.position, goal, reserved_states)
            all_paths[rob.id] = path # still assigning the robot path 

            if path: 
                # Reserve all positions along the path at their times
                for t, pos in enumerate(path, start=1):  # start at time 1 (robot moves at t=1) since t0 is statring pos 
                    reserved_states.add((pos.x, pos.y, t))
                
                end = path[-1]
                final_time = len(path) #time robot arrives at goal
                for waiting_time in range(1, 3): # reserve the goal for 2 seconds after arrival to prevent crash 
                    reserved_states.add((end.x, end.y, final_time + waiting_time))

        return all_paths


    @staticmethod
    def manhattan(pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)


class TaskAssigner:
    """Assign orders to robots optimally"""
    
    @staticmethod
    def greedy_assignment(warehouse: Warehouse) -> Dict[int, int]:
        pass
    
    @staticmethod
    def optimal_assignment(warehouse: Warehouse) -> Dict[int, int]:
        """
        Optimal task assignment using Hungarian algorithm or auction-based method
        
        Returns: dict of robot_id -> order_id
        """

        #Same idea as baseline, we only get orders that are unassigned completely to prevent race condition 
        idle_robots = [r for r in warehouse.robots if r.state == "idle" and r.battery > r.battery_thresh]
        active_order_ids = {r.order_id for r in warehouse.robots if r.order_id is not None}
        orders = [
            o for o in warehouse.orders 
            if o.completion_time is None and o.id not in active_order_ids
        ]
     
        
        n_robots = len(idle_robots)
        n_orders = len(orders)

        if not idle_robots or not orders: #no avail robots or no pending orders
            return {}

        cost_matrix = np.zeros((n_robots, n_orders)) #init blank cost matrix

        for i, robot in enumerate(idle_robots):
            for j, order in enumerate(orders):
                curr_cost = PathPlanner.manhattan(robot.position, order.item_location) #Cost matrix will store distance from robot to order

                # priority: int = 1  # 1=normal, 2=high, 3=urgent
                # My logic here is that urgent orders can get big negative distance to prioritize them 
                # Might have to tweak this if I have time 

                order_priority = (order.priority - 1) * 50 
                cost_matrix[i, j] = curr_cost - order_priority

        row_indexes, col_indexes = linear_sum_assignment(cost_matrix)
        #scikit returns indices of optimal assignment not values 
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html 

        robot_to_order = {}

        for robot_idx, order_idx in zip(row_indexes, col_indexes):
            robot = idle_robots[robot_idx].id
            order = orders[order_idx].id
            robot_to_order[robot] = order #assign the dict robot -> order 

        return robot_to_order

class OrderPredictor:
    """ML model to predict order patterns"""
    
    def __init__(self, history_num_samples = 24 * 7 ):
        self.time_model = RandomForestRegressor(random_state=42) #Same as our notebook
        self.trained_time_model = False
        # self.last_update_time = 0

        self.shelf_visit_freq = {} #want ti store pos -> visit count 
        self.decay_amnt = 0.95 # equiv to 5% decay per call (60 step) to adapt to movin shelf sampling

        self.history = []
        self.history_num_samples = history_num_samples
    
    def update_shelf_visit_freq(self, pos): #If a shelf x is visited, increment its count in the dict
        dict_key = (pos.x, pos.y)
        self.shelf_visit_freq[dict_key] = float(self.shelf_visit_freq.get(dict_key, 0) + 1)

    def apply_decay_to_freq(self): #just decay the count by time so if its old its forgotten
        for shelf_pos in list(self.shelf_visit_freq.keys()):
            self.shelf_visit_freq[shelf_pos] *= self.decay_amnt
            if self.shelf_visit_freq[shelf_pos] < 0.01:
                del self.shelf_visit_freq[shelf_pos] #No point decaying one so small justremove 

    def update_hourly(self, current_time, order_count): #can call every hour to update models
        # Convert minutes to hours 1 dt = 1 minute
        current_time_hours = current_time / 60.0
        hour_index = int(current_time_hours)
        
        curr_hour_of_week = current_time_hours % 168 
        curr_hour_of_day = current_time_hours % 24
        curr_day_week = int(current_time_hours // 24) % 7 

        # print("Debug: ", current_time, current_time_hours, curr_hour_of_week, curr_hour_of_day, curr_day_week)
        #Might have to add cycles here TODO add if perf bad 
        #https://towardsdatascience.com/how-to-handle-cyclical-data-in-machine-learning-3e0336f7f97c/
        
        self.history.append({
            'time' : current_time, 
            'hour_of_day' : curr_hour_of_day,
            'day_of_week' : curr_day_week,
            'order_count' : order_count
        })

        #if training takes too long add 
        # if len(self.history) > self.history_num_samples:
        #     self.history.pop(0)  #keep history size manageable

    def prep_data (self, df):
        df = df.copy()
        df['prev_hour_count'] = df['order_count'].shift(1) #move one down so we have prev hour count 
        df = df.dropna() #just incase we shifted and got blank 

        #Add cyclical features 
        #Apparently useful for recurring patterns see https://mlpills.substack.com/p/issue-89-encoding-cyclical-features for ref
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)

        all_feats = ['hour_sin', 'hour_cos', 'day_of_week', 'prev_hour_count'] #Add altogether to make up our features 
        X = df[all_feats].values
        y = df['order_count'].values
        return X, y


    def train(self):
        if len(self.history) < 24: #start training after 1 day of data
            return #nothing to update
        
        df = pd.DataFrame(self.history)
        X, y = self.prep_data(df)

        if len(X) > 0:
            self.time_model.fit(X, y)
            self.trained_time_model = True
            # print("OrderPredictor: Updated time model with", len(X), "samples.")
            # print("Feature importances:", self.time_model.feature_importances_)
            # print("model error", np.std(y - self.time_model.predict(X)))
    


    def predict_next_hour(self, current_time: float, curr_hour_count):
        """Predict order volume for next hour"""
        if not self.trained_time_model:
            return 10  # No model trained yet just return a number
        
        next_hour_time = current_time + 60  # next hour in minutes
        next_hour_of_day = (next_hour_time / 60) % 24 
        next_day_week = int(next_hour_time // 1440) % 7  #1440 minutes in day

        #Add in the cyclical features for hour of day 
        next_hour_sin = np.sin(2 * np.pi * next_hour_of_day / 24.0)
        next_hour_cos = np.cos(2 * np.pi * next_hour_of_day / 24.0)

        to_predict = np.array([[next_hour_sin, next_hour_cos, next_day_week, curr_hour_count]])
        predicted_count = self.time_model.predict(to_predict)[0]

        # print("Given current time:", current_time, "and curr hour count:", curr_hour_count, "predicted next hour count:", predicted_count)


        return max(0, int(predicted_count)) #model sometimes gave negative values when not trained well

    
    def predict_hotspots(self, current_time: float) -> List[Position]:
        """Predict which warehouse areas will be busy"""
        if not self.shelf_visit_freq:
            return [] #no data yet, so no hotspots
        
        shelves_sorted = sorted(self.shelf_visit_freq.keys(), key=lambda x: self.shelf_visit_freq[x], reverse=True)
        top_shelves = shelves_sorted[:5] #top 5 most visited shelves

        return [Position(x, y) for (x, y) in top_shelves]

class WarehouseSimulator:
    """Main simulation controller"""
    
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.path_planner = PathPlanner()
        self.task_assigner = TaskAssigner()
        self.dock_locations = self.find_cell_type(CellType.LOADING_DOCK)
        self.charging_stations = self.find_cell_type(CellType.CHARGING_STATION)
        self.shelf_locations = self.find_cell_type(CellType.SHELF)
    
        self.order_schedule = self._generate_daily_schedule()
        self._initialize_hotspots()

        self.next_order_index = 0

        self.order_pred = OrderPredictor() 
        self.current_orders_for_hour = 0
        self.last_time_checked = 0
        self.predicted_next_hour_orders = 0
        
        

    #Daily schedule generated using the given synthetic data generator
    def _generate_daily_schedule(self):

        schedule_rng = np.random.RandomState(SEED)

        gen = SyntheticDataGenerator()

        df = gen.generate_poisson_events(
            n_days=config.N_DAYS, 
            base_rate=config.BASE_RATE,  #base orders per hour         
            peak_hours=config.PEAK_HOURS,
            peak_multiplier=config.PEAK_MULTIPLIER,    #how much busier are we at peak,
            seed=config.SEED
        )
        
        arrival_times = []
    
        STEPS_PER_HOUR = config.STEPS_PER_HOUR # granularity of simulation steps per hour (how many env steps per hour in the dataset)
        #reminder : 1 dt = 1 minute in sim so 60 steps per hour

        for hour, row in df.iterrows():
            count = int(row['event_count']) # order count for this hour
            if count > 0:
                start_time = hour * STEPS_PER_HOUR
                end_time = (hour + 1) * STEPS_PER_HOUR
                
                # Generate random times within this hour
                times = schedule_rng.uniform(start_time, end_time, count)
                arrival_times.extend(times)
                
        return sorted(arrival_times) #sort them so we process morn to night 
    
    #Data generator for choosing favoured shelves 
    def _initialize_hotspots(self, hotspot_ratio: float = 0.2, hotspot_weight: float = 5.0):
        shelves = np.argwhere(self.warehouse.grid == CellType.SHELF.value)
        n_shelves = len(shelves)
        
        #Randomly select hotspot shelves
        n_hotspots = max(1, int(n_shelves * hotspot_ratio))
        hotspot_indices = np.random.choice(n_shelves, size=n_hotspots, replace=False)

        self.shelf_weights = np.ones(n_shelves) #Give favoured shelfs more weight
        self.shelf_weights[hotspot_indices] = hotspot_weight

        self.hotspot_shelves = set()
        for idx in hotspot_indices:
            y, x = shelves[idx]
            self.hotspot_shelves.add(Position(x, y))
    
        self.shelf_probabilities = self.shelf_weights / self.shelf_weights.sum() #put in range 0-1

    #Find all the positions of a given cell type
    def find_cell_type(self, cell_type: CellType) -> List[Position]:
        positions = []
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                if self.warehouse.grid[y, x] == cell_type.value:
                    positions.append(Position(x, y))
        return positions

    #Main robot movement logic for ML spatial check given temporal trigger
    def ml_robot_movement(self):
        high_load_thresh = 10  #orders per hour if have time make this dynamic based on past data
        idle_robots = [r for r in self.warehouse.robots if r.state == "idle"] #Apply logic to only idle robots

        if self.predicted_next_hour_orders >= high_load_thresh: #Trigger, if there is high load coming in the next hour 
            # print("Detected high load")

            hotspots = self.order_pred.predict_hotspots(self.warehouse.time)
            if hotspots:
                for i, robot in enumerate(idle_robots):
                    robot.battery_thresh = 30.0 #reset since we will be moving robots
                    if robot.battery > robot.battery_thresh + 10: #have enough battery to move
                        curr_hotspot = hotspots[i % len(hotspots)]

                        nearby_cell_hotspot = []
                        for dx in range(-3, 4): #scan around the hotspot to find an empty cell for us 
                            for dy in range(-3, 4):
                                pos_to_go = Position(curr_hotspot.x + dx, curr_hotspot.y + dy)
                                if self.warehouse.is_valid_position(pos_to_go):
                                    nearby_cell_hotspot.append(pos_to_go) #Want to Move robot to that cell by hotspot 

                        if nearby_cell_hotspot:
                            #Scatter robots around the hotspot to avoid congestion and improve number of orders we get out 
                            target_index = (i * 3) % len(nearby_cell_hotspot) 
                            robot.target = nearby_cell_hotspot[target_index]
                            robot.state = "moving"


            else: #fall back to old middle warehouse strategy (robots wait in middle)
                existing_path_reservations = set()
                warehouse_middle = self.warehouse.width // 2, self.warehouse.height // 2

                for robot in idle_robots:
                    robot.battery_thresh = 30.0  #reset to normal threshold
                    if robot.battery > robot.battery_thresh + 10: #have enough battery to move
                        x_area_possible = np.random.randint(-5, 5)
                        y_area_possible = np.random.randint(-5, 5)

                        robot_target = Position(
                            min(self.warehouse.width -1, max(0, warehouse_middle[0] + x_area_possible)),
                            min(self.warehouse.height -1, max(0, warehouse_middle[1] + y_area_possible))
                        )

                        if self.warehouse.is_valid_position(robot_target):
                            robot.target = robot_target
                            robot.state = "moving"

        else:
            #not many orders can charge more 
            for robot in idle_robots:
                if robot.battery < 60.0:
                    robot.battery_thresh = 60.0

    def step(self, dt: float = 1.0):
        self.warehouse.time += dt
        
        while (self.next_order_index < len(self.order_schedule) and  #still have orders to process
               self.order_schedule[self.next_order_index] <= self.warehouse.time): #dont schedule orders in future
            
            self._generate_random_order()
            self.next_order_index += 1
            self.current_orders_for_hour += 1


        #ML Data update
        if self.warehouse.time - self.last_time_checked >= 60: #atleast every hour
            self.order_pred.update_hourly(self.warehouse.time, self.current_orders_for_hour)

            self.order_pred.train() #retrain 

            self.predicted_next_hour_orders = self.order_pred.predict_next_hour(self.warehouse.time, self.current_orders_for_hour) #predict next hour
            
            self.order_pred.apply_decay_to_freq() #decay our freqyency track every hour

            # print("Time:", self.warehouse.time, "Ended hour with ", self.current_orders_for_hour, 
            #       "orders. Predicted next hour orders:", self.predicted_next_hour_orders)
            
            #Update robots movement (no planning done here just target assign)
            self.ml_robot_movement()

            self.last_time_checked = self.warehouse.time
            self.current_orders_for_hour = 0 #reset for next hour   

        # # Generate orders
        # if np.random.random() < 0.1:  # 10% chance per step
        #     self._generate_random_order()
        
        robots_to_plan = [] #going to plan altogether now so we can avoid collisions
        goals_to_plan = []

        #Check if there is a peak coming in the next hour using temporal model 
        is_peak_coming = self.predicted_next_hour_orders > 10 #was 15, check if still good
        
        #If peak is coming strict charging charge everyone idle below 80 
        #else can relax no peak incomming just normal orders
        opportunity_thresh = 80.0 if is_peak_coming else 40.0

        critical_thresh = 25.0 #To prevent robots going flat make sure theyre forced to charge 

        robots_need_charge = []
        charging_stations = self.charging_stations 


        for robot in self.warehouse.robots:
            if robot.state == "charging":
                #IF already charging, go until 90 if there is demand or 100 if relaxed period 
                robot.battery = min(100.0, robot.battery + CHARGE_RATE * dt)
                stop_charge_thresh = 90.0 if is_peak_coming else 100.0
                
                if robot.battery >= stop_charge_thresh:
                    robot.state = "idle"
                continue 

        
            should_charge = False
            if robot.battery <= critical_thresh:
                should_charge = True #have to have to charge now 
                
            elif robot.state == "idle" and robot.battery <= opportunity_thresh:
                should_charge = True #should charge if robot isnt busy and we can gain some battery 

            if should_charge: #If a robot marked to be charged
                if charging_stations: #

                    occupied_positions = {r.position for r in self.warehouse.robots} #make sure dont plan to one thats in use
                    
                    free_stations = [pos for pos in charging_stations if pos not in occupied_positions]
                    
                    candidates = free_stations if free_stations else charging_stations #prioritise an open close one otherwise any one is fine
                    closest_station = min(candidates, key=lambda pos: self.path_planner.manhattan(robot.position, pos))
                    robot.target = closest_station
                    
                    #If robot is idle or moving and not carrying an item mark to be charged at given station
                    if robot.state in ["idle", "moving"] and not robot.carrying_item:
                        robot.state = "moving"
                        robots_need_charge.append(robot)
                
                        path = self.path_planner.a_star(self.warehouse, robot.position, robot.target)
                        if path:
                            robot.path = path
                             
        #add to be planned
        robots_to_plan.extend(robots_need_charge)
        goals_to_plan.extend([r.target for r in robots_need_charge])

        #Assign orders to robots
        assignments = self.task_assigner.optimal_assignment(self.warehouse)
        
        for robot in self.warehouse.robots:
            if robot.state == "idle" and robot.id in assignments: #robots been assigned, set the order and target
                order_id = assignments[robot.id] 
                order = next(o for o in self.warehouse.orders if o.id == order_id)

                robot.target = order.item_location
                robot.order_id = order.id
                robots_to_plan.append(robot)
                goals_to_plan.append(order.item_location)

            #Move idle robots off docks to random empty cells
            #Had a bug where a robot would deliver and go idle on the dock causing a lock up 
            elif robot.state == "idle" and robot.position in self.dock_locations and robot.target is None:
                if robot.battery > robot.battery_thresh: #if we have enough battery to move, get off the dock
                    empty_cells = []
                    for y in range(self.warehouse.height):
                        for x in range(self.warehouse.width):
                            pos = Position(x, y)
                            if (self.warehouse.grid[y, x] == CellType.EMPTY.value and 
                                pos not in self.dock_locations and
                                not any(r.position == pos for r in self.warehouse.robots)):
                                empty_cells.append(pos)
                    
                    if empty_cells:
                        robot.target = random.choice(empty_cells)
                        robots_to_plan.append(robot)
                        goals_to_plan.append(robot.target)

            #Replan and double check to make sure robots have a path
            elif robot.state in ["picking", "delivering", "moving"] and robot.target:
                if not robot.path:
                    robots_to_plan.append(robot)
                    goals_to_plan.append(robot.target)


        if robots_to_plan:
            existing_path_reservations = set() #start with no reservations
            for robot in self.warehouse.robots:
                if robot.path and robot.id not in [r.id for r in robots_to_plan]:
                    for t, pos in enumerate(robot.path, start=1): 
                        existing_path_reservations.add((pos.x, pos.y, t)) #for each pos in the path add it to be reserved for this robot
            
            #plan using our prioritised planning algorithm
            panned_paths = self.path_planner.prioritised_planning(self.warehouse, robots_to_plan, goals_to_plan, existing_path_reservations)

            for rob in robots_to_plan:
                if rob.id in panned_paths:
                    rob.path = panned_paths[rob.id]
                    if rob.path:
                        rob.state = "moving"


        reserved_next_positions = set()  # x, y of hard obstacles 
        approved_moves = {}  # robot id -> position stores approved moves (ones allowed)

        #Sort by task priority like we did above
        sorted_robots = sorted(self.warehouse.robots, key=lambda r: 0 if r.carrying_item else 1)

        for robot in sorted_robots:
            next_pos = robot.position #default to staying where you are
        
            if robot.path and robot.state == "moving": #if we have a path then we move
                candidate = robot.path[0]
                
                #Check to make sure the next cell isnt claimed by a robot that is more important
                if (candidate.x, candidate.y) in reserved_next_positions:
                    next_pos = robot.position
                
                #This is needed to prevent swaps 
                elif any(r.position == candidate and r.id != robot.id for r in self.warehouse.robots):
                    next_pos = robot.position
                
                #stayt still if the next cell is invalid 
                elif not self.warehouse.is_valid_position(candidate):
                    next_pos = robot.position
                
                else:
                    next_pos = candidate #can move 

            approved_moves[robot.id] = next_pos
            reserved_next_positions.add((next_pos.x, next_pos.y))

        for robot in self.warehouse.robots:
            approved_next_pos = approved_moves[robot.id]
            self.move_robot_logic(robot, approved_next_pos, dt) 

        
    def _generate_random_order(self):
        """Generate a random order"""
        shelves = np.argwhere(self.warehouse.grid == CellType.SHELF.value)
        if len(shelves) > 0:
            shelf_idx = np.random.choice(len(shelves), p=self.shelf_probabilities)
            shelf = shelves[shelf_idx]
            order = Order(
                id=len(self.warehouse.orders) + len(self.warehouse.completed_orders),
                item_location=Position(shelf[1], shelf[0]),
                priority=np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            )
            self.warehouse.add_order(order)

            self.order_pred.update_shelf_visit_freq(order.item_location) #update our freqyency track

    def move_robot_logic(self, robot: Robot, next_pos: Position, dt: float):

        if robot.state == "moving" and robot.path:
            robot.battery = max(0.0, robot.battery - DRAIN_RATE * dt) #drain battery as we move

        if robot.battery <= 0.0:
            robot.state = "flat" #flat robots dont move
            robot.path = []
            if robot.position in self.dock_locations:
                print(f"Robot {robot.id} went flat on dock {robot.position.x} {robot.position.y}")
            return


        if next_pos != robot.position: #Robot successfully moved 
            robot.position = next_pos
            robot.move_steps += 1
            
            if robot.path and robot.path[0] == next_pos:
                robot.path.pop(0)
                
        else:
            if robot.state == "moving" and robot.path: #robot pos didnt change so wait
                robot.wait_steps += 1
                
                if robot.wait_steps > 2 and robot.wait_steps < 5:
                    robot.path = [] # Force a replan to prevent a deadlock (had this happen with occupied chargers)
                    
        
                elif robot.wait_steps >= 5: #Here we move to a random cell nxt to us to prevent replanning the same bad path or to unblock a robot
                    neighbors = self.warehouse.get_neighbors(robot.position)
                    occupied = {r.position for r in self.warehouse.robots}
                    valid_escape_routes = [n for n in neighbors if n not in occupied]
                    
                    if valid_escape_routes:
                        escape_node = valid_escape_routes[random.randint(0, len(valid_escape_routes)-1)]
                        robot.path = [escape_node] 
                        robot.wait_steps = 0

        if robot.target is not None and robot.position == robot.target: #We have arrived at our target
            
            if robot.position in self.charging_stations:
                robot.state = "charging"
                robot.path = []
                return

            if not robot.carrying_item:
                robot.state = "picking"
                robot.path = []
            else: 
                robot.state = "delivering"
                robot.path = []

       
        if robot.state == "picking": #collecting an order to deliver
            robot.carrying_item = True
            if self.dock_locations:
                #Same as above, try closest dock otherwise pick any oppen one 
                occupied_positions = {r.position for r in self.warehouse.robots}
                free_docks = [d for d in self.dock_locations if d not in occupied_positions]
                candidates = free_docks if free_docks else self.dock_locations
                
                if not free_docks:
                    target_dock = candidates[random.randint(0, len(candidates)-1)]
                else:
                    target_dock = min(candidates, key=lambda pos: self.path_planner.manhattan(robot.position, pos)) #Closest dock
                
                robot.target = target_dock

        elif robot.state == "delivering":  # robot on the dock cell
            if robot.order_id is not None:
                order_idx = -1
                for i, o in enumerate(self.warehouse.orders):
                    if o.id == robot.order_id:
                        order_idx = i
                        break
                
                if order_idx != -1:
                    order = self.warehouse.orders.pop(order_idx)
                    order.completion_time = self.warehouse.time
                    self.warehouse.completed_orders.append(order)
                else:
                    print(f"error robot is at the dock but delivering an order that doesnt exist anymore (reassigned)")

            #order is completed, reset robot state
            robot.carrying_item = False
            robot.target = None
            robot.path = []
            robot.order_id = None
            robot.state = "idle" 
    
    def get_metrics(self) -> Dict:
        completed = self.warehouse.completed_orders
        prio_times = {1: [], 2: [], 3: []}
        for o in completed:
            if o.completion_time:
                duration = o.completion_time - o.arrival_time
                prio_times[o.priority].append(duration)

        total_times = [t for sublist in prio_times.values() for t in sublist]

        # aggregate wait/move across robots
        total_wait = sum(r.wait_steps for r in self.warehouse.robots)
        total_move = sum(r.move_steps for r in self.warehouse.robots)
        coord_overhead = (total_wait / (total_wait + total_move)) if (total_wait + total_move) > 0 else 0.0

        # battery metrics
        battery_levels = [r.battery for r in self.warehouse.robots]
        avg_battery = np.mean(battery_levels) if battery_levels else 0
        min_battery = np.min(battery_levels) if battery_levels else 0
        charging_robots = sum(1 for r in self.warehouse.robots if r.state == "charging")

        metrics = {
            'total_completed': len(completed),
            'avg_time_all': np.mean(total_times) if total_times else 0,
            'pending_orders': len(self.warehouse.orders),
            'coord_overhead': coord_overhead,
            'total_wait_steps': total_wait,
            'total_move_steps': total_move,
            'avg_battery': avg_battery,
            'min_battery': min_battery,
            'charging_robots': charging_robots,
        }
        for p in [1, 2, 3]:
            metrics[f'avg_time_P{p}'] = np.mean(prio_times[p]) if prio_times[p] else 0
            metrics[f'count_P{p}'] = len(prio_times[p])
        return metrics
    
def main():
    """Example usage"""
    np.random.seed(config.SEED)  # ensure reproducible runs
    random.seed(config.SEED)
    print("Warehouse Robot Fleet Coordination - Starter Code")
    print("=" * 50)
    
    # Create warehouse
    warehouse = Warehouse(width=config.WAREHOUSE_WIDTH, height=config.WAREHOUSE_HEIGHT)
    
    # Add robots
    for i in range(config.NUM_ROBOTS):
        warehouse.add_robot(Position(i, 0))
    
    print(f"Created warehouse: {warehouse.width}x{warehouse.height}")
    print(f"Added {len(warehouse.robots)} robots")
    
    # Create simulator
    sim = WarehouseSimulator(warehouse)
    
    # Run simulation
    print("\nRunning simulation...")
    # Calculate total steps: N_DAYS * 24 hours * STEPS_PER_HOUR
    total_steps = config.N_DAYS * 24 * config.STEPS_PER_HOUR
    for step in range(total_steps):
        sim.step(dt=1.0)
        
        if step % config.PRINT_INTERVAL == 0:
            metrics = sim.get_metrics()
            print(f"Step {step}: {metrics}")

    
    #test the a* 
    # start = Position(0, 0)
    # goal = Position(5, 5)
    # path = PathPlanner.a_star(warehouse, start, goal)
    # print(f"Path from {start} to {goal}:")
    # for pos in path:
    #     print(f"({pos.x}, {pos.y})", end=" -> ")
    


if __name__ == "__main__":
    main()
