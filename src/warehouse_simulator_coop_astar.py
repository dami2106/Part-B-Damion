"""
Warehouse Robot Fleet Coordination - Starter Code
Problem 3: Multi-Robot Task Assignment and Pathfinding

This is a skeleton to get you started quickly.
Feel free to modify, extend, or completely rewrite!
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional
import heapq
from enum import Enum
from src.synthetic_data import SyntheticDataGenerator   

from scipy.optimize import linear_sum_assignment
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
        self.total_collisions = 0  #Added this to see how many times we wait for block to be open 
        
        # Initialize warehouse layout
        self._create_layout()
        
    def _create_layout(self):
        
        for i in range(2, self.height - 2, 3):
            for j in range(2, self.width - 2, 3):
                self.grid[i, j] = CellType.SHELF.value # Place shelves in a grid pattern
         
        #Exttra shelves to increase density (hopefully more clashes makes it harder)
        for i in [3, 6, 9]:
            for j in [4, 7, 10]:
                if self.grid[i, j] == CellType.EMPTY.value:
                    self.grid[i, j] = CellType.SHELF.value
        
        # Loading docks: top middle, right middle, bottom middle, left middle
        dock_positions = [
            (0, self.width // 2),              # Top middle
            (self.height // 2, self.width - 1), # Right middle
            (self.height - 1, self.width // 2), # Bottom middle
            (self.height // 2, 0),              # Left middle
        ]
        for y, x in dock_positions:
            self.grid[y, x] = CellType.LOADING_DOCK.value
        
        charging_positions = [
            (0, 0),
            (0, self.width - 1),
            (self.height - 1, 0),
            (self.height - 1, self.width - 1),
        ]
        for y, x in charging_positions:
            self.grid[y, x] = CellType.CHARGING_STATION.value
        
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

        frontier = []
        count = 0 # to break ties in priority queue python limitation (incremement for each push)
        heapq.heappush(frontier, (0, count, start_time, start)) #(f score, count, start time, position)


        came_from   = {(start, start_time): None} #map node, time to parent for getting path at end 
        cost_so_far = {(start, start_time): 0} #g score for each node and time  (cost from start to node)

        curr_node = None 
        final_state = None
        max_time = start_time + 200  #had an issue where we got into inf loop, so limit search time



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

                # Check if this path to next_node at next_time is better
                
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
    def conflict_based_search(warehouse: Warehouse,
                             robots: List[Robot],
                             goals: List[Position],
                             existing_reservations: Set[Tuple[int, int, int]] = None) -> Dict[int, List[Position]]:
        """
        Multi-robot pathfinding with collision avoidance
        """

        if not robots or not goals:
            return {} # no roobts to plan or no goals to plan to 
        
        robot_goal_pairs = list(zip(robots, goals))
        robot_goal_pairs.sort(key=lambda rg: PathPlanner.robot_priority(rg[0]))    

        all_paths = {} #Store path map robot -> path 
        reserved_states = set() #remember to store x, y, time now 
        
        # Add existing reservations (from robots with paths that aren't being replanned)
        if existing_reservations:
            reserved_states.update(existing_reservations)

        # new - reserve pos at t = 0 to prevent conflicts 
        for r in robots:
            reserved_states.add((r.position.x, r.position.y, 0))


        for rob, goal in robot_goal_pairs:
            path = PathPlanner.a_star(warehouse, rob.position, goal, reserved_states)
            all_paths[rob.id] = path # still assigning the robot path 

            if path: 
                # Reserve all positions along the path at their respective times
                for t, pos in enumerate(path, start=1):  # start at time 1 (robot moves at t=1)
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
        """
        Greedy task assignment: assign each order to nearest available robot
        
        Returns: dict of robot_id -> order_id
        """
        assignment = {}
        
        available_robots = [r for r in warehouse.robots if r.state == "idle" and r.battery > r.battery_thresh]
        pending_orders = [o for o in warehouse.orders if o.completion_time is None]
        
        #sort orders by priority (higher first)
        pending_orders.sort(key=lambda o: o.priority, reverse=True)


        # Simple greedy: for each order, find closest robot
        for order in pending_orders:
            if not available_robots:
                break
            
            # Find closest robot
            min_dist = float('inf')
            best_robot = None
            for robot in available_robots:
                dist = abs(robot.position.x - order.item_location.x) + \
                       abs(robot.position.y - order.item_location.y)
                if dist < min_dist:
                    min_dist = dist
                    best_robot = robot
            
            if best_robot:
                assignment[best_robot.id] = order.id
                available_robots.remove(best_robot)
        
        return assignment
    
    @staticmethod
    def optimal_assignment(warehouse: Warehouse) -> Dict[int, int]:
        """
        Optimal task assignment using Hungarian algorithm or auction-based method
        
        Returns: dict of robot_id -> order_id
        """

        idle_robots = [r for r in warehouse.robots if r.state == "idle" and r.battery > r.battery_thresh]
        n_robots = len(idle_robots)
        orders = [o for o in warehouse.orders if o.completion_time is None] #incomplete orders have no completion 
        n_orders = len(orders)

        if not idle_robots or not orders: #no avail robots or no pending orders
            return {}

        cost_matrix = np.zeros((n_robots, n_orders))

        for i, robot in enumerate(idle_robots):
            for j, order in enumerate(orders):
                curr_cost = PathPlanner.manhattan(robot.position, order.item_location)

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
    
    def __init__(self):
        self.model = None
        
    def train(self, historical_orders: np.ndarray):
        """Train model to predict order volume and patterns"""
        # TODO: Implement time series forecasting
        # - Seasonality (time of day, day of week)
        # - Trends
        # - Anomaly detection for flash sales
        pass
    
    def predict_next_hour(self, current_time: float) -> np.ndarray:
        """Predict order volume for next hour"""
        # TODO: Return predicted order counts
        # Placeholder: assume 10 orders per hour with variation
        return np.random.poisson(10, size=1)
    
    def predict_hotspots(self, current_time: float) -> List[Position]:
        """Predict which warehouse areas will be busy"""
        # TODO: Predict where robots should pre-position
        # Based on historical patterns
        return []


class WarehouseSimulator:
    """Main simulation controller"""
    
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.path_planner = PathPlanner()
        self.task_assigner = TaskAssigner()
        self.order_predictor = OrderPredictor()
        self.dock_locations = self.find_cell_type(CellType.LOADING_DOCK)
        self.charging_stations = self.find_cell_type(CellType.CHARGING_STATION)
        self.shelf_locations = self.find_cell_type(CellType.SHELF)
    
        self.order_schedule = self._generate_daily_schedule()
        self.next_order_index = 0
        
        
        self._initialize_hotspots()

    #Daily schedule generated using the given synthetic data generator
    def _generate_daily_schedule(self):

        schedule_rng = np.random.RandomState(SEED)

        gen = SyntheticDataGenerator()
        
        #One day for now with peaks at 9, and 5 
        df = gen.generate_poisson_events(
            n_days=config.N_DAYS, 
            base_rate=config.BASE_RATE,  #base orders per hour         
            peak_hours=config.PEAK_HOURS,
            peak_multiplier=config.PEAK_MULTIPLIER,    #how much busier are we at peak,
            seed=config.SEED
        )
        
        arrival_times = []
    
        STEPS_PER_HOUR = config.STEPS_PER_HOUR # granularity of simulation steps per hour (how many env steps per hour in the dataset)
        
        for hour, row in df.iterrows():
            count = int(row['event_count']) # order count for this hour
            if count > 0:
                start_time = hour * STEPS_PER_HOUR
                end_time = (hour + 1) * STEPS_PER_HOUR
                
                # Generate random times within this hour
                times = schedule_rng.uniform(start_time, end_time, count)
                arrival_times.extend(times)
                
        return sorted(arrival_times) #sort them so we process morn to night 
    
    def _initialize_hotspots(self, hotspot_ratio: float = 0.2, hotspot_weight: float = 5.0):
        shelves = np.argwhere(self.warehouse.grid == CellType.SHELF.value)
        n_shelves = len(shelves)
        
        # Randomly select hotspot shelves
        n_hotspots = max(1, int(n_shelves * hotspot_ratio))
        hotspot_indices = np.random.choice(n_shelves, size=n_hotspots, replace=False)

        self.shelf_weights = np.ones(n_shelves)
        self.shelf_weights[hotspot_indices] = hotspot_weight #weights to the hotspot shelves

        self.hotspot_shelves = set()
        for idx in hotspot_indices:
            y, x = shelves[idx]
            self.hotspot_shelves.add(Position(x, y))
    
        self.shelf_probabilities = self.shelf_weights / self.shelf_weights.sum() #put in range 0-1
    def find_cell_type(self, cell_type: CellType) -> List[Position]:
        positions = []
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                if self.warehouse.grid[y, x] == cell_type.value:
                    positions.append(Position(x, y))
        return positions


    def step(self, dt: float = 1.0):
        self.warehouse.time += dt
        
        while (self.next_order_index < len(self.order_schedule) and  #still have orders to process
               self.order_schedule[self.next_order_index] <= self.warehouse.time): #dont schedule orders in future
            
            self._generate_random_order()
            self.next_order_index += 1

        # # Generate orders
        # if np.random.random() < 0.1:  # 10% chance per step
        #     self._generate_random_order()
        
        robots_to_plan = [] #going to plan altogether now so we can avoid collisions
        goals_to_plan = []

        robots_needing_charge = []
        charging_stations = self.charging_stations 

        for robot in self.warehouse.robots: #if robot is on charge let it charge 
            if robot.state == "charging":
                robot.battery = min(100.0, robot.battery + CHARGE_RATE * dt) # Charge rate
                if robot.battery >= 100.0:
                    robot.state = "idle" #Fully charged 
                continue 

            # Check active robots if battery is nearly flat
            if robot.state == "idle" and robot.battery <= robot.battery_thresh:
                # Find nearest charger
                if charging_stations:
                    closest_station = min(charging_stations, key=lambda pos: self.path_planner.manhattan(robot.position, pos))
                    robot.target = closest_station
                    robot.state = "moving"
                    robots_needing_charge.append(robot)
                    robots_to_plan.append(robot)
                    goals_to_plan.append(robot.target)

        # if robots_needing_charge:
        #     goals = [r.target for r in robots_needing_charge]
        #     paths = self.path_planner.conflict_based_search(self.warehouse, robots_needing_charge, goals)
        #     for r in robots_needing_charge:
        #         if r.id in paths:
        #             r.path = paths[r.id] #For robots that need to charge, set their paths


        # assignments = self.task_assigner.greedy_assignment(self.warehouse) # assign based on first match 
        assignments = self.task_assigner.optimal_assignment(self.warehouse) # optimal assignment
        
        

        for robot in self.warehouse.robots:
            if robot.state == "idle" and robot.id in assignments:
                order_id = assignments[robot.id] 
                order = next(o for o in self.warehouse.orders if o.id == order_id)

                robot.target = order.item_location
                robot.order_id = order.id #so we know when dropped off etc 

                robots_to_plan.append(robot)
                goals_to_plan.append(order.item_location)

            #dont replan every step (may remove if buggy)
            elif robot.state in ["picking", "delivering"]:
                
                if robot.target and not robot.path:
                    robots_to_plan.append(robot)
                    goals_to_plan.append(robot.target)
        

        #  re-plan for robots already moving or picking 
        for robot in self.warehouse.robots:
            if robot.state in ["moving", "picking"] and robot.target is not None:
                if not robot.path: #make sure we dont double plan for idle robots above
                    robots_to_plan.append(robot)
                    goals_to_plan.append(robot.target)

        if robots_to_plan:
            existing_path_reservations = set()
            for robot in self.warehouse.robots:
                if robot.path and robot.id not in [r.id for r in robots_to_plan]:
                    for t, pos in enumerate(robot.path, start=1):
                        existing_path_reservations.add((pos.x, pos.y, t)) #reserve path positions for other bots, no overlap ever
            
            panned_paths = self.path_planner.conflict_based_search(self.warehouse, robots_to_plan, goals_to_plan, existing_path_reservations)

            for rob in robots_to_plan:
                if rob.id in panned_paths:
                    rob.path = panned_paths[rob.id]
                    if rob.path:
                        rob.state = "moving"
                    elif rob.position == rob.target:
                        pass #already at target no need to move
                    else:
                        pass

        robot_positions = {r.position for r in self.warehouse.robots} #track occupied cells
        for robot in self.warehouse.robots:
            self._move_robot(robot, robot_positions, dt) #finally move all 

        
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

    def _move_robot(self, robot: Robot, robot_positions: Set[Position], dt: float):
            """Move robot one step along its path"""
            
            if robot.state == "moving" and robot.path:
                robot.battery = max(0.0, robot.battery - DRAIN_RATE * dt) # drain battery 

            if robot.battery <= 0.0:
                robot.state = "flat"
                robot.path = []
                return

            # have a path and either moving or need to move 
            if robot.path and robot.state in ["moving", "idle"]:
                robot.state = "moving"
                next_pos = robot.path[0] 

                if next_pos.x == robot.position.x and next_pos.y == robot.position.y:
                    robot.wait_steps += 1 #waiting in same cell robot didnt move this turn 
                else:
                    robot.move_steps += 1 #robot moved this turn
    
                if robot.position in robot_positions:
                    robot_positions.remove(robot.position) #free up current cell so dont conflict a*
                
                robot.position = next_pos 
                robot_positions.add(robot.position)
                
                robot.path.pop(0) #Remove next step from the path to move 

            if robot.target is not None and robot.position == robot.target: #Arrived at target 
                
                if robot.position in self.charging_stations: #hack to set charging state without adding a moving to charge state
                    robot.state = "charging"
                    robot.path = []
                    return

                if not robot.carrying_item:
                    robot.state = "picking" #Picking item, path reset (on shelf)
                    robot.path = []
                else: 
                    robot.state = "delivering" #Delivering now path reset (on dock)
                    robot.path = []

            if robot.state == "picking": #on shelf picking order 
                robot.carrying_item = True
                if self.dock_locations:
                    # Find nearest dock
                    closest_dock = min(self.dock_locations, key=lambda pos: self.path_planner.manhattan(robot.position, pos))
                    robot.target = closest_dock 


            elif robot.state == "delivering": #on dock delivering 
                if robot.position == robot.target and robot.order_id is not None:
                    order_idx = -1
                    for i, o in enumerate(self.warehouse.orders):
                        if o.id == robot.order_id:
                            order_idx = i
                            break
                    
                    if order_idx != -1:
                        order = self.warehouse.orders.pop(order_idx)
                        order.completion_time = self.warehouse.time
                        self.warehouse.completed_orders.append(order)
                        #Order processed go back to idle 

                    # reset
                    robot.carrying_item = False
                    robot.target = None
                    robot.path = []
                    robot.order_id = None
                    robot.state = "idle"
                    
            elif robot.state == "idle":
                pass # robot is idle, wait a step 
    
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
    
    print("\nNext steps:")
    print("1. Implement A* pathfinding")
    print("2. Implement multi-robot coordination (CBS)")
    print("3. Implement optimal task assignment")
    print("4. Add ML for demand prediction")
    print("5. Create visualization")
    print("6. Benchmark vs baseline")


if __name__ == "__main__":
    main()
