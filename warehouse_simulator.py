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
from synthetic_data import SyntheticDataGenerator   

from scipy.optimize import linear_sum_assignment

CHARGE_RATE = 5.0
DRAIN_RATE = 0.1
SEED = 42

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
    battery: float = 100.0  # battery level percentage
    battery_thresh: float = 20.00 #when to top up 

    move_steps: int = 0
    wait_steps: int = 0
    

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
        """Create warehouse layout with shelves, docks, charging stations"""
        # TODO: Design warehouse layout
        # Example: shelves in grid pattern, loading dock at bottom, charging stations
        
        # Add some shelves (example)
        for i in range(2, self.height - 2, 3):
            for j in range(2, self.width - 2, 3):
                self.grid[i, j] = CellType.SHELF.value
        
        # Loading dock at bottom
        self.grid[-1, self.width // 2] = CellType.LOADING_DOCK.value
        
        # Charging stations
        self.grid[0, 0] = CellType.CHARGING_STATION.value
        self.grid[0, -1] = CellType.CHARGING_STATION.value
        
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
    def a_star(warehouse: Warehouse,  #map of warehouse 2d arr 
               start: Position,  #start pos of calling robot 
               goal: Position,   #goal pos of calling robot 
               reserved_positions: Set[Position] = None #can populate this list with reserved grid pos to avoid collisions
               ) -> List[Position]:
        
        if reserved_positions is None:
            reserved_positions = set()

        frontier = []
        count = 0 # to break ties in priority queue python limitation (incremement for each push)

        heapq.heappush(frontier, (0, count, start)) #(f score, count, position)
        came_from = {start: None} #map node to parent for getting path at end 
        cost_so_far = {start: 0} #g score for each node (cost from start to node)

        curr_node = None 

        while frontier:
            curr_node = heapq.heappop(frontier)[2]
            if curr_node == goal:
                break 

            for next_node in warehouse.get_neighbors(curr_node):
                if next_node in reserved_positions and next_node != goal:
                    continue

                new_cost = cost_so_far[curr_node] + 1

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost

                    priority = new_cost + PathPlanner.manhattan(next_node, goal)
                    
                    count += 1
                    heapq.heappush(frontier, (priority, count, next_node))
                    came_from[next_node] = curr_node

        if curr_node != goal:
            return []  # No path found havent reached goal empty path 
        
        path = []
        while curr_node is not None:
            path.append(curr_node)
            curr_node = came_from[curr_node]
        
        path.reverse() #planned from agent to goal, need to reverse to get agent next move 
        if len(path) > 0:
            path.pop(0)
        return path


    
    @staticmethod
    def conflict_based_search(warehouse: Warehouse,
                             robots: List[Robot],
                             goals: List[Position]) -> Dict[int, List[Position]]:
        """
        Multi-robot pathfinding with collision avoidance
        
        TODO: Implement CBS or similar algorithm
        - Find paths for all robots
        - Resolve conflicts (same position at same time)
        - Return dict of robot_id -> path
        """
        # Placeholder
        paths = {}
        for robot, goal in zip(robots, goals):
            paths[robot.id] = PathPlanner.a_star(warehouse, robot.position, goal)
        return paths
    
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

    #Daily schedule generated using the given synthetic data generator
    def _generate_daily_schedule(self):
        gen = SyntheticDataGenerator()
        
        #One day for now with peaks at 9, and 5 
        df = gen.generate_poisson_events(
            n_days=7, 
            base_rate=10,  #base orders per hour         
            peak_hours=[9, 17],
            peak_multiplier=3.0,    #how much busier are we at peak,
            seed=SEED
        )
        
        arrival_times = []
    
        STEPS_PER_HOUR = 60 # granularity of simulation steps per hour (how many env steps per hour in the dataset)
        
        for hour, row in df.iterrows():
            count = int(row['event_count']) # order count for this hour
            if count > 0:
                start_time = hour * STEPS_PER_HOUR
                end_time = (hour + 1) * STEPS_PER_HOUR
                
                # Generate random times within this hour
                times = np.random.uniform(start_time, end_time, count)
                arrival_times.extend(times)
                
        return sorted(arrival_times) #sort them so we process morn to night 
    

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
        
        #Battery code here 
        robots_need_charge = []
        charging_stations = self.charging_stations

        for robot in self.warehouse.robots:
            if robot.state == "charging":
                robot.battery = min(100.0, robot.battery + CHARGE_RATE * dt)
                if robot.battery >= 100.0:
                    robot.state = "idle" #done charging 
                continue 
                
            if robot.state == "idle" and robot.battery <= robot.battery_thresh:
                robots_need_charge.append(robot)    
                if charging_stations:
                    robot.target = charging_stations[0] #go to first charging station 
                    robot.state = "moving"

        if robots_need_charge:
            for robot in robots_need_charge:
                 path = self.path_planner.a_star(self.warehouse, robot.position, robot.target)
                 if path:
                     robot.path = path 


        assignments = self.task_assigner.greedy_assignment(self.warehouse) # assign based on first match 
        # assignments = self.task_assigner.optimal_assignment(self.warehouse) # optimal assignment
        
        for robot in self.warehouse.robots:
            if robot.state == "idle" and robot.id in assignments:
                order_id = assignments[robot.id] 
                order = next(o for o in self.warehouse.orders if o.id == order_id)


                robot.target = order.item_location
                robot.path = PathPlanner.a_star(self.warehouse, robot.position, robot.target)
                robot.order_id = order.id #so we know when dropped off etc 
                
                if len(robot.path) > 1:
                    robot.state = "moving"
        
        robot_positions = {r.position for r in self.warehouse.robots}

        # Move robots
        for robot in self.warehouse.robots:
            self._move_robot(robot, robot_positions, dt)
        
    def _generate_random_order(self):
        """Generate a random order"""
        shelves = np.argwhere(self.warehouse.grid == CellType.SHELF.value)
        if len(shelves) > 0:
            shelf = shelves[np.random.randint(len(shelves))]
            order = Order(
                id=len(self.warehouse.orders) + len(self.warehouse.completed_orders),
                item_location=Position(shelf[1], shelf[0]),
                priority=np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            )
            self.warehouse.add_order(order)

    def _move_robot(self, robot: Robot, robot_positions: Set[Position], dt: float):
        
        #Battery drain every movement step 
        if robot.state == "moving" and robot.path:
            robot.battery = max(0.0, robot.battery - DRAIN_RATE * dt)

        if robot.battery <= 0.0:
            robot.state = "flat"
            robot.path = []
            robot.target = None
            return

        #Robot already moving
        if robot.state == "moving":
            if not robot.path:
                if robot.target is not None and robot.position == robot.target: # found target move state 
                    if not robot.carrying_item:
                        robot.state = "picking"
                    else:
                        robot.state = "delivering"
                return

            next_pos = robot.path[0]

            # naive way to prevent collision just wait until the robot on our path has left 
            if next_pos in robot_positions and next_pos != robot.position: 
                robot.wait_steps += 1
                return 

            # advance
            prev_pos = robot.position
            robot_positions.remove(robot.position) 
            robot.position = next_pos
            robot_positions.add(robot.position) #update the robot cells list 
            robot.path.pop(0)

            # classify step as wait or move 
            if robot.position.x == prev_pos.x and robot.position.y == prev_pos.y:
                robot.wait_steps += 1
            else:
                robot.move_steps += 1
            
            #again check target  (if we reached etiher picking or delivering)
            if robot.position == robot.target:
                if not robot.carrying_item:
                    robot.state = "picking"
                else:
                    robot.state = "delivering"

        #get new item from shelf  need to deliver 
        elif robot.state == "picking":
            robot.carrying_item = True
            
            
            if self.dock_locations:
                robot.target = self.dock_locations[0]
        
            robot.path = PathPlanner.a_star(self.warehouse, robot.position, robot.target)
            if robot.path:
                robot.state = "moving"
     
        elif robot.state == "delivering": #robot on the dock cell 
            
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

            #Order is completed, reset robot state
            robot.carrying_item = False
            robot.target = None
            robot.path = []
            robot.order_id = None
            robot.state = "idle"
            
        
        elif robot.state == "idle":
             #robot idle, but has a target maybe didnt find poath so re plan 
             if robot.target and not robot.path:
                 robot.path = PathPlanner.a_star(self.warehouse, robot.position, robot.target)
                 if robot.path:
                     robot.state = "moving"

    
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

        metrics = {
            'total_completed': len(completed),
            'avg_time_all': np.mean(total_times) if total_times else 0,
            'pending_orders': len(self.warehouse.orders),
            'coord_overhead': coord_overhead,
            'total_wait_steps': total_wait,
            'total_move_steps': total_move,
        }
        for p in [1, 2, 3]:
            metrics[f'avg_time_P{p}'] = np.mean(prio_times[p]) if prio_times[p] else 0
            metrics[f'count_P{p}'] = len(prio_times[p])
        return metrics
    

def main():
    """Example usage"""
    np.random.seed(SEED)  # ensure reproducible runs
    print("Warehouse Robot Fleet Coordination - Starter Code")
    print("=" * 50)
    
    # Create warehouse
    warehouse = Warehouse(width=50, height=50)
    
    # Add robots
    for i in range(10):
        warehouse.add_robot(Position(i, 0))
    
    print(f"Created warehouse: {warehouse.width}x{warehouse.height}")
    print(f"Added {len(warehouse.robots)} robots")
    
    # Create simulator
    sim = WarehouseSimulator(warehouse)
    
    # Run simulation
    print("\nRunning simulation...")
    # 24 hours * 60 steps/hour = 1440 steps
    for step in range(1440):
        sim.step(dt=1.0)
        
        if step % 20 == 0:
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
