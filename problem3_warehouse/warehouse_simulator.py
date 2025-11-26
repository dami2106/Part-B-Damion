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
        
        available_robots = [r for r in warehouse.robots if r.state == "idle"]
        pending_orders = [o for o in warehouse.orders if o.completion_time is None]
        
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
        
        TODO: Implement optimal assignment
        - Consider: distance, priority, battery levels
        - Minimize total completion time
        """
        # TODO: Use scipy.optimize.linear_sum_assignment or similar
        return TaskAssigner.greedy_assignment(warehouse)


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
    
    def find_cell_type(self, cell_type: CellType) -> List[Position]:
        positions = []
        for y in range(self.warehouse.height):
            for x in range(self.warehouse.width):
                if self.warehouse.grid[y, x] == cell_type.value:
                    positions.append(Position(x, y))
        return positions


    def step(self, dt: float = 1.0):
        self.warehouse.time += dt
        
        # Generate orders
        if np.random.random() < 0.1:  # 10% chance per step
            self._generate_random_order()
        
        assignments = self.task_assigner.greedy_assignment(self.warehouse) # assign based on first match 
        
        for robot in self.warehouse.robots:
            if robot.state == "idle" and robot.id in assignments:
                order_id = assignments[robot.id] 
                order = next(o for o in self.warehouse.orders if o.id == order_id)


                robot.target = order.item_location
                robot.path = PathPlanner.a_star(self.warehouse, robot.position, robot.target)
                robot.current_order_id = order.id #so we know when dropped off etc 
                
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
        """Move robot one step along its path"""# State: Moving to Target
        if robot.state == "moving":
            if robot.path and robot.position == robot.path[0]:
                robot.path.pop(0) #Easiest condition just follow path

            if not robot.path:
                if robot.target is not None and robot.position == robot.target: # found target move state 
                    if not robot.carrying_item:
                        robot.state = "picking"
                    else:
                        robot.state = "delivering"
                return

            next_pos = robot.path[0]

            #naive way to prevent collision just wait until the robot on our path has left 
            if next_pos in robot_positions and next_pos != robot.position: 
                self.warehouse.total_collisions += 1
                return 

            robot_positions.remove(robot.position) 
            robot.position = next_pos
            robot_positions.add(robot.position) #update the robot cells list 
            
            #again check target 
            if robot.position == robot.target:
                if not robot.carrying_item:
                    robot.state = "picking"
                else:
                    robot.state = "delivering"

        #get new item from shelf 
        elif robot.state == "picking":
            robot.carrying_item = True
            
            
            if self.dock_locations:
                robot.target = self.dock_locations[0]
        
            robot.path = PathPlanner.a_star(self.warehouse, robot.position, robot.target)
            if robot.path:
                robot.state = "moving"
     
        elif robot.state == "delivering": #robot on the dock cell 
            
            if robot.current_order_id is not None:
                
                order_idx = -1
                for i, o in enumerate(self.warehouse.orders):
                    if o.id == robot.current_order_id:
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
            robot.current_order_id = None
            robot.state = "idle"
            
        
        elif robot.state == "idle":
             #robot idle, but has a target maybe didnt find poath so re plan 
             if robot.target and not robot.path:
                 robot.path = PathPlanner.a_star(self.warehouse, robot.position, robot.target)
                 if robot.path:
                     robot.state = "moving"

    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        completed = self.warehouse.completed_orders
        if not completed:
            return {
                'avg_completion_time': 0,
                'throughput': 0,
                'collisions': 0
            }
        
        completion_times = [o.completion_time - o.arrival_time 
                          for o in completed if o.completion_time]
        
        return {
            'avg_completion_time': np.mean(completion_times) if completion_times else 0,
            'throughput': len(completed),
            'orders_pending': len(self.warehouse.orders),
            'collisions': self.warehouse.total_collisions
        }



def main():
    """Example usage"""
    np.random.seed(42)  # ensure reproducible runs
    print("Warehouse Robot Fleet Coordination - Starter Code")
    print("=" * 50)
    
    # Create warehouse
    warehouse = Warehouse(width=20, height=20)
    
    # Add robots
    for i in range(5):
        warehouse.add_robot(Position(i, 0))
    
    print(f"Created warehouse: {warehouse.width}x{warehouse.height}")
    print(f"Added {len(warehouse.robots)} robots")
    
    # Create simulator
    sim = WarehouseSimulator(warehouse)
    
    # Run simulation
    print("\nRunning simulation...")
    for step in range(100):
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
