import heapq
import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

NUM_WAYPOINTS = 4
WAYPOINT_BOUNDS = (-10.0, 10.0)
ARRIVAL_TOLERANCE = 0.3
ANGULAR_GAIN = 1.5
FORWARD_SPEED = 0.5

# Known obstacle geometry from worlds/empty_world.world: two walls form a
# plus sign centered at the origin. Each rect is (xmin, ymin, xmax, ymax),
# inflated past the wall's real size by a safety margin (wall half-thickness
# + robot half-width + buffer) so a "clear" path actually keeps the box off
# the wall, not just off its exact collision mesh.
MARGIN = 0.5
V_ARM_RECT = (-0.25 - MARGIN, -7.0 - MARGIN, 0.25 + MARGIN, 7.0 + MARGIN)
H_ARM_RECT = (-7.0 - MARGIN, -0.25 - MARGIN, 7.0 + MARGIN, 0.25 + MARGIN)
ARM_RECTS = [V_ARM_RECT, H_ARM_RECT]

# For the route grid and A* search: world x/y run -10..10, shown/indexed as
# a 21x21 grid of (row, col) with row 0 = north (y=10), col 0 = west (x=-10).
GRID_MIN = -10
GRID_MAX = 10
GRID_SIZE = GRID_MAX - GRID_MIN + 1


def world_to_grid(x, y):
    row = round(GRID_MAX - y)
    col = round(x - GRID_MIN)
    return row, col


def grid_to_world(cell):
    row, col = cell
    return (col + GRID_MIN, GRID_MAX - row)


def is_wall_cell(x, y):
    in_v_arm = abs(x) <= 0.25 and abs(y) <= 7.0
    in_h_arm = abs(y) <= 0.25 and abs(x) <= 7.0
    return in_v_arm or in_h_arm


def print_route_grid(logger, start, route):
    size = GRID_MAX - GRID_MIN + 1
    grid = [
        ['#' if is_wall_cell(x, GRID_MAX - row) else '.' for x in range(GRID_MIN, GRID_MAX + 1)]
        for row in range(size)
    ]

    def draw_line(p0, p1):
        r0, c0 = world_to_grid(*p0)
        r1, c1 = world_to_grid(*p1)
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for i in range(steps + 1):
            t = i / steps
            r = round(r0 + (r1 - r0) * t)
            c = round(c0 + (c1 - c0) * t)
            if grid[r][c] == '.':
                grid[r][c] = '*'

    points = [start] + route
    for p0, p1 in zip(points, points[1:]):
        draw_line(p0, p1)

    sr, sc = world_to_grid(*start)
    grid[sr][sc] = 'S'
    for gap in route[:-1]:
        gr, gc = world_to_grid(*gap)
        grid[gr][gc] = 'o'
    wr, wc = world_to_grid(*route[-1])
    grid[wr][wc] = 'W'

    lines = ['Planned route (S=start, o=turn point, W=waypoint, #=wall):']
    lines.extend(' '.join(row) for row in grid)
    logger.info('\n' + '\n'.join(lines))


def point_is_free(x, y):
    for xmin, ymin, xmax, ymax in ARM_RECTS:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return False
    return True


def generate_waypoints(count, bounds):
    low, high = bounds
    points = []
    while len(points) < count:
        x = random.uniform(low, high)
        y = random.uniform(low, high)
        if point_is_free(x, y):
            points.append((x, y))
    return points


def yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def cell_is_free(cell):
    row, col = cell
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        return False
    return point_is_free(*grid_to_world(cell))


def neighbors(cell):
    row, col = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            n = (row + dr, col + dc)
            if cell_is_free(n):
                yield n, math.hypot(dr, dc)


def heuristic(cell, goal_cell):
    return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])


def astar(start_cell, goal_cell):
    """Classic grid A*: 8-directional movement, Euclidean heuristic, over
    the same grid used to print the route."""
    open_set = [(heuristic(start_cell, goal_cell), 0.0, start_cell)]
    came_from = {}
    best_g = {start_cell: 0.0}
    visited = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor, step_cost in neighbors(current):
            tentative_g = g + step_cost
            if tentative_g < best_g.get(neighbor, math.inf):
                best_g[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(
                    open_set,
                    (tentative_g + heuristic(neighbor, goal_cell), tentative_g, neighbor),
                )

    return None


def simplify_cells(cells):
    """Drop cells that lie in the middle of a straight run, keeping only
    the points where the path's direction actually changes."""
    if len(cells) <= 2:
        return cells
    simplified = [cells[0]]
    for i in range(1, len(cells) - 1):
        prev_dir = (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
        next_dir = (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])
        if prev_dir != next_dir:
            simplified.append(cells[i])
    simplified.append(cells[-1])
    return simplified


def find_route(start, goal):
    """A* over the grid from start's cell to goal's cell, simplified down
    to just the turn points, then converted back to world coordinates."""
    start_cell = world_to_grid(*start)
    goal_cell = world_to_grid(*goal)

    path_cells = astar(start_cell, goal_cell)
    if not path_cells:
        return [goal]

    turn_cells = simplify_cells(path_cells)[1:]  # drop the start cell itself
    route = [grid_to_world(cell) for cell in turn_cells]
    route[-1] = goal  # end exactly on the real goal, not its rounded cell
    return route or [goal]


class Mover(Node):
    def __init__(self):
        super().__init__('mover')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.timer = self.create_timer(0.1, self.move_callback)

        self.x = None
        self.y = None
        self.yaw = None

        self.waypoints = generate_waypoints(NUM_WAYPOINTS, WAYPOINT_BOUNDS)
        self.waypoint_index = 0
        self.route = None
        self.route_index = 0
        self.get_logger().info(f'Waypoints: {self.waypoints}')

    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def move_callback(self):
        if self.x is None or self.waypoint_index >= len(self.waypoints):
            return

        if self.route is None:
            goal = self.waypoints[self.waypoint_index]
            self.route = find_route((self.x, self.y), goal)
            self.route_index = 0

            row, col = world_to_grid(*goal)
            self.get_logger().info(
                f'Navigating to waypoint {self.waypoint_index} at (row {row}, col {col})'
            )
            print_route_grid(self.get_logger(), (self.x, self.y), self.route)

        target_x, target_y = self.route[self.route_index]
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)

        if distance < ARRIVAL_TOLERANCE:
            self.route_index += 1
            if self.route_index >= len(self.route):
                goal_x, goal_y = self.waypoints[self.waypoint_index]
                self.get_logger().info(
                    f'Reached waypoint {self.waypoint_index}: '
                    f'({goal_x:.2f}, {goal_y:.2f})'
                )
                self.waypoint_index += 1
                self.route = None
                if self.waypoint_index >= len(self.waypoints):
                    self.publisher_.publish(Twist())
                    self.get_logger().info('All waypoints visited.')
            return

        angle_to_target = math.atan2(dy, dx)
        angle_diff = math.atan2(
            math.sin(angle_to_target - self.yaw), math.cos(angle_to_target - self.yaw)
        )

        msg = Twist()
        msg.angular.z = ANGULAR_GAIN * angle_diff
        msg.linear.x = FORWARD_SPEED if abs(angle_diff) < 0.5 else 0.0
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Mover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
