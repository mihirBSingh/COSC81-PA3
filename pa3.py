#!/usr/bin/env python

# Author: AQL
# Date: 2025/03/31

# imports
import numpy as np
import tf_transformations
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseArray, Twist
from collections import deque, defaultdict
from typing import List, Tuple, Dict
import tf2_ros
import math
from rclpy.duration import Duration

# topics
NODE_NAME = "planner"
MAP_TOPIC = "map"
POSE_TOPIC = "pose"
POSE_SEQUENCE_TOPIC = "pose_sequence"
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'

# ref frames
MAP_FRAME_ID = "map"
BASE_LINK_FRAME_ID = "rosbot/base_link"

USE_SIM_TIME = True

# constants
FREQUENCY = 10 #Hz.
ANGULAR_VELOCITY = np.pi/4 # rad/s
LINEAR_VELOCITY = 0.125 # m/s
GRID_OFFSET = 3 # number of grids - padding for the robot to not hit the wall
ANGLE_THRESHOLD = 0.1 # radians
DISTANCE_THRESHOLD = 0.1 # meters

class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution
        self.width = width
        self.height = height

    def cell_at(self, r, c):
        return self.grid[r, c]
    
    def is_valid(self, r, c):
        # Check if (r, c) is out of bounds
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return False

        # Check the current cell
        if self.cell_at(r, c) == 100:
            return False

        # Check all cells within the GRID_OFFSET square
        for dr in range(-GRID_OFFSET, GRID_OFFSET + 1):
            for dc in range(-GRID_OFFSET, GRID_OFFSET + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    if self.cell_at(rr, cc) == 100:
                        return False

        return True


class Plan(Node):
    def __init__(self, map_frame_id=MAP_FRAME_ID, node_name=NODE_NAME, context=None):
        super().__init__(node_name, context=context)

        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])
        
        # Rate at which to operate the while loop.
        self.rate = self.create_rate(FREQUENCY)
        
        # for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # subscribers
        self._map_sub = self.create_subscription(OccupancyGrid, MAP_TOPIC, self.map_callback, 10)
        
        # publishers
        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 10)
        self.pose_array_pub = self.create_publisher(PoseArray, POSE_SEQUENCE_TOPIC, 10)
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 10)

        self.linear_velocity = LINEAR_VELOCITY
        self.angular_velocity = ANGULAR_VELOCITY
        
        self.map = None
        self.map_frame_id = map_frame_id
        self.map_origin = None
        self.occupancy_grid = None

    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)
        
    # adapted from PA0
    def rotate(self, angle):
        duration = Duration(seconds=abs(angle/ANGULAR_VELOCITY))
        angular_velocity = math.copysign(1, angle) * ANGULAR_VELOCITY
        rclpy.spin_once(self)
        
        start_time = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self)
            # check if traveled given distance based on time
            if self.get_clock().now() - start_time >= duration:
                break
            
            # publish massage 
            self.move(0.0, angular_velocity)
            
    # from lecture
    def move_forward(self, duration):
        """Function to move_forward for a given duration."""
        # Setting velocities. 
        twist_msg = Twist()
        twist_msg.linear.x = LINEAR_VELOCITY

        duration = Duration(seconds=duration)
        rclpy.spin_once(self)
        start_time = self.get_clock().now()

        # Loop.
        while rclpy.ok():
            rclpy.spin_once(self)
            # Check if traveled of given distance based on time.
            # self.get_logger().info(f"{start_time} {self.get_clock().now()} {duration}")
            if self.get_clock().now() - start_time >= duration:
                break

            # Publish message.
            self._cmd_pub.publish(twist_msg)

        # Traveled the required distance, stop.
        self.stop()

    def map_callback(self, msg):
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.map_origin = msg.info.origin
        self.occupancy_grid = msg
        print(f"Got map: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}")

    # adapted from PA1 code
    def move_along_sequence(self, poses):
        
        # check that sequence is recieved
        if not poses:
            print("No poses to move along.")
            self.stop()
            return
        
        for i, target_pose in enumerate(poses):
            # print(f"Moving to pose {i}: ({target_pose.pose.position.x:.3f}, {target_pose.pose.position.y:.3f})")
            
            while rclpy.ok():
                # get current pose
                try:
                    transform = self.tf_buffer.lookup_transform(
                        MAP_FRAME_ID, BASE_LINK_FRAME_ID, rclpy.time.Time(), Duration(seconds=1.0))
                    curr_x = transform.transform.translation.x
                    curr_y = transform.transform.translation.y
                    q = transform.transform.rotation
                    curr_yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print(f"Cannot get current pose: {e}")
                    self.stop()
                    return

                # find angle to target
                dx = target_pose.pose.position.x - curr_x
                dy = target_pose.pose.position.y - curr_y
                target_angle = np.arctan2(dy, dx)
                angle_diff = (target_angle - curr_yaw + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]

                # find distance to target
                distance = np.sqrt(dx**2 + dy**2)

                # rotate
                # if abs(angle_diff) > ANGLE_THRESHOLD:
                #     print("asds")
                self.rotate(angle_diff)
                
                # move forward if close enough in angle
                if abs(angle_diff) <= ANGLE_THRESHOLD:
                    if distance > DISTANCE_THRESHOLD:
                        move_duration = distance / self.linear_velocity
                        self.move_forward(move_duration)
                    else:
                        # print(f"Reached pose {i}")
                        self.stop()
                        break
                
                self.rate.sleep()       
    
    # adapted from class lecture code about transformation
    def find_start_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(MAP_FRAME_ID, BASE_LINK_FRAME_ID,  rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            
            # don't really need to worry about orientation here
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            return (x, y)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f"cannot get current pose: {e}")
            return None
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        x_rel = x - self.map_origin.position.x
        y_rel = y - self.map_origin.position.y
        col = int(x_rel / self.map.resolution)
        row = int(y_rel / self.map.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = col * self.map.resolution + self.map_origin.position.x + self.map.resolution / 2
        y = row * self.map.resolution + self.map_origin.position.y + self.map.resolution / 2
        return x, y

    # BFS or DFS algorithm to find path from start to goal position
    def find_path(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float], algorithm: str = "BFS") -> List[PoseStamped]:
        
        # if no map then get one
        if not self.map:
            print("had to get map")
            self.map = Grid(self.occupancy_grid.data, self.occupancy_grid.info.width, 
                           self.occupancy_grid.info.height, self.occupancy_grid.info.resolution)
            self.map_origin = self.occupancy_grid.info.origin

        # convert world to grid coords
        start_row, start_col = self.world_to_grid(start_pos[0], start_pos[1])
        goal_row, goal_col = self.world_to_grid(goal_pos[0], goal_pos[1])

        # validate start and goal positions
        if not self.map.is_valid(start_row, start_col) or not self.map.is_valid(goal_row, goal_col):
            first = not self.map.is_valid(start_row, start_col)
            second = not self.map.is_valid(goal_row, goal_col)
            print(f"Invalid start position: {first}, Invalid goal position: {second}")
            return []

        # data structures for BFS and DFS
        visited = set()
        parent = {} 
        if algorithm == "BFS":
            frontier = deque([(start_row, start_col)])
        else:  # DFS
            frontier = [(start_row, start_col)]
        
        visited.add((start_row, start_col))
        parent[(start_row, start_col)] = None

        # 8 possible directions for movement
        directions =  [(1, 0), (0, -1), (-1, 0), (0, 1), (1, 1), (-1, -1), (-1, 1), (1, -1)] 
        print("Start searching...")
        
        # search using algorithm selected
        while frontier:
            if algorithm == "BFS":
                current = frontier.popleft()
            else:  # DFS
                current = frontier.pop()

            if current == (goal_row, goal_col):
                break

            curr_row, curr_col = current
            for dr, dc in directions:
                next_row, next_col = curr_row + dr, curr_col + dc
                if (next_row, next_col) not in visited and self.map.is_valid(next_row, next_col):
                    visited.add((next_row, next_col))
                    parent[(next_row, next_col)] = current
                    if algorithm == "BFS":
                        frontier.append((next_row, next_col))
                    else:  # DFS
                        frontier.append((next_row, next_col))

        # rebuild path
        if (goal_row, goal_col) not in parent:
            print("No path found")
            return []
        path = []
        current = (goal_row, goal_col)
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()

        # convert path to PoseStamped messages
        poses = []
        for i, (row, col) in enumerate(path):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.map_frame_id
            
            # Set position
            x, y = self.grid_to_world(row, col)
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # Set orientation (pointing to next point, last point keeps previous orientation)
            if i < len(path) - 1:
                next_x, next_y = self.grid_to_world(path[i + 1][0], path[i + 1][1])
                angle = np.arctan2(next_y - y, next_x - x)
                quaternion = tf_transformations.quaternion_from_euler(0, 0, angle)
                
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]      
            else:
                # Use previous orientation for last point
                quaternion = poses[-1].pose.orientation if poses else tf_transformations.quaternion_from_euler(0, 0, 0)
            
                pose.pose.orientation.x = quaternion.x
                pose.pose.orientation.y = quaternion.y
                pose.pose.orientation.z = quaternion.z
                pose.pose.orientation.w = quaternion.w
            
            poses.append(pose)
            
        # publish poses
        self.publish_pose_sequence(poses)
            
        return poses
    
    # sequence of poses for robot to go from start to goal position
    def publish_pose_sequence(self, poses):
        # Publish the path as a PoseArray
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = self.map_frame_id
        pose_array_msg.poses = [pose.pose for pose in poses]
        
        # Print the poses for debugging
        for i, pose in enumerate(pose_array_msg.poses):
            print(f"Pose {i}: ({pose.position.x}, {pose.position.y}), Orientation: ({pose.orientation.x}, {pose.orientation.y}, {pose.orientation.z}, {pose.orientation.w})")
        print("pose array", len(pose_array_msg.poses))
        
        # publish the PoseArray message
        self.pose_array_pub.publish(pose_array_msg)

    # just to see individual poses - good for debugging/testing
    def publish_individual_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame_id
        
        pose_msg.pose.position.x = 1.0
        pose_msg.pose.position.y = 1.0
        quaternion = tf_transformations.quaternion_from_euler(0, 0, np.pi/2, 'rxyz')
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(pose_msg)

    def spin(self):
        while rclpy.ok():
            # get user goal input
            goal = input("Enter goal position (x,y) where for 0 < x and y < 200 and x and y are both floats: ")
            x, y = goal.strip("() ").split(",")
            x = float(x)
            y = float(y)
            if x < 0 or x > 200 or y < 0 or y > 200:
                print("Invalid goal position. Using default goal position (5, 5).")
                x = 5.0
                y = 5.0
            goal_pos = (x, y)
            
            # get algorithm
            algorithm = input("Enter algorithm - choose 1 for BFS and 2 for DFS): ")
            if algorithm == "1":
                algorithm = "BFS"
            elif algorithm == "2":
                algorithm = "DFS"
            else:
                print("Invalid algorithm choice. Using BFS as default.")
                algorithm = "BFS"
                
            # get start pose
            start_pose = self.find_start_pose()
            print(start_pose)
            if start_pose is None:
                print("error getting start pose")
            
            path_poses = self.find_path(start_pose, goal_pos, algorithm=algorithm)
            print("following path")
            self.move_along_sequence(path_poses)
                       
def main(args=None):
    rclpy.init(args=args)
    p = Plan()
    
    # basically just wait to see if the map is received
    while p.occupancy_grid is None:
        rclpy.spin_once(p)
    if p.occupancy_grid is not None:
        rclpy.spin_once(p)
        p.spin()
        
    rclpy.shutdown()

if __name__ == "__main__":
    main()