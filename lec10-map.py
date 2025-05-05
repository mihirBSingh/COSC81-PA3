#!/usr/bin/env python

# Author: AQL
# Date: 2025/03/31

import numpy as np

import tf_transformations

import rclpy # module for ROS APIs
from rclpy.node import Node
# http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

NODE_NAME = "planner"

MAP_TOPIC = "map"
POSE_TOPIC = "pose"

MAP_FRAME_ID = "map"

USE_SIM_TIME = True

class Grid:
    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution

    def cell_at(self, r, c):
        return self.grid[r, c]

class Plan(Node):
    def __init__(self, map_frame_id=MAP_FRAME_ID, node_name=NODE_NAME, context=None):
        super().__init__(node_name, context=context)

        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])

        self.sub = self.create_subscription(OccupancyGrid, MAP_TOPIC, self.map_callback, 1)

        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 1)

        self.map = None # the variable containing the map.

        self.map_frame_id = map_frame_id

    def map_callback(self, msg):
        print("CALLBACK")
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        print(self.map.cell_at(0,1))

    def publish_pose(self):
        """Example of publishing an arrow, without orientation."""
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

def main(args=None):
    # 1st. initialization of node.
    rclpy.init(args=args)

    p = Plan()

    # try:
    while rclpy.ok():
        p.publish_pose()

        rclpy.spin_once(p)
    # except KeyboardInterrupt:
    #     print("SOMETHING")
    #     pass

    rclpy.shutdown()
    


if __name__ == "__main__":
    main()
