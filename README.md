# COSC81, Spring 2025, PA3 - Path Planning, Mihir Singh
## How To Run

### Installation
To run the Path Planning program you must first install Docker [here](https://docs.docker.com/desktop/setup/install/mac-install/) for Mac or [here](https://docs.docker.com/desktop/setup/install/windows-install/) for Windows. You should also have Python3 installed.

Then you can follow the instructions from [this](https://github.com/quattrinili/vnc-ros) repository from Professor Quattrini-Li to set up the container that contains ROS and other tools.

After installing vnc-ros, you must also install the following stage maps in vnc-ros/workspace/src. You can install them [here](https://canvas.dartmouth.edu/files/14044565/download?download_frd=1). There is an empty map world and a corrider map world. 

### Running Program
Navigate to the 'vnc-ros' directory on your computer in your terminal and run the following command:\
```docker compose up```

Then when you see something like:\
```rosbot-gazebo  | [INFO] [spawner-9]: process has finished cleanly [pid 359]```

Open your localhost8080 port in your broswer and click to vnc.html. You should see a screen with a connect button. Click on the connect button. Since we will not be using Gazebo, you can stop the Gazebo container in your Docker dashboard.

You must make sure that you have Stage installed. 

To install stage, run the following command in a newly opened terminal and first run ```docker compose exec ros bash```:

Then run the following commands:\
```bash install_stage.sh```
```source ../install/setup.bash```

You should also have the new map installed [here](https://canvas.dartmouth.edu/files/14060087/download?download_frd=1)

Now we will begin the process of running the path planning algorithm. For every terminal you must first run the command:\
```docker compose exec ros bash```

You should already have (this is the only terminal where you don't run ```docker compose exec ros bash```):\
Terminal 1: ```docker compose up```

Now, run the following commands:\
Terminal 2: ```ros2 launch stage_ros2 stage.launch.py world:=/root/catkin_ws/src/pa3/maze enforce_prefixes:=false one_tf_tree:=true```

Terminal 3: ```python3 pa3_path_planning.py```\
Running this before the map ensures that its subscriber will grab the map which is important.

Terminal 4: ```ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=pa3/maze.yml```

Terminal 5:
1. ```ros2 run nav2_util lifecycle_bringup map_server```
2. ```ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap "{map_url: pa3/maze.yml}"```
3. ```ros2 run tf2_ros static_transform_publisher 2 2 0 0 0 0 map rosbot/odom```

Terminal 6:```ros2 run rviz2 rviz2```

In rviz2 add the map, pose_sequence, and odom topics to view the map, the planned path and the robot movement respectively

## Credits
I wrote the code for this assignment using some code from lecture examples provided by Professor Quattrini-Li as well as some of my old programming assignments. I talked to TA Luyang Zhao about how to get the map to show up in rviz as well. Finally, I benefited from help requests in the class Slack.

