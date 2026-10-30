#!/bin/bash
# robot_control's chess executor on MOCK hardware: UR driver (mock), MoveIt,
# a fake Hand-E and robot_control; sends one chess move (pawn to storage +
# castling) and waits for robot_move_done.  Isolated ROS domain, localhost
# only, fake robot IP - nothing reaches the real robot.  Only the process
# groups started here are stopped.  Logs: /tmp/chess_mock_robot/
#   bash chess_game/tools/mock_robot/run.sh [~/chess_ws]
WS=${1:-~/chess_ws}
HERE=$(cd "$(dirname "$0")" && pwd)
L=/tmp/chess_mock_robot
mkdir -p $L
export ROS_DOMAIN_ID=77 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
source /opt/ros/jazzy/setup.bash
source $WS/install/setup.bash
PGIDS=()
start() { setsid "$@" & PGIDS+=($!); }
start bash -c "ros2 launch cocohrip_control start_robot_control.launch.py use_mock_hardware:=true robot_ip:=127.0.0.1 launch_rviz:=false > $L/ur.log 2>&1"
sleep 10
start bash -c "ros2 launch cocohrip_moveit_config move_group.launch.py > $L/move_group.log 2>&1"
start bash -c "python3 $HERE/fake_gripper.py > $L/gripper.log 2>&1"
sleep 12
start bash -c "ros2 launch robot_control new_launch.py > $L/robot_control.log 2>&1"
sleep 15
timeout 300 python3 $HERE/send_move.py 240
for g in "${PGIDS[@]}"; do kill -INT -- -$g 2>/dev/null; done
sleep 6
for g in "${PGIDS[@]}"; do kill -9 -- -$g 2>/dev/null; done
