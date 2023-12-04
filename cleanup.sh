#!/bin/bash

# stopping all rosnodes
rosnode kill --all
# # stopping the gazebo client aka gui
killall gzclient
# # stopping the gazebo server
killall gzserver
killall -9 roslaunch
killall -9 roslaunch
killall -9 roslaunch
killall ssd_server.bin 
killall ssd_server.sh

# # kill all screen
killall screen

# # lets get a bit more drastic # commented for subproccess to funciton properly
pkill -f ros/noetic
pkill -f /home/jacob/catkin_ws
pkill -f /home/jacob/PX4-Autopilot

sleep 10


