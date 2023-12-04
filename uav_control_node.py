#!/usr/bin/env python

import rospy
import sys, getopt
import numpy as np
from scipy.optimize import minimize
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped, PointStamped, Point, Vector3
from nav_msgs.msg import Path, Odometry
from gazebo_msgs.msg import ModelStates

class ControlNode:
    def __init__(self, ID=0):

        # Initialize the global variable to None
        self.subscriber_message_uavpose = Point(0,0,0)
        self.subscriber_message_uavvel = Twist()
        self.control_uavvel = Twist()

        
        self.goal_position = Point(0,0,0)
        self.t_value = 0
        self.stop = False
        self.reset = True
        self.ID=ID

        # Initialize the ROS node
        rospy.init_node('Control_node', anonymous=True)

        # Create a publisher that runs at 50 Hz
        self.publisher_velocity = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        rospy.Timer(rospy.Duration(1.0/20), self.publish_velocity)

        # Create a subscriber
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/actions", TwistStamped, self._subscriber_agent_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/stop", Bool, self._subscriber_stop_callback)
        rospy.Subscriber("uav_" + str(self.ID) + "/agent/reset", Bool, self._subscriber_reset_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_pose_callback)

    def _reset_takeoff(self):
        goal = Point(0,0,1)
        # calculate velocity to reach the goal based on uav position
        # get the current position
        position = self.subscriber_message_uavpose
        # calculate the velocity
        velocity = Twist(Vector3(0,0,0), Vector3(0,0,0))
        #clip velocity to max 1 m/s
        velocity.linear.x = np.clip(goal.x, -1, 1)
        velocity.linear.y = np.clip(goal.y, -1, 1)
        velocity.linear.z = np.clip(goal.z, -1, 1)

        
        return velocity

    def _subscriber_reset_callback(self, msg:Bool):
        self.reset = msg.data
        print("Reset: ", self.reset)
            

    def publish_velocity(self, event):
        # Publish the message
        velocity = self.control_uavvel
        
        if self.stop == True:
            velocity = Twist(Vector3(0,0,0), Vector3(0,0,0))
    
        if self.reset == True:
            velocity = self._reset_takeoff()

        self.publisher_velocity.publish(velocity)


    def _subscriber_stop_callback(self, msg:Bool):
        self.stop = msg.data
        print("Stop: ", self.stop)
    
    def _gazebo_pose_callback(self, msg:ModelStates):
            # Get the list of model names
            model_name = msg.name
            name = 'quadrotor' #+ str(self.ID)
            #print(name)

            # Get the index of the 'quadrotor' model
            try:
                quadrotor_index = model_name.index(name)
            except ValueError:
                rospy.logwarn("%s model not found in ModelStates",name)
                return

            # Get the pose of the 'quadrotor' model
            quadrotor_pose = msg.pose[quadrotor_index]
            quadrotor_velocity = msg.twist[quadrotor_index]

            # Print the pose information
            self.subscriber_message_uavpose = quadrotor_pose.position
            self.subscriber_message_uavvel = quadrotor_velocity
            #rospy.loginfo("quadrotor Pose:\n{}".format(self.subscriber_message_uavpose))


    # subscribe the action from the agent
    def _subscriber_agent_callback(self, msg:TwistStamped):

        self.reset = False
        self.stop = False
        received_vel = msg.twist

        # sum up the linear and z angular velocity
        uavvel = Twist(Vector3(0,0,0), Vector3(0,0,0))
        

        if self.stop == False:
            uavvel.linear.x = received_vel.linear.x + self.subscriber_message_uavvel.linear.x
            uavvel.linear.y = received_vel.linear.y + self.subscriber_message_uavvel.linear.y
            uavvel.linear.z = received_vel.linear.z + self.subscriber_message_uavvel.linear.z
            uavvel.angular.z = received_vel.angular.z + self.subscriber_message_uavvel.angular.z

        #limit the velocity to 1 m/s in x and y direction and 0.5 m/s in z direction and 0.174 rad/s in z direction
        uavvel = self.velocity_limiter(uavvel, 1, 1, 0.5, 0.174)
        

        # Publish the message
        self.publisher_velocity.publish(uavvel)

        self.control_uavvel = uavvel

    def velocity_limiter(self, uavvel, velx, vely, velz, angz):
        if uavvel.linear.x > velx:
            uavvel.linear.x = velx
        elif uavvel.linear.x < -velx:
            uavvel.linear.x = -velx
        if uavvel.linear.y > vely:
            uavvel.linear.y = vely
        elif uavvel.linear.y < -vely:
            uavvel.linear.y = -vely
        if uavvel.linear.z > velz:
            uavvel.linear.z = velz
        elif uavvel.linear.z < -velz:
            uavvel.linear.z = -velz
        if uavvel.angular.z > angz:
            uavvel.angular.z = angz
        elif uavvel.angular.z < -angz:
            uavvel.angular.z = -angz
        
        return uavvel

    def run(self):
        # Spin the node to receive and process messages
        rospy.spin()

def get_id(argv):
    ID=0
    opts, args = getopt.getopt(argv,":i:",["index="])
    for opt, arg in opts:
        # print(opt," ",arg)
        if opt in ("-i","--index"):
            ID = arg
    return ID

if __name__ == '__main__':
    id=get_id(sys.argv[1:])
    print("ID: ",id)
    try:
        node = ControlNode(ID=id)
        node.run()
    except rospy.ROSInterruptException:
        pass
