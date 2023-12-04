import time

import gym
import numpy as np
import rospy

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, PointStamped, Point
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode, CommandBool

from gazebo_msgs.msg import ModelStates

from gazebo_msgs.srv import DeleteModel, SpawnModel


from gym import spaces
from std_srvs.srv import Empty, EmptyRequest
from turtlesim.msg import Pose


def lmap(v, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


class UAVEnv(gym.Env):
    def __init__(
        self,
        index = 0,  # env index
        rate = 10,
        steps_thr = 1024*10,  # simulation refresh rate
    ):
        super().__init__()
        self.index = index

        #rospy.init_node("UAVEnv" + str(index))

        self.rate = rospy.Rate(rate)
        self._create_ros_pub_sub_srv()
        

        
        # Action dimension and boundaries
        self.act_dim = 4 # [vx, vy, vz, wz]
        self.action_space = spaces.Box(
            low=-np.ones(self.act_dim), high=np.ones(self.act_dim), dtype=np.float32
        )
        self.act_bnd = {"vx": (-0.1, 0.1), "vy": (-0.1, 0.1), "vz":(-0.1, 0.1), "wz":(-0.175, 0.175)} # What are the boudaries for the velocity and angular velocity

        self.obs_dim = 15  # [dist, UAVPose(3), WaypointPose(3), prev_action(4), velocity(4)]
        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_dim), high=np.ones(self.obs_dim), dtype=np.float32
        )
        self.obs_bnd = {"xyz": 10.}  # cubic space between (-100 to 100)

        self.rew_w = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )  # [dist, delta_theta, cos(delta_theta), sin(delta_theta), x, y, theta]
        self.dist_threshold = 0.25  # done if dist < dist_threshold, final goal radius distance

        self.goal_name = "GoalPoint"
        self.uav_pose = np.zeros(3)
        self.goal = np.zeros(3)
        self.prev_actions = np.zeros(4)
        self.uav_velocity = np.zeros(4)
        

        self.steps = 0
        self.total_ts = 0
        self.steps_thr = steps_thr
        self.prev_dist = 0
        self.got_goal = 1
        

        self.reset()
        rospy.loginfo("[ENV Node " + str(self.index) + " ] Initialized")

    def step(self, action):
        if not rospy.is_shutdown():
        
            self.steps += 1
            self.total_ts += 1

            #self.publish_action(np.zeros(6))
            self.publish_action(action) # send the action to the Bezier node
            observation = self.observe() # get the observation
            reward = self.compute_reward(observation) # compute the reward
            done = self.is_terminal(observation) # check if the episode is done

            # Check if the UAV is out of the boundaries
            #if self.steps >= 50:
            if not(-11.0 <= self.uav_pose[0] <= 11 and -11 <= self.uav_pose[1] <= 11 and -11 <= self.uav_pose[2] <= 11):
                done = True
                reward = -self.steps_thr
                print(reward)
            
            info = {}
            self.rate.sleep()
            return observation, reward, done, info
        else:
            rospy.logerr("rospy is shutdown")

    def reset(self): # reset the training  - send the vehicle to init postion ToDo
        self.steps = 0
        
        self.stop_publisher.publish(Bool(True))  # TODO ADD Action to stop the UAV
        # self._reset_bezier("stop")
        print("Reseting")
        rospy.sleep(2)

        self.reset_publisher.publish(Bool(True))
        self.reset_world()
        

        rospy.sleep(3)

        if self.got_goal ==1:
            try:
                self._clear_goal()
            except:
                None
            self.set_goal()
            self.got_goal=0
        

        #self.unpause_sim()
        #self.set_uav_mode("MANUAL")
        #time.sleep(1)

        # Land the UAV
        # rospy.wait_for_service("/mavros/cmd/land")
        # while (self.land_client.call(True).success == False):
        #     rospy.loginfo("Landing")
        #     rospy.sleep(1)
        # rospy.loginfo("Landed")

        # # Arming the UAV
        # rospy.wait_for_service("/mavros/cmd/arming")
        # while (self.arming_client.call(True).success == False):
        #     rospy.loginfo("Arming")
        #     rospy.sleep(1)
        # rospy.loginfo("Armed")

        # #send pose to the UAV
        # self.pose = PositionTarget()
        # self.pose.header.stamp = rospy.Time.now()
        # self.pose.header.frame_id = ''
        # self.pose.coordinate_frame = 1
        # self.pose.type_mask = 3064
        # self.pose.position.x = 0
        # self.pose.position.y = 0
        # self.pose.position.z = 10000.0
        # self.pose.yaw = 0
        # self.local_pose_pub.publish(self.pose)

        # rospy.wait_for_service("/mavros/cmd/arming")
        # while (self.arming_client.call(True).success == False):
        #     rospy.loginfo("Arming")
        #     rospy.sleep(1)
        # rospy.loginfo("Armed")

        # rospy.sleep(1)

        # self.set_uav_mode("OFFBOARD")
        #self.stop_publisher.publish(Bool(False))
        print("New Goal =",self.goal)

        return self.observe()
    

    def publish_action(self, action, type="default"): # action changed
        action = self._proc_action(action)

        sender_action = TwistStamped()
        sender_action.header.stamp = rospy.Time.now()
        sender_action.twist.linear.x = action[0]
        sender_action.twist.linear.y = action[1]
        sender_action.twist.linear.z = action[2]
        sender_action.twist.angular.z = action[3]

        #heck if values are nan
        if np.isnan(sender_action.twist.linear.x):
            sender_action.twist.linear.x = 0
        if np.isnan(sender_action.twist.linear.y):
            sender_action.twist.linear.y = 0
        if np.isnan(sender_action.twist.linear.z):
            sender_action.twist.linear.z = 0
        if np.isnan(sender_action.twist.angular.z):
            sender_action.twist.angular.z = 0

        self.prev_actions = action
        self.prev_dist = np.linalg.norm(self.uav_pose - self.goal, 2)

        self.agent_actions_publisher.publish(sender_action)
        

    def observe(self, n_std=0.1):
        relative_dist = np.linalg.norm(self.uav_pose - self.goal, 2)
        

        return np.concatenate([np.array([relative_dist]), self.uav_pose, self.goal, self.prev_actions, self.uav_velocity]) + np.random.normal(
            0, n_std, self.obs_dim
        )

    def compute_reward(self, obs):
        
        distance_reward = np.clip(np.dot(self.rew_w, -np.abs(obs)/18), -1, 0) #18 is the mas distance between the UAV and the goal, therefore the max distance reward is -1
        action_reward = np.clip(-self.prev_dist/18, -1, 0)
        angle = self.compute_angle(self.goal, self.uav_pose)
        #print("Angle ", angle)
        angle_reward = np.clip(-np.abs(angle)/np.pi, -1, 0)
        reward = (distance_reward*2 + action_reward + angle_reward)
        #print("Reward Dist ", distance_reward, "Reward final ", final, "Final ", final/2, "Dist diff ", self.prev_dist-dist_now)
        print(reward)
        #reward is nan
        if np.isnan(reward):
            reward = prev_reward
        prev_reward = reward
        return reward

    def set_goal(self):
        #self.goal = self._random_goal()
        self.goal = np.array([8,8,8])

    def is_terminal(self, observation):
        done = observation[0] <= self.dist_threshold or self.steps_thr - self.steps <=0 #or (self.uav_pose[2]<3 and self.steps >500)
        #print(self.steps, self.total_ts)

        if observation[0] <= self.dist_threshold:
            self.got_goal=1
        return done

    def _create_ros_pub_sub_srv(self):

        #Control UAV position to reset simulation
        # self.position_publisher = rospy.Publisher(
        #     "/mavros/setpoint_position/local", PoseStamped, queue_size=1
        # )
        # self.local_pose_pub = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)

        # #Service to arm the UAV
        # self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        # #Service to land the UAV
        # self.land_client = rospy.ServiceProxy("mavros/cmd/land", CommandBool)

        #Action pub to the Bezier node
        # self.bezier_publisher = rospy.Publisher("/bezier/parameters", Path, queue_size=10)

        # TODO convert from bezier to velocities.
        self.agent_actions_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/actions", TwistStamped, queue_size=1)
        self.stop_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/stop", Bool, queue_size=1)
        self.reset_publisher = rospy.Publisher("uav_" + str(self.index) + "/agent/reset", Bool, queue_size=1)

        #Get the current UAV position
        #rospy.Subscriber(
        #    "/mavros/local_position/pose", PoseStamped, self._mavros_pose_callback
        #)

        #Get the current UAV postion from Gazebo
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._gazebo_pose_callback
        )

        #Mavros Services
        # self.set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        #Gazebo Services for pause and return simulation

        self.delete = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy(
            "/gazebo/reset_simulation", Empty
        )
        self.reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        time.sleep(0.1)

    # Mavros Subscriber deactivated !!! ----------
    #def _mavros_pose_callback(self, msg: PoseStamped): 
    #    self.uav_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    #-----------------------------------------    

    def _gazebo_pose_callback(self, msg:ModelStates):
        # Get the list of model names
        model_name = msg.name

        # Get the index of the 'quadrotor' model
        try:
            quadrotor_index = model_name.index("quadrotor")
        except ValueError:
            rospy.logwarn("'quadrotor' model not found in ModelStates")
            return

        # Get the pose of the 'quadrotor' model
        quadrotor_pose = msg.pose[quadrotor_index]
        quadrotor_velocity = msg.twist[quadrotor_index]
        
        # Print the pose information
        self.uav_pose = np.array([quadrotor_pose.position.x, quadrotor_pose.position.y, quadrotor_pose.position.z])
        self.uav_velocity = np.array([quadrotor_velocity.linear.x, quadrotor_velocity.linear.y, quadrotor_velocity.linear.z, quadrotor_velocity.angular.z])
        #rospy.loginfo("quadrotor Pose:\n{}".format(self.uav_pose))

    def _proc_action(self, action, noise_std=0.3): # generates action between -1 and 1 and then scale up to the boundaries
        proc = action + np.random.normal(0, noise_std, action.shape)
        proc = np.clip(proc, -1, 1)
        
        # scale the action to the boundaries
        proc[0] = lmap(proc[0], [-1, 1], self.act_bnd["vx"])
        proc[1] = lmap(proc[1], [-1, 1], self.act_bnd["vx"])
        proc[2] = lmap(proc[2], [-1, 1], self.act_bnd["vz"])
        proc[3] = lmap(proc[3], [-1, 1], self.act_bnd["wz"])

        return proc

    def _random_goal(self):
        goal = np.random.uniform(-1, 1, 3)
        goal[0:2] = lmap(goal[0:2], [-1, 1], self.act_bnd["XY"])
        goal[2] = lmap(goal[2], [-1, 1], self.act_bnd["Z"])
        #goal = self.obs_bnd["xyz"]*(goal)
        return goal

    def _clear_goal(self):
        self.goal=np.zeros(3)
        #kill_obj = KillRequest()
        #kill_obj.name = self.goal_name
        #self.kill_srv(kill_obj)


    def reset_world(self):
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as err:
            print("/gazebo/reset_world service call failed", err)

    def pause_sim(self):
        """pause simulation with ros service call"""
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as err:
            print("/gazebo/pause_physics service call failed", err)

        rospy.logdebug("PAUSING FINISH")

    def unpause_sim(self):
        """unpause simulation with ros service call"""
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as err:
            print("/gazebo/unpause_physics service call failed", err)

        rospy.logdebug("UNPAUSING FiNISH")

    def set_uav_mode(self,new_mode):
        mode = SetMode()
        mode.custom_mode = new_mode
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            set_mode_response = self.set_mode(0, mode.custom_mode)  # 0 is the base mode
            if set_mode_response.mode_sent:
                rospy.loginfo("Changed to %s mode",new_mode)
            else:
                rospy.logwarn("Failed to change to %s mode",new_mode)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))
        

    # def _reset_bezier(self, mode="reset"):
    #     msg0 = PoseStamped()
    #     msg0.header.frame_id="P0"
    #     msg0.pose.position = Point(0,0,5)

    #     msg1 = PoseStamped()
    #     msg1.header.frame_id="P1"
    #     msg1.pose.position = msg0.pose.position
            
    #     msg2 = PoseStamped()
    #     msg2.header.frame_id="P2"
    #     msg2.pose.position = msg0.pose.position

    #     msg3 = PoseStamped()
    #     msg3.header.frame_id="GOAL"
    #     msg3.pose.position= Point(self.goal[0],self.goal[1],self.goal[2])
                
    #     if mode == "stop":
    #         msg0.pose.position = Point(self.uav_pose[0], self.uav_pose[1], self.uav_pose[2])
    #         msg1.pose.position = msg0.pose.position
    #         msg2.pose.position = msg0.pose.position
    #         #print("Stop sended", msg0.pose.position)

    #     sender_msg = Path()
    #     sender_msg.header.frame_id = "map"
    #     sender_poses = []
    #     sender_poses.append(msg0)
    #     sender_poses.append(msg1)
    #     sender_poses.append(msg2)
    #     sender_poses.append(msg3)
    #     sender_msg.poses = sender_poses

    #     self.bezier_publisher.publish(sender_msg) 



    @classmethod
    def compute_angle(cls, goal_pos: np.array, obs_pos: np.array) -> float:
        pos_diff = obs_pos - goal_pos
        goal_yaw = np.arctan2(pos_diff[1], pos_diff[0]) - np.pi
        ang_diff = goal_yaw - obs_pos[2]

        if ang_diff > np.pi:
            ang_diff -= 2 * np.pi
        elif ang_diff < -np.pi:
            ang_diff += 2 * np.pi

        return ang_diff

    def render(self):
        raise NotImplementedError

    def close(self):
        rospy.signal_shutdown("Training Complete") 


    def _goal_pose_callback(self, msg: Pose):
        self.goal = np.array([msg.x, msg.y, msg.theta])


if __name__ == "__main__":
    rospy.init_node("UAVEnv")

    env = UAVEnv()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        action = np.ones_like(action)  # [thrust, ang_vel]
        obs, reward, terminal, info = env.step(action)
        
    env.reset()
    print("Finish test")
