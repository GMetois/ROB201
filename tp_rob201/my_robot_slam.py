"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        self._size_area = (800, 800)
        self.tiny_slam = TinySlam(x_min=- self._size_area[0],
                                  x_max=self._size_area[0],
                                  y_min=- self._size_area[1],
                                  y_max=self._size_area[1],
                                  resolution=2)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # storage for frontier exploration
        self.frontier_list = np.array([])

        # storage of the current goal
        self.near_goal = False
        self.current_goal = self.current_goal = np.random.normal(0, 50, 3) + self.odometer_values()

    def control(self):
        """
        Main control function executed at each time step
        """
        self.counter += 1

        # Compute new command speed to perform obstacle avoidance

        #Init robot pos (439.0, 195.0)
        #Map Size (1113, 750)


        #Updating the map
        self.tiny_slam.update_map(self.lidar(),self.odometer_values())

        #Localisation
        self.tiny_slam.localise(self.lidar(),self.odometer_values())


        #Setting the new goal if necessary
        if self.near_goal :
            
            #Getting the frontiers
            print("Getting the frontiers")
            self.frontier_list = self.tiny_slam.frontier(self.odometer_values())
            
            self.near_goal = False
            if len(self.frontier_list) > 0 :
                spot = self.frontier_list[0]
                spot_world = self.tiny_slam._conv_map_to_world(spot[0], spot[1])
                self.current_goal = [spot_world[0], spot_world[1], 0]
                print("New goal is ", self.current_goal)
            
            else :
                self.current_goal = np.random.normal(0, 50, 3) + self.odometer_values()
                print("No frontier found, random goal")

        if (np.linalg.norm((self.odometer_values() - self.current_goal)[:1]) < 15) :
            self.near_goal = True

        print("Goal is ", self.current_goal)
        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), self.odometer_values(), self.current_goal)

        return command
