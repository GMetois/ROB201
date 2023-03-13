import math
from abc import abstractmethod
from enum import IntEnum

from spg.agent.agent import Agent

from place_bot.entities.robot_base import RobotBase
from place_bot.entities.lidar import Lidar, LidarParams
# from place_bot.entities.odometer import Odometer
# from place_bot.entities.odometer_v2 import OdometerV2
from place_bot.entities.odometer import Odometer, OdometerParams
from place_bot.utils.utils import normalize_angle

import matplotlib.pyplot as plt


class RobotAbstract(Agent):
    """
    This class should be used as a parent class to create your own Robot class.
    It is a BaseAgent class with 3 sensors, 1 sensor of position and one mandatory functions control()
    """

    class SensorType(IntEnum):
        LIDAR = 0
        ODOMETER = 1

    def __init__(self,
                 should_display_lidar=False,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        super().__init__(interactive=True, lateral=False, radius=10)

        base = RobotBase()
        self.add(base)

        self.base.add(Lidar(lidar_params=lidar_params, invisible_elements=self._parts))
        self.base.add(Odometer(odometer_params=odometer_params))

        self._should_display_lidar = should_display_lidar

        if self._should_display_lidar:
            plt.figure(self.SensorType.LIDAR)
            plt.axis([-300, 300, 0, 300])
            plt.title("lidar measurements")
            plt.ion()

    @abstractmethod
    def control(self):
        """
        This function is mandatory in the class you have to create that will inherit from this class.
        This function should return a command which is a dict with values for the actuators.
        For example:
        command = {"forward": 1.0,
                   "rotation": -1.0}
        """
        pass

    def lidar(self):
        """
        Give access to the value of the lidar sensor.
        """
        return self.sensors[self.SensorType.LIDAR.value]

    def lidar_is_disabled(self):
        return self.lidar().is_disabled()

    def odometer_is_disabled(self):
        return self.sensors[self.SensorType.ODOMETER].is_disabled()

    def odometer_values(self):
        """
        Give the estimated pose after integration of odometer's delta
        """
        return self.sensors[self.SensorType.ODOMETER].get_sensor_values()

    def true_position(self):
        """
        Give the true orientation of the robot, in pixels
        You must NOT use this value for your calculation in the control() function, instead you
        should use the position estimated by the odometer sensor.
        But you can use it for debugging or logging.
        """
        return self.position

    def true_angle(self):
        """
        Give the true orientation of the robot, in radians between 0 and 2Pi.
        You must NOT use this value for your calculation in the control() function, instead you
        should use the position estimated by the odometer sensor.
        But you can use it for debugging or logging.
        """
        return normalize_angle(self.angle)

    def true_velocity(self):
        """
        Give the true velocity of the robot, in pixels per second
        You must NOT use this value for your calculation in the control() function, instead you
        should use the position estimated by the odometer sensor.
        But you can use it for debugging or logging.
        """
        return self.base.velocity

    def true_angular_velocity(self):
        """
        Give the true angular velocity of the robot, in radians per second
        You must NOT use this value for your calculation in the control() function, instead you
        should use the position estimated by the odometer sensor.
        But you can use it for debugging or logging.
        """
        return self.base.angular_velocity

    def display(self):
        if self._should_display_lidar:
            self.display_lidar()

    def display_lidar(self):
        if self.lidar().get_sensor_values() is not None:
            plt.figure(self.SensorType.LIDAR)
            plt.cla()
            plt.axis([-math.pi, math.pi, 0, self.lidar().max_range])
            plt.plot(self.lidar().get_ray_angles(), self.lidar().get_sensor_values(), "g.:")
            plt.grid(True)
            plt.title("lidar measurements", fontsize=16)
            plt.draw()
            plt.pause(0.001)
