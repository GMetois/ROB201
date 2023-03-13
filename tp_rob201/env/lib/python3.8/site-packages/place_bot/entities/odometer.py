import math

import numpy as np

from spg.agent.sensor.internal import InternalSensor
from place_bot.utils.utils import rad2deg, normalize_angle


class OdometerParams:
    """
    Class containing parameters for the Odometer sensor

    Parameters:
    - param1 (float): Influence of translation on translation
    - param2 (float): Influence of rotation on translation
    - param3 (float): Influence of translation on rotation
    - param4 (float): Influence of rotation on rotation
    """
    param1 = 0.3  # 0.3  # meter/meter, influence of translation to translation
    param2 = 0.1  # 0.1  # meter/degree, influence of rotation to translation
    param3 = 0.04  # 0.04 # degree/meter, influence of translation to rotation
    param4 = 0.01  # 0.01 # degree/degree, influence of rotation to rotation


class Odometer(InternalSensor):
    """
      Odometer sensor returns a numpy array containing:
      - dist_travel, the distance of the travel of the robot during one step
      - alpha, the relative angle of the current position seen from the previous reference frame of the robot
      - theta, the variation of orientation (or rotation) of the robot during the last step in the reference frame

      For the noise model, it is inspired by these pages:
      - https://blog.lxsang.me/post/id/16
      - https://www.mrpt.org/tutorials/programming/odometry-and-motion-models/probabilistic_motion_models/
      - https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf  page 113

      I could not use directly the model given in the first link, because my robot slides a bit sideways.
      This causes calculation problems. If the robot moves sideways, it will be calculated that the robot has done
      a 90° degree rotation, moved straight and then another 90° in the other direction. This distorts the
      calculations of the noise...
    """

    def __init__(self, odometer_params: OdometerParams = OdometerParams(), **kwargs):
        """
        Initialize the Odometer sensor instance.

        Parameters:
        - odometer_params: an OdometerParams instance containing parameters for the sensor
        - kwargs: other keyword arguments
        """
        super().__init__(**kwargs)
        self._noise = True

        self.param1 = odometer_params.param1
        self.param2 = odometer_params.param2
        self.param3 = odometer_params.param3
        self.param4 = odometer_params.param4

        self._values = self._default_value
        self._dist = 0
        self._alpha = 0
        self._theta = 0
        self.prev_angle = None
        self.prev_position = None

    def _compute_raw_sensor(self):
        """
        Compute the distance traveled, relative angle, and variation of orientation for the robot.
        """
        # DIST_TRAVEL
        if self.prev_position is None:
            self.prev_position = self._anchor.position

        travel_vector = self._anchor.position - self.prev_position
        self._dist = math.sqrt(travel_vector[0] ** 2 + travel_vector[1] ** 2)

        has_translated = True
        if abs(self._dist) < 1e-5:
            has_translated = False

        # ALPHA
        if self.prev_angle is None:
            self.prev_angle = self._anchor.angle

        if has_translated:
            alpha = math.atan2(travel_vector[1], travel_vector[0]) - self.prev_angle
        else:
            alpha = 0
        self._alpha = normalize_angle(alpha)

        # THETA
        theta = self._anchor.angle - self.prev_angle
        self._theta = normalize_angle(theta)

        # UPDATE
        self.prev_position = self._anchor.position
        self.prev_angle = self._anchor.angle

        if self._noise:
            self._apply_my_noise()

        self.integration()

    def integration(self):
        """
        Compute a new position of the robot by adding noisy displacement to the previous
        position. It updates self._values.
        """
        x, y, orient = tuple(self._values)
        new_x = x + self._dist * math.cos(self._alpha + orient)
        new_y = y + self._dist * math.sin(self._alpha + orient)
        new_orient = orient + self._theta

        new_orient = normalize_angle(new_orient)

        self._values = np.array([new_x, new_y, new_orient])

    def _apply_normalization(self):
        pass

    @property
    def _default_value(self) -> np.ndarray:
        return np.zeros(self.shape)

    def get_sensor_values(self):
        return self._values

    def draw(self):
        pass

    @property
    def shape(self) -> tuple:
        return 3,

    def _apply_noise(self):
        """
        Overload of an internal function of _apply_noise of the class InternalSensor
        As we have to do more computation (integration) after this function, we cannot use it.
        In the function update() of sensor.pu in SPG, we call first _compute_raw_sensor() then _apply_noise. That all.
        We will use the function _apply_my_noise below instead.
        """
        pass

    def _apply_my_noise(self):
        """
        Overload of an internal function of _apply_noise of the class InternalSensor
        """
        sd_trans = self.param1 * self._dist + self.param2 * rad2deg(abs(self._theta))
        sd_rot = self.param3 * self._dist + self.param4 * rad2deg(abs(self._theta))

        noisy_alpha = self._alpha + np.random.normal(0, sd_rot * sd_rot)
        noisy_dist_travel = self._dist + np.random.normal(0, sd_trans * sd_trans)
        noisy_theta = self._theta + np.random.normal(0, sd_rot * sd_rot)

        self._dist = noisy_dist_travel
        self._alpha = noisy_alpha
        self._theta = noisy_theta

    def is_disabled(self):
        return self._disabled
