from abc import ABC, abstractmethod
from typing import Type, Union

from spg.playground import Playground

from place_bot.entities.robot_abstract import RobotAbstract


class WorldAbstract(ABC):
    """
    It is abstract class to construct every worlds used in the directory worlds
    """

    def __init__(self, robot: Union[RobotAbstract, None]):
        self._size_area = None
        self._robot = robot
        self._playground: Union[Playground, Type[None]] = None

    @property
    def robot(self):
        return self._robot

    @property
    def size_area(self):
        return self._size_area

    @property
    def playground(self):
        return self._playground
