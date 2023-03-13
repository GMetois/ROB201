from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.simu_world.simulator import Simulator

from worlds.world_intermediate_01 import MyWorldIntermediate01
from worlds.world_complete_01 import MyWorldComplete01
from worlds.world_complete_02 import MyWorldComplete02


class MyWorld(MyWorldComplete02):
    pass


class MyRobot(RobotAbstract):

    def control(self):
        pass


if __name__ == "__main__":
    print("")
    my_robot = MyRobot
    my_world = MyWorld(robot=my_robot)

    simulator = Simulator(the_world=my_world,
                          use_mouse_measure=True)

    simulator.run()
