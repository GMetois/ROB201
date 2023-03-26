import numpy as np

""" A set of robotics control functions """


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    span = 30
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    index = np.where(angles == 0)[0][0]
    indexlist = np.array([index+i for i in range(-span,span,1)])
    mindistance = np.min([distances[i] for i in indexlist])
    forward = 0
    rotation = 0

    if mindistance < 70 :
        forward = 0
        indexangle = np.where(distances == np.max(distances))
        rotation = angles[indexangle]
        rotation = rotation/np.pi
        print(rotation)
    else :
        forward = 0.2
        rotation = 0

    command = {"forward": forward,
               "rotation": rotation}

    return command

def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
   # TODO for TP2

    command = {"forward": 0,
               "rotation": 0}

    return command
