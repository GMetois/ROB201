import numpy as np

""" A set of robotics control functions """


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    span = 15
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    index = np.where(angles == 0)[0][0]
    indexlist = np.array([index+i for i in range(-span,span,1)])
    mindistance = np.min([distances[i] for i in indexlist])
    forward = 0
    rotation = 0

    if mindistance < 70 :
        forward = 0
        #indexangle = np.where(distances == np.max(distances))
        #rotation = angles[indexangle]
        #rotation = rotation/np.pi
        rotation = np.random.uniform(-1,1)
    
    else :
        forward = 1
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
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    ecart = [goal[0]-pose[0], goal[1]-pose[1]]
    Kgoal = 1
    pregrad = Kgoal/np.linalg.norm(ecart)
    gradient = np.array([pregrad*ecart[0], pregrad*ecart[1]])
    gradient_angle = np.arctan2(gradient[0], gradient[1])
    gradient_norme = np.linalg.norm(gradient)
    velocity = np.clip(0.01*np.log(np.linalg.norm(ecart)), -1, 1)
    rotation = (gradient_angle-pose[2])/np.pi
    print("Velocity : "  ,velocity, " Rotation : ", rotation)

    command = {"forward": velocity,
               "rotation": rotation}

    return command
