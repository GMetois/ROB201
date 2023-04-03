import numpy as np
import time

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
    #Paramètres
    dchang = 30
    rmin = 10
    dsafe = 50

    #Acquisition des données.
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    ecart = [goal[0]-pose[0], goal[1]-pose[1]]
    ecart_norm = np.linalg.norm(ecart)
    
    #Détection de l'obstacle le plus proche.
    index = np.argmin(distances)
    mindist = distances[index]
    minangle = angles[index]
    obstacle_position = np.array([pose[0] + mindist*np.cos(minangle+pose[2]), pose[1] + mindist*np.sin(minangle+pose[2])])
    if mindist < dsafe :
        #print("Wall Detected")
        Kobs = 10000
        pregrad = Kobs/(mindist**3)*((1/mindist)-(1/dsafe))
        gradient_obstacle = pregrad*(obstacle_position - np.array([pose[0],pose [1]]))
    else :
        gradient_obstacle = np.array([0,0])
    print("Gradient Obstacle : ", gradient_obstacle)


    #values,angles = lidar.get_sensor_values(), lidar.get_ray_angles()
    #min_val = np.argmin(values)
    #obstacle_pos = np.array([pose[0] + values[min_val]*np.cos(angles[min_val])+pose[2],pose[1] + values[min_val]*np.sin(angles[min_val])+pose[2]])
    #obs_dist = np.linalg.norm(obstacle_pos-np.array([pose[0],pose[1]]))
    #gradient_obstacle = 1e3/(obs_dist**3)*(1/obs_dist-1/dsafe)*(obs_dist-pose[:1]) if obs_dist < dsafe else np.array([0,0])


    #Cas éloigné - Potentiel conique.
    if ecart_norm > dchang :
        Kcone = 0.1
        pregrad = Kcone/np.linalg.norm(ecart)
        gradient = np.array([pregrad*ecart[0], pregrad*ecart[1]])
        print("Old Gradient : ", gradient)
        gradient = gradient - gradient_obstacle
        print("New Gradient : ", gradient)
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(gradient_norme*np.cos(gradient_angle-pose[2]), -1, 1)
        #velocity = np.clip(gradient_norme, -1, 1)
        rotation = np.clip(gradient_norme*np.sin(gradient_angle-pose[2]), -1, 1)

    #Cas proche - Potentiel quadratique.
    elif rmin < ecart_norm <= dchang :
        print("Approaching the goal")
        Kquad = 0.1/dchang
        gradient = np.array([Kquad*ecart[0], Kquad*ecart[1]])
        print("Old Gradient : ", gradient)
        gradient = gradient - gradient_obstacle
        print("New Gradient : ", gradient)
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(gradient_norme*np.cos(gradient_angle-pose[2]), -1, 1)
        #velocity = np.clip(gradient_norme, -1, 1)
        rotation = np.clip(gradient_norme*np.sin(gradient_angle-pose[2]), -1, 1)
    
    #Cas touché - On s'arrête.
    elif ecart_norm <= rmin :
        velocity = 0
        rotation = 0

    print("Position : ", pose[0], pose[1], "Velocity : " , velocity, " Rotation : ", rotation)

    command = {"forward": velocity,
               "rotation": rotation}

    return command
