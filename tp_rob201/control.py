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
    dchang = 200
    rmin = 20
    dsafe = 50

    #Acquisition des données.
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    ecart = [goal[0]-pose[0], goal[1]-pose[1]]
    ecart_norm = np.linalg.norm(ecart)
    
    #Détection de l'obstacle le plus proche.
    index = np.where(distances == np.min(distances))[0][0]
    mindist = distances[index]
    minangle = angles[index]
    #print("Minangle : ", minangle)
    if mindist < dsafe :
        #print("Wall Detected")
        Kobs = 5000000
        pregrad = Kobs/(mindist**3)*((1/mindist)-(1/dsafe))
        gradient_obstacle = np.array([pregrad*mindist*np.cos(minangle), pregrad*mindist*np.sin(minangle)])
    else :
        gradient_obstacle = np.array([0,0])
    print("Gradient Obstacle : ", gradient_obstacle)

    #Cas éloigné - Potentiel conique.
    if ecart_norm > dchang :
        Kcone = 100
        pregrad = Kcone/np.linalg.norm(ecart)
        gradient = np.array([pregrad*ecart[0], pregrad*ecart[1]])
        print("Old Gradient : ", gradient)
        gradient = gradient - gradient_obstacle
        print("New Gradient : ", gradient)
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(0.01*np.log(gradient_norme), -1, 1)
        #velocity = np.clip(gradient_norme, -1, 1)
        rotation = np.clip((gradient_angle-pose[2])/np.pi, -1, 1)

    #Cas proche - Potentiel quadratique.
    elif rmin < ecart_norm <= dchang :
        print("Approaching the goal")
        Kquad = 1/dchang
        gradient = np.array([Kquad*ecart[0], Kquad*ecart[1]])
        print("Old Gradient : ", gradient)
        gradient = gradient - gradient_obstacle
        print("New Gradient : ", gradient)
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        #velocity = np.clip(0.05*np.log(gradient_norme), -1, 1)
        velocity = np.clip(gradient_norme, -1, 1)
        rotation = np.clip((gradient_angle-pose[2])/np.pi, -1, 1)
    
    #Cas touché - On s'arrête.
    elif ecart_norm <= rmin :
        velocity = 0
        rotation = 0

    print("Velocity : " , velocity, " Rotation : ", rotation)

    command = {"forward": velocity,
               "rotation": rotation}

    return command
