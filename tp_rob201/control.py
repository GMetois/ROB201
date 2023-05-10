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
        rotation = np.random.uniform(-1,1)
    
    else :
        forward = 1
        rotation = 0

    command = {"forward": forward,
               "rotation": rotation}

    command = {"forward": 0,
               "rotation": 0}

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
    
    #Conversion des coordonnées de l'obstacles dans la position du monde.
    obstacle_position = np.array([pose[0] + mindist*np.cos(minangle+pose[2]), pose[1] + mindist*np.sin(minangle+pose[2])])
    
    #Si on est proche de l'obstacle, on le prend en compte.
    if mindist < dsafe :
        Kobs = 10000
        pregrad = Kobs/(mindist**3)*((1/mindist)-(1/dsafe))
        gradient_obstacle = pregrad*(obstacle_position - np.array([pose[0],pose[1]]))
    #Au dessus du seuil, on l'ignore.
    else :
        gradient_obstacle = np.array([0,0])

    #Cas éloigné - Potentiel conique.
    if ecart_norm > dchang :
        Kcone = 0.1
        
        #Calcul du gradient.
        pregrad = Kcone/np.linalg.norm(ecart)
        gradient = np.array([pregrad*ecart[0], pregrad*ecart[1]])
        #Ajout du gradient obstacle.
        gradient = gradient - gradient_obstacle
        #Caractéristiques du gradient pour le contrôle.
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(gradient_norme*np.cos(gradient_angle-pose[2]), -1, 1)
        rotation = np.clip(gradient_norme*np.sin(gradient_angle-pose[2]), -1, 1)

    #Cas proche - Potentiel quadratique.
    elif rmin < ecart_norm <= dchang :
        Kquad = 0.1/dchang
        gradient = np.array([Kquad*ecart[0], Kquad*ecart[1]])
        gradient = gradient - gradient_obstacle
        gradient_angle = np.arctan2(gradient[1], gradient[0])
        gradient_norme = np.linalg.norm(gradient)
        velocity = np.clip(gradient_norme*np.cos(gradient_angle-pose[2]), -1, 1)
        rotation = np.clip(gradient_norme*np.sin(gradient_angle-pose[2]), -1, 1)
    
    #Cas touché - On s'arrête.
    elif ecart_norm <= rmin :
        velocity = 0
        rotation = 0

    #Ligne de contrôle général.
    print("Position : ", pose[0], pose[1], "Velocity : " , velocity, " Rotation : ", rotation)

    command = {"forward": velocity,
               "rotation": rotation}

    return command
