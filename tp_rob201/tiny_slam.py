""" A simple robotics navigation code including SLAM, exploration, planning"""

import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
import heapq
import sys

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        # Given : constructor
        self.x_min_world = x_min
        self.x_max_world = x_max
        self.y_min_world = y_min
        self.y_max_world = y_max
        self.resolution = resolution

        self.x_max_map, self.y_max_map = self._conv_world_to_map(
            self.x_max_world, self.y_max_world)

        self.occupancy_map = np.zeros(
            (int(self.x_max_map), int(self.y_max_map)))

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _conv_world_to_map(self, x_world, y_world):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """
        x_map = (x_world - self.x_min_world) / self.resolution
        y_map = (y_world - self.y_min_world) / self.resolution

        if isinstance(x_map, float):
            x_map = int(x_map)
            y_map = int(y_map)
        elif isinstance(x_map, np.ndarray):
            x_map = x_map.astype(int)
            y_map = y_map.astype(int)

        return x_map, y_map

    def _conv_map_to_world(self, x_map, y_map):
        """
        Convert from map coordinates to world coordinates
        x_map, y_map : list of x and y coordinates in cell numbers (~pixels)
        """
        x_world = self.x_min_world + x_map *  self.resolution
        y_world = self.y_min_world + y_map *  self.resolution

        if isinstance(x_world, np.ndarray):
            x_world = x_world.astype(float)
            y_world = y_world.astype(float)

        return x_world, y_world

    def add_map_line(self, x_0, y_0, x_1, y_1, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        # convert to pixels
        x_start, y_start = self._conv_world_to_map(x_0, y_0)
        x_end, y_end = self._conv_world_to_map(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_map or y_start < 0 or y_start >= self.y_max_map:
            return

        if x_end < 0 or x_end >= self.x_max_map or y_end < 0 or y_end >= self.y_max_map:
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        # add value to the points
        self.occupancy_map[points[0], points[1]] += val

    def add_map_points(self, points_x, points_y, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """
        x_px, y_px = self._conv_world_to_map(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_map),
                                np.logical_and(y_px >= 0, y_px < self.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        self.occupancy_map[x_px, y_px] += val


    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        angles = lidar.get_ray_angles() + pose[2]
        distances = lidar.get_sensor_values()
        score = 0
        max_range = lidar.max_range

        #Filtrage des points en dehors de la mesure du laser
        remove_index = np.argwhere(distances < max_range)
        distances_corrected, angles_corrected = distances[remove_index], angles[remove_index]

        #Conversion dans le repère global de l'odométrie
        x_poses = pose[0] + distances_corrected*np.cos(angles_corrected)
        y_poses = pose[1] + distances_corrected*np.sin(angles_corrected)

        #Suppression des points hors carte
        x_map, y_map = self._conv_world_to_map(x_poses, y_poses)
        remove_index_x = np.where((0 < x_map) & (x_map < self.x_max_map))
        remove_index_y = np.where((0 < y_map) & (y_map < self.y_max_map))
        remove_index = np.intersect1d(remove_index_x, remove_index_y)
        x_map = x_map[remove_index]
        y_map = y_map[remove_index]
        
        score = np.sum(self.occupancy_map[x_map, y_map])

        return score

    def get_corrected_pose(self, odom, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        odom_pose = np.zeros(3)

        if odom_pose_ref != None :
            odom_pose[0] = odom[0] + odom_pose_ref*np.cos(odom[2] + odom_pose_ref[2])
            odom_pose[1] = odom[1] + odom_pose_ref*np.sin(odom[2] + odom_pose_ref[2])
            odom_pose[2] = odom[2] + odom_pose_ref[2]
        else :
            odom_pose[0] = odom[0] + self.odom_pose_ref*np.cos(odom[2] + self.odom_pose_ref[2])
            odom_pose[1] = odom[1] + self.odom_pose_ref*np.sin(odom[2] + self.odom_pose_ref[2])
            odom_pose[2] = odom[2] + self.odom_pose_ref[2]

        return odom_pose

    def localise(self, lidar, odom):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        counter = 0
        N = 10
        sigma = 50
        best_score = 0
        save_pos = odom
        while (counter < N) :
            (offset_x, offset_y) = np.random.normal(0, sigma, 2)
            offset_angle = np.random.normal(0, 0.2)
            new_pos = (odom[0] + offset_x, odom[1] + offset_y, odom[2] + offset_angle)
            score = self.score(lidar, new_pos)
            counter += 1
            if (score > best_score) :
                best_score = score
                counter = 0
                save_pos = new_pos

        self.odom_pose_ref = save_pos
        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        distances = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()
        max_range = lidar.max_range
        index = np.where(distances < max_range-10)
        distances = distances[index]
        angles = angles[index]
        
        corrected_angles = angles + pose[2]
        coordinates_robot_x = pose[0] + np.cos(corrected_angles)*distances
        coordinates_robot_y = pose[1] + np.sin(corrected_angles)*distances
        coordinates_robot_x_far = pose[0] + np.cos(corrected_angles)*distances*0.99
        coordinates_robot_y_far = pose[1] + np.sin(corrected_angles)*distances*0.99
        lenght = len(coordinates_robot_x)
        
        for i in range(lenght) :
            self.add_map_line(pose[0], pose[1], coordinates_robot_x_far[i], coordinates_robot_y_far[i], -2)
        
        self.add_map_points(coordinates_robot_x, coordinates_robot_y, +4)
        self.display(pose)

        #Seuillage
        self.occupancy_map[self.occupancy_map > 10] = 10
        self.occupancy_map[self.occupancy_map < -10] = -10


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        """
        # TODO for TP5

        path = [start, goal]  # list of poses
        return path

    def display(self, robot_pose):
        """
        Screen display of map and robot pose, using matplotlib
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        plt.cla()
        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world, self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")

        delta_x = np.cos(robot_pose[2]) * 10
        delta_y = np.sin(robot_pose[2]) * 10
        plt.arrow(robot_pose[0], robot_pose[1], delta_x, delta_y,
                  color='red', head_width=5, head_length=10, )

        # plt.show()
        plt.pause(0.001)

    def display2(self, robot_pose):
        """
        Screen display of map and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """

        img = cv2.flip(self.occupancy_map.T, 0)
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img2 = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        pt2_x = robot_pose[0] + np.cos(robot_pose[2]) * 20
        pt2_y = robot_pose[1] + np.sin(robot_pose[2]) * 20
        pt2_x, pt2_y = self._conv_world_to_map(pt2_x, -pt2_y)

        pt1_x, pt1_y = self._conv_world_to_map(robot_pose[0], -robot_pose[1])

        # print("robot_pose", robot_pose)
        pt1 = (int(pt1_x), int(pt1_y))
        pt2 = (int(pt2_x), int(pt2_y))
        cv2.arrowedLine(img=img2, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow("map slam", img2)
        cv2.waitKey(1)

    def save(self, filename):
        """
        Save map as image and pickle object
        filename : base name (without extension) of file on disk
        """

        plt.imshow(self.occupancy_map.T, origin='lower',
                   extent=[self.x_min_world, self.x_max_world,
                           self.y_min_world, self.y_max_world])
        plt.clim(-4, 4)
        plt.axis("equal")
        plt.savefig(filename + '.png')

        with open(filename + ".p", "wb") as fid:
            pickle.dump({'occupancy_map': self.occupancy_map,
                         'resolution': self.resolution,
                         'x_min_world': self.x_min_world,
                         'x_max_world': self.x_max_world,
                         'y_min_world': self.y_min_world,
                         'y_max_world': self.y_max_world}, fid)

    def load(self, filename):
        """
        Load map from pickle object
        filename : base name (without extension) of file on disk
        """
        # TODO

    def get_neighbors(self, current):

        voisins = []
        x_map, y_map = self._conv_world_to_map(current[0], current[1])
        for i in range(-1,1,1) :
            for j in range(-1,-1,1):
                pos = [x_map+i, y_map+j]
                if ( (i,j) != (0,0) and self.occupancy_map[pos[0], pos[1]] < 0 and 0 < pos[0] and pos[0] < self.x_max_map and 0 < pos[1] and pos[1] < self.y_max_map):
                    voisins.append(pos)
        return voisins
    
    
    #Reconstruct the followed path in the world coordinates
    def reconstruct_path(self, cameFrom, start, goal) :
        path = []
        current = goal
        while current != start :
            current_world = self._conv_map_to_world(current[0], current[1])
            path = [current_world] + path
            current = cameFrom[current]
        return path
    
    def heuristic(self, a, b):
        return np.linalg.norm(b-a)

    def A_Star(self, start, goal) :
        cameFrom = np.zeros((self.x_max_map, self.y_max_map))
        
        gScore = np.full((self.x_max_map, self.y_max_map), sys._float_info.max, float)
        gScore[start] = 0
        
        fScore = np.full((self.x_max_map, self.y_max_map), sys._float_info.max, float)
        fScore[start] = self.heuristic(start, goal)
        
        openSet = heapq.heapify([(fScore[start], start)])

        notempty = True

        while(notempty) :
            current = heapq.heappop(openSet)
            if current == IndexError :
                notempty == False
                break

            if current[1] == goal :
                return self.reconstruct_path(cameFrom, current)
            
            voisins = self.get_neighbors(current)

            for i in voisins :
                tentative_gScore = gScore[current] + self.heuristic(current, i)
                if tentative_gScore < gScore[i] :
                    cameFrom[i] = current
                    gScore[i] = tentative_gScore
                    fScore = tentative_gScore + self.heuristic(i, goal)
                
                ispresent = False
                for j in openSet :
                    if j[1] == i :
                        ispresent == True
                
                if(not ispresent) :
                    heapq.heappush(openSet, (fScore(i), i))

        return False
    

    # If the point has a undiscovered neighbor, then it's a frontier
    def is_frontier(self, pose):
        voisins = self.get_neighbors(pose)
        frontier = False

        for i in voisins:
            if self.occupancy_map[i] == 0 :
                frontier == True
        
        return frontier


    #Frontier detection implemented following the paper at https://arxiv.org/pdf/1806.03581.pdf
    #
    #Convention de notation :
    #
    # 1 = Map-Open-List
    # 2 = Map-Close-List
    # 3 = Frontier-Open-List
    # 4 = Frontier-Close-List

    def frontier(self, location):
        #Initialising some variables
        pose = self._conv_world_to_map(location[0], location[1])
        result = np.array([])
        
        #A queue for analyzed emplacements
        m_queue = [pose]
        #A map for the emplacements' marking storage
        marking = np.zeros((self.x_max_map, self.y_max_map))
        marking[pose] = 1

        #While every KNOWN emplacement have not been analysed
        while(len(m_queue) != 0) :
            #Pop the first element of the queue
            p = m_queue[0]
            m_queue = m_queue[1:]
            print("Getting an element from the m_queue", p)
            
            #If p has been analysed, do nothing.
            if(marking[p] == 2):
                print("Already analyzed, doing nothing")
                continue

            #Else, under the condition that the emplacement is a frontier.
            if(self.is_frontier(p) == True):
                print("Starting building a frontier from p")
                #New queue for frontier neighbors
                f_queue = []
                #Collection of the frontier's emplacement
                NewFrontier = []
                #Appending the new element
                f_queue = f_queue + [p]
                #Marking the element as part of a undiscovered frontier
                marking[p] = 3
                #While there's still elements of the frontier not analysed.
                while(len(f_queue) != 0) :
                    
                    #Pop elements of the queue
                    q = f_queue[0]
                    f_queue = f_queue[1:]

                    #If already analysed
                    if (marking[q] in (2, 4)):
                        continue
                    #Else if the emplacement is part of a frontier (hence the current one)
                    if (self.is_frontier(q) == True):
                        #Add it to the frontier
                        NewFrontier = NewFrontier + [q]

                        #Getting its neighbors and adding them to the queue if possible.
                        voisins = self.get_neighbors(q)
                        for i in voisins :
                            if (marking[i] not in (2, 3, 4)) :
                                f_queue = f_queue + [i]
                                marking[i] = 3
                    
                    #Characterizing the current emplacement as analyzed.
                    marking[q] = 4

                #Loading the frontier in the result
                result = np.append(result, NewFrontier)
                #Marking the elements of the frontier according to their new status
                for i in NewFrontier :
                    marking[i] = 2
            
            #Getting the neighbors of the current position
            voisins = self.get_neighbors(p)
            
            #If the neighbors have not been analyzed, and have an open and practicable space, then add them to the queue
            for v in voisins :
                if (marking[v] not in (1, 2)):
                    voisinsbis = self.get_neighbors(v)
                    Open_Space = False
                    for j in voisinsbis :
                        if (self.occupancy_map[j] < 0) :
                            Open_Space == True
                    if Open_Space :
                        marking[v] = 1
                        m_queue = m_queue + [v]
            #Marking the current emplacement as analyzed.
            marking[p] = 2

        #Returning the result.
        return(result)