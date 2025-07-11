from GVD.voronoi.voronoi import GeneralizedVoronoi, run_type
from GVD.voronoi.geometry import Line, Triangle
from GVD.voronoi.image import PolygonDetector
from GVD.voronoi.astar import Astar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scipy.interpolate as si
from DPP import DeceptivePathPlanner
from tag_ID import TagID as id

class Planner:
    def __init__(self, tags, frame_width, frame_height):
        self.tags = tags
        self.robot_tag_width = self.measureTag(id.ROBOT_ID)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.deceptivepath = None
        self.failed = False

        self.computeGVD()
        if self.failed:
            return
        self.deceptivepath = self.findDeceptivePath()
        #print(self.deceptivepath)
        self.deceptivepath = self.smooth_path_with_spline(self.deceptivepath)
        #print(self.deceptivepath)
        #self.displayPath()

    def measureTag(self, id):
        width = None
        for tag in self.tags:
            if tag.tag_id == id:
                width = np.hypot(tag.corners[0][0] - tag.corners[1][0], tag.corners[0][1] - tag.corners[1][1])
        return width

    def smooth_path_with_spline(self, path, resolution=0.01):
        # remove really close points
        i = 1
        while i < len(path):
            if distance(path[0], path[1]) <= 0.05:
                path = np.delete(path, i)
            else:
                i += 1
            
        path = np.array(path)
        if path.ndim != 2 or path.shape[0] < 3:
            return path
        
        # Normalized parameter along the path
        t = np.linspace(0, 1, len(path))
        
        # Create splines
        spl_x = si.make_interp_spline(t, path[:, 0], k=3)
        spl_y = si.make_interp_spline(t, path[:, 1], k=3)
        
        # Generate new points - ensure we reach the end point
        t_new = np.linspace(0, 1, int(1/resolution)+1)  # Includes endpoint
        
        # Interpolate
        x_smooth = spl_x(t_new)
        y_smooth = spl_y(t_new)
        
        # Combine and convert to integers
        smoothed_path = np.column_stack((x_smooth, y_smooth))
        
        # Ensure we keep the exact goal position
        smoothed_path[-1] = path[-1]  # Force last point to match original goal
        # After smoothing, verify the goal was reached
        #print("Original goal:", path[-1])
        #print("Smoothed goal:", smoothed_path[-1])
        
        return smoothed_path

    def displayPath(self):
        fig, ax = plt.subplots()
        result = self.deceptivepath
        vertices = self.vor_result.vertices
        points = self.vor_result.points

        ax.plot(points[:,0], points[:,1], '.')
        ax.plot(vertices[:,0], vertices[:,1], 'o')

        segments = []
        for vertex in self.vor_result.ridge_vertices:
            vertex = np.asarray(vertex)
            segments.append(vertices[vertex])
            #print(vertices[vertex])
        ax.add_collection(LineCollection(segments, colors='g'))
        
        segments = []
        for i in range(len(result)-1):
            seg = [result[i], result[i+1]]
            segments.append(seg)
        ax.add_collection(LineCollection(segments, colors='r', lw=2.0))

        plt.show(block=False)
        plt.pause(0.001) 
    
    def findDeceptivePath(self):
        dpplanner = DeceptivePathPlanner(self.edges, self.nodes)
        prior = (0.6, 0.4)
        PBNE_paths = dpplanner.attacker_strategy(self.robot_node_id, 
                                                 self.real_goal_node_id, 
                                                 self.fake_goal_node_id, 
                                                 prior)
        idx_path = PBNE_paths['xi_1_I']
        return self.vor_result.vertices[idx_path]

    def computeGVD(self):
        self.computeBoundaries()
        if self.failed:
            return
        #print('boundaries found')

        # locate start and end
        robot = None
        real_goal = None
        fake_goal = None
        for tag in self.tags:
            if tag.tag_id == id.ROBOT_ID:
                robot = [(tag.center[0] - self.ox)/self.arena_width, 
                         ((self.frame_height - tag.center[1]) - self.oy)/self.arena_height]
            elif tag.tag_id == id.REAL_GOAL_ID:
                real_goal = [(tag.center[0] - self.ox)/self.arena_width, 
                             ((self.frame_height - tag.center[1]) - self.oy)/self.arena_height]
            elif tag.tag_id == id.FAKE_GOAL_ID:
                fake_goal = [(tag.center[0] - self.ox)/self.arena_width, 
                             ((self.frame_height - tag.center[1]) - self.oy)/self.arena_height]
        if robot is None or real_goal is None or fake_goal is None:
            print('Missing robot, fake goal, or real goal')
            self.failed = True
            return

        self.computePolygons()
        if self.failed:
            return
        GeneralizedVoronoi.rdp_epsilon = 0.0064
        Line.point_distance = 0.008
        Triangle.distance_trash = 0.02
        vor = GeneralizedVoronoi()
        vor.add_polygons(self.polygons)
        vor.add_boundaries(self.boundaries)
        vor.add_points([robot, real_goal, fake_goal])
        vor_result = vor.run(run_type.optimized)
        #print('voronoi diagram done')
        #print(vor_result)
        #print(len(vor_result.vertices))

        # saving robot and goals to ID later
        self.robot_pos = robot
        self.real_goal_pos = real_goal
        self.fake_goal_pos = fake_goal
        

        #return None
        # not actually using A*, just using its helper functions
        astar = Astar(vor_result, robot, real_goal, fake_goal) 
        self.vor_result = vor_result
        astar.run()
        self.edges, self.nodes = self.resultToGraph(vor_result)
        return None

    def resultToGraph(self, result):
        edges = []
        nodes = []

        point_pairs = np.array(result.vertices[result.ridge_vertices])
        #print(result.ridge_vertices)
        point_diff = point_pairs[:, 0, :] - point_pairs[:, 1, :]
        w = np.linalg.norm(point_diff, axis=1)
        #print(w)
        edges = list(zip(result.ridge_vertices[:, 0], result.ridge_vertices[:, 1], w))
        #print(edges)

        ids = range(len(result.vertices))
        nodes = list(zip(ids, result.vertices[:, 0], result.vertices[:, 1]))
        #print(nodes)
        
        self.robot_node_id = np.where(np.isclose(result.vertices, self.robot_pos))[0][0]
        self.real_goal_node_id = np.where(np.isclose(result.vertices, self.real_goal_pos))[0][0]
        self.fake_goal_node_id = np.where(np.isclose(result.vertices, self.fake_goal_pos))[0][0]
        #raise Exception('lol')
        return edges, nodes


    def computeBoundaries(self):
        left = None
        bottom = None
        right = None
        top = None
        for tag in self.tags:
            if tag.tag_id == id.GROUND_LEFT_ID:
                left = tag.corners[0][0]
                bottom = self.frame_height - tag.corners[0][1]
                #bl = [tag.corners[0][0], tag.corners[0][1]]
            elif tag.tag_id == id.GROUND_RIGHT_ID:
                right = tag.corners[2][0]
                top = self.frame_height - tag.corners[2][1]
                #tr = [tag.corners[1][0], tag.corners[1][1]]

        if left is None or right is None:
            print('missing one or both boundaries')
            self.failed = True
            return
        
        bl = [0, 0]
        br = [right - left, 0]
        tl = [0, top - bottom]
        tr = [right - left, top - bottom]
        #print(bl, br, tl, tr)
        #self.boundaries = [Line([bl, br]), Line([br, tr]), Line([tr, tl]), Line([bl, tl])]
        self.boundaries = [Line([[0.0, 0.0], [1.0, 0.0]]), Line([[1.0, 0.0], [1.0, 1.0]]), 
                           Line([[1.0, 1.0], [0.0, 1.0]]), Line([[0.0, 0.0], [0.0, 1.0]])]
        # save values for normalizing other objects
        self.ox = left
        self.oy = bottom
        self.arena_height = abs(top - bottom)
        self.arena_width = abs(right - left)
        return 
    
    def computePolygons(self):
        # create black/white image for PolygonDetector
        from PIL import Image, ImageDraw
        img = Image.new('L', (int(self.frame_width), int(self.frame_height)), 0) #black background
        draw = ImageDraw.Draw(img)

        obstacleFound = False
        self.obstacle_corners = []
        for tag in self.tags:
            if tag.tag_id == id.OBSTACLE_ID:
                #print(tag.corners)
                self.obstacle_corners.append(tag.corners)
                corners = [(x, y) for [x, y] in tag.corners]
                draw.polygon(corners, fill=255) # may need to change later if we want more shapes than squares
                obstacleFound = True
        
        if not obstacleFound:
            print('no obstacles detected')
            self.failed = True
            return

        img.save("BW_image.png")
        #print('saved image')

        # run polygon detector as normal
        PolygonDetector.gray_thresh_boundary = 3
        pd = PolygonDetector('BW_image.png', [214, 255])
        PolygonDetector.area_threshold = 400
        PolygonDetector.rdp_epsilon = 0.01
        polygons = pd.run(bound=[self.frame_width, self.frame_height])
        self.polygons = self.normalizePolygons(polygons)
        #print(self.polygons)
        return
    
    def normalizePolygons(self, polygons):
        #print(polygons)
        newPolygons = []
        for polygon in polygons:
            newPolygon = []
            for pt in polygon:
                pt = [(pt[0] - self.ox)/self.arena_width, 
                      (pt[1] - self.oy)/self.arena_height]
                newPolygon.append(pt)
            newPolygons.append(newPolygon)
        return newPolygons


def distance(a, b):
    return np.linalg.norm([a[0] - b[0], a[1] - b[1]])
