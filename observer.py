"""
observer.py

Defines two observer classes for the deception game:
 - CPUObserver: automatic allocation based on progress towards guessed goal,
   with obstacle-aware geodesic distance via a visibility graph and Dijkstra.
 - ManualObserver: manual allocation adjusted via keyboard input.

Usage:
    from observer import CPUObserver, ManualObserver
"""

import numpy as np
import heapq


def _orientation(a, b, c):
    """Return >0 if points a,b,c are clockwise, <0 if counterclockwise, 0 if colinear."""
    return (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])


def _on_segment(a, b, c):
    """Return True if point b lies on segment a-c."""
    return (min(a[0],c[0]) <= b[0] <= max(a[0],c[0]) and
            min(a[1],c[1]) <= b[1] <= max(a[1],c[1]))


def _segments_intersect(p1, p2, q1, q2):
    """Return True if segment p1-p2 intersects segment q1-q2."""
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1*o2 < 0 and o3*o4 < 0:
        return True
    # colinear cases
    if o1 == 0 and _on_segment(p1, q1, p2): return True
    if o2 == 0 and _on_segment(p1, q2, p2): return True
    if o3 == 0 and _on_segment(q1, p1, q2): return True
    if o4 == 0 and _on_segment(q1, p2, q2): return True
    return False


class CPUObserver:
    """
    CPU-based observer that allocates points based on progress towards a guessed goal.
    Uses obstacle-aware geodesic distances computed by building a visibility graph
    over start, goal, and obstacle vertices, then running Dijkstra.
    """
    def __init__(self, goal_positions, priors, obstacle_polygons=None):
        """
        goal_positions: list of two np.array([x,y]) for the two goals.
        priors: tuple (p1, p2) initial belief over goals.
        obstacle_polygons: list of list of (x, y) tuples for rectangular obstacles.
        """
        self.goals = [np.array(g) for g in goal_positions]
        self.guess = 0 if priors[0] >= priors[1] else 1
        self.obstacles = obstacle_polygons or []
        # history
        self.prev_pos = None
        self.prev_dist = None
        self.last_time = None

    def _visible(self, p, q):
        """Return True if line segment p-q doesn't intersect any obstacle edge."""
        for poly in self.obstacles:
            n = len(poly)
            for i in range(n):
                if _segments_intersect(p, q, poly[i], poly[(i+1)%n]):
                    return False
        return True

    def _geodesic_dist(self, start, goal):
        """
        Compute shortest collision-free distance from start->goal using visibility graph.
        """
        start = tuple(start)
        goal = tuple(goal)
        # direct line?
        if self._visible(start, goal):
            return np.linalg.norm(np.array(start) - np.array(goal))

        # nodes: start, goal, and all obstacle vertices
        nodes = [start, goal]
        for poly in self.obstacles:
            for v in poly:
                nodes.append(tuple(v))
        N = len(nodes)

        # build adjacency
        adj = {i: [] for i in range(N)}
        for i in range(N):
            for j in range(i+1, N):
                if self._visible(nodes[i], nodes[j]):
                    d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
                    adj[i].append((j, d))
                    adj[j].append((i, d))

        # Dijkstra
        dist = [float('inf')] * N
        dist[0] = 0
        pq = [(0, 0)]
        while pq:
            cd, u = heapq.heappop(pq)
            if u == 1:
                break
            if cd > dist[u]:
                continue
            for v, w in adj[u]:
                nd = cd + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist[1]

    def update(self, current_pos, timestamp):
        """
        Update observer with robot's current position.

        current_pos: np.array([x, y])
        timestamp: float, e.g. time.time()

        Returns: (sigma1, sigma2, dt)
        """
        pos = np.array(current_pos)
        if self.prev_pos is None:
            # initialize
            gd = self._geodesic_dist(pos, self.goals[self.guess])
            self.prev_pos = pos.copy()
            self.prev_dist = gd
            self.last_time = timestamp
            return 0.5, 0.5, 0.0

        dt = timestamp - self.last_time
        self.last_time = timestamp

        d_now = self._geodesic_dist(pos, self.goals[self.guess])
        prog = max(0.0, self.prev_dist - d_now)
        l = np.linalg.norm(pos - self.prev_pos)

        sigma1 = (prog / l) if l > 1e-6 else (1.0 if prog > 0 else 0.0)
        sigma1 = float(np.clip(sigma1, 0.0, 1.0))
        sigma2 = 1.0 - sigma1

        self.prev_dist = d_now
        self.prev_pos = pos.copy()

        return sigma1, sigma2, dt


class ManualObserver:
    """
    Manual observer: operator adjusts allocation via adjust(delta);
    update() returns current allocation.
    """
    def __init__(self, goal_positions, priors, initial_sigma=(0.5,0.5)):
        self.sigma1, self.sigma2 = initial_sigma
        # keep for API consistency
        self.goals = goal_positions
        self.guess = 0 if priors[0] >= priors[1] else 1

    def update(self, *_):
        """Returns (sigma1, sigma2, dt=0.0)."""
        return self.sigma1, self.sigma2, 0.0

    def adjust(self, delta):
        """Adjust sigma1 by delta; sigma2 = 1 - sigma1."""
        self.sigma1 = float(np.clip(self.sigma1 + delta, 0.0, 1.0))
        self.sigma2 = 1.0 - self.sigma1
