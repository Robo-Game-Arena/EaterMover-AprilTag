"""
observer.py

Defines two observer classes for the deception game using pixel-based coordinates:

- CPUObserver: automatic allocation based on progress towards a guessed goal,
  with obstacle-aware geodesic distance via a visibility graph and Dijkstra.

- ManualObserver: manual allocation adjusted via keyboard input.

Usage:
    from observer import CPUObserver, ManualObserver

    # goal_positions: list of two np.array([x_pix, y_pix])
    # priors: tuple (p1, p2)
    # obstacle_polygons: list of list of (x_pix, y_pix) tuples

    obs = CPUObserver(goal_positions, priors, obstacle_polygons)
    sigma1, sigma2, dt = obs.update(current_pos, timestamp)

    # Or for manual:
    obs = ManualObserver(goal_positions, priors)
    obs.adjust(delta)
    sigma1, sigma2, dt = obs.update(None, timestamp)
"""

import numpy as np
import heapq


def _orientation(a, b, c):
    """Return >0 if points a, b, c are clockwise, <0 if counterclockwise, 0 if colinear."""
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _on_segment(a, b, c):
    """Return True if point b lies on segment a-c (colinear case)."""
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))


def _segments_intersect(p1, p2, q1, q2):
    """Return True if segment p1-p2 intersects segment q1-q2."""
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    # General case
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    # Colinear cases
    if o1 == 0 and _on_segment(p1, q1, p2): return True
    if o2 == 0 and _on_segment(p1, q2, p2): return True
    if o3 == 0 and _on_segment(q1, p1, q2): return True
    if o4 == 0 and _on_segment(q1, p2, q2): return True
    return False


class CPUObserver:
    """
    CPU-based observer that allocates points based on progress towards a guessed goal.
    Uses pixel-based coordinates and computes obstacle-aware geodesic distances
    via a visibility graph over start, goal, and obstacle vertices, then Dijkstra.

    Parameters:
    - goal_positions: list of two np.array([x_pix, y_pix]) for the goal tags
    - priors: tuple (p1, p2) initial belief over goals
    - obstacle_polygons: list of list of (x_pix, y_pix) tuples for rectangular obstacles
    """
    def __init__(self, goal_positions, priors, obstacle_polygons=None):
        self.goals = [np.array(g, dtype=float) for g in goal_positions]
        self.guess = 0 if priors[0] >= priors[1] else 1
        self.obstacles = obstacle_polygons or []
        # History for computing progress
        self.prev_pos = None
        self.prev_dist = None
        self.last_time = None

    def _visible(self, p, q):
        """Return True if the segment p->q does not intersect any obstacle."""
        p = tuple(p)
        q = tuple(q)
        for poly in self.obstacles:
            n = len(poly)
            for i in range(n):
                a = tuple(poly[i])
                b = tuple(poly[(i + 1) % n])
                if _segments_intersect(p, q, a, b):
                    return False
        return True

    def _geodesic_dist(self, start, goal):
        """Compute shortest collision-free distance from start to goal in pixel space."""
        s = tuple(start)
        g = tuple(goal)
        # Direct line clear?
        if self._visible(s, g):
            return np.linalg.norm(start - goal)

        # Build nodes: start, goal, obstacle vertices
        nodes = [s, g]
        for poly in self.obstacles:
            for v in poly:
                nodes.append(tuple(v))
        N = len(nodes)

        # Build adjacency list
        adj = {i: [] for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                if self._visible(nodes[i], nodes[j]):
                    d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
                    adj[i].append((j, d))
                    adj[j].append((i, d))

        # Dijkstra from 0 to 1
        dist = [float('inf')] * N
        dist[0] = 0.0
        pq = [(0.0, 0)]  # (distance, node_index)
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
        Update observer with the robot's current pixel position.

        Returns:
        - sigma1, sigma2: allocations for goal1 and goal2 (sum to 1)
        - dt: elapsed time since last update
        """
        pos = np.array(current_pos, dtype=float)
        # DEBUG: print incoming position
        print(f"[DEBUG] update(): pos={pos}, prev_pos={self.prev_pos}")
        if self.prev_pos is None:
            # First call: initialize history
            d0 = self._geodesic_dist(pos, self.goals[self.guess])
            print(f"[DEBUG] init: guess={self.guess}, d0={d0:.2f}")
            self.prev_pos = pos.copy()
            self.prev_dist = d0
            self.last_time = timestamp
            return 0.5, 0.5, 0.0

        # Compute dt
        dt = timestamp - self.last_time
        self.last_time = timestamp

        # Geodesic distances
        d_now = self._geodesic_dist(pos, self.goals[self.guess])
        prog = max(0.0, self.prev_dist - d_now)
        l = np.linalg.norm(pos - self.prev_pos)
        # DEBUG: print distance diagnostics
        print(f"[DEBUG] prev_dist={self.prev_dist:.2f}, d_now={d_now:.2f}, prog={prog:.2f}, l={l:.2f}")


        # Allocation ratio
        if l > 1e-6:
            sigma1 = prog / l
        else:
            sigma1 = 1.0 if prog > 0 else 0.0
        sigma1 = float(np.clip(sigma1, 0.0, 1.0))
        sigma2 = 1.0 - sigma1
        # DEBUG: print resulting allocation
        print(f"[DEBUG] sigma1={sigma1:.2f}, sigma2={sigma2:.2f}, dt={dt:.3f}")

        # Update history
        self.prev_dist = d_now
        self.prev_pos = pos.copy()

        return sigma1, sigma2, dt


class ManualObserver:
    """
    Manual observer: operator adjusts allocation via adjust(delta);
    update() returns current allocation and zero dt.

    Parameters:
    - goal_positions: list of two positions (unused internally)
    - priors: tuple (p1, p2) (used only to pick guess index)
    - initial_sigma: starting allocation tuple
    """
    def __init__(self, goal_positions, priors, initial_sigma=(0.5, 0.5)):
        self.sigma1, self.sigma2 = initial_sigma
        self.goals = goal_positions
        self.guess = 0 if priors[0] >= priors[1] else 1

    def update(self, *_):
        """Returns (sigma1, sigma2, dt=0.0)."""
        return self.sigma1, self.sigma2, 0.0

    def adjust(self, delta):
        """Adjust sigma1 by delta (clamped to [0,1]), sigma2 = 1 - sigma1."""
        self.sigma1 = float(np.clip(self.sigma1 + delta, 0.0, 1.0))
        self.sigma2 = 1.0 - self.sigma1
