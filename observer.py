"""
observer.py

Defines two observer classes for the deception game using pixel-based coordinates:

- CPUObserver: automatic allocation based on progress towards a guessed goal,
  with obstacle-aware geodesic distance via a visibility graph and Dijkstra.
  Skips computation when robot hasn’t moved between frames (movement threshold).
  Stops timer and further updates once the robot reaches its guessed goal.

- ManualObserver: manual allocation adjusted via keyboard input.

Usage:
    from observer import CPUObserver, ManualObserver
    obs = CPUObserver(goal_positions, priors, obstacle_polygons)
    sigma1, sigma2, dt = obs.update(current_pos, timestamp)

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
    CPU-based observer allocating points based on progress toward a guessed goal.
    Skips computation when movement between frames is negligible.
    Stops updates once the robot reaches its guessed goal.

    Parameters:
    - goal_positions: list of two np.array([x_pix, y_pix]) for the goal tags
    - priors: tuple (p1, p2) initial belief over goals
    - obstacle_polygons: list of lists of (x_pix, y_pix) tuples for rectangular obstacles
    """
    def __init__(self, goal_positions, priors, obstacle_polygons=None):
        self.goals = [np.array(g, dtype=float) for g in goal_positions]
        self.guess = 0 if priors[0] >= priors[1] else 1
        self.obstacles = obstacle_polygons or []
        # History
        self.prev_pos = None
        self.prev_dist = None
        self.last_time = None
        # Movement threshold (pixels)
        self.movement_epsilon = 1e-3
        # Goal reach threshold
        self.goal_epsilon = 2.0  # in pixels
        # Finished flag
        self._finished = False
        # Last-known allocations
        self.last_sigma1 = 0.5
        self.last_sigma2 = 0.5

    def _visible(self, p, q):
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
        s = tuple(start)
        g = tuple(goal)
        if self._visible(s, g):
            return np.linalg.norm(start - goal)
        nodes = [s, g]
        for poly in self.obstacles:
            for v in poly:
                nodes.append(tuple(v))
        N = len(nodes)
        adj = {i: [] for i in range(N)}
        for i in range(N):
            for j in range(i + 1, N):
                if self._visible(nodes[i], nodes[j]):
                    d = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
                    adj[i].append((j, d))
                    adj[j].append((i, d))
        dist = [float('inf')] * N
        dist[0] = 0.0
        pq = [(0.0, 0)]
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
        # DEBUG: incoming values
        print(f"[DEBUG] CPUObserver.update: pos={current_pos}, finished={self._finished}")
        # Stop if goal reached
        if self._finished:
            print("[DEBUG] Goal already reached; no further updates.")
            return self.last_sigma1, self.last_sigma2, 0.0

        pos = np.array(current_pos, dtype=float)
        # Skip if no movement
        if self.prev_pos is not None:
            movement = np.linalg.norm(pos - self.prev_pos)
            print(f"[DEBUG] movement={movement:.3f}px")
            if movement < self.movement_epsilon:
                return self.last_sigma1, self.last_sigma2, 0.0
        # Initialize on first valid frame
        if self.prev_pos is None:
            d0 = self._geodesic_dist(pos, self.goals[self.guess])
            print(f"[DEBUG] init: guess={self.guess}, dist={d0:.2f}px")
            self.prev_pos = pos.copy()
            self.prev_dist = d0
            self.last_time = timestamp
            self.last_sigma1, self.last_sigma2 = 0.5, 0.5
            return 0.5, 0.5, 0.0
        # Compute dt
        dt = timestamp - self.last_time
        self.last_time = timestamp
        # Geodesic distances
        d_now = self._geodesic_dist(pos, self.goals[self.guess])
        prog = max(0.0, self.prev_dist - d_now)
        l = np.linalg.norm(pos - self.prev_pos)
        print(f"[DEBUG] prev_dist={self.prev_dist:.2f}, d_now={d_now:.2f}, prog={prog:.2f}, l={l:.2f}, dt={dt:.3f}")
        # Check goal reach
        if d_now <= self.goal_epsilon:
            print(f"[DEBUG] Goal reached (d_now={d_now:.2f}px) - stopping observer.")
            self._finished = True
            return self.last_sigma1, self.last_sigma2, 0.0
        # Allocation ratio
        if l > self.movement_epsilon:
            sigma1 = prog / l
        else:
            sigma1 = 1.0 if prog > 0 else 0.0
        sigma1 = float(np.clip(sigma1, 0.0, 1.0))
        sigma2 = 1.0 - sigma1
        print(f"[DEBUG] sigma1={sigma1:.2f}, sigma2={sigma2:.2f}")
        # Update history
        self.prev_dist = d_now
        self.prev_pos = pos.copy()
        self.last_sigma1, self.last_sigma2 = sigma1, sigma2
        return sigma1, sigma2, dt


class ManualObserver:
    """
    Manual observer: operator adjusts allocation via adjust(delta);
    update() returns current allocation and zero dt.
    """
    def __init__(self, goal_positions, priors, initial_sigma=(0.5, 0.5)):
        self.sigma1, self.sigma2 = initial_sigma
        self.goals = goal_positions
        self.guess = 0 if priors[0] >= priors[1] else 1

    def update(self, *_):
        print(f"[DEBUG] ManualObserver.update: sigma1={self.sigma1:.2f}, sigma2={self.sigma2:.2f}")
        return self.sigma1, self.sigma2, 0.0

    def adjust(self, delta):
        self.sigma1 = float(np.clip(self.sigma1 + delta, 0.0, 1.0))
        self.sigma2 = 1.0 - self.sigma1
        print(f"[DEBUG] ManualObserver.adjust: new sigma1={self.sigma1:.2f}")
