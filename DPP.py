import numpy as np
import networkx as nx
from typing import List, Tuple, Dict

class DeceptivePathPlanner:
    def __init__(self, edges: List[Tuple[int, int, float]], nodes: List[Tuple[int, float, float]]):
        """
        Initialize the graph with nodes and edges.
        
        Args:
            edges: List of (u, v, weight) tuples.
            nodes: List of (node_id, x_coord, y_coord) tuples.
        """
        self.G = nx.Graph()
        for u, v, w in edges:
            self.G.add_edge(u, v, weight=w)
        self.node_positions = {node[0]: (node[1], node[2]) for node in nodes}

    def find_shortest_path(self, start: int, goal: int) -> List[int]:
        """Compute the shortest path using Dijkstra's algorithm."""
        return nx.shortest_path(self.G, start, goal, weight='weight')

    def compute_progress(self, path: List[int], goal: int) -> List[float]:
        """Calculate the progress function c_t (minimum distance to g1 along the path)."""
        distances = [nx.shortest_path_length(self.G, node, goal, weight='weight') for node in path]
        return [min(distances[:i+1]) for i in range(len(distances))]

    def defender_strategy(self, ct_prev: float, edge: Tuple[int, int], current_node: int, goal: int) -> float:
        """
        Defender's strategy σ*: Probability of allocating to g1.
        
        Args:
            ct_prev: Previous progress (c_{t-1}).
            edge: Current edge (u, v) being traversed.
            current_node: Attacker's current position.
            goal: Defender's goal (g1).
        
        Returns:
            Probability of allocating to g1.
        """
        u, v = edge
        ct = nx.shortest_path_length(self.G, v, goal, weight='weight')
        progress = ct_prev - ct
        return max(0, min(1, progress / self.G.edges[edge]['weight']))

    def attacker_strategy(self, start: int, g1: int, g2: int, prior: Tuple[float, float]) -> Dict[str, List[int]]:
        """
        Compute PBNE paths for both Attacker types.
        
        Args:
            start: Initial position (s0).
            g1: Primary goal.
            g2: Secondary goal.
            prior: (b1, b2) prior probabilities for g1 and g2.
        
        Returns:
            Dictionary with paths for primary/secondary Attackers.
        """
        # Secondary Attacker's path (ξ2)
        xi_2 = self.find_shortest_path(start, g2)
        
        # Primary Attacker's paths (ξ1_I and ξ1_II)
        ct = self.compute_progress(xi_2, g1)
        
        # Phase I: Defender allocates fully to g1
        t_I = next((i for i, c in enumerate(ct) if c < ct[0]), len(xi_2))
        
        # Phase II: Defender mixes allocations
        t_II = next((i for i, c in enumerate(ct[t_I:], t_I) if c == ct[t_I]), len(xi_2))
        
        # Construct ξ1_I and ξ1_II
        if t_I < len(xi_2):
            xi_1_I = xi_2[:t_I] + self.find_shortest_path(xi_2[t_I], g1)
        else:
            xi_1_I = xi_2.copy()
        
        if t_II < len(xi_2):
            xi_1_II = xi_2[:t_II] + self.find_shortest_path(xi_2[t_II], g1)
        else:
            xi_1_II = xi_2.copy()
        
        return {
            'xi_2': xi_2,          # Secondary Attacker's path
            'xi_1_I': xi_1_I,      # Primary Attacker's deceptive path
            'xi_1_II': xi_1_II,    # Primary Attacker's direct path
            't_I': t_I,
            't_II': t_II
        }

    def payoff(self, path: List[int], goal: int, defender_goal: int) -> float:
        """
        Compute the total cost (payoff) for a given path.
        
        Args:
            path: Path taken by the Attacker.
            goal: Attacker's true goal.
            defender_goal: Defender's allocation goal.
        
        Returns:
            Total cost (sum of edge weights where Defender allocates to true goal).
        """
        total_cost = 0.0
        ct_prev = nx.shortest_path_length(self.G, path[0], goal, weight='weight')
        
        for i in range(1, len(path)):
            u, v = path[i-1], path[i]
            edge_weight = self.G.edges[(u, v)]['weight']
            alloc_prob = self.defender_strategy(ct_prev, (u, v), v, defender_goal)
            
            if defender_goal == goal:
                total_cost += edge_weight * alloc_prob
            
            ct_prev = min(ct_prev, nx.shortest_path_length(self.G, v, goal, weight='weight'))
        
        return total_cost

# Example Usage
if __name__ == "__main__":
    # Example graph data (replace with your actual data)
    nodes = [
        (372, 741544.469, 3739192.157),  # s0
        (477, 741432.308, 3739261.639),  # g1
        (318, 741295.869, 3739186.843)   # g2
    ]
    
    edges = [
        (372, 477, 70.275),  # s0 -> g1
        (372, 318, 76.107),  # s0 -> g2
        (477, 318, 80.0)     # g1 -> g2 (example)
    ]
    
    planner = DeceptivePathPlanner(edges, nodes)
    
    # Prior beliefs (b1, b2)
    prior = (0.6, 0.4)
    
    # Compute PBNE paths
    equilibrium_paths = planner.attacker_strategy(372, 477, 318, prior)
    
    print("Secondary Attacker's path (ξ2):", equilibrium_paths['xi_2'])
    print("Primary Attacker's deceptive path (ξ1_I):", equilibrium_paths['xi_1_I'])
    print("Primary Attacker's direct path (ξ1_II):", equilibrium_paths['xi_1_II'])
    
    # Compute payoffs
    v1 = planner.payoff(equilibrium_paths['xi_1_I'], 477, 477)  # Primary Attacker cost
    v2 = planner.payoff(equilibrium_paths['xi_2'], 318, 477)    # Secondary Attacker cost
    
    print(f"Primary Attacker's cost (v1): {v1:.2f}")
    print(f"Secondary Attacker's cost (v2): {v2:.2f}")