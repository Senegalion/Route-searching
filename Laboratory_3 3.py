import random
import math
from collections import deque


# Scenario 1: All direct connections
def create_cities_all_direct(n):
    cities = set()
    for _ in range(n):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        cities.add((x, y))
    return cities


# Scenario 2: 80% of possible roads available
def create_cities_partial_direct(n):
    cities = set()
    for _ in range(n):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        cities.add((x, y))
    return cities


def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def create_graph(cities):
    graph = {}
    for city1 in cities:
        graph[city1] = []
        for city2 in cities:
            if city1 != city2:
                distance = calculate_distance(city1, city2)
                graph[city1].append((city2, distance))
    return graph


def bfs(graph, start):
    visited = set()
    queue = deque()
    queue.append(start)
    paths = {start: [start]}
    costs = {start: 0}  # Initialize costs with start node's cost as 0

    while queue:
        node = queue.popleft()
        current_path = paths[node]
        current_cost = costs[node]

        visited.add(node)

        if len(visited) == len(graph):  # Check if all cities have been visited
            return current_path, current_cost

        for neighbor, distance in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                paths[neighbor] = current_path + [neighbor]
                costs[neighbor] = current_cost + distance  # Calculate cumulative cost based on the parent's cost

    return [], float('inf')  # No path found


def dfs(graph, start):
    visited = set()
    path = []
    stack = deque([(start, [start], 0)])  # Include cost in the stack

    while stack:
        node, current_path, current_cost = stack.pop()

        if node not in visited:
            visited.add(node)
            path.append(node)

            if len(visited) == len(graph):  # Check if all cities have been visited
                return current_path, current_cost

            for neighbor, distance in reversed(graph[node]):  # Process neighbors in reverse order
                if neighbor not in visited:
                    stack.append((neighbor, current_path + [neighbor], current_cost + distance))  # Calculate cumulative cost

    return [], float('inf')  # No path found


def mst(graph):
    start = list(graph.keys())[0]
    visited = set([start])
    mst_edges = []
    total_cost = 0

    while len(visited) < len(graph):
        min_edge = None
        for node in visited:
            for neighbor, weight in graph[node]:
                if neighbor not in visited and (min_edge is None or weight < min_edge[1]):
                    min_edge = (node, weight, neighbor)

        if min_edge:
            mst_edges.append((min_edge[0], min_edge[2], min_edge[1]))
            total_cost += min_edge[1]
            visited.add(min_edge[2])

    return mst_edges, total_cost


def greedy_search(graph, start):
    visited = set([start])
    path = [start]
    total_distance = 0

    while len(visited) < len(graph):
        current_city = path[-1]
        min_distance = float('inf')  # Initialize min_distance to infinity
        next_city = None

        for neighbor, distance in graph[current_city]:
            if neighbor not in visited and distance < min_distance:
                min_distance = distance
                next_city = neighbor

        if next_city:
            visited.add(next_city)
            path.append(next_city)
            total_distance += min_distance

    return path, total_distance


def bidirectional_search(graph, start, end):
    forward_visited = set([start])
    backward_visited = set([end])
    forward_queue = deque([(start, [start], 0)])  # Include cost in the queue
    backward_queue = deque([(end, [end], 0)])  # Include cost in the queue

    while forward_queue and backward_queue:
        forward_node, forward_path, forward_cost = forward_queue.popleft()
        backward_node, backward_path, backward_cost = backward_queue.popleft()

        # Check if the two searches meet
        intersection = forward_visited.intersection(backward_visited)
        if intersection:
            intersection_node = intersection.pop()
            total_distance = forward_cost + backward_cost
            return forward_path + backward_path[::-1][1:], total_distance  # Combine the paths

        for neighbor, distance in graph[forward_node]:
            if neighbor not in forward_visited:
                forward_queue.append((neighbor, forward_path + [neighbor], forward_cost + distance))
                forward_visited.add(neighbor)

        for neighbor, distance in graph[backward_node]:
            if neighbor not in backward_visited:
                backward_queue.append((neighbor, backward_path + [neighbor], backward_cost + distance))
                backward_visited.add(neighbor)

    return None, None  # No path found


# Step 1: Create a set of cities
print("Step 1:")
cities_all_direct = create_cities_all_direct(3)
cities_partial_direct = create_cities_partial_direct(3)
print("Cities (All Direct):", cities_all_direct)
print("Cities (Partial Direct):", cities_partial_direct)

# Step 2: Create a weighted graph
print("\nStep 2:")
graph_all_direct = create_graph(cities_all_direct)
graph_partial_direct = create_graph(cities_partial_direct)
print("Graph (All Direct):", graph_all_direct)
print("Graph (Partial Direct):", graph_partial_direct)

# Step 3a: Full search using BFS and DFS
print("\nStep 3a:")
start_city = list(graph_all_direct.keys())[0]
bfs_path, bfs_cost = bfs(graph_all_direct, start_city)
dfs_path, dfs_cost = dfs(graph_all_direct, start_city)
print("BFS Path (All Direct):", bfs_path)
print("BFS Cost (All Direct):", bfs_cost)
print("DFS Path (All Direct):", dfs_path)
print("DFS Cost (All Direct):", dfs_cost)
print()

start_city = list(graph_partial_direct.keys())[0]
bfs_path, bfs_cost = bfs(graph_partial_direct, start_city)
dfs_path, dfs_cost = dfs(graph_partial_direct, start_city)
print("BFS Path (Partial Direct):", bfs_path)
print("BFS Cost (Partial Direct):", bfs_cost)
print("DFS Path (Partial Direct):", dfs_path)
print("DFS Cost (Partial Direct):", dfs_cost)

# Step 3b: Approximate solution using Minimum Spanning Tree
print("\nStep 3b:")
mst_edges, mst_cost = mst(graph_all_direct)
print("MST Edges (All Direct):", mst_edges)
print("MST Cost (All Direct):", mst_cost)
print()

mst_edges, mst_cost = mst(graph_partial_direct)
print("MST Edges (Partial Direct):", mst_edges)
print("MST Cost (Partial Direct):", mst_cost)

# Step 3c: Approximate solution using Greedy Search
print("\nStep 3c:")
start_city = list(graph_all_direct.keys())[0]
greedy_path, greedy_cost = greedy_search(graph_all_direct, start_city)
print("Greedy Path (All Direct):", greedy_path)
print("Greedy Cost (All Direct):", greedy_cost)
print()

start_city = list(graph_partial_direct.keys())[0]
greedy_path, greedy_cost = greedy_search(graph_partial_direct, start_city)
print("Greedy Path (Partial Direct):", greedy_path)
print("Greedy Cost (Partial Direct):", greedy_cost)

# Step 4: Bidirectional search
print("\nStep 3d:")
start_city = list(graph_all_direct.keys())[0]
end_city = list(graph_all_direct.keys())[-1]
bidirectional_path, bidirectional_cost = bidirectional_search(graph_all_direct, start_city, end_city)
print("Bidirectional Path (All Direct):", bidirectional_path)
print("Bidirectional Cost (All Direct):", bidirectional_cost)
print()

start_city = list(graph_partial_direct.keys())[0]
end_city = list(graph_partial_direct.keys())[-1]
bidirectional_path, bidirectional_cost = bidirectional_search(graph_partial_direct, start_city, end_city)
print("Bidirectional Path (Partial Direct):", bidirectional_path)
print("Bidirectional Cost (Partial Direct):", bidirectional_cost)
