import sys

# Define the graph
graph = {
    'A': {'B': 2, 'G': 6},
    'B': {'A': 2, 'C': 7, 'E': 2},
    'C': {'B': 7, 'D': 3, 'F': 3},
    'D': {'C': 3, 'H': 2},
    'E': {'B': 2, 'F': 2, 'G': 1},
    'F': {'C': 3, 'E': 2, 'H': 2},
    'G': {'A': 6, 'E': 1, 'H': 4},
    'H': {'D': 2, 'G': 4}
}

# Dijkstra's algorithm function
def dijkstra(graph, start, end):
    distances = {node: sys.maxsize for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0
    unvisited_nodes = list(graph.keys())

    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda node: distances[node])
        unvisited_nodes.remove(current_node)

        for neighbor, weight in graph[current_node].items():
            if distances[current_node] + weight < distances[neighbor]:
                distances[neighbor] = distances[current_node] + weight
                previous_nodes[neighbor] = current_node

    # Reconstruct the shortest path from start to end
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    return distances[end], path

# User input for start and end nodes
start_node = input("Enter the start node: ")
end_node = input("Enter the end node: ")

# Check if the input nodes are valid
if start_node in graph and end_node in graph:
    shortest_distance, shortest_path = dijkstra(graph, start_node, end_node)
    print(f"Shortest distance from {start_node} to {end_node}: {shortest_distance}")
    print("Shortest path:", " -> ".join(shortest_path))
else:
    print("Invalid start or end node.")