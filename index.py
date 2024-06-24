import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations
from matplotlib.animation import FuncAnimation

# Data input
depot = (106.84513, -6.208763)  # Koordinat depot
nodes = [
    (106.842709, -6.207489), 
    (106.84513, -6.210963), 
    (106.84863, -6.206763),
    (106.850134, -6.209234), 
    (106.844256, -6.205345), 
    (106.846789, -6.208965),
    (106.849312, -6.207894), 
    (106.841234, -6.206321), 
    (106.843345, -6.209876),
    (106.845987, -6.210123), 
]
truck_capacity = 10  # kapasitas truk dalam m^3
num_trucks = 2  # jumlah truk yang tersedia
carbon_emission_per_km = 2.64  # kg/km
fuel_consumption_per_km = 0.5  # liter/km
fuel_cost_per_liter = 10000  # rp

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

# Matriks jarak antara semua titik
epsilon = 1e-6  # Small value to prevent division by zero
distance_matrix = np.zeros((len(nodes) + 1, len(nodes) + 1))
all_nodes = [depot] + nodes
for i, node1 in enumerate(all_nodes):
    for j, node2 in enumerate(all_nodes):
        distance_matrix[i, j] = euclidean_distance(node1, node2) + epsilon

# Implementasi ACO untuk VRP
class AntColonyOptimizationVRP:
    def __init__(self, distance_matrix, num_trucks, truck_capacity, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_constant=100):
        self.distance_matrix = distance_matrix
        self.num_trucks = num_trucks
        self.truck_capacity = truck_capacity
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.num_nodes = distance_matrix.shape[0]
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))
    
    def run(self):
        self.best_route = None
        self.best_distance = float('inf')
        self.iteration_routes = []
        
        for iteration in range(self.num_iterations):
            all_routes = []
            all_distances = []
            
            for ant in range(self.num_ants):
                route, distance = self.construct_solution()
                all_routes.append(route)
                all_distances.append(distance)
                
                if distance < self.best_distance:
                    self.best_route = route
                    self.best_distance = distance
            
            self.update_pheromone(all_routes, all_distances)
            self.iteration_routes.append(self.best_route.copy())
        
        return self.best_route, self.best_distance
    
    def construct_solution(self):
        route = []
        current_node = 0
        remaining_capacity = self.truck_capacity
        visited = set()
        distance_travelled = 0
        
        while len(visited) < self.num_nodes - 1:
            probabilities = self.calculate_transition_probabilities(current_node, visited)
            next_node = self.choose_next_node(probabilities)
            distance_travelled += self.distance_matrix[current_node, next_node]
            route.append(next_node)
            visited.add(next_node)
            current_node = next_node
            
            # Check capacity constraint
            remaining_capacity -= 1
            if remaining_capacity == 0:
                route.append(0)  # Return to depot
                distance_travelled += self.distance_matrix[current_node, 0]
                current_node = 0
                remaining_capacity = self.truck_capacity
        
        route.append(0)  # Return to depot
        distance_travelled += self.distance_matrix[current_node, 0]
        
        return route, distance_travelled
    
    def calculate_transition_probabilities(self, current_node, visited):
        probabilities = np.zeros(self.num_nodes)
        for node in range(self.num_nodes):
            if node not in visited:
                probabilities[node] = (self.pheromone_matrix[current_node, node] ** self.alpha) * ((1 / self.distance_matrix[current_node, node]) ** self.beta)
        probabilities /= probabilities.sum()
        return probabilities
    
    def choose_next_node(self, probabilities):
        return np.random.choice(range(self.num_nodes), p=probabilities)
    
    def update_pheromone(self, all_routes, all_distances):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for route, distance in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                self.pheromone_matrix[route[i], route[i + 1]] += self.pheromone_constant / distance

# Jalankan algoritma ACO
aco_vrp = AntColonyOptimizationVRP(distance_matrix, num_trucks, truck_capacity)
best_route, best_distance = aco_vrp.run()

# Animasi rute terbaik
fig, ax = plt.subplots(figsize=(10, 10))
G = nx.DiGraph()
pos = {i: all_nodes[i] for i in range(len(all_nodes))}
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')

def update(num):
    ax.clear()
    route = aco_vrp.iteration_routes[num]
    edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    G.clear()
    G.add_edges_from(edges)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
    ax.set_title(f"Iteration {num + 1}")

ani = FuncAnimation(fig, update, frames=len(aco_vrp.iteration_routes), interval=500, repeat=False)
plt.show()

# Hitung metrik lainnya
total_trash_volume = truck_capacity * num_trucks
total_carbon_emission = best_distance * carbon_emission_per_km
total_fuel_consumption = best_distance * fuel_consumption_per_km
total_fuel_cost = total_fuel_consumption * fuel_cost_per_liter

print(f"Jarak Optimal: {best_distance:.2f} km")
print(f"Jumlah Kubik Sampah: {total_trash_volume} m^3")
print(f"Jumlah Emisi Karbon: {total_carbon_emission:.2f} kg")
print(f"Jumlah Truk Optimal: {num_trucks}")
print(f"Jumlah Bensin: {total_fuel_consumption:.2f} liter")
print(f"Biaya Bahan Bakar: Rp {total_fuel_cost:.2f}")
