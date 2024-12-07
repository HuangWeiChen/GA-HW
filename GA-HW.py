import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import time
import csv

# Get cities info
def getCity(filepath):
    cities = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            cities.append([float(row[0]), float(row[1])])
    return cities

def create_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
    return distance_matrix

def genetic_algorithm_parallel(distance_matrix, population_size, cities, num_generations, crossover_rate, mutation_rate):
    population = generate_initial_population(population_size, cities)
    best_score = -np.inf
    best_individual = None

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(individual, distance_matrix) for individual in population]
        best_idx = np.argmax(fitness_scores)

        if fitness_scores[best_idx] > best_score:
            best_score = fitness_scores[best_idx]
            best_individual = population[best_idx]

        # Stopping condition
        if generation == num_generations - 1:
            break

        selected_parents = selection(population, fitness_scores)

        # Ensure an even number of parents
        if len(selected_parents) % 2 != 0:
            selected_parents.append(selected_parents[0])

        # Parallel generation of offspring
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(
                process_crossover_mutation,
                [(selected_parents[i], selected_parents[i + 1], crossover_rate, mutation_rate)
                 for i in range(0, len(selected_parents), 2)]
            )

        new_population = [child for pair in results for child in pair]
        population = new_population[:population_size]

    return best_individual, 1 / best_score  # Return shortest distance

def process_crossover_mutation(parent1, parent2, crossover_rate, mutation_rate):
    """Process crossover and mutation for parallel execution"""
    child1, child2 = crossover(parent1, parent2, crossover_rate)
    child1 = mutation(child1, mutation_rate)
    child2 = mutation(child2, mutation_rate)
    return child1, child2

def calculate_fitness(individual, distance_matrix):
    total_distance = sum(distance_matrix[individual[i - 1]][individual[i]] for i in range(len(individual)))
    return 1 / total_distance  # Use inverse of total distance to improve fitness scaling

def generate_initial_population(population_size, cities):
    population = []
    num_cities = len(cities)
    for _ in range(population_size // 2):
        population.append(np.random.permutation(num_cities).tolist())  # Random generation
    
    for _ in range(population_size // 2):  # Greedy initialization
        start_city = np.random.randint(num_cities)
        unvisited = set(range(num_cities))
        unvisited.remove(start_city)
        path = [start_city]
        while unvisited:
            last_city = path[-1]
            next_city = min(unvisited, key=lambda x: np.linalg.norm(np.array(cities[last_city]) - np.array(cities[x])))
            path.append(next_city)
            unvisited.remove(next_city)
        population.append(path)
    return population

def validate_path(individual):
    """Validate and fix the path"""
    seen = set()
    new_individual = []
    for city in individual:
        if city not in seen:
            new_individual.append(city)
            seen.add(city)
    missing = [city for city in range(len(individual)) if city not in seen]
    return new_individual + missing

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]
        return validate_path(child1), validate_path(child2)
    return parent1, parent2

def mutation(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def selection(population, fitness_scores, elite_size=2):
    # Retain elites
    elite_indices = np.argsort(fitness_scores)[-elite_size:]
    elites = [population[i] for i in elite_indices]

    # Roulette wheel selection for the rest
    prob = np.array(fitness_scores) / np.sum(fitness_scores)
    selected_indices = np.random.choice(len(population), len(population) - elite_size, p=prob, replace=False)
    selected = [population[i] for i in selected_indices]

    return elites + selected

def plot_route(cities, best_route, best_distance):
    plt.figure(figsize=(10, 8))
    best_route_cities = [cities[i] for i in best_route] + [cities[best_route[0]]]
    x, y = zip(*best_route_cities)

    # Plot the route
    plt.plot(x, y, marker='o', color='blue', label=f"Shortest Distance: {best_distance:.2f}")
    plt.scatter(*zip(*cities), color='red', label="Cities")

    # Label cities
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=8, color='green')

    plt.title("Traveling Salesman Problem - Best Route")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid()
    plt.show()

# Main program
if __name__ == "__main__":
    cities = getCity("large.csv")
    start_time = time.time()
    distance_matrix = create_distance_matrix(cities)
    print("LOADING......")

    population_size = 100
    num_generations = 200
    crossover_rate = 0.7
    mutation_rate = 0.2

    best_route, best_distance = genetic_algorithm_parallel(
        distance_matrix,
        population_size,
        cities,
        num_generations,
        crossover_rate,
        mutation_rate
    )
    end_time = time.time()
    print("Best route:", best_route)
    print("Shortest distance:", best_distance)
    print(f"Total time: {end_time - start_time:.6f} seconds")

    plot_route(cities, best_route, best_distance)