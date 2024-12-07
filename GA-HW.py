import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import time
import csv

# get cities info
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

        # 停止條件
        if generation == num_generations - 1:
            break

        selected_parents = selection(population, fitness_scores)

        # 確保選出的父代數量是偶數，避免索引越界
        if len(selected_parents) % 2 != 0:
            selected_parents.append(selected_parents[0])  # 複製第一個父代湊成偶數

        # 並行生成子代
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.starmap(
                process_crossover_mutation,
                [(selected_parents[i], selected_parents[i + 1], crossover_rate, mutation_rate)
                 for i in range(0, len(selected_parents), 2)]
            )

        new_population = [child for pair in results for child in pair]

        # 維持種群數量
        population = new_population[:population_size]

    return best_individual, 1 / best_score  # 返回最短距離

def process_crossover_mutation(parent1, parent2, crossover_rate, mutation_rate):
    """單獨處理交叉與突變，用於並行處理"""
    child1, child2 = crossover(parent1, parent2, crossover_rate)
    child1 = mutation(child1, mutation_rate)
    child2 = mutation(child2, mutation_rate)
    return child1, child2

def calculate_fitness(individual, distance_matrix):
    total_distance = sum(distance_matrix[individual[i-1]][individual[i]] for i in range(len(individual)))
    return 1 / total_distance

def generate_initial_population(population_size, num_cities):
    return [np.random.permutation(num_cities).tolist() for _ in range(population_size)]

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]
        return child1, child2
    return parent1, parent2

def mutation(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def selection(population, fitness_scores, elite_size=2):
    # 保留精英
    elite_indices = np.argsort(fitness_scores)[-elite_size:]
    elites = [population[i] for i in elite_indices]
    
    # 剩余选择
    prob = np.array(fitness_scores) / np.sum(fitness_scores)
    selected_indices = np.random.choice(len(population), len(population) - elite_size, p=prob, replace=False)
    selected = [population[i] for i in selected_indices]
    
    return elites + selected


# 繪製路徑圖
def plot_route(cities, best_route, best_distance):
    plt.figure(figsize=(10, 8))
    best_route_cities = [cities[i] for i in best_route] + [cities[best_route[0]]]
    x, y = zip(*best_route_cities)

    # 繪製路徑
    plt.plot(x, y, marker='o', color='blue', label=f"Shortest Distance: {best_distance:.2f}")
    plt.scatter(*zip(*cities), color='red', label="Cities")

    # 標示城市編號
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i), fontsize=8, color='green')

    plt.title("Traveling Salesman Problem - Best Route")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend()
    plt.grid()
    plt.show()

# 測試程式碼
if __name__ == "__main__":
    # 假設距離矩陣
    num_cities = 50
    cities = getCity("large.csv")
    
    start_time = time.time()
    distance_matrix = create_distance_matrix(cities)
    print("LOADING......")
    #distance_matrix = np.random.rand(num_cities, num_cities)
    #np.fill_diagonal(distance_matrix, 0)  # 自己到自己的距離為0

    population_size = 100
    num_generations = 200
    crossover_rate = 0.7
    mutation_rate = 0.4

    best_route, best_distance = genetic_algorithm_parallel(
        distance_matrix,
        population_size,
        len(cities),
        num_generations,
        crossover_rate,
        mutation_rate
    )
    end_time = time.time()
    print("Best route:", best_route)
    print("Shortest distance:", best_distance)
    print(f"Total time: {end_time - start_time:.6f} seconds")

    # 繪製最佳路徑
    plot_route(cities, best_route, best_distance)