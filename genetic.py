import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Define the neural network architecture
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),  # Input layer (4 features)
        keras.layers.Dense(8, activation='relu'),  # Hidden layer
        keras.layers.Dense(3, activation='softmax')  # Output layer (3 classes)
    ])
    return model

# Flatten and unflatten functions for genetic representation
def flatten_weights(model):
    return np.concatenate([w.flatten() for w in model.get_weights()])

def unflatten_weights(model, flat_weights):
    shapes = [w.shape for w in model.get_weights()]
    new_weights = []
    index = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(flat_weights[index:index+size].reshape(shape))
        index += size
    model.set_weights(new_weights)

# Fitness function (Evaluate model accuracy)
def evaluate_fitness(model, weights, x_train, y_train):
    unflatten_weights(model, weights)
    loss, acc = model.evaluate(x_train, y_train, verbose=0)
    return acc  # Accuracy as fitness score

# Selection (Tournament Selection)
def select_parents(population, fitness, num_parents=2):
    parents = []
    for _ in range(num_parents):
        tournament = np.random.choice(len(population), size=3, replace=False)
        best = tournament[np.argmax(fitness[tournament])]
        parents.append(population[best])
    return parents

# Crossover (Uniform Crossover)
def crossover(parent1, parent2):
    mask = np.random.rand(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child

# Mutation (Small Random Changes)
def mutate(weights, mutation_rate=0.1):
    mutation_mask = np.random.rand(len(weights)) < mutation_rate
    weights[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.1
    return weights

# Genetic Algorithm
def genetic_algorithm(x_train, y_train, pop_size=10, generations=20, mutation_rate=0.1):
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    population = [flatten_weights(model) for _ in range(pop_size)]
    fitness = np.zeros(pop_size)
    
    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")

        # Evaluate fitness
        for i in range(pop_size):
            fitness[i] = evaluate_fitness(model, population[i], x_train, y_train)
        
        print(f"Best Fitness: {max(fitness):.4f}")

        # Select parents & create offspring
        new_population = []
        for _ in range(pop_size // 2):  # Each iteration creates 2 children
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2), crossover(parent2, parent1)
            new_population += [mutate(child1, mutation_rate), mutate(child2, mutation_rate)]

        population = new_population

    # Return the best individual
    best_weights = population[np.argmax(fitness)]
    unflatten_weights(model, best_weights)
    return model

# Load dataset (Iris dataset as an example)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_iris()
x = StandardScaler().fit_transform(data.data)  # Normalize data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Run Genetic Algorithm
best_model = genetic_algorithm(x_train, y_train)

# Evaluate Best Model
test_acc = best_model.evaluate(x_test, y_test, verbose=0)[1]
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
