import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

harDataset_file_path = r"Path\to\dataset-HAR-PUC-Rio.csv"

# read the data, store data in DataFrame titled harDataset_data 
# skip row 122077 having junks in z4 value 
harDataset_data = pd.read_csv(harDataset_file_path, on_bad_lines='skip', delimiter=';', decimal=",",
                            dtype={'user': 'str', 'gender':'str', 'age':'int', 'HowTallInMeters':'float', 
                                   'weight':'int', 'BodyMassIndex':'float', 
                                   'x1':'int', 'y1':'int', 'z1':'int', 'x2':'int', 'y2':'int', 'z2':'int', 
                                   'x3':'int', 'y3':'int', 'z3':'int', 'x4':'int', 'y4':'int', 'z4':'int', 
                                   'class':'str'}, skiprows=[122077])
harDataset_array = np.array(harDataset_data)
harDataset_df = pd.DataFrame(
harDataset_array,
index=list(range(len(harDataset_data))),
columns=['User', 'Gender', 'Age', 'HowTallInMeters', 'Weight', 'BodyMassIndex', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'Class']
)

# debora: 0, katia: 1, wallace: 2, jose_carlos: 3
harDataset_df['User'] = harDataset_df['User'].map({'debora': 0, 'katia': 1, 'wallace': 2, 'jose_carlos': 3})
# Male: 0, Female: 1 mapping
harDataset_df['Gender'] = harDataset_df['Gender'].map({'Man': 0, 'Woman': 1})
# sitting-down: 1, standing-up: 2, standing: 3, walking: 4, sitting: 5
harDataset_df['Class'] = harDataset_df['Class'].map({'sittingdown': 1, 'standingup': 2, 'standing': 3, 'walking': 4, 'sitting': 5})

##################################### NORMALIZATION ###############################################
# 0 to 1 scaling on metrics' fields (x1 to z4)
harDataset_df[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']]= minmax_scale(harDataset_df[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']], feature_range=(0, 1)) 
###################################################################################################

# load data
df_sitting = harDataset_df[harDataset_df['Class'] == 5] # filter for class 4 (sitting)
df_others = harDataset_df[harDataset_df['Class'] != 5] # filter for all other classes


# calculate mean vectors
mean_sitting = df_sitting.loc[:, 'x1':'z4'].mean().values
mean_others = [df_others[df_others['Class'] == i].loc[:, 'x1':'z4'].mean().values for i in range(1,5)]


def fitness(individual):
    v = np.array(individual)
    c = 0  # or any constant value you want to choose
    cos_sim_sitting = np.dot(v, mean_sitting) / (np.linalg.norm(v) * np.linalg.norm(mean_sitting))
    cos_sim_others = np.mean([np.dot(v, mean) / (np.linalg.norm(v) * np.linalg.norm(mean)) for mean in mean_others])
    return (cos_sim_sitting + c * (1 - cos_sim_others)) / (1 + c),


# initialize population
def initialize_population(pop_size, ind_size):
    return [[random.uniform(0, 1) for _ in range(ind_size)] for _ in range(pop_size)]


# tournament selection
def selection(population, fitnesses, tourn_size):
    selected = []
    for _ in range(2):  # Select 2 parents
        individuals = random.choices(population, k=tourn_size)
        fitnesses_ind = [fitnesses[population.index(i)] for i in individuals]
        selected.append(individuals[fitnesses_ind.index(max(fitnesses_ind))])
    return selected


# uniform crossover
def crossover(parent1, parent2):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return [child1, child2]

# mutation
def mutation(individual, mu, sigma):
    for i in range(len(individual)):
        if random.random() < 0.1:  # mutation probability
            individual[i] += random.gauss(mu, sigma)
            individual[i] = max(min(individual[i], 1), 0)  # ensure within range (clipping)
    return individual

def genetic_algorithm(population_size, crossover_prob, mutation_prob, fitness_fn, max_generations, 
                      no_improv_generations=100, improvement_threshold=0.001, elitism_rate=0.1):
    # Initialize the population
    population = initialize_population(population_size, 12)
    best_fitnesses = []
    stagnant_gen_count = 0
    for generation in range(max_generations):
        fitnesses = [fitness_fn(i)[0] for i in population]
        best_fitness = max(fitnesses)
        best_fitnesses.append(best_fitness)
        print(f'Generation {generation}, Best Solution: {best_fitness}')
        
        # Check if improvement threshold or stagnation limit has been reached
        if len(best_fitnesses) > 1:
            if (best_fitnesses[-1] - best_fitnesses[-2]) / best_fitnesses[-2] < improvement_threshold:
                stagnant_gen_count += 1
            else:
                stagnant_gen_count = 0  # Reset stagnant generation count when improvement happens
                
            if stagnant_gen_count >= no_improv_generations:
                print('Improvement stopped')
                return best_fitnesses, population, fitnesses
        
        # Continue with the genetic algorithm
        next_population = []
        # Elitism: Preserve the fittest individuals.
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
        elites = sorted_population[:int(len(population) * elitism_rate)]
        # Run the GA for the rest of the population.
        for _ in range((len(population) - len(elites)) // 2):
            if random.random() < crossover_prob: # Only crossover if the random number is less than crossover probability
                parents = selection(population, fitnesses, 3)
                children = crossover(*parents)
                next_population += [mutation(c, 0, 0.2) if random.random() < mutation_prob else c for c in children] # Only mutate if the random number is less than mutation probability
            else:
                next_population += random.sample(population, 2) # If not crossing over, then just carry forward individuals to next generation
        # Combine elites and next generation.
        population = elites + next_population
    
    return best_fitnesses, population, fitnesses


 
# determine parameter values
population_size = 200
crossover_prob = 0.1
mutation_prob = 0.01
generations = 500


# run the genetic algorithm
best_fitnesses, population, fitnesses = genetic_algorithm(population_size=population_size, crossover_prob=crossover_prob, mutation_prob=mutation_prob, fitness_fn=fitness, max_generations=generations)


# calculate average best fitness
average_best_fitness = np.mean(best_fitnesses)
print('Average Best Solution: ', average_best_fitness)


plt.figure(figsize=(10,5))
plt.plot(best_fitnesses)
plt.title('Best Solution Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Solution')
plt.show()
