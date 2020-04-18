import random
import math
import time
from train import NetworkInstance
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
import torch.nn.functional as F
import torch.optim as optim
def generate_population(size, x_boundaries, y_boundaries):
    lower_x, upper_x = x_boundaries
    lower_y, upper_y = y_boundaries
    population = []
    for i in range(size):
        individual = {
            "x": random.uniform(lower_x, upper_x),
            "y": random.uniform(lower_y, upper_y)
        } 
        population.append(individual)
    return population

def apply_function(individual):
    x = individual["x"]
    y = individual["y"]
    return math.sin(math.sqrt(x ** 2 + y ** 2))
def sorted_population_by_fitness(population):
    return sorted(population, key = apply_function)
def crossover(individual_a, individual_b):
    xa = individual_a["x"]
    ya = individual_a["y"]

    xb = individual_b["x"]
    yb = individual_b["y"]

    return {"x": (xa + xb)/2, "y": (ya + yb)/2}
def mutate(individual):
    next_x = individual["x"] + random.uniform(-0.05, 0.05)
    next_y = individual["y"] + random.uniform(-0.05, 0.05)
    lower_boundary, upper_boundary = (-4, 4)
    next_x = min(max(next_x, lower_boundary), upper_boundary)
    next_y = min(max(next_y, lower_boundary), upper_boundary)

    return {"x": next_x, "y": next_y}
def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum
    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)
    draw = random.uniform(0, 1)
    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual
def make_next_generation(previous_population):
    next_generation = []
    sorted_by_fitness_population = sorted_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_function(individual) for individual in previous_population)

    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)

        individual = crossover(first_choice, second_choice)
        individual = mutate(individual)
        next_generation.append(individual)
    return next_generation
def set_params_range(params_range):
    '''Set params range for all kinds of hyperparameters in the network'''
    params_range["n_hidden"]["lower"] = 5
    params_range["n_hidden"]["upper"] = 50
    params_range["dropout"]["lower"] = 0.2
    params_range["dropout"]["upper"] = 0.8
    params_range["learning_rate"]["lower"] = -6 #this param is searched on log scale
    params_range["learning_rate"]["upper"] = 1
    params_range["L2_regularization"]["lower"] = -6
    params_range["L2_regularization"]["upper"] = 1
    lower_x, upper_x = x_boundaries
    lower_y, upper_y = y_boundaries
    population = []
    # for i in range(size):
    #     individual = {
    #         "x": random.uniform(lower_x, upper_x),
    #         "y": random.uniform(lower_y, upper_y)
    #     } 
    #     population.append(individual)
    return population
def gcn_generate_individual():
    x = {}
    x = {"seed": 42,
        "nfeat": 1433,
        "nclass": 7,
        "epochs": 10,
        "n_hidden": random.randint(5, 50),
        "dropout": random.uniform(0.2, 0.8),
        "lr": 10 ** random.uniform(-6, 1),
        "weight_decay": 10 ** random.uniform(-6, 0),
        "L2_regularization": 10 ** random.uniform(-6, 1)}
    return x
def gcn_generate_population(size):
    population = []
    for i in range(size):
        population.append(gcn_generate_individual())
    return population
def gcn_crossover(A, B):
    '''Perform crossover between individual A and individual B'''
    #For every param, toss a coin to decide whether to inherit that param from father (A), or mother (B)
    child = {}
    fixed_list = ["seed", "nfeat", "nclass", "epochs"]
    param_list = ["n_hidden", "dropout", "weight_decay", "L2_regularization", "lr"]
    for param in fixed_list:
        child[param] = A[param]
    for param in param_list:
        p = random.uniform(0, 1)
        child[param] = A[param] if p > 0.5 else B[param]
    return child
def a_random(param):
    if param == "n_hidden":
        return random.randint(5, 50)
    elif param == "dropout":
        return random.uniform(0.2, 0.8)
    elif param == "lr":
        return 10 ** random.uniform(-6, 1)
    elif param == "weight_decay":
        return 10 ** random.uniform(-6, 0)
    elif param == "L2_regularization":
        return 10 ** random.uniform(-6, 1)
def gcn_mutation(A):
    '''Perform mutation on individual A'''
    param_list = ["n_hidden", "dropout", "weight_decay", "L2_regularization", "lr"]
    for param in param_list:
        p = random.uniform(0, 1) #toss a coin, whether to mutate this param
        if p < 0.1:
            A[param] = a_random(param)
    return A
def choice_by_fitness(sorted_population):
    offset = 0
    normalized_fitness_sum = fitness_sum
    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)
    draw = random.uniform(0, 1)
    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual
def main():
    generations = 100
    population = generate_population(size = 10, x_boundaries = (-4, 4), y_boundaries = (-4, 4))
    i = 1
    while True:
        print(f" Generation {i}")
        for individual in population:
            print(individual)
        if i == generations:
            break
        i += 1
        population = make_next_generation(population)
    best_individual = sorted_population_by_fitness(population)[-1]
    print("\n FINAL RESULT")
    print(best_individual, apply_function(best_individual))

def train(network):
    t_total = time.time()
    model = network.model
    optimizer = network.optimizer
    #train
    epochs = network.params["epochs"]
    for i in range(epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(i+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
def sorted_by_fitness_population(population):
    return sorted(population, key = lambda x: x["acc_test"])
def test(network):
    model = network.model
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()
def set_params(params):
    params["epochs"] = 200
    params["lr"] = 0.01
    params["weight_decay"] = 5e-4
    params["n_hidden"] = 16
    params["dropout"] = 0.2
    params["seed"] = 42
if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
    params = {}
    params["nfeat"] = 1433
    #params["nclass"] = labels.max().item() + 1
    params["nclass"] = 7
    set_params(params)
    pop = gcn_generate_population(10)
    print(pop[0], pop[8])
    a = gcn_crossover(pop[0], pop[8])
    print(a)
    print(gcn_mutation(a))
    generations = 10
    size = 20
    for i in range(1):
        population = gcn_generate_population(20)
        for indi in population:
            net = NetworkInstance(**indi)
            train(net)
            indi["acc_test"] = test(net)
        population = sorted_by_fitness_population(population)
        for indi in population:
            print(indi)
    #print(gcn_crossover(pop[0], pop[8]))
    
    # net = NetworkInstance(**a)
    # train(net)
    # test(net)

    #train.do()
    #main()
