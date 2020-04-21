import random
import math
import time
from train import NetworkInstance
from utils import load_data, accuracy
from pygcn.models import GCN
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import choice
import numpy as np
import matplotlib.pyplot as plt

global nfeat
global nclass



def gcn_generate_individual():
    global nfeat
    global nclass
    x = {}
    x = {"seed": 42,
        "nfeat": nfeat,
        "nclass": nclass,
        "epochs": 40,
        "n_hidden": random.randint(5, 100),
        "dropout": random.uniform(0.0, 0.8),
        "lr": random.uniform(-5, 0),
        "weight_decay": random.uniform(-5, 0)
        }
    return x
def gcn_sample_individual():
    global nfeat
    global nclass
    x = {}
    x = {"seed": 42,
        "nfeat": nfeat,
        "nclass": nclass,
        "epochs": 10,
        "n_hidden": 16,
        "dropout": 0.5,
        "lr": np.log10(0.01),
        "weight_decay": -4 + np.log10(5)
        }
    return x
def gcn_generate_population(size):
    population = []
    population.append(gcn_sample_individual())
    for i in range(1, size):
        population.append(gcn_generate_individual())
    return population
def gcn_crossover(A, B):
    '''Perform crossover between individual A and individual B'''
    #For every param, toss a coin to decide whether to inherit that param from father (A), or mother (B)
    child = {}
    fixed_list = ["seed", "nfeat", "nclass", "epochs"]
    param_list = ["n_hidden", "dropout", "weight_decay", "lr"]
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
        return random.uniform(0.1, 0.8)
    elif param == "lr":
        return random.uniform(-4, 0)
    elif param == "weight_decay":
        return random.uniform(-6, 0)
def winsorize(indi):
    indi["n_hidden"] = int(min(100, max(indi["n_hidden"], 5)))
    indi["dropout"] = min(1, max(indi["dropout"], 0))
    return indi
def gcn_mutation(A):
    '''Perform mutation on individual A'''
    param_list = ["n_hidden", "dropout", "weight_decay","lr"]
    for param in param_list:
        p = random.uniform(0, 1) #toss a coin, whether to mutate this param
        if p < 0.05:
            #phi = random.uniform(-1, 1)
            A[param] = (A[param] + a_random(param))/2.0
    return winsorize(A)
def get_fitness(network, n_tries):
    s = 0
    #Adaptive n_tries
    for i in range(n_tries):
        p = train(network)
        if (p < 0.5):
            return p
        s += train(network)
    s /= n_tries
    return s
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
    return acc_val.item()
        # print('Epoch: {:04d}'.format(i+1),
        #     'loss_train: {:.4f}'.format(loss_train.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()),
        #     'loss_val: {:.4f}'.format(loss_val.item()),
        #     'acc_val: {:.4f}'.format(acc_val.item()),
        #     'time: {:.4f}s'.format(time.time() - t))
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
def sorted_by_fitness_population(population):
    return sorted(population, key = lambda x: x["acc_val"])
def choice_by_fitness(sorted_population, prob):
    return choice(sorted_population, p = prob)
def fitness(indi):
    #return indi["acc_val"]/(1 - indi["acc_val"])
    return indi["acc_val"]
def gcn_make_next_generation(sorted_population, size):
    fitness_sum = 0
    p = [-1 for i in range(len(sorted_population))]
    new_population = []
    for i in range(len(sorted_population)):
        fitness_sum +=  i#fitness(sorted_population[i])
    for i in range(len(sorted_population)):
        #p[i] = fitness(sorted_population[i])/fitness_sum
        p[i] = i/fitness_sum

    #When fitnesses range are close, top individuals should be rewarded more

    #print(p)
    for i in range(size):
        father = choice_by_fitness(sorted_population, p)
        mother = choice_by_fitness(sorted_population, p)
        child = gcn_crossover(father, mother)
        child = gcn_mutation(child)
        new_population.append(child)
    return new_population
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
    params["lr"] = -2
    params["weight_decay"] = -4
    params["n_hidden"] = 16
    params["dropout"] = 0.2
    params["seed"] = 42
def print_statistics(population):
    #Mean
    f = []
    for indi in population:
        f.append(fitness(indi))
    print("Maximum fitness: ", max(f))
    print("Average fitness: ", np.mean(f))
    print("Medium fitness: ", np.median(f))
def short_print(indi):
    print("n_hidden: {:.2f}, dropout: {:.3f}, weight_decay: {:.5f}, lr: {:.4f}, acc: {:.3f}".format(
        indi["n_hidden"], indi["dropout"], indi["weight_decay"], indi["lr"], indi["acc_val"]
    ))
def simulate():
    simulations = 1
    generations = 10
    values = np.zeros(generations)
    size = 15
    itr = range(generations)
    best = 0
    for s in range(simulations):
        print("===================Simulation- {0}===================".format(s))
        population = gcn_generate_population(size)
        optimality_tracking = []
        for i in range(generations):
            print("--Generation ", i)
            for indi in population:
                net = NetworkInstance(**indi)
                indi["acc_val"] = get_fitness(net, 2)
            population = sorted_by_fitness_population(population)
            # for indi in population:
            #     short_print(indi)
            print_statistics(population)
            optimality_tracking.append(population[-1]["acc_val"])
            if population[-1]["acc_val"] > best:
                best = population[-1]["acc_val"]
                optimal_sol = population[-1]
            if (i != generations - 1):
                population = gcn_make_next_generation(population, size)
        values += np.array(optimality_tracking)
    optimal_sol["epochs"] = 200
    print('best = ', best)
    print('optimal_sol = ', optimal_sol)
    s = 0
    for i in range(100):
        net = NetworkInstance(**optimal_sol)
        train(net)
        s += test(net)
    s /= 100
    print(s)
    values /= simulations
    plt.plot(itr, values, lw = 0.5)
    plt.show()
    print(values)

if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data("pubmed")

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
    params = {}
    params["nfeat"] = features.shape[1]
    params["nclass"] = labels.max().item() + 1
    nfeat = params["nfeat"]
    nclass = params["nclass"]
    #params["nclass"] = 7
    #set_params(params)
    #pop = gcn_generate_population(10)
    #print(pop[0], pop[8])
    #a = gcn_crossover(pop[0], pop[8])
    #print(a)
    #print(gcn_mutation(a))
    simulate()
    # res = 0
    # for i in range(20):
    #     params["epochs"] = 200
    #     params["n_hidden"] = 16
    #     params["dropout"] = 0.5
    #     params["weight_decay"] = -4 + np.log10(5)
    #     params["lr"] = np.log10(0.01)
    #     params["seed"] = 42
    #     net = NetworkInstance(**params)
    #     train(net)
    #     res += test(net)
    # res /= 20
    # print(res)
    

    # generations = 10
    # size = 30
    # population = gcn_generate_population(20)
    # for i in range(10):
    #     print("=================================Gen #{0}===========================================".format(i))
    #     print(len(population))
    #     for indi in population:
    #         net = NetworkInstance(**indi)
    #         #train(net)
    #         indi["acc_val"] = train(net)
    #         #indi["acc_test"] = test(net)
    #     population = sorted_by_fitness_population(population)
    #     for indi in population:
    #         #print(indi)
    #         short_print(indi)
    #     print_statistics(population)
    #     if (i != 9):
    #         population = gcn_make_next_generation(population)
    # population[-1]["epochs"] = 100
    # net = NetworkInstance(**(population[-1]))
    # train(net)
    # print(test(net))
