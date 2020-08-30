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
import torch
import config as cf
global nfeat
global nclass


time_out = cf.time_out
simulations = cf.simulations
generations = cf.generations
size = cf.size
alpha = cf.alpha
early_stop = cf.early_stop
min_epoch = cf.min_epoch
def gcn_generate_individual():
    global nfeat
    global nclass
    x = {}
    x = {"seed": cf.seed,
        "nfeat": nfeat,
        "nclass": nclass,
        "epochs": random.randint(cf.epochs_low, cf.epochs_high),
        "n_hidden": random.randint(cf.n_hidden_low, cf.n_hidden_high),
        "dropout": random.uniform(cf.dropout_low, cf.dropout_high),
        "lr": random.uniform(cf.lr_low, cf.lr_high),
        "weight_decay": random.uniform(cf.weight_decay_low, cf.weight_decay_high)
        }
    return x
def gcn_sample_individual():
    global nfeat
    global nclass
    x = {}
    x = {"seed": cf.seed,
        "nfeat": nfeat,
        "nclass": nclass,
        "epochs": cf.sample_epochs,
        "n_hidden": cf.sample_nhidden,
        "dropout":  cf.sample_dropout,
        "lr": cf.sample_lr,
        "weight_decay": cf.sample_weight_decay
        }
    return x
def gcn_generate_population(size):
    population = []
    #population.append(gcn_sample_individual())
    for i in range(0, size):
        population.append(gcn_generate_individual())
    return population
def gcn_crossover(A, B):
    '''Perform crossover between individual A and individual B'''
    #For every param, toss a coin to decide whether to inherit that param from father (A), or mother (B)
    child = {}
    fixed_list = ["seed", "nfeat", "nclass"]
    param_list = ["n_hidden", "dropout", "weight_decay", "lr", "epochs"]
    for param in fixed_list:
        child[param] = A[param]
    for param in param_list:
        p = random.uniform(0, 1)
        child[param] = A[param] if p > 0.5 else B[param]
    return child
def a_random(param):
    if param == "n_hidden":
        return random.randint(cf.n_hidden_low, cf.n_hidden_high)
    elif param == "dropout":
        return random.uniform(cf.dropout_low, cf.dropout_high)
    elif param == "lr":
        return random.uniform(cf.lr_low, cf.lr_high)
    elif param == "weight_decay":
        return random.uniform(cf.weight_decay_low, cf.weight_decay_high)
    elif param == "epochs":
        return random.uniform(cf.epochs_low, cf.epochs_high)
def cut_tail(indi):
    indi["n_hidden"] = int(min(cf.n_hidden_high, max(indi["n_hidden"], cf.n_hidden_low)))
    indi["dropout"] = min(cf.dropout_high, max(indi["dropout"], cf.dropout_low))
    indi["epochs"] = int(min(cf.epochs_high, max(indi["epochs"], cf.epochs_low)))
    return indi
def gcn_mutation(A):
    '''Perform mutation on individual A'''
    param_list = ["n_hidden", "dropout", "weight_decay","lr", "epochs"]
    for param in param_list:
        p = random.uniform(0, 1) #toss a coin, whether to mutate this param
        if p < cf.mutation_rate:
            phi = random.uniform(0, 1)
            A[param] = A[param]*phi + (a_random(param))*(1 - phi)
    return cut_tail(A)
def get_fitness(network):
    s, _ = train(network)
    return s
def get_fitness_time(network):
    s, a = train(network)
    t = max(a/time_out - 1, 0)
    return (s - alpha*t), s, a
def train(network):
    t_total = time.time()
    model = network.model
    optimizer = network.optimizer
    losses_val = []
    #pull out params for easier use
    epochs = network.params["epochs"]  
    torch.cuda.manual_seed(network.params["seed"])
    torch.manual_seed(network.params["seed"])
    np.random.seed(network.params["seed"])
    
    iter_since_best = 0
    best_loss_val = 10**9
    #min_epoch_ = min(min_epoch, epochs)
    for i in range(epochs):
        #train
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        #evaluate on validation set
        model.eval()
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        #count number of iterations since best loss
        if (loss_val.item() < best_loss_val):
            iter_since_best = 0
            best_loss_val = loss_val.item()
        else:
            iter_since_best += 1
        losses_val.append(loss_val.item())
        acc_val = accuracy(output[idx_val], labels[idx_val])
        #early stop
        if i > min_epoch and iter_since_best > early_stop:
            print("Early stopping at...", i, " over ", epochs)
            break
    return acc_val.item(), time.time() - t_total
def sorted_by_fitness_population(population):
    return sorted(population, key = lambda x: x["fitness"])
def choice_by_fitness(sorted_population, prob):
    return choice(sorted_population, p = prob)
def fitness(indi):
    return indi["fitness"]
def gcn_make_next_generation(sorted_population, size):
    fitness_sum = 0
    p = [-1 for i in range(len(sorted_population))]
    new_population = []
    min_fitness = fitness(sorted_population[0]) - cf.epsilon
    for i in range(len(sorted_population)):
        fitness_sum += (fitness(sorted_population[i]) - min_fitness)  #fitness(sorted_population[i])
    for i in range(len(sorted_population)):
        #p[i] = fitness(sorted_population[i])/fitness_sum
        p[i] = (fitness(sorted_population[i]) - min_fitness)/(fitness_sum)

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
def print_statistics(population):
    #Mean
    f = []
    for indi in population:
        f.append(fitness(indi))
    print("Maximum fitness: ", max(f))
    print("Average fitness: ", np.mean(f))
    print("Medium fitness: ", np.median(f))
def short_print(indi):
    # print("epochs: {:.2f}, n_hidden: {:.2f}, dropout: {:.3f}, weight_decay: {:.5f}, lr: {:.4f}, accuracy: {:.3f}, time: {:.3f}, fitness: {:.3f}".format(
    #     indi["epochs"], indi["n_hidden"], indi["dropout"], indi["weight_decay"], indi["lr"], indi["acc_val"], indi["time"], indi["fitness"]
    # ))
     print("epochs: {:.2f}, n_hidden: {:.2f}, dropout: {:.3f}, weight_decay: {:.5f}, lr: {:.4f}, fitness: {:.3f}".format(
        indi["epochs"], indi["n_hidden"], indi["dropout"], indi["weight_decay"], indi["lr"], indi["fitness"]
    ))
def print_population(population):
    for indi in population:
        short_print(indi)
def update_best(population, best, optimal_sol):
    if population[-1]["fitness"] > best:
        best = population[-1]["fitness"]
        print("best = ", best)
        optimal_sol = population[-1]
    return best, optimal_sol
def run(indi):
    net = NetworkInstance(**indi)
    if cf.countTime:
        indi["fitness"], indi["acc_val"], indi["time"] = get_fitness_time(net)
    else:
        indi["fitness"] = get_fitness(net)
def run_ga(values, best, optimal_sol, countTime):
    population = gcn_generate_population(size)
    times = []
    optimality_tracking = []
    t = time.time()
    for i in range(generations):
        print("--Generation ", i)
        for indi in population:
            run(indi)      
        population = sorted_by_fitness_population(population)
        print_population(population)
        print_statistics(population)
        best, optimal_sol = update_best(population, best, optimal_sol)
        optimality_tracking.append(best)
        if (i != generations - 1):
            population = gcn_make_next_generation(population, size)
        times.append(time.time() - t)
    print(optimality_tracking)
    print(times)
    values += np.array(optimality_tracking)
    print('best = ', best)
    return values, best, optimal_sol, times
def simulate():
    values = np.zeros(generations)
    itr = range(generations)
    optimal_sol = gcn_generate_individual()
    best = 0
    times_total = np.zeros(generations)
    for s in range(simulations):
        print("===================Simulation- {0}===================".format(s))
        values, best, optimal_sol, times = run_ga(values, best, optimal_sol, countTime = False)
        times_total += times
    times_total /= simulations
    
    print('best = ', best)
    print('optimal_sol = ', (optimal_sol))
    acc = 0
    time_taken = 0
    for i in range(10):
        net = NetworkInstance(**optimal_sol)
        _, t = train(net)
        time_taken += t
        acc += test(net)
    print('Test accuracy: ', acc/10)
    print('Training time: ', time_taken/10)
    values /= simulations
    plt.plot(itr, values, lw = 0.5)
    plt.show()
    print(values)

if __name__ == '__main__':
    np.random.seed(cf.seed)
    torch.cuda.manual_seed(cf.seed)
    torch.manual_seed(cf.seed) #seed for training GCN, keep it the same as in original paper
    random.seed(1) # this is seed for GA
    adj, features, labels, idx_train, idx_val, idx_test = load_data("cora")
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    nfeat = features.shape[1]
    nclass = labels.max().item() + 1
    simulate()