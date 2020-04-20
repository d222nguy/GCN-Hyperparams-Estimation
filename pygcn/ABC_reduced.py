import random
import math
import time
from train import NetworkInstance
from utils import load_data, accuracy
from pygcn.models import GCN
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import choice
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
W = 0.5
c1 = 0.8
c2 = 0.9
n_iterations = 20
target_error = 1e-6
n_particles = 30
class Bee(object):
    def __init__(self):
        self.pos = abc_generate_individual()
        self.fitness = get_fitness(self.pos)
        self.trial = 0
        self.prob = 0
    def update_bee(self, pos, fitness):
        if fitness >= self.fitness:
            self.pos = pos
            self.fitness = fitness
            self.trial = 0
        else:
            self.trial += 1
    def reset_bee(self, max_trials):
        if self.trial >= max_trials:
            self.__reset_bee()
    def __reset_bee(self):
        self.pos = abc_generate_individual()
        self.fitness = get_fitness(self.pos)
        self.trial = 10
        self.prob = 0
    def __str__(self):
        return "Position: n_hidden: {:.2f}, dropout: {:.3f}, lr: {:.4f}, weight_decay: {:.5f}, acc: {:.3f}".format(self.pos["n_hidden"], self.pos["dropout"], self.pos["lr"], self.pos["weight_decay"], self.fitness)
def get_fitness(params):
    net = NetworkInstance(**params)
    fitness = train(net)
    return fitness
class EmployeeBee(Bee):
    def explore(self, max_trials):
        pass
    def compute_prob(self, sum_fitness):
        self.prob = self.fitness / sum_fitness
class OnlookerBee(Bee):
    def onlook(self, best_food_sources, max_trials):
        p = []
        s = sum(map(lambda x: x.fitness, best_food_sources))
        for bee in best_food_sources:
            p.append(bee.fitness/s)
        candidate = np.random.choice(best_food_sources, p = p)
        self.__exploit(candidate.pos, candidate.fitness, max_trials)
    def __exploit(self, candidate, fitness, max_trials):
        pass
class ABC(object):
    def __init__(self,colony_size = 12, n_iter = 10, max_trials = 6):
        self.colony_size = colony_size
        self.n_iter = n_iter
        self.max_trials = max_trials
        self.optimal_solution = None
        self.optimality_tracking = []

    def __reset_algorithm(self):
        self.optimal_solution = None
        self.optimality_tracking = []
    def __update_optimality_tracking(self):
        self.optimality_tracking.append(self.optimal_solution.fitness)
    def __update_optimal_solution(self):
        n_optimal_solution = max(self.onlooker_bees + self.employee_bees, key = lambda bee: bee.fitness)
        #print('n_optimal_solution fitness: ', n_optimal_solution.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        else:
            if n_optimal_solution.fitness > self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)
    def __initialize_employees(self):
        self.employee_bees = []
        for itr in range(self.colony_size // 2):
            self.employee_bees.append(EmployeeBee())
    def __initialize_onlookers(self):
        self.onlooker_bees = []
        for itr in range(self.colony_size // 2):
            self.onlooker_bees.append(OnlookerBee())
    def __employee_bees_phase(self):
        #For each employee bee: Explore around its source!
        for bee in self.employee_bees:
            if bee.trial <= self.max_trials:
                n_pos = bee.pos.copy()
                a_random_source = random.choice(self.employee_bees).pos
                param = random.choice(["lr", "weight_decay", "n_hidden", "dropout"])
                phi = np.random.uniform(low=-1, high=1)
                n_pos[param] = n_pos[param] + phi * (n_pos[param] - a_random_source[param])
                n_pos = winsorize(n_pos)
                #print('npos: ', n_pos)
                fitness = get_fitness(n_pos)
                bee.update_bee(n_pos, fitness)
    def __calculate_probabilities(self):
        sum_fitness = sum(map(lambda bee: bee.fitness, self.employee_bees))
        for bee in self.employee_bees:
            bee.compute_prob(sum_fitness)
            #self.probs.append(bee.prob)
    def __select_best_food_sources(self):
        self.best_food_sources = list(filter (lambda bee: bee.prob > np.random.uniform(low = 0, high = 1), self.employee_bees))
        while not self.best_food_sources:
            self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low = 0, high = 1), self.employee_bees))
        # print("len = ", len(self.best_food_sources))
        # print("=========List of best food sources==============")
        # for bee in self.best_food_sources:
        #     print(bee)
        # print("=========End of list of best sources=============")
    def __onlooker_bees_phase(self):
        #for each onlooker bee: choose a source (based on probabilities) to exploit

        #Calculate probabilities
        p = []
        s = sum(map(lambda x: x.fitness, self.best_food_sources))
        for bee in self.best_food_sources:
            p.append(bee.fitness/s)

        for bee in self.onlooker_bees:
            if bee.trial <= self.max_trials:
                #Pick candidate
                candidate = np.random.choice(self.best_food_sources, p = p)
                n_pos = candidate.pos.copy()
                a_random_source = random.choice(self.employee_bees).pos
                param = random.choice(["lr", "weight_decay", "n_hidden", "dropout"])
                phi = np.random.uniform(low=-1, high=1)
                n_pos[param] = n_pos[param] + phi * (n_pos[param] - a_random_source[param])
                n_pos = winsorize(n_pos)
                fitness = get_fitness(n_pos)
                bee.update_bee(n_pos, fitness)
        #TODO: Correct this (exploit function still doesn't have function body)
    def __scout_bee_phase(self):
        for bee in self.employee_bees + self.onlooker_bees:
            bee.reset_bee(self.max_trials)
    def optimize(self):
        self.__reset_algorithm()
        self.__initialize_employees()
        self.__initialize_onlookers()
        # for bee in self.onlooker_bees:
        #     print(bee.pos, bee.fitness)
        for itr in range(self.n_iter):
            #print("**************************Iteration {0}***************************".format(itr))
            # print("========Employee bees=======")
            # for bee in self.employee_bees:
            #     print(bee)
            # print("========Onlooker bees=======")
            # for bee in self.onlooker_bees:
            #     print(bee)
            self.__employee_bees_phase()
            self.__update_optimal_solution()
            self.__calculate_probabilities()
            # print("========Employee bees, after employee phase=======")
            # for bee in self.employee_bees:
            #     print(bee)
            # print('============Probabilities============')
            # for bee in self.employee_bees:
            #     print(bee.prob)
            self.__select_best_food_sources()

            self.__onlooker_bees_phase()
            # print("========Onlooker bees, after onlooker phase=======")
            # for bee in self.onlooker_bees:
            #     print(bee)
            self.__scout_bee_phase()
    
            self.__update_optimal_solution()
            self.__update_optimality_tracking()
            print('Optimal solution: ', self.optimal_solution.pos, self.optimal_solution.fitness)
            print("iter: {} = cost: {}"
                .format(itr, "%04.03e" % self.optimal_solution.fitness))

def abc_generate_individual():
    x = {}
    x = {"seed": 42,
        "nfeat": 1433,
        "nclass": 7,
        "epochs": 10,
        "n_hidden": random.randint(5, 80),
        "dropout": random.uniform(0.1, 0.8),
        "lr": random.uniform(-6, 1),
        "weight_decay": random.uniform(-6, 1)
        }
    return x

def scale(a, indi):
    '''Scale all individual elements by scalar a'''
    indi["n_hidden"] = int(a * indi["n_hidden"]) #Todo: make sure value after scale stay in MaxMin range
    indi["dropout"] = a * indi["dropout"]
    indi["lr"] *= a
    indi["weight_decay"] *= a
    return indi
def add(a, b):
    c = a.copy()
    for key in a:
        c[key] = a[key] + b[key]
    return c
def winsorize(indi):
    indi["n_hidden"] = int(min(100, max(indi["n_hidden"], 5)))
    indi["dropout"] = min(1, max(indi["dropout"], 0))
    return indi
def subtract(a, b):
    c = a.copy()
    for key in a:
        c[key] = a[key] - b[key]
    return c

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
    #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    return acc_val.item()
        # print('Epoch: {:04d}'.format(i+1),
        #     'loss_train: {:.4f}'.format(loss_train.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()),
        #     'loss_val: {:.4f}'.format(loss_val.item()),
        #     'acc_val: {:.4f}'.format(acc_val.item()),
        #     'time: {:.4f}s'.format(time.time() - t))
    # print("Optimization Finished!")
def sorted_by_fitness_population(population):
    return sorted(population, key = lambda x: x.fitness)
def choice_by_fitness(sorted_population, prob):
    return choice(sorted_population, p = prob)
def fitness(indi):
    #return indi["acc_val"]/(1 - indi["acc_val"])
    return indi["acc_val"]
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
if __name__ == '__main__':
   
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
    params = {}
    params["nfeat"] = 1433 #features.shape[1]
    params["nclass"] = 7 #labels.max().item() + 1
    #params["nclass"] = 7
    set_params(params)

    #simulate:
    simulations = 8
    n_iter = 10
    values = np.zeros(n_iter)
    itr = range(n_iter)
    for _ in range(simulations):
        abc = ABC(n_iter = n_iter, colony_size = 8)
        abc.optimize()
        values += np.array(abc.optimality_tracking)
    values /= simulations
    plt.plot(itr, values, lw = 0.5)
    plt.show()
    print(values)


