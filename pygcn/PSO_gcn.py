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
W = 0.5
c1 = 0.8
c2 = 0.9
n_iterations = 10
target_error = 1e-6
n_particles = 10
class Particle():
    def __init__(self):
        #self.position = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1) ** (bool(random.getrandbits(1))) * random.random() * 50])
        self.position = pso_generate_individual()
        self.pbest_position = self.position
        self.pbest_value = float('-inf')
        self.velocity = pso_initial_velocity()
    # def __str__(self):
    #     print("I am at", self.position, " my pbest is ", self.pbest_position)
    def move(self):
        self.position = winsorize(add(self.position, self.velocity))
    def get_fitness(self):
        net = NetworkInstance(**self.position)
        self.fitness = train(net)
        #self.position["acc_val"] = self.fitness
        return self.fitness
    def __str__(self):
        return "Position: n_hidden: {:.2f}, dropout: {:.3f}, lr: {:.4f}, weight_decay: {:.5f}, acc: {:.3f}".format(self.position["n_hidden"], self.position["dropout"], self.position["lr"], self.position["weight_decay"], self.fitness)

class Space():
    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('-inf')
        self.gbest_position = pso_generate_individual()
    def print_particles(self):
        for particle in self.particles:
            #particle.__str__()
            print(particle)
    def fitness(self, particle):
        return 1 #particle.position[0] ** 2 + particle.position[1] ** 2 + 1
    def set_pbest(self):
        for particle in self.particles:
            particle.get_fitness()
            #print(particle)
            fitness_candidate = particle.fitness
            if (particle.pbest_value < fitness_candidate):
                particle.pbest_value = fitness_candidate
                particle.pbest_position = particle.position
    def set_gbest(self):
        for particle in self.particles:
            best_fitness_candidate = particle.fitness
            if (self.gbest_value < best_fitness_candidate):
                self.gbest_value = best_fitness_candidate
                self.gbest_position = particle.position
    def move_particles(self):
        for particle in self.particles:
            global W
            new_velocity = add(add(scale(W, particle.velocity), scale(c1 * random.random(), subtract(particle.pbest_position, particle.position))), scale(c2 * random.random(),  subtract(self.gbest_position, particle.position)))
            particle.velocity = new_velocity
            particle.move()
    def sort_particles_by_fitness(self):
        self.particles = sorted_by_fitness_population(self.particles)

def set_params_range(params_range):
    '''Set params range for all kinds of hyperparameters in the network'''
    params_range["n_hidden"]["lower"] = 5
    params_range["n_hidden"]["upper"] = 50
    params_range["dropout"]["lower"] = 0.1
    params_range["dropout"]["upper"] = 0.8
    params_range["learning_rate"]["lower"] = -6 #this param is searched on log scale
    params_range["learning_rate"]["upper"] = 1
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
def pso_generate_individual():
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
def pso_initial_velocity():
    x = {}
    x = {"seed": 0,
        "nfeat": 0,
        "nclass": 0,
        "epochs": 0,
        "n_hidden": 0,
        "dropout": 0,
        "lr": 0,
        "weight_decay": 0
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
    indi["n_hidden"] = min(100, max(indi["n_hidden"], 5))
    indi["dropout"] = min(1, max(indi["dropout"], 0))
    return indi
def subtract(a, b):
    c = a.copy()
    for key in a:
        c[key] = a[key] - b[key]
    return c

def a_random(param):
    if param == "n_hidden":
        return random.randint(5, 50)
    elif param == "dropout":
        return random.uniform(0.1, 0.8)
    elif param == "lr":
        return random.uniform(-6, 1)
    elif param == "weight_decay":
        return random.uniform(-6, 0)

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
    return acc_val.item()
        # print('Epoch: {:04d}'.format(i+1),
        #     'loss_train: {:.4f}'.format(loss_train.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()),
        #     'loss_val: {:.4f}'.format(loss_val.item()),
        #     'acc_val: {:.4f}'.format(acc_val.item()),
        #     'time: {:.4f}s'.format(time.time() - t))
    # print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
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
    params["nfeat"] = features.shape[1]
    params["nclass"] = labels.max().item() + 1
    #params["nclass"] = 7
    set_params(params)


    #search_space.print_particles()
    iteration = 0
    simulations = 10
    values = np.zeros(n_iterations)
    itr = range(n_iterations)
    for _ in range(simulations):
        optimality_tracking = []
        iteration = 0
        search_space = Space(1, target_error, n_particles)
        particles_vector = [Particle() for _ in range(search_space.n_particles)]
        search_space.particles = particles_vector   
        while (iteration < n_iterations):
            print("=============================Iteration {0}=========================".format(iteration))
            search_space.set_pbest()
            search_space.set_gbest()
            search_space.sort_particles_by_fitness()
            search_space.print_particles()
            print('Best position: ', search_space.gbest_position)
            print('Best fitness: ', search_space.gbest_value)
            # if (abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
            #     break
            optimality_tracking.append(search_space.gbest_value)
            search_space.move_particles()
            iteration += 1
        print('opt track: ', optimality_tracking)
        values += np.array(optimality_tracking)
        print('values: ', values)
        #print("The best solution is: ", search_space.gbest_position, " in n_iterations: ", iteration)
    values /= simulations
    plt.plot(itr, values, lw = 0.5)
    plt.show()
    print(values)
