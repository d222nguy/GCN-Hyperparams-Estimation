import numpy as np 
from scipy import optimize
from deap.benchmarks import schwefel

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass
import matplotlib.pyplot as plt
from copy import deepcopy
import config
@add_metaclass(ABCMeta)
class ObjectiveFunction(object):
    def __init__(self, name, dim, minf, maxf):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf
    def sample(self):
        return np.random.uniform(low = self.minf, high = self.maxf, size = self.dim)

    def custom_sample(self):
        return np.repeat(self.minf, repeats = self.dim) + np.random.uniform(low = 0, high = 1, size = self.dim) * np.repeat(self.maxf - self.minf, repeats = self.dim)
    @abstractmethod
    def evaluate(self, x):
        pass

class Sphere(ObjectiveFunction):
    def __init__(self, dim):
        super(Sphere, self).__init__('Sphere', dim, -100.0, 100.0)
    def evaluate(self, x):
        return sum(np.power(x, 2))

@add_metaclass(ABCMeta)
class ArtificialBee(object):
    TRIAL_INITIAL_DEFAULT_VALUE = 0
    INITIAL_DEFAULT_PROBABILITY = 0.0
    def __init__(self, obj_function):
        self.pos = obj_function.custom_sample()
        self.obj_function = obj_function
        self.minf = obj_function.minf
        self.maxf = obj_function.maxf
        self.fitness = obj_function.evaluate(self.pos)
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY
    def evaluate_boundaries(self, pos):
        if (pos < self.minf).any() or (pos > self.maxf).any():
            pos[pos > self.maxf] = self.maxf
            pos[pos < self.minf] = self.minf
        return pos
    def update_bee(self, pos, fitness):
        if fitness <= self.fitness:
            print('Improved!')
            self.pos = pos
            self.fitness = fitness
            self.trial = 0
        else:
            self.trial += 1
    def reset_bee(self, max_trials):
        if self.trial >= max_trials:
            self.__reset_bee()
    def __reset_bee(self):
        self.pos = self.obj_function.custom_sample()
        self.fitness = self.obj_function.evaluate(self.pos)
        self.trial = ArtificialBee.TRIAL_INITIAL_DEFAULT_VALUE
        self.prob = ArtificialBee.INITIAL_DEFAULT_PROBABILITY

class EmployeeBee(ArtificialBee):
    def explore(self, max_trials):
        #print('==========================================')
        if self.trial <= max_trials:
            component = np.random.choice(self.pos)
            print('component = ', component)
            print('self.pos = ', self.pos)
            phi = np.random.uniform(low=-1, high=1, size = len(self.pos))
            n_pos = self.pos + (self.pos - component) * phi
            n_pos = self.evaluate_boundaries(n_pos)
            n_fitness = self.obj_function.evaluate(n_pos)
            self.update_bee(n_pos, n_fitness)
    def get_fitness(self):
        return 1/(1 + self.fitness) if self.fitness >= 0 else 1 + np.abs(self.fitness)
    def compute_prob(self, max_fitness):
        self.prob = self.get_fitness() / max_fitness

class OnLookerBee(ArtificialBee):
    def onlook(self, best_food_sources, max_trials):
        # for source in best_food_sources:
        #     print(source.pos)
        candidate = np.random.choice(best_food_sources)
        self.__exploit(candidate.pos, candidate.fitness, max_trials)
    def __exploit(self, candidate, fitness, max_trials):
        if self.trial <= max_trials:
            component = np.random.choice(candidate)
            phi = np.random.uniform(low=-1, high=1, size = len(candidate))
            n_pos = candidate + (candidate - component) * phi
            n_pos = self.evaluate_boundaries(n_pos)
            n_fitness = self.obj_function.evaluate(n_pos)

            if n_fitness <= fitness:
                self.pos = n_pos
                self.fitness = n_fitness
                self.trial = 0
            else:
                self.trial += 1
class ABC(object):
    def __init__(self, obj_function, colony_size = 30, n_iter = 50, max_trials = 10):
        self.colony_size = colony_size
        self.obj_function = obj_function
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
        n_optimal_solution = min(self.onlooker_bees + self.employee_bees, key = lambda bee: bee.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        else:
            if n_optimal_solution.fitness < self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)
    def __initialize_employees(self):
        self.employee_bees = []
        for itr in range(self.colony_size // 2):
            self.employee_bees.append(EmployeeBee(self.obj_function))
    def __initialize_onlookers(self):
        self.onlooker_bees = []
        for itr in range(self.colony_size // 2):
            self.onlooker_bees.append(OnLookerBee(self.obj_function))
    def __employee_bees_phase(self):
        #print('================================')
        #print(len(self.employee_bees))
        for bee in self.employee_bees:
            bee.explore(self.max_trials)
        # map(lambda bee: bee.explore(self.max_trials), self.employee_bees)
    def __calculate_probabilities(self):
        sum_fitness = sum(map(lambda bee: bee.get_fitness(), self.employee_bees))
        for bee in self.employee_bees:
            bee.compute_prob(sum_fitness)
        #map(lambda bee: bee.compute_prob(sum_fitness), self.employee_bees)
    def __select_best_food_sources(self):
        self.best_food_sources = list(filter (lambda bee: bee.prob > np.random.uniform(low = 0, high = 1), self.employee_bees))
        while not self.best_food_sources:
            self.best_food_sources = list(filter(lambda bee: bee.prob > np.random.uniform(low = 0, high = 1), self.employee_bees))
    def __onlooker_bees_phase(self):
        for bee in self.onlooker_bees:
            bee.onlook(self.best_food_sources, self.max_trials)
        # map(lambda bee: bee.onlook(self.best_food_sources, self.max_trials), self.onlooker_bees)
    def __scout_bee_phase(self):
        for bee in self.employee_bees + self.onlooker_bees:
            bee.reset_bee(self.max_trials)
        # map(lambda bee: bee.reset_bee(self.max_trials), self.onlooker_bees + self.employee_bees)
    def optimize(self):
        self.__reset_algorithm()
        self.__initialize_employees()
        self.__initialize_onlookers()
        for itr in range(self.n_iter):
            self.__employee_bees_phase()
            self.__update_optimal_solution()
            self.__calculate_probabilities()
            self.__select_best_food_sources()

            self.__onlooker_bees_phase()
            self.__scout_bee_phase()
    
            self.__update_optimal_solution()
            self.__update_optimality_tracking()
            print('Optimal solution: ', self.optimal_solution.pos)
            print("iter: {} = cost: {}"
                .format(itr, "%04.03e" % self.optimal_solution.fitness))
def get_objective(objective, dimension=30):
    objectives = {'Sphere': Sphere(dimension)}
                #   'Rastrigin': Rastrigin(dimension),
                #   'Rosenbrock': Rosenbrock(dimension),
                #   'Schwefel': Schwefel(dimension)}
    return objectives[objective]


def simulate(obj_function, colony_size=30, n_iter=50,
             max_trials=10, simulations=1):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    box_optimal = []
    for _ in range(simulations):
        optimizer = ABC(obj_function=get_objective(obj_function),
                        colony_size=colony_size, n_iter=n_iter,
                        max_trials=max_trials)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
        box_optimal.append(optimizer.optimal_solution.fitness)
        print(optimizer.optimal_solution.pos)
    values /= simulations

    plt.plot(itr, values, lw=0.5, label=obj_function)
    plt.legend(loc='upper right')


def main():
    plt.figure(figsize=(10, 7))
    print("Hello!")
    simulate('Sphere')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    main()