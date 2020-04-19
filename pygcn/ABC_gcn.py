import numpy as np 
from scipy import optimize
from deap.benchmarks import schwefel

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass

@add_metaclass(ABCMeta)
class ObjectiveFunction(object):
    def __init__(self, name, dim, minf, maxf):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf
    def sample(self):
        return np.random.uniform(low = self.minf, high = self.maxf, size = self.dim)