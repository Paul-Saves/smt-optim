import numpy as np
from smt_optim.benchmarks.base import BenchmarkProblem

class ZDT1(BenchmarkProblem):

    def __init__(self):
        super().__init__()

        self.num_dim = 2 #To change
        self.num_cstr = 0
        self.num_obj = 2
        self.num_fidelity = 1
        self.bounds = np.array([[1e-10,1]]*self.num_dim)
        self.costs = [1,1]

        self.objective = [self.f1,self.f2]
        self.constraints = []

        self.tags = [
            "zdt"
        ]

    def f1(self, x):
        res = x[0]
        return res
    
    def f2(self, x):
        res = 1-np.sqrt(x[0]/(1+9*np.sum(x[1:])/(self.num_dim-1)))
        return res
