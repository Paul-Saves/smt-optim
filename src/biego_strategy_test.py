from smt_optim.acquisition_strategies.biego import BiEGO

import numpy as np

from smt_optim.core import Problem
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.acquisition_strategies.multiobj import MultiObj
from smt_optim.core import ObjectiveConfig, DriverConfig
from smt_optim.core import Driver

from smt_optim.benchmarks.registry import list_problems

import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

bounds=np.array([[-4, 4],[-4,4]])

n_accuracy=1000

L=list_problems(tags=["zdt"])

surrogate=SmtAutoModel


def get_DoE(state):
    Y=state.dataset.export_as_dict()["obj"]
    D=state.dataset.export_as_dict()["x"]
    return (D,Y)

def Dominates(p,q):
    #Returns True if point p strictly dominates point q, else returns False
    return (p[0]<q[0] and p[1]<q[1])

def ParetoFront(D,Y):
    #Given a DoE (D,Y), returns the list of indices of non-dominated points, sorted by ascending value of f1
    t=len(D)
    front=[]
    for i in range(t):
        if all([not Dominates(q,Y[i]) for q in Y]):
            front.append((Y[i][0],i))
    front.sort()
    return [p[1] for p in front]

def run_benchmark(bproblem):
    name=bproblem.name
    print("Running BiEGO on benchmark problem",name)
    num_dim=bproblem.num_dim
    num_obj=bproblem.num_obj
    num_cstr=bproblem.num_cstr
    bounds=bproblem.bounds
    objective=bproblem.objective

    assert(num_obj==2)
    assert(num_cstr==0)

    f1=objective[0]
    f2=objective[1]

    max_budget=20*num_dim
    n_init=2*num_dim+1

    obj_config1 = ObjectiveConfig(
        [f1],
        type="minimize",
        surrogate=surrogate,
    )

    obj_config2 = ObjectiveConfig(
        [f2],
        type="minimize",
        surrogate=surrogate,
    )

    prob_definition = Problem(
        obj_configs=[obj_config1,obj_config2],
        design_space=bounds,            # problem bounds
        costs=[1,1]
    )


    opt_config = DriverConfig(
        max_iter = max_budget - n_init,
        max_budget = max_budget,
        nt_init = n_init,
        verbose = True,
        scaling = True,
        seed=42,
    )

    strategy_kwargs = {
        "n_start":n_init,
        "n_accuracy":n_accuracy
    }

    driver = Driver(prob_definition, opt_config, strategy=BiEGO, strategy_kwargs=strategy_kwargs)


    state = driver.optimize()
    return state

class PymooProblem(ElementwiseProblem):

    def __init__(self,bprob):
        self.prob=bprob
        super().__init__(n_var=self.prob.num_dim,
                         n_obj=self.prob.num_obj,
                         xl=np.array([coord_bounds[0] for coord_bounds in self.prob.bounds]),
                         xu=np.array([coord_bounds[1] for coord_bounds in self.prob.bounds]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = self.prob.objective[0](x)
        f2 = self.prob.objective[1](x)

        out["F"] = [f1, f2]

def run_benchmark_pymoo(bproblem):
    pymoo_problem=PymooProblem(bproblem)
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=25,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 1000)
    res = minimize(pymoo_problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)

    X = res.X
    F = res.F
    return (X,F)

data=[]

for bproblem in L:
    state=run_benchmark(bproblem)
    X,F=run_benchmark_pymoo(bproblem)
    data.append((bproblem.name,state,X,F))


_,axs=plt.subplots(3,3)

for i in range(3):
    for j in range(3):
        index = i*3+j
        ax=axs[i][j]

        D,Y=get_DoE(data[index][1])
        pareto_points = [(D[i],Y[i]) for i in ParetoFront(D,Y)]
        ax.scatter([p[1][0] for p in pareto_points],[p[1][1] for p in pareto_points])

        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]
        ax.scatter([p[1][0] for p in pareto_points_pymoo],[p[1][1] for p in pareto_points_pymoo])

        ax.title.set_text(data[index][0])

plt.show()

