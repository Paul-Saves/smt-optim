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
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV

from naiveBiEGO import NaiveBiEGO
from smt.sampling_methods import LHS
import smt.design_space as ds
import random

L=list_problems(tags=["zdt"])

surrogate=SmtAutoModel

a=1 # Set to 1 for testing, set to 3 for running Benchmark

# Parameters

n_accuracy=100 # Precision on composite acquisition function
seed=420 # Seed for generating the initial DoE
budget_factor=20 # Budget: budget_factor * dim
init_factor=2 # Initial DoE size: init_factor * dim + 1
min_factor=2 # Initial calls to determine min(f1): min_factor * dim + 1 (same for min(f2))
max_so_iter_factor=2 # Max calls to a single-objective subproblem: max_so_iter_factor * dim + 1
soformulation_naive="Normalized"
soformulation_composite="Product"
multi_start_factor=10 # Number of multistart calls for acquisition function optimization: multi_start_factor * dim
test_number=0


def get_DoE(state):
    Y=state.dataset.export_as_dict()["obj"]
    D=state.dataset.export_as_dict()["x"]
    return (D,Y)

def Dominates(p,q):
    #Returns True if point p strictly dominates point q, else returns False
    return (p[0]<q[0] and p[1]<=q[1]) or (p[0]<=q[0] and p[1]<q[1])

def ParetoFront(D,Y):
    #Given a DoE (D,Y), returns the list of indices of non-dominated points, sorted by ascending value of f1
    t=len(D)
    front=[]
    for i in range(t):
        if all([not Dominates(q,Y[i]) for q in Y]):
            front.append((Y[i][0],i))
    front.sort()
    return [p[1] for p in front]

def run_all_benchmarks(bproblem):
    name=bproblem.name
    num_dim=bproblem.num_dim
    bounds=bproblem.bounds
    objective=bproblem.objective

    f1=objective[0]
    f2=objective[1]

    max_budget=budget_factor*num_dim
    n_init=init_factor*num_dim+1

    float_vars = []
    for idx in range(bounds.shape[0]):
        float_vars.append(
            ds.FloatVariable(bounds[idx, 0], bounds[idx, 1])
        )
    design_space = ds.DesignSpace(float_vars)

    F=lambda x: (f1(x),f2(x))
    #init DoE using seed and LHS
    sampler = LHS(xlimits=design_space.get_unfolded_num_bounds(),
                              criterion="ese",
                              seed=seed, )
    doe = sampler(n_init)
    D=[x for x in doe]
    Y=[F(x) for x in D]
    Ni0=min_factor*num_dim+1
    Ng=max_budget-n_init-Ni0*2
    Ni=max_so_iter_factor*num_dim+1
    Ni0=min_factor*num_dim+1
    soformulation=soformulation_naive

    #run min f1 and min f2
    #TODO

    run_benchmark(bproblem) # with DoE
    NaiveBiEGO(F,D,Y,Ng,Ni,0,bounds,n_multistart=multi_start_factor*num_dim,soformulation=soformulation)
    run_benchmark_pymoo(bproblem) # with DoE



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

    max_budget=budget_factor*num_dim
    n_init=init_factor*num_dim+1
    n_min=min_factor*num_dim+1
    n_so=max_so_iter_factor*num_dim+1

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
        seed=seed,
    )

    strategy_kwargs = {
        "n_multi_start":multi_start_factor*num_dim,
        "n_init":n_init,
        "n_accuracy":n_accuracy,
        "so_formulation":soformulation_composite,
        "single_objective_max_calls":n_so,
        "min_max_calls":n_min
    }

    driver = Driver(prob_definition, opt_config, strategy=BiEGO, strategy_kwargs=strategy_kwargs)


    state = driver.optimize()
    return state

def run_benchmark_naive(bproblem):
    name=bproblem.name
    print("Running naive BiEGO on benchmark problem",name)
    num_dim=bproblem.num_dim
    bounds=bproblem.bounds
    objective=bproblem.objective

    f1=objective[0]
    f2=objective[1]

    max_budget=budget_factor*num_dim
    n_init=init_factor*num_dim+1

    float_vars = []
    for idx in range(bounds.shape[0]):
        float_vars.append(
            ds.FloatVariable(bounds[idx, 0], bounds[idx, 1])
        )
    design_space = ds.DesignSpace(float_vars)

    F=lambda x: (f1(x),f2(x))
    sampler = LHS(xlimits=design_space.get_unfolded_num_bounds(),
                              criterion="ese",
                              seed=seed, )
    doe = sampler(n_init)
    D=[x for x in doe]
    Y=[F(x) for x in D]
    Ng=max_budget-n_init
    Ni=max_so_iter_factor*num_dim+1
    Ni0=min_factor*num_dim+1
    soformulation=soformulation_naive

    return NaiveBiEGO(F,D,Y,Ng,Ni,Ni0,bounds,n_multistart=multi_start_factor*num_dim,soformulation=soformulation)

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
        pop_size=1000,
        n_offsprings=250,
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

def run_benchmark_pymoo_with_budget(bproblem):
    num_dim=bproblem.num_dim
    pymoo_problem=PymooProblem(bproblem)
    algorithm = NSGA2(
        pop_size=budget_factor,
        n_offsprings=budget_factor,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", num_dim)
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

for bproblem in L[:a**2]:
    state=run_benchmark(bproblem)
    pareto_points_naive,D_naive,Y_naive=run_benchmark_naive(bproblem)
    X,F=run_benchmark_pymoo(bproblem)
    X2,F2=run_benchmark_pymoo_with_budget(bproblem)
    data.append((bproblem.name,state,X,F,pareto_points_naive,D_naive,Y_naive,X2,F2))

print(f"Test {test_number}: n_accuracy = {n_accuracy}, seed = {seed}, budget = {budget_factor} * dim, n_init = {init_factor} * dim + 1, n_min = {min_factor} * dim + 1, n_single_objective_max = {max_so_iter_factor} * dim + 1, n_multistart = {multi_start_factor} * dim, single-objective formulation = {soformulation_composite} for composite, {soformulation_naive} for naive")

fig,axs=plt.subplots(3,3)
fig.set_size_inches(20,14)

for i in range(a):
    for j in range(a):
        index = i*3+j
        ax=axs[i][j]

        D,Y=get_DoE(data[index][1])
        D_naive=data[index][5]
        Y_naive=data[index][6]
        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]


        T=[i for i in range(len(D))]
        IGD_ac=[]
        IGD_naive=[]
        for t in T:
            pareto_points = [(D[i],Y[i]) for i in ParetoFront(D[:t+1],Y[:t+1])]
            pareto_points_naive = [(D_naive[i],Y_naive[i]) for i in ParetoFront(D_naive[:t+1],Y_naive[:t+1])]
            IGD_ac.append(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points])))
            IGD_naive.append(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])))

        ax.semilogy(T,IGD_ac,label="composite biEGO")
        ax.semilogy(T,IGD_naive,label="naive biEGO")
        ax.title.set_text(f"{data[index][0]}")
        ax.legend(loc="best")
        ax.set_xlabel("budget")
        ax.set_ylabel("IGD+")

plt.show()

fig,axs=plt.subplots(3,3)
fig.set_size_inches(20,14)

for i in range(a):
    for j in range(a):
        index = i*3+j
        ax=axs[i][j]

        D,Y=get_DoE(data[index][1])
        D_naive=data[index][5]
        Y_naive=data[index][6]
        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]


        T=[i for i in range(len(D))]
        HV_ac=[]
        HV_naive=[]
        ref_point=np.array([1,1])
        for t in T:
            pareto_points = [(D[i],Y[i]) for i in ParetoFront(D[:t+1],Y[:t+1])]
            pareto_points_naive = [(D_naive[i],Y_naive[i]) for i in ParetoFront(D_naive[:t+1],Y_naive[:t+1])]
            HV_ac.append(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points])))
            HV_naive.append(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])))

        ax.plot(T,HV_ac,label="composite biEGO")
        ax.plot(T,HV_naive,label="naive biEGO")
        ax.title.set_text(f"{data[index][0]}")
        ax.legend(loc="best")
        ax.set_xlabel("budget")
        ax.set_ylabel("Hypervolume")

plt.show()

fig,axs=plt.subplots(3,3)
fig.set_size_inches(20,14)

for i in range(a):
    for j in range(a):
        index = i*3+j
        ax=axs[i][j]
        ref_point=np.array([1,1])

        X,F=data[index][2],data[index][3]
        pareto_points_pymoo = [(X[i],F[i]) for i in ParetoFront(X,F)]
        ax.scatter([p[1][0] for p in pareto_points_pymoo],[p[1][1] for p in pareto_points_pymoo],marker=".",s=5,color="red",label="Optimal Pareto front")

        D,Y=get_DoE(data[index][1])
        pareto_points = [(D[i],Y[i]) for i in ParetoFront(D,Y)]
        ax.scatter([p[1][0] for p in pareto_points],[p[1][1] for p in pareto_points],color="blue",label="Composite acquisition function")

        pareto_points_naive=data[index][4]
        ax.scatter([p[1][0] for p in pareto_points_naive],[p[1][1] for p in pareto_points_naive],color="green",label="Naive biEGO",marker="+",s=60)

        X2,F2=data[index][7],data[index][8]
        pareto_points_pymoo2 = [(X[i],F[i]) for i in ParetoFront(X2,F2)]
        ax.scatter([p[1][0] for p in pareto_points_pymoo2],[p[1][1] for p in pareto_points_pymoo2],marker="x",color="purple",label="Pymoo with same budget")

        ax.title.set_text(f"""{data[index][0]} IGD+: {round(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points])),2)} (biEGO), {round(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])),2)} (naive), {round(IGDPlus(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo2])),2)} (pymoo)\nHypervolume: {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points])),2)} (biEGO), {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_naive])),2)} (naive), {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo2])),2)} (pymoo), {round(HV(ref_point=ref_point).do(np.array([pareto_point[1] for pareto_point in pareto_points_pymoo])),2)} (true front)""")
        ax.legend(loc="best")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")

plt.show()

