from time import perf_counter
from typing import Callable

import numpy as np
from scipy import optimize as so, stats as stats

import smt.design_space as ds

from smt_optim.acquisition_functions import log_ei
from smt_optim.acquisition_strategies import AcquisitionStrategy
# from smt_optim.surrogate_models.smt import SmtMFK

from smt_optim.core.state import State
from smt_optim.subsolvers.multistart import mixvar_multistart_minimize

from smt_optim.utils.get_fmin import get_fmin

from smt_optim.subsolvers import multistart_minimize

from smt_optim.acquisition_functions.multi_obj import init_bi_obj_cei

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

def PositivePart(x):
    return max(x,0)

def build_composite_expected_improvement(state,kwargs):

    phi=kwargs["phi"]

    def composite_expected_improvement(mu: float, s2: float, f_min: float, n_expectancy=1000) -> float:
        """
        Expected Improvement composite acquisition function.

        Parameters
        ----------
        mu: np.array
            Mean prediction.
        s2: np.array
            Variance prediction.
        f_min: float
            Best minimum objective value in training data.
        phi: np.array -> float

        Returns
        -------
        float
            Expected Improvement value.
        """

        S=np.atleast_1d(0.0)
        for i in range(n_expectancy):
            sampleZ = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,1]]))
            S+=PositivePart(f_min-phi(mu+s2*sampleZ))
        ei=S/n_expectancy

        return ei[0]
    
    models=state.obj_models
    f_min=min([phi(y) for y in state.scaled_dataset.export_data([0,1],0)])

    def cei(x_pred):
        s = np.array([
            np.sqrt(models[0].predict_variances(x_pred)).item(),
            np.sqrt(models[1].predict_variances(x_pred)).item()
        ])

        y = np.array([
            models[0].predict_values(x_pred).item(),
            models[1].predict_values(x_pred).item(),
        ])
        return composite_expected_improvement(y,s,f_min)
    return cei

class BiEGO(AcquisitionStrategy):
    def __init__(self, state: State, **kwargs):
        super().__init__()


        self.acq_func1 = kwargs.get("acq_func", log_ei) #Acquisition function for min(f1) (to be modified to take only f1 as a parameter)
        self.acq_func2 = kwargs.get("acq_func", log_ei) #Acquisition function for min(f2) (same for f2)
        self.acq_func_gen3 = kwargs.get("acq_func_bi", init_bi_obj_cei) #Composite acquisition function for min(f1,f2)
        self.n_start = kwargs.pop("n_start", 20)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))
        self.current_calls = 0
        self.current_subcalls = 0
        self.single_obj_max_calls = kwargs.pop("single_obj_max_calls",5)
        self.acq_func_gen1 = lambda state : lambda x : self.acq_func1(state.obj_models[0].predict_values(x).item(),state.obj_models[0].predict_variances(x).item(),min(state.scaled_dataset.export_data([0],0)))
        self.acq_func_gen2 = lambda state : lambda x : self.acq_func2(state.obj_models[1].predict_values(x).item(),state.obj_models[1].predict_variances(x).item(),min(state.scaled_dataset.export_data([1],0)))

        self.r = None
        self.X = None #TODO init this
        self.W = None #TODO init this




    def validate_config(self, state):
        pass


    def get_infill(self, state):

        #Init
        if self.current_calls < self.single_obj_max_calls:
            self.current_calls+=1
            return self.get_infill_custom(state,self.acq_func_gen1)
        elif self.current_calls < 2*self.single_obj_max_calls:
            self.current_calls+=1
            return self.get_infill_custom(state,self.acq_func_gen2)

        #Main loop
        else:
            if self.current_subcalls == 0 or self.current_subcalls == self.single_obj_max_calls:
                self.current_subcalls = 0
                r=(0,0) # Choose r
            self.current_calls+=1
            self.current_calls+=1
            return self.get_infill_custom(state,self.acq_func_gen3)
    
    def get_infill_custom(self,state,acq_func_gen,**kwargs):
        self.seed = state.iter

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_start)

        ac_func = acq_func_gen(state,kwargs)

        def sp_wrapper(x):
            x = x.reshape(1, -1)
            return -ac_func(x)

        res = multistart_minimize(sp_wrapper,
                                    bounds=np.array([[0, 1]] * state.problem.num_dim),
                                    constraints=[],
                                    n_start=self.n_start,
                                    seed=self.seed,
                                    tol=self.sp_tol,
                                    method=self.sp_method, )

        next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill

