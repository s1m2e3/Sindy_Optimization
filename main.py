# from BFGS import BFGS
# from CG import CG
from PGM import PGM_solver
from system import system_creator
import numpy as np

def f():
    return
def grad_f():
    return
def g_prox:
    return 

def main():

    states = np.random.rand(10,3)
    comp = True
    sys = system_creator(states,comp)
    initial_guess = np.random.rand((sys.lib.shape[1],states.shape[1]))
    pgm = PGM_solver(sys,initial_guess)
    pgm.solve()

if __name__=="__main__":
    main()


