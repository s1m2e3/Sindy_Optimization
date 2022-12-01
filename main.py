# from BFGS import BFGS
# from CG import CG
from PGM import PGM_solver
from system import system_creator
import numpy as np

def f(A,x,b):
    return np.sum(np.power((np.dot(A*x)-b),2))
def grad_f(A,x,b):
    vec= 2*(np.dot(A*x)-b)
    vec = vec.reshape((len(vec),1))
    return np.sum(np.multiply(vec,A),axis=0)
def g_prox(x,lamb):
    return  (np.abs(x)-lamb*np.ones(x.shape))*np.sign(x)

def lorentz_system(sigma,rho,beta,n_points):
    low = -20
    high = 20
    
    states = np.random.uniform(low,high,size = (n_points,3))
    diff = states
    diff[:,0] = sigma*(states[:,1]-states[:,0])
    diff[:,1] = states[:,0](rho-states[:,2])-states[:,1]
    diff[:,2] = states[:,0]*states[:,1]-beta*states[:,2]

    return states, diff

def vortex_system (mu,omega,A,lamb,n_points):
        
    
def main():

    states = np.random.rand(10,3)
    b = states.diff()
    comp = True
    sys = system_creator(states,comp)
    initial_guess = np.random.rand((sys.lib.shape[1],states.shape[1]))
    pgm = PGM_solver(sys,initial_guess,b)
    pgm.add_functions(f,grad_f,g_prox)
    pgm.solve()

if __name__=="__main__":
    main()


