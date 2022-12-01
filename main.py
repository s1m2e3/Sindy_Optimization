# from BFGS import BFGS
# from CG import CG
from PGM import PGM_solver
from system import system_creator
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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
    diff[:,1] = states[:,0]*(rho-states[:,2])-states[:,1]
    diff[:,2] = states[:,0]*states[:,1]-beta*states[:,2]

    return states, diff

#def linear_system(n_points):

#def vortex_system (mu,omega,A,lamb,n_points):
        
    
def main():

    n_points =[100]

    for points in n_points:
        
        sigma = np.random.normal(loc = 0,scale = 10)
        rho = np.random.normal(loc = 0,scale = 10)
        beta = np.random.normal(loc = 0,scale = 10)
        states,diff = lorentz_system(sigma,rho,beta,points)
        
        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        # Data for a three-dimensional scattered
        ax.scatter3D(diff[:,0],diff[:,1], diff[:,2],c=diff[:,0]*diff[:,1],cmap="summer")
        plt.savefig("orig_lorentz_"+str(points)+".png")
        '''

        sys = system_creator(states,True)
        initial_guess = np.random.rand((sys.lib.shape[1],states.shape[1]))
        rho = 0.9
        lamb = 1e-3
        maxiter = 1e5
        epsilon = 1e-5
        pgm = PGM_solver(sys,initial_guess,diff)
        pgm.PGMD(f,grad_f,g_prox,lamb,rho,maxiter,epsilon)
        pgm.solve()

if __name__=="__main__":
    main()


