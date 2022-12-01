# from BFGS import BFGS
# from CG import CG
from PGM import PGM_solver
from system import system_creator
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns

def f(A,x,b):
    return np.sum(np.power((A@x-b),2))/len(x)
def grad_f(A,x,b):
    vec= 2*(A@x-b)/len(x)
    vec = vec.reshape((len(vec),1))
    comp1 = np.sum(np.multiply(vec,A),axis=0) 
    comp = (np.matmul(A.T,np.matmul(A,x))+np.matmul(np.matmul(x.T,A.T),A)-np.matmul(A.T,b)-np.matmul(b.T,A))/len(x)
    return comp1
def g_prox(x,lamb,gamma):
    prev = np.abs(x)-gamma*lamb*np.ones(x.shape)
    return  np.maximum(prev,np.zeros(x.shape))*np.sign(x)

def lorentz_system(sigma,rho,beta,n_points):
    
    low = -10
    high = 10
    states = np.random.uniform(low,high,size = (n_points,3))
    diff = states
    diff[:,0] = sigma*(states[:,1]-states[:,0])
    diff[:,1] = states[:,0]*(rho-states[:,2])-states[:,1]
    diff[:,2] = states[:,0]*states[:,1]-beta*states[:,2]

    return states, diff

#def linear_system(n_points):

#def vortex_system (mu,omega,A,lamb,n_points):
        
    
def main():

    n_points =[1000]

    for points in n_points:
        
        sigma = np.random.normal(loc = 0,scale = 10)
        rho = np.random.normal(loc = 0,scale = 10)
        beta = np.random.normal(loc = 0,scale = 10)
        states,diff = lorentz_system(sigma,rho,beta,points)
        states = normalize(states)
        diff = np.nan_to_num(diff,nan=0,posinf=0,neginf=0)
        diff = normalize(diff)
        
        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        # Data for a three-dimensional scattered
        ax.scatter3D(diff[:,0],diff[:,1], diff[:,2],c=diff[:,0]*diff[:,1],cmap="summer")
        plt.savefig("orig_lorentz_"+str(points)+".png")
        '''

        sys = system_creator(states,True)
        sys.lib = np.nan_to_num(sys.lib,nan=0,posinf=0,neginf=0)
        initial_guess = np.zeros((sys.lib.shape[1],states.shape[1]))
        rho = 0.5
        lamb = 1e-1
        maxiter = 1e4
        epsilon = 1e-4
        solution ={}
        obj_func = {}
        for col in range(initial_guess.shape[1]):
            pgm = PGM_solver(sys,initial_guess[:,col],diff[:,col])
            solution[col] , obj_func[col]=pgm.PGMD(f,grad_f,g_prox,lamb,rho,maxiter,epsilon)
            plt.figure()
            sns.set_theme()
            plt.plot(obj_func[col]-obj_func[col][-1])
            plt.ylabel('objective function reduction')
            plt.xlabel('number of iterations')
            plt.title("equation_pred_"+str(col))
            plt.savefig("equation_pred_"+str(col)+".png")
            plt.close()
if __name__=="__main__":
    main()


