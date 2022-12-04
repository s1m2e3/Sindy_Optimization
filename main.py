# from BFGS import BFGS
# from CG import CG
from PGM import PGM_solver
from system import system_creator
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns
import time

def f(A,x,b):
    return np.sum(np.power((A@x-b),2))/len(x)
def grad_f(A,x,b):
    #vec = 2*(A@x-b)/len(x)
    #vec = vec.reshape((len(vec),1))
    #comp1 = np.sum(np.multiply(vec,A),axis=0) 
    comp = (np.matmul(A.T,np.matmul(A,x))+np.matmul(np.matmul(x.T,A.T),A)-np.matmul(A.T,b)-np.matmul(b.T,A))/len(x)
    return comp
def g_prox(x,lamb,gamma):
    return  np.maximum(np.abs(x)-gamma*lamb*np.ones(x.shape),np.zeros(x.shape))*np.sign(x)

def lorentz_system(sigma,rho,beta,n_points):
    
    #low = -10
    #high = 10
    states =np.random.normal(loc = 0,scale = 10,size = (n_points,3)) 
    diff = states
    diff[:,0] = sigma*(states[:,1]-states[:,0])
    diff[:,1] = states[:,0]*(rho-states[:,2])-states[:,1]
    diff[:,2] = states[:,0]*states[:,1]-beta*states[:,2]

    return states, diff

#def linear_system(n_points):

#def vortex_system (mu,omega,A,lamb,n_points):
        
    
def main():

    n_points =[1000,5000,10000]

    for points in n_points:
        
        #create lorentz system prediction
        sigma = np.random.normal(loc = 0,scale = 10)
        rho = np.random.normal(loc = 0,scale = 10)
        beta = np.random.normal(loc = 0,scale = 10)
        states,diff = lorentz_system(sigma,rho,beta,points)
        #states,states_norm = normalize(states,return_norm=True)
        diff = np.nan_to_num(diff,nan=0,posinf=0,neginf=0)
        diff,diff_norm = normalize(diff,return_norm=True)
        
        
        #create library
        sys = system_creator(states,True)
        labels = sys.lib.columns.to_list()
        sys.lib,lib_norm = normalize(sys.lib,return_norm=True)
        
        #set parameters
        initial_guess = np.zeros((sys.lib.shape[1],states.shape[1]))
        rho = 0.5
        lamb = 1e-4
        maxiter = 1e5
        epsilon = 1e-4

        #create storage
        solution ={"reg":{},"fista":{}}
        obj_func = {"reg":{},"fista":{}}
        time_per_step = {"reg":{},"fista":{}}
        active_columns = {"reg":{},"fista":{}}
        opt_gap = {"reg":{},"fista":{}}
        
        sns.set()
        for col in range(3):
            pgm = PGM_solver(sys,initial_guess[:,col],diff[:,col])
            solution['fista'][col] ,obj_func['fista'][col],time_per_step['fista'][col],active_columns['fista'][col],opt_gap['fista'][col]=pgm.PGMD(f,grad_f,g_prox,lamb,rho,maxiter,epsilon)
            sparse_index = (np.nonzero(solution['fista'][col]))[0]
            for index in sparse_index:
                print("sparse representation for equation %d found by PGMB FISTA:%s " %(col,labels[index]))
            plt.figure()
            plt.plot(np.multiply(sys.lib@solution['fista'][col],lib_norm[col]),'o')
            plt.plot(np.multiply(diff[:,col],diff_norm[col]),'x')
            plt.legend(["prediction","real"])
            plt.show()
            print("average time per step by PGMD FISTA",np.mean(time_per_step["fista"][col]))
            print("MSE found by PGMB FISTA:",obj_func['fista'][col][-1])
            solution['reg'][col] ,obj_func['reg'][col],time_per_step['reg'][col],active_columns['reg'][col],opt_gap['reg'][col]=pgm.PGMD_FISTA(f,grad_f,g_prox,lamb,rho,maxiter,epsilon)
            sparse_index = (np.nonzero(solution['reg'][col]))[0]
            for index in sparse_index:
                print("sparse representation for equation %d found by PGMB:%s " %(col,labels[index]))
            print("average time per step by PGMD",np.mean(time_per_step["reg"][col]))
            print("MSE found by PGMB:",obj_func['reg'][col][-1])
            plt.figure()
            plt.plot(sys.lib@solution['reg'][col],'o')
            plt.plot(diff[:,col],'x')
            plt.legend(["prediction","real"])
            plt.show()
            plt.figure(figsize=(20,10))
            plt.semilogy(obj_func['reg'][col])
            plt.semilogy(obj_func['fista'][col])
            plt.ylabel('objective function reduction')
            plt.xlabel('number of iterations')
            plt.title("equation_pred_"+str(col))
            plt.legend(["regular PGMD","FISTA PGMD"])
            plt.savefig("equation_pred_"+str(col)+str(points)+".png")
            plt.close()

            plt.figure(figsize=(20,10))
            active_columns['reg'][col]=[len(i) for i in active_columns['reg'][col]]
            active_columns['reg'][col],bins=np.unique(active_columns['reg'][col],return_counts=True)
            plt.bar(x=bins,height=active_columns['reg'][col])
            active_columns['fista'][col]=[len(i) for i in active_columns['fista'][col]]
            active_columns['fista'][col],bins=np.unique(active_columns['fista'][col],return_counts=True)
            plt.bar(x=bins,height=active_columns['fista'][col])
            
            plt.ylabel('number of active terms')
            plt.xlabel('number of iterations')
            plt.title("equation_pred_"+str(col)+"active_terms")
            plt.legend(["regular PGMD","FISTA PGMD"])
            plt.savefig("active_terms_equation_pred_"+str(col)+str(points)+".png")
            plt.close()

            


if __name__=="__main__":
    main()


