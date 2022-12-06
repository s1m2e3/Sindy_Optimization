# from BFGS import BFGS
# from CG import CG
from PGM import PGM_solver
import pandas as pd
from system import system_creator
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import seaborn as sns
import time

from scipy.integrate import solve_ivp


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])


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

def lorentz_system(t,alpha,sigma,rho,beta):
    x,y,z = alpha
    return [sigma*(y-x),rho*x-y-x*z,x*y-beta*z]

#def linear_system(n_points):

#def vortex_system (mu,omega,A,lamb,n_points):
        
    
def main():

    n_points =[100,200,500]
    hertz = 10
    for points in n_points:
        
        #create lorentz system prediction
        sigma = np.random.normal(loc = 0,scale = 1)
        rho = np.random.normal(loc = 0,scale = 1)
        beta = np.random.normal(loc = 0,scale = 1)
        init_point = np.random.normal(loc = 0,scale = 1,size=3)
        time_length = points//hertz
        time_points =np.linspace(0,time_length,points)
        sol = solve_ivp(lorentz_system,[0,time_length],init_point,t_eval=time_points,args = [sigma,rho,beta])
        states=sol.y.T
        diff = np.zeros((states.shape))
        diff[1:,:] = np.diff(states,axis=0)/(1/hertz)        
        diff,diff_norm = normalize(diff,return_norm=True)
        
        
        #create library
        print("started library creation")
        sys = system_creator(states,True)
        print("finished library creation")
        labels = sys.lib.columns.to_list()
        sys.lib,lib_norm = normalize(sys.lib,return_norm=True)
        
        #set parameters
        initial_guess = np.zeros((sys.lib.shape[1],states.shape[1]))
        A2 = 2*np.matmul(sys.lib.T,sys.lib)/len(initial_guess)
        A2 = np.nan_to_num(A2,nan=0,posinf=0,neginf=0)
        lip=np.real((np.max(np.linalg.eig(A2)[0])))
        rho = 0.5
        lamb = 5e-3
        maxiter = 1e4
        epsilon = 1e-4

        #create storage
        solution ={"reg":{},"fista":{}}
        obj_func = {"reg":{},"fista":{}}
        time_per_step = {"reg":{},"fista":{}}
        active_columns = {"reg":{},"fista":{}}
        opt_gap = {"reg":{},"fista":{}}
        indexes = np.arange(len(labels))+0.3
        sns.set()
        for col in range(3):
            pgm = PGM_solver(sys,initial_guess[:,col],diff[:,col])
            solution['reg'][col] ,obj_func['reg'][col],time_per_step['reg'][col],active_columns['reg'][col],opt_gap['reg'][col]=pgm.PGMD(f,grad_f,g_prox,lamb,rho,lip,maxiter,epsilon)
            print(solution['reg'][col])
            sparse_index = (np.nonzero(solution['reg'][col]))[0]
            for index in sparse_index:
                print("sparse representation for equation %d found by PGMB :%s " %(col,labels[index]))
            print("average time per step by PGMD ",np.mean(time_per_step["reg"][col]))
            print("MSE found by PGMB :",obj_func['reg'][col][-1])
            solution['fista'][col] ,obj_func['fista'][col],time_per_step['fista'][col],active_columns['fista'][col],opt_gap['fista'][col]=pgm.PGMD_FISTA(f,grad_f,g_prox,lamb,rho,lip,maxiter,epsilon)
            print(solution['fista'][col])
            sparse_index = (np.nonzero(solution['fista'][col]))[0]
            for index in sparse_index:
                print("sparse representation for equation %d found by PGMB FISTA:%s " %(col,labels[index]))
            print("average time per step by PGMD FISTA",np.mean(time_per_step["fista"][col]))
            print("MSE found by PGMB FISTA:",obj_func['fista'][col][-1])
            
            plt.figure(figsize=(20,10))
            plt.plot(np.multiply(sys.lib@solution['reg'][col],lib_norm[col]),'o')
            plt.plot(np.multiply(sys.lib@solution['fista'][col],lib_norm[col]),'o')
            plt.plot(np.multiply(diff[:,col],diff_norm[col]),'x')
            
            plt.ylabel("state_derivative_"+str(col))
            plt.title("State %s Derivative Regression " %(str(col)))
            plt.legend(["prediction PGMD","prediction FISTA","real"])
            plt.savefig("states_"+str(col)+"_regression_"+str(points)+".png")
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.semilogy(obj_func['reg'][col])
            plt.semilogy(obj_func['fista'][col])
            plt.ylabel('objective function reduction')
            plt.xlabel('number of iterations')
            plt.title("equation_pred_"+str(col))
            plt.legend(["regular PGMD","FISTA PGMD"])
            plt.savefig("equation_pred_"+str(col)+str(points)+".png")
            plt.close()

            df_reg = pd.DataFrame([labels,solution['reg'][col]]).T
            df_reg.columns = ['label','value']
            df_reg['solver']='pgmd'
            df_reg.value = np.abs(df_reg.value)
            lengths=np.arange(len(df_reg))
            df_fista = pd.DataFrame([labels,solution['fista'][col]]).T
            df_fista.columns =['label','value']
            df_fista['solver']='pgmd-fista'
            df_fista.value = np.abs(df_fista.value)
            
            plt.figure(figsize=(20,10))
            plt.bar(x=lengths,height=df_reg['value'],alpha=0.5,tick_label=df_reg['label'])
            plt.bar(x=lengths,height=df_fista['value'],alpha=0.5,tick_label=df_reg['label'])
            plt.ylabel('absolute value of active terms')
            plt.xlabel('type of term')
            plt.title("equation_pred_"+str(col)+"active_terms")
            plt.legend(["regular PGMD","FISTA PGMD"])
            plt.savefig("active_terms_equation_pred_"+str(col)+str(points)+".png")
            plt.close()

            


if __name__=="__main__":
    main()


