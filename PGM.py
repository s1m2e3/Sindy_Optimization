import numpy as np
import pandas as pd

class PGM_solver:
    
   def __init__(self,system,initial_guess,b):
       
      self.system = system
      self.system.lib = np.nan_to_num(self.system.lib,nan=0,posinf=0,neginf=0)
      self.initial_guess = initial_guess
      self.b = b
   
   def backtracking(self,x,grad_f,g_prox,lip,rho,lamb): 
      gamma = 1/lip
      A=self.system.lib
      b=self.b
      while np.linalg.norm(grad_f(A,g_prox(x-gamma*grad_f(A,x,b),lamb,gamma),b)-grad_f(A,x,b))>1/(2*gamma)*np.linalg.norm(g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)-x):
         gamma = gamma*rho
         x = g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)
      return gamma
         
   def PGMD(self,f,grad_f,g_prox,lamb,rho,maxiter,epsilon):

      x = self.initial_guess
      A = self.system.lib
      b = self.b
      A2 = 2*np.matmul(A.T,A)/len(x)
      A2 = np.nan_to_num(A2,nan=0,posinf=0,neginf=0)
      lip=np.real((np.max(np.linalg.eig(A2)[0])))
      gamma = 1/lip
      num=0
      obj_function=[]
      #for k in range(int(maxiter)):
      while(1/gamma*np.linalg.norm(g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)-x))> epsilon:
         gamma = self.backtracking(x,grad_f,g_prox,lip,rho,lamb)
         x = g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)
         obj_function.append(f(A,x,b))
         print(x)
      print("final_loss",f(A,x,b))
      print("if x is 0 ",np.sum(np.power(b,2))/len(x))
      print("least squares solution ",f(A,np.linalg.lstsq(A,b,rcond=None)[0],b)) 
      print("pseudo inverse solution ",f(A,np.matmul(np.linalg.pinv(A),b),b)) 
      print("finished")
      return x,obj_function
