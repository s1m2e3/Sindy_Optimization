import numpy as np
import pandas as pd

class PGM_solver:
    
   def __init__(self,system,initial_guess,b):
       
      self.system = system
      self.system.lib = np.nan_to_num(self.system.lib,nan=0,posinf=0,neginf=0)
      self.initial_guess = initial_guess
      self.b = b

   
   def backtracking(self,x,grad_f,g_prox,lip,rho,lamb): 
      print("started backtracking")
      gamma = 1
      A=self.system.lib
      b=self.b

      while np.linalg.norm(grad_f(A,g_prox(x-gamma*grad_f(A,x,b),lamb),b)-grad_f(A,x,b))>= 1/2*gamma*np.linalg.norm(g_prox(x-gamma*grad_f(A,x,b),lamb)-x):
         gamma = gamma*rho
         x = g_prox(x-gamma*grad_f(A,x,b),lamb)
         print(np.linalg.norm(grad_f(A,g_prox(x-gamma*grad_f(A,x,b),lamb),b)-grad_f(A,x,b)))
      print("finished backtracking")         
      return gamma
         
   def PGMD(self,f,grad_f,g_prox,lamb,rho,maxiter,epsilon):

      x = self.initial_guess
      A = self.system.lib
      b = self.b
      
      A2 = np.matmul(A.T,A)
      A2 = np.nan_to_num(A2,nan=0,posinf=0,neginf=0)
      lip=np.real((np.max(np.linalg.eig(A2)[0])))
      gamma = 1
      num=0
      while(1/gamma*np.linalg.norm(g_prox(x-gamma*grad_f(A,x,b),lamb)-x))>= epsilon:
         
         gamma = self.backtracking(x,grad_f,g_prox,lip,rho,lamb)
         x = g_prox(x-gamma*grad_f(A,x,b),lamb)
         num+=1
         print("one_step",f(A,x,b))
      print("finished") 

      return x,f(A,x,b)   
