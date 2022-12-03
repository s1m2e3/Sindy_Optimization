import numpy as np
import pandas as pd
import time

class PGM_solver:
    
   def __init__(self,system,initial_guess,b):
       
      self.system = system
      self.system.lib = np.nan_to_num(self.system.lib,nan=0,posinf=0,neginf=0)
      self.initial_guess = initial_guess
      self.b = b
   
   def backtracking(self,x,grad_f,g_prox,lip,rho,lamb): 
      gamma = 2/lip
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
      gamma = 1
      obj_function=[]
      time_per_step = []
      active_columns = []
      optimality_gap = []
      
      #for k in range(maxiter):
      while(1/gamma*np.linalg.norm(g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)-x)) > epsilon:
         current= time.time()
         gamma = self.backtracking(x,grad_f,g_prox,lip,rho,lamb)
         x = g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)
         obj_function.append(f(A,x,b))
         final = time.time()
         time_per_step.append(final-current)
         active_columns.append(np.nonzero(x))
         optimality_gap.append(1/gamma*np.linalg.norm(g_prox(x-gamma*grad_f(A,x,b),lamb,gamma)-x)-epsilon)
         print(x)
      return x,obj_function,time_per_step,active_columns,optimality_gap
   

   def PGMD_FISTA(self,f,grad_f,g_prox,lamb,rho,maxiter,epsilon):
      y = self.initial_guess
      A = self.system.lib
      b = self.b
      A2 = 2*np.matmul(A.T,A)/len(y)
      A2 = np.nan_to_num(A2,nan=0,posinf=0,neginf=0)
      lip=np.real((np.max(np.linalg.eig(A2)[0])))
      gamma = 1
      obj_function=[]
      
      xk_1 = y
      tk_1 = 1
      time_per_step = []
      active_columns = []
      optimality_gap = []
      for k in range(int(maxiter)):
      #while(1/gamma*np.linalg.norm(g_prox(y-gamma*grad_f(A,y,b),lamb,gamma)-y)) > epsilon:
         current= time.time()
         gamma = self.backtracking(xk_1,grad_f,g_prox,lip,rho,lamb) 
         xk = g_prox(y-gamma*grad_f(A,y,b),lamb,gamma)
         tk = (1 + np.sqrt(1+4*tk_1**2))/2
         y = xk + ((tk_1-1)/tk)*(xk-xk_1) 
         tk_1 = tk 
         xk_1 = xk
         final = time.time()
         obj_function.append(f(A,xk,b))
         time_per_step.append(final-current)
         active_columns.append(np.nonzero(xk))
         optimality_gap.append(1/gamma*np.linalg.norm(g_prox(xk-gamma*grad_f(A,xk,b),lamb,gamma)-xk)-epsilon)
         print(xk)
         #print(np.round(f(A,xk,b),3),np.nonzero(xk),np.round(1/gamma*np.linalg.norm(g_prox(y-gamma*grad_f(A,y,b),lamb,gamma)-y),3))

      return xk,obj_function,time_per_step,active_columns,optimality_gap