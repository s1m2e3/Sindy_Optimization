import numpy as np
import pandas as pd

class PGM_solver:
    
   def __init__(self,system,initial_guess,b):
       
      self.system = system
      self.initial_gues = initial_guess
      self.b = b

   def f(self,func,A,x,b):
      return func(A,x,b)       
   def grad_f(self,func,A,x,b):
      return func(A,x,b)       
   def g_prox(self,func,x,lamb):
      return func(x,lamb)       
   
   def backtracking(self,x,grad_f,g_prox,lip,rho,lamb): 

      gamma = 1/lip
      
      while np.norm(grad_f(g_prox(x-gamma*grad_f(x),lamb))-grad_f(x))>= 1/2*gamma*np.norm(g_prox(x-gamma*grad_f(x),lamb)-x):
         
         gamma = gamma*rho
         x = g_prox(x-gamma*grad_f(x),lamb)

      return gamma
         
   def PGMD(self,f,grad_f,g_prox,lamb,rho,maxiter,epsilon):

      lip = np.linalg.norm(np.dot(A.T,A))
      x = self.initial_guess
      A = self.system
      b = self.b
      
      gamma = 1/lip

      while (1/gamma*np.norm(g_prox(x-gamma*grad_f(x),lamb)-x))>= epsilon:
         
         gamma = self.backtracking(x,grad_f,g_prox,lip,rho,lamb)
         x = g_prox(x-gamma*grad_f(x),lamb)

      return x   
