import torch 
import numpy as np
from scipy.special import comb
from sympy import *
from sympy import symbols, Poly
import pandas as pd

class system_creator:

    def __init__(self,states,comp=False,tuning=False,n_degree=3) -> None:
       
       self.states = states
       self.comp = comp
       self.tuning = tuning
       self.n_degree = n_degree
       self.lib = self.create()
       
    def create(self):
        
        '''
        Consider only polynomial functions , trigonometric, exponential and logaritmic functions 
        Dictionary without composition is a matrix with matrices:
        polinomial | sin(polinomial)|cos(polinomial)|tan(polinomial)|exp(polinomial)|log(polinomial)
        Dictionary with composition has the following size: choose number of compositions out of 6 functions
        '''

        pol_comb = int(comb(self.n_degree+self.states.shape[1],self.n_degree)-1)
        base_matrix_size = pol_comb*6
        if self.comp:
            compositions=2
            max_size = base_matrix_size
            while max_size<self.states.shape[0]:
                max_size = comb(base_matrix_size,compositions)
                compositions+=1
            
        print("number of compositions is:",compositions)
        lib = self.compose(compositions,pol_comb,self.states,self.n_degree)
        return lib
        
    def compose(self,compositions,matrix_cols,states,degree):
        
        base = np.zeros((states.shape[0],matrix_cols))
        string = "x:"+str(states.shape[1])
        variables = symbols(string)
        exp = variables[0]
        
        for var in variables[1:]:
            exp = var+exp
        dictionary = self.pol(exp,degree,variables,states)
        
        mat_pol = pd.DataFrame.from_dict(dictionary)
        mat_cos = np.cos(mat_pol)
        string_1 = "cos("
        string_2 = ")"
        mat_cos.columns = [string_1+str(col)+string_2 for col in mat_cos.columns] 
        mat_tan = np.tan(mat_pol)
        string_1 = "tan("
        string_2 = ")"
        mat_tan.columns = [string_1+str(col)+string_2 for col in mat_tan.columns] 
        
        mat_sin = np.sin(mat_pol)
        string_1 = "sin("
        string_2 = ")"
        mat_sin.columns = [string_1+str(col)+string_2 for col in mat_sin.columns] 
        
        mat_exp = np.exp(mat_pol)
        string_1 = "exp("
        string_2 = ")"
        mat_exp.columns = [string_1+str(col)+string_2 for col in mat_exp.columns] 
        
        mat_log = np.log(mat_pol)
        string_1 = "log("
        string_2 = ")"
        mat_log.columns = [string_1+str(col)+string_2 for col in mat_log.columns] 
        
        elems = ['pol','sin','cos','tan','exp','log']
        base_dict = dict(zip(elems,[mat_pol,mat_sin,mat_cos,mat_tan,mat_exp,mat_log]))
        if self.comp:
            combinations = [(elems[i],elems[j]) for i in range(len(elems)) for j in range(len(elems))]
            combinations = combinations[1:]
            base_dict = self.compose_dict(base_dict,combinations,degree,compositions)
        df=pd.DataFrame(np.ones((states.shape[0],1)))
        df.columns = ["constant"]
        for shape in base_dict:
            df = pd.concat((df,base_dict[shape]),axis=1)
        return df


    def pol(self,exp,degree,variables,states):
        dictionary = {}
        for i in range(degree):
            p = Poly((exp)**(i+1))
            prods = [prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]
            for term in prods:
                vector = np.zeros(states.shape[0])
                symbols_list = list(term.free_symbols)
                indexes = [variables.index(var) for var in symbols_list]
                for j in range(len(states)):
                    replace = [(symbols_list[i],states[j][indexes[i]]) for i in range(len(symbols_list))]
                    vector[j] = term.subs(replace)
                dictionary[term] = vector
        return dictionary
    
    def compose_dict(self,base_dict,combinations,degree,compositions):
        
        for pair in combinations:
            
            first = pair[0]
            second = pair[1]
            
            if first == "pol":
                '''
                string = "x:"+str(base_dict[second].shape[1])
                variables = symbols(string)
                exp = variables[0]
                for var in variables[1:]:
                    exp = var+exp
                base_dict[pair]=self.pol(exp,degree,variables,np.array(base_dict[second]))
                '''
                pass
            elif first == "sin":
                base_dict[pair]=np.sin(base_dict[second])
            elif first == "cos":
                base_dict[pair]=np.cos(base_dict[second])
            elif first == "tan":
                base_dict[pair]=np.tan(base_dict[second])
            elif first == "exp":
                base_dict[pair]=np.exp(base_dict[second])
            elif first == "log":
                base_dict[pair]=np.log(base_dict[second])
            
        return base_dict