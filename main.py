# from BFGS import BFGS
# from CG import CG
# from GM import GM
from system import system_creator
import numpy as np

def main():

    states = np.random.rand(10,3)
    cols = 5 
    comb = True
    sys = system_creator(cols,states,comb)
    


if __name__=="__main__":
    main()


