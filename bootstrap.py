import numpy as np
import matplotlib.pyplot as plt

def Bootstrap(y,model,N = 1000,x_gen):
    avg_betas = np.zeros_like(model(X,y))
