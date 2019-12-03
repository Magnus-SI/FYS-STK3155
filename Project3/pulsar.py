import numpy as np
import pandas as pd

class pulsardat:
    def __init__(self):
        df = pd.read_csv('pulsar_stars.csv')
        self.xlabels = df.keys()[:-1]
        self.ylabels = df.keys()[-1]
        self.df = df

    def __call__(self):
        return self.df, self.xlabels, self.ylabels
