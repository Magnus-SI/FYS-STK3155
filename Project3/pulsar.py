import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class pulsardat:
    def __init__(self):
        """
        Reads the datafile for the pulsar data set,
        and sorts into x and y with labels
        """
        df = pd.read_csv('pulsar_stars.csv')
        self.xlabels = df.keys()[:-1]
        self.ylabels = df.keys()[-1]
        self.df = df

    def pre_process(self):
        """
        Pre process the data, and overwrites self.df with the pre processed one
        """
        x = self.df[self.xlabels].values
        y = self.df[self.ylabels].values

        scaler = StandardScaler()

        scaled_x = scaler.fit_transform(x)

        scaled_df = pd.DataFrame(scaled_x, columns=self.xlabels)

        scaled_df[self.ylabels] = y

        self.df = scaled_df

    def __call__(self):
        self.pre_process()
        return self.df, self.xlabels, self.ylabels
