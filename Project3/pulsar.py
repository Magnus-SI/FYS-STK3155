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

    def corrplot(self):
        """
        Creates a correlation plot of the data, pre_process before doing this?
        """
        plt.figure()
        df = self.df
        varlabels = ["X%i"%i for i in range(len(self.xlabels))] + ['y']
        dat = df.values
        corr = np.corrcoef(dat.T).round(2)
        sns.heatmap(data = corr, annot = True, cmap = 'viridis', xticklabels = varlabels, yticklabels = varlabels)
        plt.savefig('corrplot.pdf')

    def pairplot(self):
        df, xl, yl = self()
        fig = sns.pairplot(df,hue="target_class")
        plt.savefig("pairplot.png")

    def __call__(self):
        self.pre_process()
        return self.df, self.xlabels, self.ylabels


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    data = pulsardat()
    data.corrplot()
    data.pairplot()

    # df, xl, yl = data()
    # fig = sns.pairplot(df,hue="target_class")
    # plt.savefig("pairplot.png")
