import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#boxplot
def get_boxplot(df,xlabel,path):
    sns.boxplot(df)
    plt.xlabel(xlabel)
    plt.savefig(path)
    plt.show()
    plt.clf()
    plt.close()

def get_scatter_plot(df,a,ylabel,path):

    plt.scatter(df.index,df[a])
    plt.xlabel("Date Time")
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()
