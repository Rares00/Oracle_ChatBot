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


# plot and compare them
def compare_graph(x_data1,y_data1,x_data2,y_data2,x_data3,y_data3, x_label, y_label, title, path):
    """
    Takes data from two data frames and compare
    Was used for comparing plot with outliers and without outliers
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    ax1.set_title = title
    ax1.scatter(x_data1,y_data1)
    ax1.scatter(x_data2, y_data2, c='r')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid()

    ax2.set_title('After Drop')
    ax2.scatter(x_data3,y_data3)
    ax2.set_xlabel(x_label)
    ax2.grid()
    plt.xticks(rotation=30)
    plt.savefig(path)
#Source module taken from a different project I did, Rares Nitu

def data_filter_ml(df, start, end, collumns):
    """
    This function is used to filter the data in the context of using it for a prediction model
    in our context we do not need it because the data is already pre-procesed but considering
    we want the program to be as flexible as possible we used this function in case other datasets 
    are goinng to be used

    This function takes the complete dataset and filter it by datetime index with a start date an end date
    and the wanted columns.
    """
    filter_data = df.loc[(df.index >= start) & (df.index <= end), collumns]
    return filter_data