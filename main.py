# coding: utf-8

from gng import GrowingNeuralGas
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import os
import shutil
import pandas as pd

# normalize the values array
def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

# dataset preparation function
def dataset():
    # read csv file
    data = pd.read_csv("dataset_transfusion.csv", header=None, sep=",")
    # shuffles the data
    data = data.sample(frac=1).reset_index(drop=True)
    # inputs
    values = data.iloc[:, :-1]
    # rearranging the values and answers
    values = normalize(values).values

    # returning values
    return values

if __name__ == '__main__':
    if os.path.exists('visualization/sequence'):
        shutil.rmtree('visualization/sequence')
        
    os.makedirs('visualization/sequence')
    n_samples = 1500

    data = None
    #data = dataset() # in case the user wants to use a specific dataset
    #data = datasets.load_breast_cancer()
    
    #data = datasets.load_iris(n_samples=n_samples, random_state=8)
    #data = datasets.make_blobs(n_samples=n_samples, random_state=8)
    data = datasets.make_moons(n_samples=n_samples, noise=.05)
    #data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    #data = StandardScaler().fit_transform(data.data) # depends on the data variable
    data = StandardScaler().fit_transform(data[0]) # depends on the data variable
    
    print('Done.')
    print('Fitting neural network...')
    gng = GrowingNeuralGas(data)
    gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=5, plot_evolution=True)
    
    print('Found %d clusters.' % gng.number_of_clusters())
    gng.plot_clusters(gng.cluster_data())