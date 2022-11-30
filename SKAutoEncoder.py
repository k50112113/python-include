import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pickle

class Trained_Model:
    def __init__(self,ae,test_x):
        self.ae=ae
        self.test_x=test_x
        
def SaveModel(filename,ae,test_x):
    with open(filename, 'wb') as fout:  
        pickle.dump(Trained_Model(ae,test_x), fout)

def ReadModel(filename):
    with open(filename, 'rb') as fin:  
        tm = pickle.load(fin)
    return tm.ae, tm.test_x
   
class MyAutoEncoder:
    def __init__(self, n_encoder, n_latent, n_decoder):
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.n_latent = n_latent
        self.hidden_layer_sizes = tuple(self.n_encoder+self.n_latent+self.n_decoder)
        
        print("Hidden Layer: "+str(self.hidden_layer_sizes))
        
        self.reg = MLPRegressor(hidden_layer_sizes = self.hidden_layer_sizes, 
                                activation = 'tanh', 
                                solver = 'adam', 
                                learning_rate_init = 0.001, 
                                max_iter = 500, 
                                tol = 0.0001, 
                                verbose = True)
    
    def fit(self,train_x):
        self.reg.fit(train_x, train_x)
    
    def encode(self,data):
        data = np.asmatrix(data)
        encoder = data
        for layer_index in range(len(self.n_encoder)+1):
            encoder = encoder*self.reg.coefs_[layer_index] + self.reg.intercepts_[layer_index]
            encoder = (np.exp(encoder) - np.exp(-encoder))/(np.exp(encoder) + np.exp(-encoder))   
        return np.asarray(encoder)

    def decode(self,data):
        data = np.asmatrix(data)
        decoder = data
        for layer_index in range(len(self.n_encoder)+1,len(self.hidden_layer_sizes)+1):
            decoder = decoder*self.reg.coefs_[layer_index] + self.reg.intercepts_[layer_index]
            if layer_index < len(self.hidden_layer_sizes):
                decoder = (np.exp(decoder) - np.exp(-decoder))/(np.exp(decoder) + np.exp(-decoder))
        return np.asarray(decoder) 
    
    def reconstruct(self,data):
        return self.reg.predict(data)
    
    def chi2(self,data):     
        reconst_data = self.reg.predict(data)
        chi2 = np.mean(np.sum((reconst_data-data)**2,axis=1)/np.sum(data**2,axis=1))
        return chi2

