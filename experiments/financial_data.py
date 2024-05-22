import torch
from plot import plot
import pandas as pd

if __name__== '__main__':

    test_data = pd.read_excel('experiments/yahoo_data.xlsx')
    test_data = torch.Tensor(test_data['Close*'].to_numpy()) #we only predict the closing value

    plot_config = {'T' : len(test_data),
                   'N' : 256,
                   'filename' : 'experiments/financial_data.pdf',
                   'experiment' : 'financial_data'}
    
    plot(test_data, plot_config)

