import os
import pandas as pd
import csv

settings = pd.read_csv (r'Data/Input/settings_houses.csv')

settings.seed = 42
settings.drop_na = 1
settings.drop_high = 0
settings.shuffle = 1
settings.dummy_coding = 1
settings.ratio = 1
settings.alpha = 0.001
settings.K=10
settings.PCA = 1
settings.eigen_delete = 1

settings.to_csv('Data/Input/settings_houses.csv', index=False)

