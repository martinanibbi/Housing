import os
import pandas as pd
import csv

# Vario il parametro alpha e studio l'andamento degli errori finali

settings = pd.read_csv (r'Data/Input/settings_houses.csv')
os.system('rm -f Data/Output/housing_results.csv')
os.system('rm -f Data/Output/predictor.csv')

settings.seed = 42
settings.drop_na = 1
settings.drop_high = 0
settings.shuffle = 1
settings.dummy_coding = 1
settings.ratio = 0
settings.K=10
settings.PCA = 0
settings.eigen_delete = 1


for i in range(30,0,-3):
	alpha=10**(i/4)
	settings.alpha = alpha
	settings.to_csv('Data/Input/settings_houses.csv', index=False)
	os.system('python housing.py')

for i in range(10):
	alpha=10**(-i/4)
	settings.alpha = alpha
	settings.to_csv('Data/Input/settings_houses.csv', index=False)
	os.system('python housing.py')
	
alpha=0
settings.alpha = alpha
settings.to_csv('Data/Input/settings_houses.csv', index=False)
os.system('python housing.py')
