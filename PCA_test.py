import os
import pandas as pd
import csv

# Cambio il numero di autovalori eliminati e studio l'andamento degli errori

settings = pd.read_csv (r'Data/Input/settings_houses.csv')
os.system('rm -f Data/Output/housing_results.csv')
os.system('rm -f Data/Output/predictor.csv')

settings.seed = 42
settings.drop_na = 1
settings.drop_high = 0
settings.shuffle = 1
settings.dummy_coding = 1
settings.ratio = 1
settings.alpha = 0.001
settings.K=10
settings.PCA = 1

for i in range(16):
	settings.eigen_delete = i
	settings.to_csv('Data/Input/settings_houses.csv', index=False)
	print('number of eigenvalues deleted:	', i)
	os.system('python housing.py')

