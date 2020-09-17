import os
import pandas as pd
import csv

# Cambio il numero di autovalori eliminati e studio l'andamento degli errori

settings = pd.read_csv (r'Data/Input/settings_houses.csv')
os.system('rm -f Data/Output/housing_results.csv')

for i in range(16):
	settings.PCA = 1
	settings.eigen_delete = i
	settings.to_csv('Data/Input/settings_houses.csv', index=False)
	print('number of eigenvalues deleted:	', i)
	os.system('python housing.py')

