import os
import pandas as pd
import csv

# Vario il parametro alpha e studio l'andamento degli errori finali

settings = pd.read_csv (r'Data/Input/settings_houses.csv')
os.system('rm -f Data/Output/housing_results.csv')

for i in range(10,0,-1):
	alpha=10**(i/4)
	settings.alpha = alpha
	settings.to_csv('Data/Input/settings_houses.csv', index=False)
	print('alpha	', alpha)
	os.system('python housing.py')

for i in range(20):
	alpha=10**(-i/4)
	settings.alpha = alpha
	settings.to_csv('Data/Input/settings_houses.csv', index=False)
	print('alpha	', alpha)
	os.system('python housing.py')
	
alpha=0
settings.alpha = alpha
settings.to_csv('Data/Input/settings_houses.csv', index=False)
print('alpha	', alpha)
os.system('python housing.py')
