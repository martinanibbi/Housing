import os
import pandas as pd
import csv

# Modifico il seed e verifico che gli errori non oscillino troppo...

settings = pd.read_csv (r'Data/Input/settings_houses.csv')
os.system('rm -f Data/Output/housing_results.csv')

for i in range(100):
	settings.seed = i
	settings.to_csv('Data/Input/settings_houses.csv',index=False)
	print('seed	', i)
	os.system('python housing.py')

final_results = pd.read_csv(r'Data/Output/housing_results.csv')
print('Average Training Error:	', final_results.training_error.mean())
print('Average Test Error:	', final_results.test_error.mean())
