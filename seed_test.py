import os
import pandas as pd
import csv

# Modifico il seed e verifico che gli errori non oscillino troppo...

settings = pd.read_csv (r'Data/Input/settings_houses.csv')
os.system('rm -f Data/Output/housing_results.csv')
os.system('rm -f Data/Output/predictor.csv')

settings.drop_na = 1
settings.drop_high = 0
settings.shuffle = 1
settings.dummy_coding = 1
settings.ratio = 1
settings.alpha = 0.001
settings.K=10
settings.PCA = 1
settings.eigen_delete = 1

for i in range(100):
	settings.seed = i
	settings.to_csv('Data/Input/settings_houses.csv',index=False)
	print('seed	', i)
	os.system('python housing.py')

final_results = pd.read_csv(r'Data/Output/housing_results.csv')
print('Average Training Error:	', final_results.training_error.mean())
print('Average Test Error:	', final_results.test_error.mean())
