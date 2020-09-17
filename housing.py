import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import scipy.optimize, scipy.stats
import csv
import os



# ************************************ Funzioni ************************************

def dummy_coding(df):
	NEAR_BAY_array = np.zeros(len(df.index))										# metodo DUMMY VARIABLES... errore si abbassa!
	NEAR_OCEAN_array = np.zeros(len(df.index))
	ISLAND_array = np.zeros(len(df.index))
	H_OCEAN_array = np.zeros(len(df.index)) 
	INLAND_array = np.zeros(len(df.index))

	for i in range(len(df.index)):
		if df.loc[i, 'ocean_proximity'] == 'NEAR BAY':
			NEAR_BAY_array[i] = 1
		elif df.loc[i, 'ocean_proximity'] == 'NEAR OCEAN':
			NEAR_OCEAN_array[i] = 1
		elif df.loc[i, 'ocean_proximity'] == '<1H OCEAN':
			H_OCEAN_array[i] = 1
		elif df.loc[i, 'ocean_proximity'] == 'ISLAND':
			ISLAND_array[i] = 1
		elif df.loc[i, 'ocean_proximity'] == 'INLAND':
			INLAND_array[i] = 1

	df.insert(len(df.columns), "NEAR BAY", NEAR_BAY_array)
	df.insert(len(df.columns), "NEAR OCEAN", NEAR_OCEAN_array)
	df.insert(len(df.columns), "<1H OCEAN", H_OCEAN_array)
	df.insert(len(df.columns), "ISLAND", ISLAND_array)
	df.insert(len(df.columns), "INLAND", INLAND_array)

	df = df.drop(columns=['ocean_proximity']).reset_index(drop=True)
	return df
	
def new_dataset(df):
	df.insert(len(df.columns), "rooms_population_rate", (df.total_rooms/df.population))
	df.insert(len(df.columns), "bedrooms_population_rate", (df.total_bedrooms/df.population)) #+ efficace di rooms
	df.insert(len(df.columns), "bedrooms_rooms_rate", (df.total_bedrooms/df.total_rooms))
	df.insert(len(df.columns), "households_population_rate",(df.households/df.population)) #++ efficace
	df.insert(len(df.columns), "households_bedrooms_rate", (df.households/df.total_bedrooms))
	return df.drop(columns=['total_bedrooms', 'population']).reset_index(drop=True)


def square_loss(v_emp, v_label):							# errore, serve array delle predizioni e dei label preassegnati
	return np.linalg.norm(v_emp - v_label)**2
	
def cost(predictor_w, matrix_S, label_y, alpha):
	return square_loss(matrix_S.dot(predictor_w),label_y) + alpha*np.linalg.norm(predictor_w)**2			# ||Sw-y||^2 + a||w||^2

def empirical_risk(predict_y, label_y):
	return square_loss(predict_y,label_y)/len(label_y)

def ridge_regression(matrix_S, label_y, alpha):
	A_matrix = (matrix_S.transpose()).dot(matrix_S) + alpha*np.identity(len(matrix_S.columns))
	B_vector = (matrix_S.transpose()).dot(label_y)
	return scipy.linalg.solve(A_matrix, B_vector)

def predictor(vector_w, matrix_S, mean, std):		# applico il predictor, riduco l'errore mettendo dei limiti a 0 e 50
	pred = matrix_S.dot(vector_w)		
	max_value = (50.0001-mean)/std
	min_value = -mean/std
	for i in range(len(pred)):
		if pred[i] < min_value:
			pred[i] = min_value
		elif pred[i] > max_value:
			pred[i] = max_value
	
	return pred			
 
def print_output(settings, risk_average_training, risk_average_test):
	print('\n' + "Ridge Regression For Housing Prices Prediction")
	print('\n'+ "Settings:" + '\n')
	for c in settings.columns:
		print(c + ":	" + str(settings.loc[0,c]))
	print('\n'+"Final Errors:")
	print("Training Error:	" + str(risk_average_training))
	print("Test Error:	" + str(risk_average_test) + '\n')






# ************************************ Lettura dati in input e preparazione variabili ************************************

df = pd.read_csv ('Data/Input/cal-housing.csv')							# leggo dati
settings = pd.read_csv ('Data/Input/settings_houses.csv')		# impostazioni necessarie (es: parametro alpha, K...)

# ******************

if settings.loc[0, 'drop_na']==1:			
	df = df.dropna()																# elimino direttamente le righe con dati mancanti 
else:																							# oppure riempio nan con valori di mediana per ogni colonna... 
	df = df.fillna(df.total_bedrooms.median())															

if settings.loc[0, 'drop_high']==1:
	df = df.drop(df[df.median_house_value > 5e5].index).reset_index(drop=True)	# elimino case con valore superiore a 500k 

if settings.loc[0, 'shuffle']==1:										
	df = df.sample(frac=1, random_state=settings.loc[0,'seed']).reset_index(drop=True)		# shuffle dei dati, seed fissato esternamente


house_value = df.median_house_value								# valore mediano di una casa in dollari per isolato --> è il label!!!
df = df.drop(columns=['median_house_value'])			# NB: non può stare nella matrice dei dati! 
house_value /= 10000															# riscalato in decine di migliaia di dollari (come guadagno medio delle famiglie)

df_copy = df.copy()																# serve per stampare i dati alla fine

# *******************

if settings.loc[0, 'ratio']==1:										# aggiungo nuove features e ed elimino quelle vecchie
	df = new_dataset(df)
	
if settings.loc[0, 'dummy_coding']==1:						# gestione delle variabili categoriche
	df = dummy_coding(df)
elif settings.loc[0, 'dummy_coding']==0:
	df['ocean_proximity'] = df['ocean_proximity'].replace(["NEAR BAY", "NEAR OCEAN", "ISLAND", "<1H OCEAN", "INLAND"],[10**2, 10**1, 10**0, 10**3, 10**4])					# ESPONENZIALE nella distanza dall'oceano...	
else:
	df = df.drop(columns=['ocean_proximity']).reset_index(drop=True)		# posso anche scegliere di non utilizzarla...

	# standardizzo i parametri... utile per confronto con PCA
for column in df:
	df[column] = (df[column] - df[column].mean())/(df[column].std()) 
  

value_mean = house_value.mean()								# devo rinormalizzare anche i label! Mi salvo media e std per ottenere previsioni alla fine...
value_std = house_value.std()
house_value = (house_value - value_mean)/(value_std)






# ******************** PCA ********************

if settings.loc[0, 'PCA']==1:
	cov_matrix = df.cov()
	e_value, e_vector = np.linalg.eig(cov_matrix)
	e_sum = e_value.sum()
	e_percentage = (e_value/e_sum)*100
	eigen_out = pd.DataFrame({'eigenvalues':e_value, 'percentage':e_percentage})
	eigen_out.to_csv("Data/Output/PCA_eigenvalues.csv")
	
	for i in range(settings.loc[0,'eigen_delete']):				# eliminazione degli autovalori deve essere una scelta esterna! 
		e_vector = np.delete(e_vector, -1, 1)

	df = df.dot(e_vector)






# ************************************ Cross Validation e Soluzione ************************************

alpha=settings.loc[0, 'alpha']															

data_size = len(house_value)						# numero di righe (m)
data_dim = len(df.columns)							# numero di colonne (d)
K = settings.loc[0, 'K']								# numero di sottoinsiemi di guale lunghezza
subset = int(data_size/K)
percentage = 1 - 1/K										# percentuale training sul totale

risk_average_training, risk_average_test = 0, 0
for k in range(K):
	if k == K-1:													# attenzione se K non è un divisore di data_size!
		test_labels = (house_value[subset*k :]).reset_index(drop=True)					
		test_matrix = (df.iloc[subset*k:, :]).reset_index(drop=True)		
		training_labels = (house_value.drop(house_value.index[subset*k :])).reset_index(drop=True)												
		training_matrix = df.drop(df.index[subset*k:]).reset_index(drop=True)
	else:
		test_labels = (house_value[subset*k : subset*(k+1)]).reset_index(drop=True)					# divido i dati e i label in test e training
		test_matrix = (df.iloc[subset*k:subset*(k+1), :]).reset_index(drop=True)						# attenzione, serve resettare il numero di righe!
		training_labels = (house_value.drop(house_value.index[subset*k : subset*(k+1)])).reset_index(drop=True)												
		training_matrix = df.drop(df.index[subset*k:subset*(k+1)]).reset_index(drop=True)

	final_w = ridge_regression(training_matrix, training_labels, alpha)
	
	training_labels = (training_labels)*value_std + value_mean
	test_labels = (test_labels)*value_std + value_mean
	
	training_prediction = predictor(final_w, training_matrix, value_mean, value_std)*value_std + value_mean
	test_prediction = predictor(final_w, test_matrix, value_mean, value_std)*value_std + value_mean
	
	risk_average_training = (risk_average_training*k + empirical_risk(training_prediction, training_labels))/(k+1)		# media aggiornata
	risk_average_test = (risk_average_test*k + empirical_risk(test_prediction, test_labels))/(k+1)			






# ************************************ Stampa Risultati ************************************
# stampo solo ultima iterazione, tanto con shuffle iniziale sono tutte equivalenti...

print_output(settings, risk_average_training, risk_average_test)	# stampo impostazioni e risultati a video

with open("Data/Output/predictor.txt", "w") as w_out:							#	salvo il predictor finale
	for w_elem in final_w:
		w_out.write(str(w_elem)+'\n')

test_df = (df_copy.iloc[subset*(K-1):, :]).reset_index(drop=True)
test_df.insert(len(test_df.columns), "correct_labels", test_labels)
test_df.insert(len(test_df.columns), "predicted_labels", test_prediction)
test_df.to_csv('Data/Output/test.csv')												# dataset "originale" (dopo drop_na, drop_high e shuffle) + test

training_df = df_copy.drop(df_copy.index[subset*(K-1):]).reset_index(drop=True)
training_df.insert(len(training_df.columns), "correct_labels", training_labels)
training_df.insert(len(training_df.columns), "predicted_labels", training_prediction)
training_df.to_csv('Data/Output/training.csv')								# dataset "originale" (dopo drop_na, drop_high e shuffle) + training


# aggiungo alle impostazioni i risultati finali e stampo su un nuovo file, aggiungendo di volta in volta
settings.insert(len(settings.columns), 'training_error', risk_average_training)
settings.insert(len(settings.columns), 'test_error', risk_average_test)
if os.path.exists('Data/Output/housing_results.csv'):
	settings.to_csv('Data/Output/housing_results.csv', mode='a', header=False, index=False)
else:
	settings.to_csv('Data/Output/housing_results.csv', mode='w', header=True, index=False)

