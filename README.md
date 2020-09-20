Housing Code

In this folder 5 python scripts are provided:
1) housing.py
2) PCA_test.py
3) seed_test.py
4) alpha_test.py
5) settings_setup.py

The first one is the one that really matters, as it holds the code that applies the Ridge Regression algorithm to the given dataset in order to predict the housing median values in the Northern California area.
The script reads data and the initial settings from the files named "cal-housing.py" and "settings_housing.csv", both located in the repository "Data/Input", and writes the final results in the repository "Data/Output".
In particular, the predicted values are stored together with the correct ones in files "test.csv" and "training.csv" and, whenever PCA is performed, the eigenvalues of the covariance matrix and their percentage on the total sum are saved in "PCA_eigenvalues.csv".
Moreover, in file "housing_results.csv" the initial settings are appended with the final training and test errors every time the algorithm is performed and the same goes with file "predictor.csv" with the predictor's elements.

The initial settings are:
1) Random seed (the results shown in the report have been obtained by seed=42)
2) Drop NaN of total bedrooms (1 for yes, 0 for substitution with the median value)
3) Drop housing values above 500k (1 for yes, 0 for no)
4) Shuffle of the dataset (1 for yes, 0 for no)
5) Dummy Coding as a substitution of the categorical feature "ocean_proximity" (1 for yes, 0 for substitution, any other option for the deletion of the feature without any substitution)
6) Rearragement of the dataset (1 for yes, 0 for no)
7) Value of hyperparameter alpha
8) Number of subsets K 
9) PCA (1 for yes, 0 for no)
10) number of deleted eigenvalues in case of PCA

The settings can be changed by hand and the best solution has been found by fixing the following ones:

1) Seed = 42
2) Drop NaN = 1
3) Drop High = 0
4) Shuffle = 1
5) Dummy Coding = 1
6) Rearrangement = 1
7) alpha = 0.001
8) K = 10
9) PCA = 1
10) Deleted eigenvalues = 1

The following 3 python scripts ran the original script multiple times by changing the settings and in particular:
1) PCA_test.py performs PCA and changes many times the number of deleted eigenvalues 
2) seed_test.py runs multiple simulations by changing the seed
3) alpha_test.py runs multiple simulations by changing the hyperparameter alpha

In all of these cases, the files "housing_results.csv" and "predictor.csv" are deleted before starting the multiple simulations.

Finally, settings.setup.py only changes file "settings_housing.csv" with the best solution previously stated.

