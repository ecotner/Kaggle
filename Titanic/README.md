# Titanic Survival Prediction

Introductory machine learning competition: predict who would survive the sinking of the Titanic based on some data (age, sex, name, ticket number, family members, etc).

## To do:
- Use clustering or regression to predict missing data values, or find some way of safely handling data with missing values
- Use k-fold cross-validation to get better idea of model generalization
- Train on combined train/validation set after selecting model
- Try a neural network and see if nonlinear interactions give any kind of performance boost
- Figure out what XGBoost is and try to incorporate it
- Write a script to automatically search hyperparameter space (just use random sampling for now)

## Submissions
- First submission: an ensemble of SVM's with different kernels. Handled missing data by simply replacing with the average over training examples which were not missing data. Result: 0.78468 accuracy on the test set. Not very impressive, but there is a lot to do to improve.