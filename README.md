# Survival-Random-Forest
SRF for survival analysis and survival prediction.

Survival Random Forest - Example

In this repository it can be found an example of how to find the best parameters of a SRF to minimize the C-Index coeficient.
The parameters to optimize are:
- Max depth
- Max features
- Min samples leaf
- Number of K-folds in the cross validation

The validation method followed to validate the model is a stratified cross validation.
In this particular example, several scenarios that differ from each other in the variables considered are tested.
For each K-fold, a RandomSearchCV is implemented to find the best parameters.
To avoid that the order of the samples influences the results, for each K-fold, the best parameters are calculated for different seed numbers that shuffle the data.

The file "Analytics_RSF.py" helps calculating the mean and standard deviation of each combination of scenario and K-folds to come up with the C-Index range for that case.

The file "Features_RSF.py" performs a permutaion of importance among the variables of each case. The predifined selected cases are the cases of each scenario that resulted in the best C-Index.

Dataset used not disclosed due to data protection policies.
