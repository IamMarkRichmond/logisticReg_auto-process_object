This code can be used to create and object which automatically conducts a logistic regression analysis
for a given set of data. It can be useful to automate it in this way if one wishes to do this for multiple
dataset. For example, classying a structure into 'healthy' or damaged for many different damages or nacelle
directions.

Straight forward to use and I think quite generally applicable provided the dataset is a dataframe object
and the classes being predicted are in the form of 0 or 1.

The code does the following things:	
	• Calculate Information Value for each variable
	• Train a logistic regression model with each of n top IV variable and calculate the accuracy
	• Automate detection of the ideal number of variables. This can be done possibly with a second order difference method of the n/accuracy curve.
	• Conduct an iterative remove of variables based on p-value
Also contains functions to:
	• Return ideal variables list in order of descending IV
	• Return model details (summary of model)
	• Return model accuracy metrics
	• Produce ROC and CM figures
	• Make predictions on new data * not yet implemented, but should be easy

This should be used the same as in main where the two inputs are given and
then the functions are called in that order. I'm planning on adding more
flexability later, but right now there isn't much.
    
input:
        Dataset - this is the data where the last column is the dependant 
                  variable and there are column names. The end column should
                  be in 0 and 1 int values, this makes the model more 
                  consistent.
        Labels - This is what 0 and 1 relate to, which makes interpretation
                 easier. 

output:
               - an object with all the LR analysis already conducted. See 
                 bullets at the top.