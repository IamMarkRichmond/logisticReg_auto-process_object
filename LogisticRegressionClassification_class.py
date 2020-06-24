# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:50:37 2017

@author: Mark
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

class logRegProcess():
    """ Implements the logistic regression process
    • Calculate Information Value for each variable
	• Train a logistic regression model with each of n top IV variable and calculate the accuracy
	• Automate detection of the ideal number of variables. This can be done possibly with a second order difference method of the n/accuracy curve.
	• Conduct an iterative remove of variables based on p-value
    Also contains funstions to:
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
    """
    def __init__(self, dataset, labels):
        """ initialise the object"""
        self.data = dataset
        columns = list(self.data.columns)[:-1]
        self.columns = columns
        self.labels = labels
        self.int_val = {}
        for i in [0, 1]:
            self.int_val[self.labels[i]]=i
        self.iv_list = []
        self.accuracies = []
        self.steps = []
        self.X = []
        self.y = []
        self.columns_short = []
        self.ideal_n = len(self.columns)-1
        # There will also later be:
        # From function fitModel(self)
#        self.classifier = classifier
#        self.X_train = X_train
#        self.X_test = X_test
#        self.y_train = y_train
#        self.y_test = y_test
        print('A logRegProcess object is born!\n') # Optional celebration

    @staticmethod
    def plot_confusion_matrix(cm,
                              target_names,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot
    
        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
    
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
    
        title:        the text to display at the top of the matrix
    
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues
    
        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
    
        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph
    
        Citiation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
        """
    #    import matplotlib.pyplot as plt
    #    import numpy as np
        import itertools
    
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
    
        if cmap is None:
            cmap = plt.get_cmap('Blues')
    
        plt.figure(figsize=(8, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #    plt.title(title)
        plt.colorbar()
    
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    
        plt.tight_layout()
        plt.ylabel('True condition')
        plt.xlabel('Predicted condition\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    @staticmethod
    def IV_calc(data, var):
        """ Calculates the information value for each variable
        Rule of thumb: <0.02: useless, 0.02 to 0.1: weak, 0.1 to 0.3: medium,
                        0.3 to 0.5: strong, >0.5: suspicious
        Input:
           data - The data is the entire dataframe
           var  - The column name
        output:
            A table of IV values, to be used in the calc_IVlist function
        """
        #import copy
        y = data.columns[-1] # assuming the last column is the dependant variable
        datac = copy.deepcopy(data)
        if datac[var].dtype == 'object':
            dataf = datac.groupby([var])[y].agg(['count', 'sum'])
            dataf.columns = ["Total", "bad"]
            dataf["good"] = dataf["Total"] - dataf["bad"]
            dataf["good"] = dataf["good"] + 1 # To remove devide by zero
            dataf["bad"] = dataf["bad"] + 1
            dataf["bad_per"] = dataf["bad"]/dataf["bad"].sum()
            dataf["good_per"] = dataf["good"]/dataf["good"].sum()
            dataf["I_V"] = (dataf["good_per"] - dataf["bad_per"]) * np.log(dataf["good_per"]/dataf["bad_per"])
            return dataf
        else:
            datac['bin_var'] = pd.qcut(datac[var].rank(method='first'),10)
            dataf = datac.groupby(['bin_var'])[y].agg(['count', 'sum'])
            dataf.columns = ["Total", "bad"]
            dataf["good"] = dataf["Total"] - dataf["bad"]
            dataf["good"] = dataf["good"] + 1 # To remove devide by zero
            dataf["bad"] = dataf["bad"] + 1
            dataf["bad_per"] = dataf["bad"]/dataf["bad"].sum()
            dataf["good_per"] = dataf["good"]/dataf["good"].sum()
            dataf["I_V"] = (dataf["good_per"] - dataf["bad_per"]) * np.log(dataf["good_per"]/dataf["bad_per"])
            return dataf
    
    def calc_IVlist(self, data, columns):
        """ Calculate the information values for all variables
        For each column, calculats the IV and appends it to the list
        
        Input:
            data - a dataframe of all the data
            columns - a list of all column names for the data
        output:
            A list of IV for each column based on deciles
        
        """
        print('Calculating information values')
        iv_list = []
        for col in columns:
            assigned_data = self.IV_calc(data=data, var=col)
            iv_val = round(assigned_data["I_V"].sum(), 3)
            iv_list.append((iv_val, col))
            outstring = ("\r"+'IV for '+str(col)+' of '+str(iv_val)+' added to iv_list')
            sys.stdout.write(outstring)
            sys.stdout.flush()
        del iv_val, col, assigned_data
        iv_list = sorted(iv_list, reverse=True)
        return iv_list

    @staticmethod
    def splitscale(X, y):
        """Splitting the dataset into a training set and a test set and scales
        Input:
            X - indipendant variables in a dataframe
            y - dependant variables in a dataframe
        output:
            X_train - array of independant variables for training
            X_test - array of independant variables for testing
            y_train - array of dependant variables for training
            y_test - array of dependant variables for testing
            *** These may also be dataframes, I'm not sure
            sc_X - The standard scalar object so that future data can be used
        
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = 0.25,
                                                            random_state = 0)
        # Feature scaling
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        return X_train, X_test, y_train, y_test, sc_X

    @staticmethod
    def buildData(dataset, iv_list, step, labels):
        """ From the dataset returns X and y based on a given number of top IV
        Input:
            dataset - the entire dataframe
            iv_list - IV list for each columns
            step - Number of variables to use. Top 'step' variabels
            labels - labels for what 0 and 1 relate to
        output:
            X - Dataframe of independant vars
            y - dataframe of dependant vars, with label names
            columns_short - A reduced list of the columns
        """ 
        columns_short = []
        for i in range(0, step):
            columns_short.append(iv_list[i][1])
        columns_short.append(dataset.columns[-1])
        dataset_short = dataset[columns_short]
    
        X = dataset_short.iloc[:, :-1].values
        y = dataset_short.iloc[:, -1].values
        y = [labels[x] for x in y]
        return X, y, columns_short

    @staticmethod
    def buildData2(dataset, columns_start, labels):
        """ Builds X and y from a reduced list of columns
        Does not use information value, intended to be used after p-value
        reduction.
        Input:
            dataset - the entire dataframe
            columns_short - a list of column names to use
            labels - labels for what 0 and 1 relate to
        output:
            X - Dataframe of independant vars
            y - dataframe of dependant vars, with label names
        """
        # retain top 45 columns, this gives optimal accuracy/variables 
        columns_short = columns_start
#        columns_short.append(dataset.columns[-1])
        dataset_short = dataset[columns_short]
    
        X = dataset_short.iloc[:, :-1].values
        y = dataset_short.iloc[:, -1].values
        y = [labels[x] for x in y]
        return X, y

    @staticmethod
    def trainModel(X_train, y_train, int_val):
        """ Creates and fits a logistic regression classifier
        Input:
            X_train - dataframe of independant training variables
            y_train - dataframe of dependant training variables
            int_val - The oposite of 'labels', which int val the label refers to
        Output:
            classifier - the trained classifier object
        """
        # Fitting Logistic REgression to the Training set
    #    from sklearn.ensemble import RandomForestClassifier
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        #lbfg and saga: convergance warning. Best to use newton-cg
        classifier = LogisticRegression(random_state = 0, solver=solvers[0])
    #    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        y_train = [int_val[i] for i in y_train] # Convert labels to 1 and 0
        classifier.fit(X_train, y_train)
        return classifier
    
    def buildModel(self, X, y):
        """ From X and y, split the data, train the classifier
        input:
            X - Dataframe of independant vars
            y - dataframe of dependant vars, with label names
        output:
            classifier - The trained logreg object from sklearn
            X_train - The training dataset, X
            X_test - The testing dataset, y
            y_train - The training dataset, y
            y_test - The testing dataset, y
        """
        X_train, X_test, y_train, y_test, _ = self.splitscale(X, y)
        int_val = self.int_val
        classifier = self.trainModel(X_train, y_train, int_val)
        return classifier, X_train, X_test, y_train, y_test

    @staticmethod
    def predValues(classifier, X_test, labels):
        """ Predicting the Test set Results
        Input:
            classifier - the trained classifier object
            X_test - the dataframe of independant variables
            labels - what 0 and 1 relate to
        Output:
            y_pred - The predicted results based on the test set
        """
        y_pred = classifier.predict(X_test)
        y_pred = [labels[i] for i in y_pred]
        return y_pred
    
    def calc_acc(self, step):
        """ Calculates the accuracy for a given model
        The classifier is kept as a local variable
        Input:
            step - When building the model, take the top 'step' number of vars
        Output:
            accuracy - The accuracy of prediction as compared to the test data
        """
        dataset = self.data
        iv_list = self.iv_list
        labels = self.labels
        X, y, columns_short = self.buildData(dataset, iv_list, step, labels)
        classifier, _, X_test, _, y_test = self.buildModel(X, y)
        y_pred = self.predValues(classifier, X_test, labels)
        # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        # This plots the accuracy vs variables plot. Uncomment this then tab 158 to 195
        accuracy = np.trace(cm) / float(np.sum(cm))
        return accuracy
    
    def accuracyIVgraph(self, gap=1):
        """ Calculates the values for the accuracy with n_vars plot
        So can see how taking the top IV variables affects accuracy
        Input:
            gap - amount to increment range by. A higher number solves quicker
        Output:
            accuracies - y-axis variable in plot
            steps - x-axis variable in plot
        """
        accuracies = []
        steps = np.arange(5, len(self.columns)+1, gap)
        steps = np.asarray(list(reversed(steps)))
        print('Calculating accuracy from '+str(len(self.columns))+' to 5 variables')
        for step in steps:
            accuracy = self.calc_acc(step)
            accuracies.append(accuracy)
            outstring = ("\r" + 'Completed for '+str(step)+' variables. Accuracy: '+str(round(accuracy, 2)))
            sys.stdout.write(outstring)
            sys.stdout.flush()
        return accuracies, steps

    def plotAccuracies(self):
        """ Plots the model accuracy for taking the top step IV variables
        Input:
            self only
        Output:
            Makes a nice plot
        """
        plt.figure('AccuracyWithVariables')
        plt.plot(self.steps, self.accuracies) 
        plt.xlabel('Number of variables')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.96, wspace=0,
                            hspace=0.22)
        
    
    def plotROC(self):
        """ Plots the Receiver operating characteristic, ROC curve
        Input:
            self only
        Output:
            Makes a nice plot
        """
        classifier = self.classifier
        int_val = self.int_val
        y_test = self.y_test
        X_test = self.X_test
        # plot ROC
        logit_roc_auc = roc_auc_score([int_val[i] for i in y_test], classifier.predict(X_test))
        fpr, tpr, thresholds = roc_curve([int_val[i] for i in y_test], classifier.predict_proba(X_test)[:,1])
        plt.figure('Receiver operating characteristic')
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
#        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.grid()

    def n_selection(self):
        """ From self.accuracies and self.steps, detect drop"""
        for i in range(0, len(self.steps)-1):
            if (self.accuracies[i]-self.accuracies[i+1])>0.01:
                print('\nSetting n_values to '+str(self.steps[i]))
                print('Accuracy is '+str(self.accuracies[i]))
                return self.steps[i]
        print('Delta not exceeded, setting n_values to 25')
        return 25 # This will be updated later

    @staticmethod
    def removeVar(columns_start, var):
        """ From the input columns, removes given column
        Input:
            columns_start - Column list at the start
            var - column index (int) to be removed
        Output:
            columns_new - a list of columns with one removed
        """
        columns_new = []
        for i in range(0, len(columns_start)):
            if i != var:
                columns_new.append(columns_start[i])
        return columns_new

    def itterativePvals(self, max_p=0.05):
        """ Conducts itterative p-value reduction to further reduce variabels
        Input:
            max_p - The limit for highest acceptable p-value
        Output:
            columns_short - A list of only the important columns
            X - dataframe of independant variables
            y - dataframe of dependant variables
            logit_model - a logit object from which information can be taken
        """
        iv_list = self.iv_list
        step = self.ideal_n
        dataset = self.data
        labels = self.labels
        X, y, columns_short = self.buildData(dataset, iv_list, step, labels)
        y_int = [self.int_val[i] for i in y]
        print('\n\nLogit output:')
        logit_model = sm.Logit(y_int, X).fit()
#        print(logit_model.summary2())
        pvals = logit_model.pvalues
        while max(pvals) > max_p:
            for j in range(0, len(pvals)):
                if pvals[j] == max(pvals):
                    print('\nRemoving '+str(columns_short[j])+' with p-value '+str(pvals[j]))
                    i = j
            columns_short = self.removeVar(columns_short, i)
            X, y = self.buildData2(dataset, columns_short, labels)
            y_int = [self.int_val[i] for i in y]
#                    print(columns_short)
            print('\nLogit output:')
            logit_model = sm.Logit(y_int, X).fit()
#                    print(logit_model.summary2())
            pvals = logit_model.pvalues
            if max(pvals) < max_p:
                break
        print('\nAll p-values below '+str(max_p))
        return columns_short, X, y, logit_model

    def reduceVars(self):
        """ Applies IV reduction then itterative p-value
        Input:
            Just self
        Output:
            Just self, assigns values to a number of global variables
        """
        if self.iv_list:
            print('Information value list already present, recalculating')
        self.iv_list = self.calc_IVlist(self.data, self.columns) # This should be part of the object
        print('\n\nInformation value list calculated')
        self.accuracies, self.steps = self.accuracyIVgraph(gap=10)
        self.ideal_n = self.n_selection()
        self.columns_short, self.X, self.y, self.logit_model = self.itterativePvals()

    def calcualteCM(self):
        """ Calculates the confusion matrix
        Input:
            Just self
        Output:
            Just self
        """
        self.y_pred = self.predValues(self.classifier, self.X_test, self.labels)
        self.cm = confusion_matrix(self.y_test, self.y_pred, labels=self.labels)

    def fitModel(self):
        """ Fits a logistic regression to the reduced list of variables
        Input:
            Just self
        Output:
            Just self
        """
        if not self.columns_short:
            print('Must first reduce columns with method reduceVars')
            return
        print('\nFitting optimised model with '+str(len(self.columns_short))+' Variables.')
        X = self.X
        y = self.y
        classifier, X_train, X_test, y_train, y_test = self.buildModel(X, y)
        self.classifier = classifier
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.calcualteCM()

    def printReport(self):
        """ Prints the classification report"""
        print(classification_report(self.y_test, self.y_pred))

    def plotCM(self):
        """ Plots the confusion matrix"""
        self.plot_confusion_matrix(self.cm, target_names=self.labels)
        plt.ylim([len(labels)-.5, -0.5])
        plt.subplots_adjust(left=0.18, bottom=0.14, right=0.99, top=0.96,
                            wspace=0, hspace=0.22)

    def returnNodes(self):
        """ Returns a list of tower nodes"""
        nodes = list(set([x[5:-2] for x in self.columns_short if len(x)>6]))
        return nodes

    def logitSummary(self):
        """ Prints a summary of the logit model"""
        print(self.logit_model.summary2())

if __name__ == '__main__':
    # logistig regression Classification process
    # Set directory
    rootdir = r'F:\Ramboll\5_DamageClassification'

    # Import the data set
    dataset = pd.read_csv(rootdir+'\damage_flattened_threeDamages.csv',
                          index_col=0, low_memory=False)

    # Convert the last column to 0 and 1 
    labels = ['healthy', 'damaged']
    temp_col = dataset.iloc[:, -1].values
    for i in range(0, len(temp_col)):
        if temp_col[i] == labels[0]:
            temp_col[i] = 0
        else:
            temp_col[i] = 1
    dataset.iloc[:, -1] = temp_col
    del temp_col, i

    # Create and run object
    test = logRegProcess(dataset, labels)
    test.reduceVars()
    test.plotAccuracies()
    test.fitModel()
    test.printReport()
    test.plotCM()
    test.plotROC()
    test.returnNodes()
    test.logitSummary()