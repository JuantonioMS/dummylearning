import pandas as pd
import numpy as np
import time

# Scikit Optimize
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

# Scikit Learn
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from ..info import Info

class ClassificationModel(Info):

    """
    Main Classification Model class:
    Classification methods's father. It contains common methods and attributes
    for usual classifications methods.

    Parameters
    ---------------------------------------------------------------------------
        data  <Data> (positional)  => Data instance with data information

    Attributes
    ---------------------------------------------------------------------------
        verbose    <bool> => Flag, if False, no info saved or printed
        data       <Data> => Data instance with data information
        report     <dict> => Dictionary with process info (value) and time (key)
        model      <None> => Inheritance. Scikit Learn model
        balanceObj <None> => Not None if self.balanceData() called

    Methods
    ---------------------------------------------------------------------------

        *Optimization__________________________________________________________
            balanceData           =>
            bayessianOptimization => Set model parameters optimized with
                                     bayessian method
                                     !WARNING implement overfitting regulations
                                     TODO implement option for deleting
                                          parameters to optimize

        *Run Model_____________________________________________________________
            runClassicModel    => Fit model with train vs. test strategy
            runProductionModel => Fit model with full dataset
            runCvModel         => TODO implement

        *Process Info__________________________________________________________
            upgradeInfo => Store process info
            getInfo     => Merge self.report dictionaries and return info <str>

        *Metrics_______________________________________________________________
            calculateMetrics => Calculate scores with confussion matrix
            getMetrics       => Merge scores. Printable str is returned

        *Ploting_______________________________________________________________
            plotCoefs            => Generate one non-zero coefs plot by class
            plotConfussionMatrix => Generate one confussion matrix by dataset
            plotRocCurve         => Generate one ROC curve by class and dataset

        *Model Info____________________________________________________________
            saveCoefs     => Save coefficients into csv file
            getParameters => Get parameters info

        *Model Report__________________________________________________________
            getMetricsReport    => Model metrics with md format
            getParametersReport => Model parameters with md format
            getInfoReport       => Process info with md format
            generateReport      => Generate md format report file
    """




    def __init__(self, data, verbose = True):

        super().__init__(verbose)
        self.upgradeInfo("Creating ClassificationModel instance")

        self.data = data # Data instance

        self.model = None # Only for avoid pylint warnings
                          #     Overwritten with inheritance

        self.balanceObj = None











    #_____________________________Optimization Section_____________________________




    def balanceData(self, method: str = "mixsampling") -> None:

        """
        Function -> balanceData
        Balance data classes wiht method selected

        Parameters
        ---------------------------------------------------------------------------
            method => mixsampling, undersampling or oversampling

        Return
        ---------------------------------------------------------------------------
            None => Modify self.balanceObj
        """

        if method == "mixsampling":
            from imblearn.combine import SMOTETomek
            self.balanceObj = SMOTETomek(sampling_strategy='auto')

        elif method == "undersampling":
            from imblearn.under_sampling import NearMiss
            self.balanceObj = NearMiss(sampling_strategy= "auto", n_neighbors=3, version=2)

        elif method == "oversampling":
            from imblearn.over_sampling import RandomOverSampler
            self.balanceObj = RandomOverSampler(sampling_strategy = "auto")

        else:
            raise NameError(f"{method} method not defined")




    def bayessianOptimization(self, searchSpace, calls):
        #! Solve overfitting problem
        self.upgradeInfo("Performing bayessian optimization")

        @use_named_args(searchSpace)
        def functionToOptimize(**params):

            self.model.set_params(**params)

            cvAucMeans = []
            for trainIndex, testIndex in StratifiedKFold(n_splits = 4,
                                                         shuffle = False).split(self.data.values,
                                                                                self.data.tags):

                trainX, trainY = self.data.values[trainIndex,], self.data.tags[trainIndex,]
                testX, testY = self.data.values[testIndex,], self.data.tags[testIndex,]

                self.model.fit(trainX, trainY)

                accuracy = accuracy_score(testY, self.model.predict(testX))
                cvAucMeans.append(accuracy)

            return -np.mean(cvAucMeans)

        results = gp_minimize(functionToOptimize,
                              searchSpace,
                              n_calls = calls,
                              verbose = False)

        parameters = dict()
        for parameter, value in zip(searchSpace, results.x):

            parameters[parameter.name] = value

        self.model.set_params(**parameters)
        print("Optimization Finished")








    #______________________________Run Model Section_______________________________




    def runClassicModel(self, testRatio = 0.2):
        self.upgradeInfo(f"Running classic model (train {1 - testRatio} vs. test {testRatio})")

        trainX, testX, trainY, testY = train_test_split(self.data.values,
                                                        self.data.tags,
                                                        test_size = testRatio)

        self.upgradeInfo("Training model")

        if self.balanceObj:
            balanceX, balanceY = self.balanceObj.fit_sample(trainX, trainY)
            self.model.fit(balanceX, balanceY)
        else:
            self.model.fit(trainX, trainY)


        self.dataset = dict()

        self.upgradeInfo("Saving training dataset")
        self.dataset["Train"] = {"values" : trainX,
                                 "tags"   : trainY}

        self.upgradeInfo("Saving test dataset")
        self.dataset["Test"] = {"values" : testX,
                                "tags"   : testY}

        self.upgradeInfo("Saving complete dataset")
        self.dataset["Complete"] = {"values" : self.data.values,
                                    "tags"   : self.data.tags}

        if self.balanceObj:
            self.upgradeInfo("Saving balanced dataset")
            self.dataset["Balanced"] = {"values" : balanceX,
                                        "tags"   : balanceY}




    def runProductionModel(self):
        self.upgradeInfo("Runing production model")

        self.upgradeInfo("Training model")

        if self.balanceObj:
            balanceX, balanceY = self.balanceObj.fit_sample(self.data.values, self.data.tags)
            self.model.fit(balanceX, balanceY)
        else:
            self.model.fit(self.data.values, self.data.tags)

        self.dataset = dict()

        self.upgradeInfo("Saving complete dataset")
        self.dataset["Production"] = {"values" : self.data.values,
                                      "tags"   : self.data.tags}

        if self.balanceObj:
            self.upgradeInfo("Saving balanced dataset")
            self.dataset["Balanced"] = {"values" : balanceX,
                                        "tags"   : balanceY}




    def runCvModel(self, testRatio = 0.2):
        pass




    #______________________________Model Info Section______________________________




    def saveCoefs(self, outfile):

        column = dict()

        if len(self.model.classes_) == 2:

            for index, label in enumerate(self.model.classes_):
                if index == 0:
                    column[label] = -self.model.coef_[0]
                else:
                    column[label] = self.model.coef_[0]

            auxData = pd.DataFrame(data = column,
                                   index = self.data.valuesName)
            auxData.loc["intercept"] = [-self.model.intercept_[0], self.model.intercept_[0]]

        else:

            for index, label in enumerate(self.model.classes_):
                column[label] = self.model.coef_[index]

            auxData = pd.DataFrame(data = column,
                                   index = self.data.valuesName)
            auxData.loc["intercept"] = self.model.intercept_

        auxData.to_csv(outfile, sep = ";")

