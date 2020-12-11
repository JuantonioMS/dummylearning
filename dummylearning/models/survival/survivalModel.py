import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit Survival
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.linear_model import CoxnetSurvivalAnalysis # Elastic-net and Lasso Regression Model
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Scikit Optimize
from sklearn.model_selection import GridSearchCV, KFold
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

# Scikit Learn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from dummylearning.utilities.info import Info

class SurvivalModel(Info):
    """
    Clase Padre de modelos
    """

    def __init__(self, data, verbose = True):

        super().__init__(verbose)

        self.data = data
        self.model = None



    def optimizePenalty(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import KFold
        from sklearn.model_selection import GridSearchCV

        pipeline = make_pipeline(self.model)
        warnings.simplefilter("ignore", ConvergenceWarning)

        pipeline.fit(self.data.values, self.data.tags)

        alphas = 10. ** np.linspace(-2, 3,50)

        cv = KFold(n_splits = 5, shuffle = True)

        grid = GridSearchCV(make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=1.0, max_iter = 1000000)),
                            param_grid = {"coxnetsurvivalanalysis__alphas" : [[alpha] for alpha in alphas]},
                            cv = cv,
                            error_score = 0.5,
                            n_jobs = -1).fit(self.data.values, self.data.tags)

        bestAlpha = grid.best_params_["coxnetsurvivalanalysis__alphas"][0]

        print("El mejor pare!", bestAlpha)
        self.model.set_params(**{"alphas" : [bestAlpha]})




    def bayesianOptimization(self, searchSpace, calls = 50):

        from skopt.utils import use_named_args

        self.upgradeInfo(f"Developing bayesian optimization (calls = {calls})")

        @use_named_args(searchSpace)
        def functionToOptimize(**params):

            if type(self).__name__ == "CoxLasso":
                model = CoxnetSurvivalAnalysis(l1_ratio = 1.0)
                params["alphas"] = [2 ** params["alphas"]]

            elif type(self).__name__ == "CoxRidge":
                model = CoxPHSurvivalAnalysis()

            elif type(self).__name__ == "CoxElasticNet":
                model = CoxnetSurvivalAnalysis()
                params["alphas"] = [params["alphas"]]

            model.set_params(**params)

            concordances = []
            for trainIndex, testIndex in KFold(n_splits = 4).split(self.data.values):

                trainX, trainY = self.data.values[trainIndex,], self.data.tags[trainIndex[:, None],]
                testX, testY = self.data.values[testIndex,], self.data.tags[testIndex[:, None],]

                trainY, testY = np.reshape(trainY, -1), np.reshape(testY, -1)

                model.fit(trainX, trainY)
                concordances.append(model.score(testX, testY))

            import random
            result =  -np.mean(concordances) + ((random.random() - 0.5) / 1000000000000)
            print(result)
            return result

        results = gp_minimize(functionToOptimize,
                              searchSpace,
                              n_random_starts = 10,
                              n_initial_points = 10,
                              n_calls = calls,
                              verbose = True)

        parameters = dict()
        for parameter, value in zip(searchSpace, results.x):

            if parameter.name != "alphas":
                parameters[parameter.name] = value

            else:
                parameters[parameter.name] = [2 ** value]

        self.model.set_params(**parameters)
        print("Optimization Finished")
        print(parameters)




    def runProductionModel(self):

        self.model.fit(self.data.values, self.data.tags)

        self.dataset = dict()
        self.dataset["Production"] = dict()
        self.dataset["Production"]["values"] = self.data.values
        self.dataset["Production"]["tags"] = self.data.tags

        #self.times = list(set(self.data.tags["Survival_in_days"]))
        #self.times.sort()
        #self.aucList, self.meanAuc = cumulative_dynamic_auc(self.data.tags, self.data.tags,
                                                            #self.model.predict(self.data.values),
                                                            #self.times[1:-1])




    def runClassicModel(self):

        trainX, testX, trainY, testY = train_test_split(self.data.values,
                                                        self.data.tags,
                                                        test_size = 0.2)

        self.model.fit(trainX, trainY)

        self.dataset = dict()
        self.dataset["Train"] = dict()
        self.dataset["Train"]["values"] = trainX
        self.dataset["Train"]["tags"] = trainY

        self.dataset["Test"] = dict()
        self.dataset["Test"]["values"] = testX
        self.dataset["Test"]["tags"] = testY

        self.dataset["Complete"] = dict()
        self.dataset["Complete"]["values"] = self.data.values
        self.dataset["Complete"]["tags"] = self.data.tags

        #self.times = list(set(testY["Survival_in_days"]))
        #self.times.sort()
        #self.aucList, self.meanAuc = cumulative_dynamic_auc(testY, testY,
                                                            #self.model.predict(testX),
                                                            #self.times[1:-1])

        #print(self.meanAuc)
