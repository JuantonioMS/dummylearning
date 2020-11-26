from dummylearning.model.classificationModel import ClassificationModel
from sklearn.ensemble import RandomForestClassifier

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

class RandomForest(ClassificationModel):

    """
    Random Forest class

    Parameters
    ---------------------------------------------------------------------------
        data <Data> (positional) => Data instance containg X and Y values

    Attributes
    ---------------------------------------------------------------------------
        model <sklearn.ensemble.RandomForestClassifier> => Random Forest model

    Methods
    ---------------------------------------------------------------------------
        bayessianOptimization => Optimization method for parameter setting
    """

    def __init__(self, data, verbose = True):
        super().__init__(data, verbose)

        self.model = RandomForestClassifier(n_jobs = -1)


    def bayessianOptimization(self, calls = 20):

        """
        Function -> bayessianOptimization
        Bayessian method for setting best parameters. Here we only set de search
        space (diferent for each model)

        Parameters
        ---------------------------------------------------------------------------
            calls <int> (default: 20) => Iteration number for bayessian process

        Info
        ---------------------------------------------------------------------------
            Params search space => n_estimators      (100:1000)
                                   max_depth         (1:20)
                                   min_samples_split (2:10)
                                   min_samples_leaf  (1:10)
                                   oob_score         (True:False)

        Return
        ---------------------------------------------------------------------------
            None => It calls super().bayessianOptimization
        """

        searchSpace = [Integer(100, 1000, name = "n_estimators"),
                       Integer(1, 20, name = "max_depth"),
                       Integer(2, 10, name = "min_samples_split"),
                       Integer(1, 10, name = "min_samples_leaf"),
                       Categorical([True, False], name = "oob_score")
                       ]

        super().bayessianOptimization(searchSpace, calls)