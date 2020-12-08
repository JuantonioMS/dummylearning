from dummylearning.models.classification.classificationModel import ClassificationModel
from sklearn.linear_model import LogisticRegression
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

class LogisticElasticNet(ClassificationModel):

    """
    Logistic regression with ElasticNet penalty class

    Parameters
    ---------------------------------------------------------------------------
        data <Data> (positional) => Data instance containg X and Y values

    Attributes
    ---------------------------------------------------------------------------
        model <sklearn.linear_model.LogisticRegression> => Logistic Regression
                                                           model setted with
                                                           elasticnet penalty
                                                           and saga solver

    Methods
    ---------------------------------------------------------------------------
        bayessianOptimization => Optimization method for parameter setting
    """

    def __init__(self, data):
        super().__init__(data)

        self.model = LogisticRegression(penalty = "elasticnet",
                                        solver = "saga",
                                        l1_ratio = 0.5,
                                        n_jobs = -1)


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
            Params search space => C        (0.001:100)
                                   l1_ratio (0.0, 1.0)

        Return
        ---------------------------------------------------------------------------
            None => It calls super().bayessianOptimization
        """

        searchSpace = [Real(0.001, 100, name = "C"),
                       Real(0.0, 1.0, name = "l1_ratio")]

        super().bayessianOptimization(searchSpace, calls)