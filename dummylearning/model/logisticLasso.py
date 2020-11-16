from dummylearning.model.classificationModel import ClassificationModel
from sklearn.linear_model import LogisticRegression
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

class LogisticLasso(ClassificationModel):

    """
    Logistic regression with Lasso penalty class

    Parameters
    ---------------------------------------------------------------------------
        data <Data> (positional) => Data instance containg X and Y values

    Attributes
    ---------------------------------------------------------------------------
        model <sklearn.linear_model.LogisticRegression> => Logistic Regression
                                                           model setted with
                                                           l1 penalty and
                                                           saga solver

    Methods
    ---------------------------------------------------------------------------
        bayessianOptimization => Optimization method for parameter setting
    """

    def __init__(self, data, verbose = True):
        super().__init__(data, verbose)

        self.model = LogisticRegression(penalty = "l1",
                                        solver = "saga",
                                        max_iter = 10000,
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
            Params search space => C (0.001:100)
        Return
        ---------------------------------------------------------------------------
            None => It calls super().bayessianOptimization
        """

        searchSpace = [Real(0.001, 100, name = "C")]

        super().bayessianOptimization(searchSpace, calls)