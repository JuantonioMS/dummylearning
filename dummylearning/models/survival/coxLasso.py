from dummylearning.models.survival.survivalModel import SurvivalModel
from sksurv.linear_model import CoxnetSurvivalAnalysis # Elastic-net and Lasso Regression Model
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

class CoxLasso(SurvivalModel):

    def __init__(self, data):
        super().__init__(data)
        self.model = CoxnetSurvivalAnalysis(l1_ratio = 1.0,
                                            max_iter = 1000000)



    def bayesianOptimization(self, calls = 50):

        searchSpace = [Real(-14, 9, name = "alphas")]

        return super().bayesianOptimization(searchSpace, calls = calls)