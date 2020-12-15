from dummylearning.models.survival.survivalModel import SurvivalModel
from sksurv.linear_model import CoxPHSurvivalAnalysis # Elastic-net and Lasso Regression Model
from skopt.space import Real


class CoxRidge(SurvivalModel):

    def __init__(self, data):
        super().__init__(data)
        self.model = CoxPHSurvivalAnalysis(n_iter = 1000000)



    def bayesianOptimization(self, calls = 50):

        searchSpace = [Real(-14, 9, name = "alpha")]

        return super().bayesianOptimization(searchSpace, calls = calls)