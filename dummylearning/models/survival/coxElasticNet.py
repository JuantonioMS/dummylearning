from dummylearning.models.survival.survivalModel import SurvivalModel
from sksurv.linear_model import CoxnetSurvivalAnalysis # Elastic-net and Lasso Regression Model
from skopt.space import Real

class CoxElasticNet(SurvivalModel):

    def __init__(self, data):
        super().__init__(data)
        self.model = CoxnetSurvivalAnalysis(max_iter = 1000000)



    def bayesianOptimization(self, calls = 50):

        searchSpace = [Real(-14, 9, name = "alphas"),
                       Real(0.0, 1.0, name = "l1_ratio")]

        return super().bayesianOptimization(searchSpace, calls = calls)