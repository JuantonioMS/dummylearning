from dummylearning.model.survivalModel import SurvivalModel
from sksurv.linear_model import CoxnetSurvivalAnalysis # Elastic-net and Lasso Regression Model
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical

class SurvivalLasso(SurvivalModel):

    def __init__(self, data):
        super().__init__(data)
        self.model = CoxnetSurvivalAnalysis(l1_ratio = 1.0,
                                            max_iter = 1000000)



    def optimize(self):

        searchSpace = [Real(0.001, 1000, name = "alphas")]

        return super().optimize(searchSpace)