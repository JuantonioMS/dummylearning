import numpy as np
import pandas as pd

from dummylearning.analysis.base import AnalysisBase


class Analysis(AnalysisBase):

    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model




    def parameters(self):
        self.upgradeInfo("Extracting model parameters")

        parametersDict = dict()

        for name, value in self.model.model.get_params().items():
            parametersDict[name] = value

        return parametersDict




    def coefficients(self):
        self.upgradeInfo("Extracting model coefficients")

        column = dict()

        print()
        print(self.model.model.coef_.shape)
        print()
        column["Yes"] = self.model.model.coef_[:, 0]

        auxData = pd.DataFrame(data = column,
                               index = self.model.data.valuesName)

        auxData.loc["offset"] = [self.model.model.offset_[0]]


        nonZeroValues = []
        nonZeroNames = []

        for name, element in zip(auxData.index, auxData["Yes"]):

            if element != 0:
                nonZeroNames.append(name)
                nonZeroValues.append(element)

        nonZeroValues, nonZeroNames = zip(*sorted(zip(nonZeroValues, nonZeroNames)))

        return nonZeroValues, nonZeroNames




    def oddsRatio(self):
        self.upgradeInfo("Extracting model odds ratios")

        nonZeroValues, nonZeroNames = self.coefficients()

        from math import exp
        nonZeroValues = [exp(float(value)) for value in nonZeroValues]

        return nonZeroValues, nonZeroNames




    def log2oddsRatio(self):
        self.upgradeInfo("Extracting model log2 odds ratios")

        nonZeroValues, nonZeroNames = self.oddsRatio()

        from math import log2
        nonZeroValues = [log2(value) for value in nonZeroValues]

        return nonZeroValues, nonZeroNames




    def rocInfo(self):
        self.upgradeInfo("Calculating model ROC curves")

        auc, mean, times = dict(), dict(), dict()

        from sksurv.metrics import cumulative_dynamic_auc

        for datasetName, dataset in self.model.dataset.items():

            serial = list(set(dataset["tags"]["Survival_in_days"]))
            serial.sort()
            times[datasetName] = serial[1:-1]

            auc[datasetName], mean[datasetName] = cumulative_dynamic_auc(dataset["tags"],
                                                                         dataset["tags"],
                                                                         self.model.model.predict(dataset["values"]),
                                                                         times[datasetName])

        return auc, mean, times