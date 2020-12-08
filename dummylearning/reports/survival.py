import numpy as np
import os, shutil

from dummylearning.info import Info
from dummylearning.plotsSurvival import Plots
from dummylearning.analysisSurvival import Analysis


class Report(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model
        self.plots = Plots(self.model)
        self.analysis = Analysis(self.model)




    def generate(self, outfile):
        self.upgradeInfo("Generating model report")

        try:
            os.mkdir(outfile)
        except FileExistsError:
            shutil.rmtree(outfile)
            os.mkdir(outfile)

        #self.saveCoefs(f"{outfile}/coeffs.csv")

        self.plots.coefficients(f"{outfile}/coefs", extension = "png")
        self.plots.oddsRatio(f"{outfile}/odds", extension = "png")
        self.plots.log2oddsRatio(f"{outfile}/log2odds", extension = "png")

        self.plots.kaplanMeier(f"{outfile}/kaplanmeier", extension = "png")
        self.plots.rocCurve(f"{outfile}/rocCurve", extension = "png")


        with open(outfile + "/report.md", "w") as file:
            file.write(f"# {type(self).__name__}\n")

            file.write("## Coefficients Info\n")

            file.write(f"![coefficients](coefs.png)\n\n\n")
            file.write(f"![odds ratio](odds.png)\n\n\n")
            file.write(f"![log2 odds ratio](log2odds.png)\n\n\n")

            file.write("## ROC curves single\n")

            for dataset in self.model.dataset:
                
                file.write(f"![{dataset} roc curve](rocCurve_{dataset}.png)\n\n\n")

            for dataset in self.model.dataset:
                
                file.write(f"![{dataset} kaplan-meier](kaplanmeier_{dataset}.png)\n\n\n")

            file.write(self.parametersReport() + "\n\n")
            file.write(self.infoReport() + "\n\n")




    def parametersReport(self):
        self.upgradeInfo("Generating parameters report format")

        message = ["### Parameters"]
        for name, value in self.analysis.parameters().items():
            message.append(f"- ***{name}:*** {value}")

        return "\n".join(message)




    def infoReport(self):
        self.upgradeInfo("Generating process info report format")

        message = ["### Process Info"]
        mergedDict = {**self.report,
                      **self.model.data.report,
                      **self.plots.report,
                      **self.analysis.report}

        keys = list(mergedDict.keys())
        keys.sort()

        for key in keys:
            if "\n" in mergedDict[key]:
                aux = mergedDict[key].replace("\n", "  \n&nbsp;&nbsp;&nbsp;&nbsp;")
                message.append(aux + "  ")
            else:
                message.append(f"{mergedDict[key]}  ")

        return "\n".join(message)