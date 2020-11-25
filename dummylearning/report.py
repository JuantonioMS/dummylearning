import numpy as np
import os, shutil

from dummylearning.info import Info
from dummylearning.plots import Plots
from dummylearning.analysis import Analysis


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
        self.plots.confussionMatrix(f"{outfile}/matrix", extension = "png")
        self.plots.rocCurve(f"{outfile}/rocCurve", extension = "png")
        self.plots.precisionRecallCurve(f"{outfile}/prCurve", extension = "png")

        with open(outfile + "/report.md", "w") as file:
            file.write(f"# {type(self).__name__} {self.model.data.tagName}\n")
            file.write(self.metricsReport() + "\n\n")

            file.write("## Coefficients Info\n")

            for label in self.model.model.classes_:
                file.write(f"![{label} coefficients](coefs_{label}.png)\n")

            file.write("## Confussion Matrix\n")

            for dataset in self.model.dataset:
                file.write(f"![{dataset} confussion matrix](matrix_{dataset}.png)\n\n\n")

            file.write("## ROC curves single\n")

            for dataset in self.model.dataset:
                for clas in self.model.model.classes_:
                    file.write(f"![{dataset} roc curve](rocCurve_{dataset}_{clas}.png)\n\n\n")

            file.write("## Precision-Recall curves single\n")

            for dataset in self.model.dataset:
                for clas in self.model.model.classes_:
                    file.write(f"![{dataset} precision-recall curve](prCurve_{dataset}_{clas}.png)\n\n\n")

            file.write(self.parametersReport() + "\n\n")
            file.write(self.infoReport() + "\n\n")




    def metricsReport(self):

        self.upgradeInfo("Generating metrics report format")

        message = ["### Metrics"]

        metrics = self.analysis.metrics()
        for dataset, metrics in metrics.items():

            message.append(f"- **{dataset}**")

            for label, values in metrics.items():

                if label == "accuracy":
                    message.append(f"  - **{label}:** {values}")

                else:
                    message.append(f"  - **{label}**")
                    for score, value in values.items():
                        message.append(f"    - **{score}:** {value}")

        return "\n".join(message)




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
                      **self.model.report,
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