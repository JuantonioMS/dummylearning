import numpy as np
import os, shutil

from dummylearning.utilities.info import Info
from dummylearning.analysis.clasification import Analysis
from dummylearning.plots.classification import Plots


class Report(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model
        self.plots = Plots(self.model)
        self.analysis = Analysis(self.model)



    def coefficientsPlotsReport(self, outfile, file):

        file.write("## Coefficients Plots\n\n")
        os.mkdir(f"{outfile}/img/coefficients")
        path = f"{outfile}/img/coefficients"
        imagePath = "./img/coefficients"

        # Raw coefficients ploting
        file.write("### Raw Coefficients\n\n")
        os.mkdir(f"{path}/raw")

        self.plots.coefficients(f"{path}/raw/coefs", extension = "png")

        for label in self.model.model.classes_:
            file.write(f"![{label} coefficients]({imagePath}/raw/coefs_{label}.png)\n\n")

        # Odds Ratios Coefficients
        file.write("### Odds Ratio Coefficients\n\n")
        os.mkdir(f"{path}/odds")

        self.plots.oddsRatios(f"{path}/odds/coefs", extension = "png")

        for label in self.model.model.classes_:
            file.write(f"![{label} odds ratio]({imagePath}/odds/coefs_{label}.png)\n\n")

        # Log2 Odds Ratios Coefficients
        file.write("### Log2 Odds Ratio Coefficients\n\n")
        os.mkdir(f"{path}/log2odds")

        self.plots.log2OddsRatios(f"{path}/log2odds/coefs", extension = "png")

        for label in self.model.model.classes_:
            file.write(f"![{label} log2 odds ratio]({imagePath}/log2odds/coefs_{label}.png)\n\n")




    def confussionMatrixPlotsReport(self, outfile, file):

        datasets = list(self.model.dataset.keys())

        file.write("## Confussion Matrix Plots\n\n")
        os.mkdir(f"{outfile}/img/confussion_matrix")
        path = f"{outfile}/img/confussion_matrix"
        imagePath = "./img/confussion_matrix"

        # Confusion matrix ploting
        file.write("### Raw Matrix\n\n")
        os.mkdir(f"{path}/raw")

        self.plots.confussionMatrix(f"{path}/raw/matrix", extension = "png")

        for dataset in datasets:
            file.write(f"#### {dataset.capitalize()} dataset\n\n")
            file.write(f"![{dataset} dataset confussion matrix]({imagePath}/raw/matrix_{dataset}.png)\n")




    def rocPlotsReport(self, outfile, file):

        datasets = list(self.model.dataset.keys())
        classes = list(self.model.model.classes_) + ["micro", "macro"]

        file.write("## ROC Curves Plots\n\n")
        os.mkdir(f"{outfile}/img/roc")
        path = f"{outfile}/img/roc"
        imagePath = "./img/roc"

        # Dataset ROC curves ploting
        file.write("### ROC Curves Plots per dataset\n\n")
        os.mkdir(f"{path}/dataset")

        self.plots.datasetRocCurve(f"{path}/dataset/roc", extension = "png")

        for dataset in datasets:
            file.write(f"![{dataset} roc curves]({imagePath}/dataset/roc_{dataset}.png)\n\n")

        # Class ROC curves ploting
        file.write("### ROC Curves Plots per class\n\n")
        os.mkdir(f"{path}/class")

        self.plots.classRocCurve(f"{path}/class/roc", extension = "png")

        for clas in classes:
            file.write(f"![{clas} roc curves]({imagePath}/class/roc_{clas}.png)\n\n")

        """
        # Effect ROC curves ploting
        file.write("### ROC Curves Plots effect per class and dataset\n\n")
        os.mkdir(f"{path}/effect")

        self.plots.effectRocCurve(f"{path}/effect/roc", extension = "png")

        for dataset in datasets:
            file.write(f"#### {dataset.capitalize()} dataset\n\n")
            for clas in self.model.model.classes_:
                file.write(f"![{dataset}_{clas} roc curves]({imagePath}/effect/roc_{dataset}_{clas}.png)\n\n")
        """

        # Single ROC curves ploting
        file.write("### ROC Curves Plots per class and dataset\n\n")
        os.mkdir(f"{path}/single")

        self.plots.rocCurve(f"{path}/single/roc", extension = "png")

        for dataset in datasets:
            file.write(f"#### {dataset.capitalize()} dataset\n\n")
            for clas in classes:
                file.write(f"![{clas} roc curves]({imagePath}/single/roc_{dataset}_{clas}.png)\n\n")




    def prPlotsReport(self, outfile, file):

        datasets = list(self.model.dataset.keys())
        classes = list(self.model.model.classes_)

        file.write("## Precision Recall Curves Plots\n\n")
        os.mkdir(f"{outfile}/img/pr")
        path = f"{outfile}/img/pr"
        imagePath = "./img/pr"

        # Dataset Precision-Recall curves ploting
        file.write("### Precision Recall curves per dataset\n\n")
        os.mkdir(f"{path}/dataset")

        self.plots.datasetPrecisionRecallCurve(f"{path}/dataset/pr", extension = "png")

        for dataset in datasets:
            file.write(f"![{dataset} pr curves]({imagePath}/dataset/pr_{dataset}.png)\n\n")

        # Class Precision-Recall curves ploting
        file.write("### Precision Recall curves per class\n\n")
        os.mkdir(f"{path}/class")

        self.plots.classPrecisionRecallCurve(f"{path}/class/pr", extension = "png")

        for clas in classes + ["micro", "macro"]:
            file.write(f"![{clas} pr curves]({imagePath}/class/pr_{clas}.png)\n\n")

        # Single Precision-Recall curves ploting
        file.write("### Precision Recall curves per class and dataset\n\n")
        os.mkdir(f"{path}/single")

        self.plots.precisionRecallCurve(f"{path}/single/pr", extension = "png")

        for dataset in datasets:
            file.write(f"#### {dataset.capitalize()} dataset\n\n")
            for clas in classes + ["micro", "macro"]:
                file.write(f"![{dataset}_{clas} pr curves]({imagePath}/single/pr_{dataset}_{clas}.png)\n\n")




    def metricsPlotsReport(self, outfile, file):
        file.write("## Metrics Plots\n\n")
        os.mkdir(f"{outfile}/img/metrics")
        path = f"{outfile}/img/metrics"
        imagePath = "./img/metrics"

        # Dataset metrics
        file.write("### Metrics per dataset\n\n")
        os.mkdir(f"{path}/dataset")

        self.plots.datasetMetrics(f"{path}/dataset/metrics", extension = "png")
        file.write(f"![metrics per dataset]({imagePath}/dataset/metrics.png)\n\n")

        # Class metrics
        file.write("### Metrics per dataset\n\n")
        os.mkdir(f"{path}/class")

        self.plots.datasetMetrics(f"{path}/class/metrics", extension = "png")
        file.write(f"![metrics per dataset]({imagePath}/class/metrics.png)\n\n")




    def generate(self, outfile):
        self.upgradeInfo("Generating model report")

        try:
            os.mkdir(outfile)
        except FileExistsError:
            shutil.rmtree(outfile)
            os.mkdir(outfile)

        #self.saveCoefs(f"{outfile}/coeffs.csv")

        os.mkdir(f"{outfile}/img")

        with open(outfile + "/report.md", "w") as file:
            file.write(f"# {type(self).__name__} {self.model.data.tagName}\n")
            file.write(self.metricsReport() + "\n\n")

            self.metricsPlotsReport(outfile, file)
            self.coefficientsPlotsReport(outfile, file)
            self.confussionMatrixPlotsReport(outfile, file)
            self.rocPlotsReport(outfile, file)
            self.prPlotsReport(outfile, file)

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