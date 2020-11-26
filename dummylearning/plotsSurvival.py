from dummylearning.info import Info
from dummylearning.analysisSurvival import Analysis
import matplotlib.pyplot as plt
import pandas as pd

class Plots(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model
        self.analysis = Analysis(self.model)




    def coefficients(self, outfile, extension = "png"):
        self.upgradeInfo("Generating coefficients plot")

        nonZeroValues, nonZeroNames = self.analysis.coefficients()

        _, ax = plt.subplots()
        ax.barh(nonZeroNames, nonZeroValues, align = "center")
        ax.axvline(0, color = "black", linewidth = 2.0)

        ax.set_xlabel("Coefficient Value")
        ax.set_title(f"Exitus Coefficients")

        ax.grid(True)

        plt.savefig(f"{outfile}.{extension}", dpi = 100, bbox_inches = "tight")
        plt.close()




    def oddsRatio(self, outfile, extension = "png"):
        self.upgradeInfo("Generating odds ratios plot")

        nonZeroValues, nonZeroNames = self.analysis.oddsRatio()

        _, ax = plt.subplots()
        ax.barh(nonZeroNames, nonZeroValues, align = "center")
        ax.axvline(1, color = "black", linewidth = 2.0)

        ax.set_xlabel("Odds Ratio")
        ax.set_title(f"Exitus Odds Ratios")

        ax.grid(True)

        plt.savefig(f"{outfile}.{extension}", dpi = 100, bbox_inches = "tight")
        plt.close()




    def log2oddsRatio(self, outfile, extension = "png"):
        self.upgradeInfo("Generating odds ratios plot")

        nonZeroValues, nonZeroNames = self.analysis.log2oddsRatio()

        _, ax = plt.subplots()
        ax.barh(nonZeroNames, nonZeroValues, align = "center")
        ax.axvline(0, color = "black", linewidth = 2.0)

        ax.set_xlabel("Log2 Odds Ratio")
        ax.set_title(f"Exitus Log2 Odds Ratios")

        ax.grid(True)

        plt.savefig(f"{outfile}.{extension}", dpi = 100, bbox_inches = "tight")
        plt.close()




    def rocCurve(self, outfile, extension = "png"):
        self.upgradeInfo("Generating ROC curve plot")

        auc, mean, times = self.analysis.rocInfo()


        for dataset in auc:

            _, ax = plt.subplots()
            ax.plot(times[dataset], auc[dataset], marker="o")

            ax.axhline(mean[dataset], linestyle="--")

            ax.set_xlabel("Days")
            ax.set_ylabel("time-dependent AUC")
            ax.set_title(f"ROC curve {dataset} dataset")
            ax.grid(True)


            plt.savefig(f"{outfile}_{dataset}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def kaplanMeier(self, outfile, extension = "png"):
        self.upgradeInfo("Generating Kaplan-Meier plot")

        from sksurv.nonparametric import kaplan_meier_estimator

        for datasetName, dataset in self.model.dataset.items():

            times, survival_prob = kaplan_meier_estimator(dataset["tags"]["Status"], dataset["tags"]["Survival_in_days"])

            _, ax = plt.subplots()
            ax.step(times, survival_prob, where="post")
            ax.set_ylabel("Est. probability of survival")
            ax.set_xlabel("Days")
            ax.set_title(f"Kaplan-Meier curve {datasetName} dataset")

            plt.savefig(f"{outfile}_{datasetName}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()
