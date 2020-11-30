from dummylearning.info import Info
import matplotlib.pyplot as plt
import pandas as pd
from dummylearning.analysisClasification import Analysis

class Plots(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model
        self.analysis = Analysis(self.model)




    def coefficients(self, outfile, extension = "png"):
        self.upgradeInfo("Generating coefficients plots")

        column = dict()

        if len(self.model.model.classes_) == 2:

            for index, clas in enumerate(self.model.model.classes_):
                if index == 0:
                    column[clas] = -self.model.model.coef_[0]
                else:
                    column[clas] = self.model.model.coef_[0]

            auxData = pd.DataFrame(data = column,
                                   index = self.model.data.valuesName)
            auxData.loc["intercept"] = [-self.model.model.intercept_[0], self.model.model.intercept_[0]]

        else:

            for index, clas in enumerate(self.model.model.classes_):
                column[clas] = self.model.model.coef_[index]

            auxData = pd.DataFrame(data = column,
                                   index = self.model.data.valuesName)
            auxData.loc["intercept"] = self.model.model.intercept_

        for clas in auxData.columns:
            nonZeroValues = []
            nonZeroNames = []

            for name, element in zip(auxData.index, auxData[clas]):

                if element != 0:
                    nonZeroNames.append(name)
                    nonZeroValues.append(element)

            nonZeroValues, nonZeroNames = zip(*sorted(zip(nonZeroValues, nonZeroNames)))

            _, ax = plt.subplots()
            ax.barh(nonZeroNames, nonZeroValues, align = "center")
            ax.axvline(0, color = "black", linewidth = 2.0)

            ax.set_xlabel("Coefficient Value")
            ax.set_title(f"{clas} Coefficient")

            ax.grid(True)

            plt.savefig(f"{outfile}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def confussionMatrix(self, outfile, extension = "png"):
        self.upgradeInfo("Generating confussion matrix plot")

        from sklearn.metrics import plot_confusion_matrix

        for dataset in self.model.dataset:
            plot = plot_confusion_matrix(self.model.model,
                                         self.model.dataset[dataset]["values"],
                                         self.model.dataset[dataset]["tags"],
                                         display_labels = self.model.model.classes_)
            plot.ax_.set_title(f"{dataset} dataset")
            plt.savefig(f"{outfile}_{dataset}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def rocCurve(self, outfile, extension = "png"):
        self.upgradeInfo("Generating ROC curves plot")

        fpr, tpr, area = self.analysis.rocInfo()

        for datasetName in fpr:
            for clas in fpr[datasetName]:

                _, ax = plt.subplots()
                ax.plot(fpr[datasetName][clas], tpr[datasetName][clas],
                        color = "darkorange",
                        lw = 2,
                        label = f"ROC curve (area = {round(area[datasetName][clas], 3)})")
                ax.plot([0, 1], [0, 1], color = "black", lw = 2, linestyle = "--")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC curve for {clas} class in {datasetName} dataset")
                ax.legend(loc = "lower right")
                ax.grid(True)

                plt.savefig(f"{outfile}_{datasetName}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
                plt.close()




    def precisionRecallCurve(self, outfile, extension = "png"):
        self.upgradeInfo("Generating ROC curves plot")

        precision, recall, area = self.analysis.prcInfo()

        for datasetName in precision:
            for clas in precision[datasetName]:

                _, ax = plt.subplots()
                ax.plot(recall[datasetName][clas],
                        precision[datasetName][clas],
                        color = "darkorange",
                        lw = 2,
                        label = f"{clas} AP = {round(area[datasetName][clas], 3)}")
                ax.plot([0, 1], [1, 0], color = "black", lw = 2, linestyle = "--")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall curve for {clas} class in {datasetName} dataset")
                ax.legend(loc = "lower right")
                ax.grid(True)

                plt.savefig(f"{outfile}_{datasetName}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
                plt.close()

    def metrics(self):
        pass




    def datasetRocCurve(self, outfile, extension = "png"):
        self.upgradeInfo("Generating ROC curves plot")

        fpr, tpr, area = self.analysis.rocInfo()

        for datasetName in fpr:

            _, ax = plt.subplots()

            for clas in fpr[datasetName]:
                ax.plot(fpr[datasetName][clas], tpr[datasetName][clas],
                        lw = 4 if clas in ["micro", "macro"] else 2,
                        label = f"ROC curve {clas} (area = {round(area[datasetName][clas], 3)})",
                        linestyle = ":" if clas in ["micro", "macro"] else "-")

            ax.plot([0, 1], [0, 1], color = "black", lw = 2, linestyle = "--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC curve for {datasetName} dataset")
            ax.legend(loc = "lower right")
            ax.grid(True)

            plt.savefig(f"{outfile}_{datasetName}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def datasetPrecisionRecallCurve(self, outfile, extension = "png"):
        self.upgradeInfo("Generating ROC curves plot")

        precision, recall, area = self.analysis.prcInfo()

        for datasetName in precision:
             _, ax = plt.subplots()
             for clas in precision[datasetName]:
                 ax.plot(recall[datasetName][clas],
                        precision[datasetName][clas],
                        lw = 4 if clas in ["micro", "macro"] else 2,
                        label = f"{clas} AP = {round(area[datasetName][clas], 3)}",
                        linestyle = ":" if clas in ["micro", "macro"] else "-",)
             ax.plot([0, 1], [1, 0], color = "black", lw = 2, linestyle = "--")
             ax.set_xlim([0.0, 1.0])
             ax.set_ylim([0.0, 1.05])
             ax.set_xlabel("Recall")
             ax.set_ylabel("Precision")
             ax.set_title(f"Precision-Recall curve for {clas} class in {datasetName} dataset")
             ax.legend(loc = "lower right")
             ax.grid(True)

             plt.savefig(f"{outfile}_{datasetName}_.{extension}", dpi = 100, bbox_inches = "tight")
             plt.close()



    def effectRocCurve(self, outfile, extension = "png"):

        fpr, tpr, area = self.analysis.accumulatedRocInfo()

        for dataset in fpr:
            for clas in fpr[dataset]:

                _, ax = plt.subplots()
                counter = 0
                for coefficient in fpr[dataset][clas]:
                    counter += 1
                    ax.plot(fpr[dataset][clas][coefficient],
                            tpr[dataset][clas][coefficient],
                            lw = 2,
                            label = f"{coefficient} area = {area[dataset][clas][coefficient]}")
                    if counter == 20:
                        break

                ax.plot([0, 1], [0, 1], color = "black", lw = 2, linestyle = "--")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("False Positive Ratio")
                ax.set_ylabel("True Positive Ratio")
                ax.set_title(f"{dataset} {clas}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                ax.grid(True)

                plt.savefig(f"{outfile}_{dataset}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
                plt.close()