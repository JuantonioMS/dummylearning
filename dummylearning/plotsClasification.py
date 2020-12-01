from dummylearning.info import Info
import matplotlib.pyplot as plt
import pandas as pd
from dummylearning.analysisClasification import Analysis

class Plots(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model
        self.analysis = Analysis(self.model)




    #_________________________________COEFFICIENTS_________________________________




    def coefficients(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> coefficients
        Plot coefficients per class

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate coefficients plots
        """

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




    def oddsRatios(self, outfile: str, extension: str = "png") -> None:
        pass




    def log2OddsRatios(self, outfile: str, extension: str = "png") -> None:
        pass




    #_______________________________CONFUSSION MATRIX______________________________




    def confussionMatrix(self, outfile: str, extension:str = "png") -> None:

        """
        Function -> confussionMatrix
        Plot confussion matrix per dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate confussion matrix plots
        """

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



    #__________________________________ROC CURVES__________________________________




    def rocCurve(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> rocCurve
        Plot ROC curves per class and dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate ROC curves plots
        """

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




    def datasetRocCurve(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> datasetRocCurve
        Plot ROC curves per dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate ROC curves plots
        """

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




    def classRocCurve(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> classRocCurve
        Plot ROC curves per class

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate ROC curves plots
        """

        self.upgradeInfo("Generating clas ROC curves plot")

        fpr, tpr, area = self.analysis.rocInfo()

        datasets = fpr.keys()
        classes = fpr[list(datasets)[0]].keys()

        for clas in classes:
            _, ax = plt.subplots()

            for dataset in datasets:
                ax.plot(fpr[dataset][clas], tpr[dataset][clas],
                        lw = 3 if dataset in ["Complete", "Production"] else 1,
                        label = f"ROC curve {dataset} (area = {round(area[dataset][clas], 3)})")

            ax.plot([0, 1], [0, 1], color = "black", lw = 2, linestyle = "--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC curve for  {clas} class")
            ax.legend(loc = "lower right")
            ax.grid(True)

            plt.savefig(f"{outfile}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def effectRocCurve(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> effectRocCurve
        Plot accumulated coefficient effect ROC curves per class and dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate ROC curves plots
        """

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




    #____________________________PRECISION RECALL CURVES___________________________




    def precisionRecallCurve(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> precisionRecallCurve
        Plot precision recall curves per class and dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate coefficients plots
        """

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




    def datasetPrecisionRecallCurve(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> datasetPrecisionRecallCurve
        Plot Precision Recall curves per dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate Precision Recall curves plots
        """

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
             ax.set_title(f"Precision-Recall curve for {datasetName} dataset")
             ax.legend(loc = "lower right")
             ax.grid(True)

             plt.savefig(f"{outfile}_{datasetName}.{extension}", dpi = 100, bbox_inches = "tight")
             plt.close()




    def classPrecisionRecallCurve(self, outfile: str, extension: str = "png"):

        """
        Function -> classPrecisionRecallCurve
        Plot Precision Recall curves per class

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate Precision Recall curves plots
        """

        self.upgradeInfo("Generation class Precision Recall curves plot")

        precision, recall, area = self.analysis.prcInfo()

        datasets = precision.keys()
        classes = precision[list(datasets)[0]].keys()

        for clas in classes:
            _, ax = plt.subplots()

            for dataset in datasets:
                ax.plot(recall[dataset][clas], precision[dataset][clas],
                        lw = 3 if dataset in ["Complete", "Production"] else 1,
                        label = f"{dataset} AP = {round(area[dataset][clas], 3)}")

            ax.plot([0, 1], [1, 0], color = "black", lw = 2, linestyle = "--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall curve for {clas} class")
            ax.legend(loc = "lower right")
            ax.grid(True)

            plt.savefig(f"{outfile}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()





