import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("../utilities/")
from dummylearning.utilities.info import Info
from dummylearning.analysis.clasification import Analysis



class Plots(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model
        self.analysis = Analysis(model)

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline

def smoothplot(x: np.array, y: np.array,k: int = 3) -> np.array:
    """
        Function -> smootPlot
        Plot interpolating curve smoothing corner points
        Parameters
        ---------------------------------------------------------------------------
            x   <np.array> (positional)   => Abscissas
            y   <np.array> (positional) => Ordinates
            k   <int> (default = 3) => Degree of interpolating polynomials
            
         Return
        ---------------------------------------------------------------------------
            xSmooth => x smoothed points
            ySmooth => y smoothed points
    """
    #create evenly spaced grid
    x_smooth = np.linspace(x.min(), x.max(), 300)
    #Compute the coefficients of interpolating curve
    spline = make_interp_spline(x, y, k=k)
    #Evaluation of the spline in the grid points
    y_smooth = spline(x_smooth)
    
    return x_smooth, y_smooth

    #____________________________________METRICS___________________________________




    def datasetMetrics(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> datasetMetrics
        Plot metrics per dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate metrics plots per dataset
        """

        self.upgradeInfo("Generating datasets metrics plots")

        metrics = self.analysis.metrics()
        datasets = list(self.model.dataset.keys())

        _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 10))

        # Accuracy
        accuracies = [metrics[dataset]["accuracy"] for dataset in datasets]

        ax[0, 0].bar(datasets, accuracies)
        ax[0, 0].axhline(0.5, color = "black", linestyle = "--")

        ax[0, 0].set_ylabel("Accuracy")
        ax[0, 0].set_xlabel("Dataset")
        ax[0, 0].set_title("Accuracy scores")
        ax[0, 0].set_ylim([0.0, 1.5])


        borders = np.linspace(-0.4, 0.4, len(self.model.model.classes_) + 1)
        width = abs(borders[0] - borders[1])
        centers = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(self.model.model.classes_))

        # Precision
        for index, clas in enumerate(self.model.model.classes_):

            precisions = [metrics[dataset][clas]["precision"] for dataset in datasets]

            ax[0, 1].bar(np.arange(len(datasets)) + centers[index] , precisions, width, label = clas)

        ax[0, 1].axhline(0.5, color = "black", linestyle = "--")

        ax[0, 1].set_ylabel("Precision")
        ax[0, 1].set_xlabel("Dataset")
        ax[0, 1].set_xticks(np.arange(len(datasets)))
        ax[0, 1].set_xticklabels(datasets)
        ax[0, 1].set_title("Precision scores")
        ax[0, 1].legend()
        ax[0, 1].set_ylim([0.0, 1.5])

        # Recall
        for index, clas in enumerate(self.model.model.classes_):

            recalls = [metrics[dataset][clas]["recall"] for dataset in datasets]

            ax[1, 0].bar(np.arange(len(datasets)) + centers[index] , recalls, width, label = clas)

        ax[1, 0].axhline(0.5, color = "black", linestyle = "--")

        ax[1, 0].set_ylabel("Recall")
        ax[1, 0].set_xlabel("Dataset")
        ax[1, 0].set_xticks(np.arange(len(datasets)))
        ax[1, 0].set_xticklabels(datasets)
        ax[1, 0].set_title("Recall scores")
        ax[1, 0].legend()
        ax[1, 0].set_ylim([0.0, 1.5])

        # F1
        for index, clas in enumerate(self.model.model.classes_):

            f1s = [metrics[dataset][clas]["f1"] for dataset in datasets]

            ax[1, 1].bar(np.arange(len(datasets)) + centers[index] , f1s, width, label = clas)

        ax[1, 1].axhline(0.5, color = "black", linestyle = "--")

        ax[1, 1].set_ylabel("F1")
        ax[1, 1].set_xlabel("Dataset")
        ax[1, 1].set_xticks(np.arange(len(datasets)))
        ax[1, 1].set_xticklabels(datasets)
        ax[1, 1].set_title("F1 scores")
        ax[1, 1].legend()
        ax[1, 1].set_ylim([0.0, 1.5])


        plt.savefig(f"{outfile}.{extension}", dpi = 100, bbox_inches = "tight")
        plt.close()




    def classMetrics(self, outfile: str, extension: str = "png") -> None:

        """
        Function -> datasetMetrics
        Plot metrics per dataset

        Parameters
        ---------------------------------------------------------------------------
            outfile   <str> (positional)   => Plot name
            extension <str> (default: png) => Image extension

        Return
        ---------------------------------------------------------------------------
            None => Generate metrics plots per dataset
        """

        self.upgradeInfo("Generating datasets metrics plots")

        metrics = self.analysis.metrics()
        classes = list(self.model.model.classes_)
        datasets = list(self.model.dataset.keys())

        _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 10))

        # Accuracy
        accuracies = [metrics[dataset]["accuracy"] for dataset in datasets]

        ax[0, 0].bar(datasets, accuracies)
        ax[0, 0].axhline(0.5, color = "black", linestyle = "--")

        ax[0, 0].set_ylabel("Accuracy")
        ax[0, 0].set_xlabel("Dataset")
        ax[0, 0].set_title("Accuracy scores")
        ax[0, 0].set_ylim([0.0, 1.5])


        borders = np.linspace(-0.4, 0.4, len(datasets) + 1)
        width = abs(borders[0] - borders[1])
        centers = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(datasets))

        # Precision
        for index, dataset in enumerate(datasets):

            precisions = [metrics[dataset][clas]["precision"] for clas in classes]

            ax[0, 1].bar(np.arange(len(classes)) + centers[index] , precisions, width, label = dataset)

        ax[0, 1].axhline(0.5, color = "black", linestyle = "--")

        ax[0, 1].set_ylabel("Precision")
        ax[0, 1].set_xlabel("Class")
        ax[0, 1].set_xticks(np.arange(len(classes)))
        ax[0, 1].set_xticklabels(classes)
        ax[0, 1].set_title("Precision scores")
        ax[0, 1].legend()
        ax[0, 1].set_ylim([0.0, 1.5])

        # Recall
        for index, dataset in enumerate(datasets):

            recalls = [metrics[dataset][clas]["recall"] for clas in classes]

            ax[1, 0].bar(np.arange(len(classes)) + centers[index] , recalls, width, label = dataset)

        ax[1, 1].axhline(0.5, color = "black", linestyle = "--")

        ax[1, 0].set_ylabel("Recall")
        ax[1, 0].set_xlabel("Class")
        ax[1, 0].set_xticks(np.arange(len(classes)))
        ax[1, 0].set_xticklabels(classes)
        ax[1, 0].set_title("Recall scores")
        ax[1, 0].legend()
        ax[1, 0].set_ylim([0.0, 1.5])

        # F1
        for index, dataset in enumerate(datasets):

            f1s = [metrics[dataset][clas]["f1"] for clas in classes]

            ax[1, 1].bar(np.arange(len(classes)) + centers[index] , f1s, width, label = dataset)

        ax[1, 1].axhline(0.5, color = "black", linestyle = "--")

        ax[1, 1].set_ylabel("F1")
        ax[1, 1].set_xlabel("Class")
        ax[1, 1].set_xticks(np.arange(len(classes)))
        ax[1, 1].set_xticklabels(classes)
        ax[1, 1].set_title("F1 scores")
        ax[1, 1].legend()
        ax[1, 1].set_ylim([0.0, 1.5])


        plt.savefig(f"{outfile}.{extension}", dpi = 100, bbox_inches = "tight")
        plt.close()


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

        totalCoefficients = self.analysis.coefficients()

        for clas, coefs in totalCoefficients.items():

            nonZeroValues, nonZeroCoefs = [], []

            for coef, value in coefs.items():

                if value != 0:
                    nonZeroCoefs.append(coef)
                    nonZeroValues.append(value)

            nonZeroValues, nonZeroCoefs = zip(*sorted(zip(nonZeroValues, nonZeroCoefs)))

            _, ax = plt.subplots()
            ax.barh(nonZeroCoefs, nonZeroValues, align = "center")
            ax.axvline(0, color = "black", linewidth = 2.0)

            ax.set_xlabel("Coefficient Value")
            ax.set_title(f"{clas} Coefficient")

            ax.grid(True)

            plt.savefig(f"{outfile}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def oddsRatios(self, outfile: str, extension: str = "png") -> None:

        self.upgradeInfo("Generating odds ratio plots")

        totalOdds = self.analysis.oddsRatios()

        for clas, coefs in totalOdds.items():

            nonZeroValues, nonZeroCoefs = [], []

            for coef, value in coefs.items():

                if value != 0:
                    nonZeroCoefs.append(coef)
                    nonZeroValues.append(value)

            nonZeroValues, nonZeroCoefs = zip(*sorted(zip(nonZeroValues, nonZeroCoefs)))

            _, ax = plt.subplots()
            ax.barh(nonZeroCoefs, nonZeroValues, align = "center")
            ax.axvline(1, color = "black", linewidth = 2.0)

            ax.set_xlabel("Odds Ratio Value")
            ax.set_title(f"{clas} Odds Ratio")

            ax.grid(True)

            plt.savefig(f"{outfile}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




    def log2OddsRatios(self, outfile: str, extension: str = "png") -> None:

        self.upgradeInfo("Generating log2 odds ratio plots")

        totalOdds = self.analysis.log2oddsRatios()

        for clas, coefs in totalOdds.items():

            nonZeroValues, nonZeroCoefs = [], []

            for coef, value in coefs.items():

                if value != 0:
                    nonZeroCoefs.append(coef)
                    nonZeroValues.append(value)

            nonZeroValues, nonZeroCoefs = zip(*sorted(zip(nonZeroValues, nonZeroCoefs)))

            _, ax = plt.subplots()
            ax.barh(nonZeroCoefs, nonZeroValues, align = "center")
            ax.axvline(0, color = "black", linewidth = 2.0)

            ax.set_xlabel("Log2 Odds Ratio Value")
            ax.set_title(f"{clas} Log2 Odds Ratio")

            ax.grid(True)

            plt.savefig(f"{outfile}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
            plt.close()




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





