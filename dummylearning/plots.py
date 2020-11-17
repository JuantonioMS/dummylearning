from dummylearning.info import Info
import matplotlib.pyplot as plt
import pandas as pd

class Plots(Info):




    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model




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

        from sklearn.metrics import roc_curve, auc

        for dataset in self.model.dataset:
            self.upgradeInfo(f"Generating {dataset} dataset ROC curves plot")

            for index, clas in enumerate(self.model.model.classes_):

                if len(self.model.model.classes_) == 2:
                    if index == 0:
                        score = -self.model.model.decision_function(self.model.dataset[dataset]["values"])
                    else:
                        score = self.model.model.decision_function(self.model.dataset[dataset]["values"])

                else:
                    score = self.model.model.decision_function(self.model.dataset[dataset]["values"])[:, index]

                fpr, tpr, _ = roc_curve(self.model.dataset[dataset]["tags"] == clas, score)
                area = auc(fpr, tpr)


                _, ax = plt.subplots()
                ax.plot(fpr, tpr,
                        color = "darkorange",
                        lw = 2,
                        label = f"ROC curve (area = {round(area, 3)})")
                ax.plot([0, 1], [0, 1], color = "black", lw = 2, linestyle = "--")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"ROC curve for {clas} class in {dataset} dataset")
                ax.legend(loc = "lower right")
                ax.grid(True)

                plt.savefig(f"{outfile}_{dataset}_{clas}.{extension}", dpi = 100, bbox_inches = "tight")
                plt.close()



    def precisionRecallCurve(self):
        pass

    def metrics(self):
        pass