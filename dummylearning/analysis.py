import numpy as np

from dummylearning.info import Info


class Analysis(Info):

    def __init__(self, model, verbose = True):
        super().__init__(verbose)

        self.model = model




    def metrics(self):
        self.upgradeInfo("Calculating model metrics")

        from sklearn.metrics import confusion_matrix

        metricsDict = dict()

        for dataset in self.model.dataset:
            self.upgradeInfo(f"Calculating {dataset} dataset metrics")

            metricsDict[dataset] = dict()

            matrix = confusion_matrix(self.model.dataset[dataset]["tags"],
                                      self.model.model.predict(self.model.dataset[dataset]["values"]),
                                      labels = self.model.model.classes_)

            metricsDict[dataset]["accuracy"] = np.sum(np.diagonal(matrix)) / np.sum(matrix)

            for index, clas in enumerate(list(self.model.model.classes_)):
                metricsDict[dataset][clas] = dict()

                # Precission
                if np.sum(matrix[:,index]) != 0:
                    precision = matrix[index, index] / np.sum(matrix[:,index])
                    metricsDict[dataset][clas]["precision"] = precision
                else:
                    metricsDict[dataset][clas]["precision"] = 0.0

                #Recall
                if np.sum(matrix[index,:]) != 0:
                    recall = matrix[index, index] / np.sum(matrix[index,:])
                    metricsDict[dataset][clas]["recall"] = recall
                else:
                    metricsDict[dataset][clas]["recall"] = 0.0            

                #F1
                if precision + recall != 0:
                    metricsDict[dataset][clas]["f1"] = (2 * precision * recall) / (precision + recall)
                else:
                    metricsDict[dataset][clas]["f1"] = 0.0

        return metricsDict




    def parameters(self):
        self.upgradeInfo("Extracting model parameters")

        parametersDict = dict()

        for name, value in self.model.model.get_params().items():
            parametersDict[name] = value

        return parametersDict