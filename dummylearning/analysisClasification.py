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




    def rocInfo(self):
        self.upgradeInfo("Calulating fpr, tpr and areas")

        from sklearn.metrics import roc_curve
        from sklearn.metrics import auc

        fpr, tpr, areas, scores = dict(), dict(), dict(), dict()

        for datasetName, dataset in self.model.dataset.items():
            fpr[datasetName] = dict()
            tpr[datasetName] = dict()
            areas[datasetName] = dict()
            scores[datasetName] = dict()

            for index, clas in enumerate(self.model.model.classes_):

                # Binary problems
                if len(self.model.model.classes_) == 2:
                    if index == 0:
                        scores[datasetName][clas] = -self.model.model.decision_function(dataset["values"])
                    else:
                        scores[datasetName][clas] = self.model.model.decision_function(dataset["values"])

                # Not binary problems
                else:
                    scores[datasetName][clas] = self.model.model.decision_function(dataset["values"])[:, index]

                fpr[datasetName][clas], tpr[datasetName][clas], _ = roc_curve(dataset["tags"] == clas, scores[datasetName][clas])
                areas[datasetName][clas] = auc(fpr[datasetName][clas], tpr[datasetName][clas])

            # Micro-average info
            microInfo = [(dataset["tags"] == clas, scores[datasetName][clas]) for clas in self.model.model.classes_]
            microTag = np.concatenate([tag for tag, _ in microInfo])
            microScore = np.concatenate([score for _, score in microInfo])

            fpr[datasetName]["micro"], tpr[datasetName]["micro"], _ = roc_curve(microTag, microScore)
            areas[datasetName]["micro"] = auc(fpr[datasetName]["micro"], tpr[datasetName]["micro"])

            #Macro-average info
            uniqueFpr = np.unique(np.concatenate([fpr[datasetName][clas] for clas in self.model.model.classes_]))

            meanTpr = np.zeros_like(uniqueFpr)
            for clas in self.model.model.classes_:
                meanTpr += np.interp(uniqueFpr, fpr[datasetName][clas], tpr[datasetName][clas])
            meanTpr /= len(self.model.model.classes_)

            fpr[datasetName]["macro"] = uniqueFpr
            tpr[datasetName]["macro"] = meanTpr
            areas[datasetName]["macro"] = auc(fpr[datasetName]["macro"], tpr[datasetName]["macro"])


        return fpr, tpr, areas




    def prcInfo(self):
        self.upgradeInfo("Calculating precision-recall curve info")

        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score

        precisions, recalls, averagePrecision, scores = dict(), dict(), dict(), dict()

        for datasetName, dataset in self.model.dataset.items():
            precisions[datasetName] = dict()
            recalls[datasetName] = dict()
            scores[datasetName] = dict()
            averagePrecision[datasetName] = dict()


            for index, clas in enumerate(self.model.model.classes_):

                # Binary problems
                if len(self.model.model.classes_) == 2:
                    if index == 0:
                        scores[datasetName][clas] = -self.model.model.decision_function(dataset["values"])
                    else:
                        scores[datasetName][clas] = self.model.model.decision_function(dataset["values"])

                # Not binary problems
                else:
                    scores[datasetName][clas] = self.model.model.decision_function(dataset["values"])[:, index]

                precisions[datasetName][clas], recalls[datasetName][clas], _ = precision_recall_curve(dataset["tags"] == clas,
                                                                                                      scores[datasetName][clas])
                averagePrecision[datasetName][clas] = average_precision_score(dataset["tags"] == clas, scores[datasetName][clas])

            # Micro-average info
            microInfo = [(dataset["tags"] == clas, scores[datasetName][clas]) for clas in self.model.model.classes_]
            microTag = np.concatenate([tag for tag, _ in microInfo])
            microScore = np.concatenate([score for _, score in microInfo])

            precisions[datasetName]["micro"], recalls[datasetName]["micro"], _ = precision_recall_curve(microTag, microScore)
            averagePrecision[datasetName]["micro"] = average_precision_score(microTag, microScore)

            #Macro-average info
            uniqueRecalls = np.unique(np.concatenate([recalls[datasetName][clas] for clas in self.model.model.classes_]))

            meanPrecisions = np.zeros_like(uniqueRecalls)
            for clas in self.model.model.classes_:
                meanPrecisions += np.interp(uniqueRecalls, recalls[datasetName][clas], precisions[datasetName][clas])
            meanPrecisions /= len(self.model.model.classes_)

            recalls[datasetName]["macro"] = uniqueRecalls
            precisions[datasetName]["macro"] = meanPrecisions

            aux = []
            lastRecall = 0
            for recall, precision in zip(recalls[datasetName]["macro"], precisions[datasetName]["macro"]):
                aux.append((recall - lastRecall) * precision)
                lastRecall = recall
            averagePrecision[datasetName]["macro"] = sum(aux)



        return precisions, recalls, averagePrecision




    def accumulatedRocInfo(self):
        self.upgradeInfo("Calulating fpr, tpr and areas for accumulated ROC curves")

        from sklearn.metrics import roc_curve
        from sklearn.metrics import auc

        fpr, tpr, areas = dict(), dict(), dict()
        order = self.coefficientsOrder()

        for datasetName, dataset in self.model.dataset.items():
            fpr[datasetName] = dict()
            tpr[datasetName] = dict()
            areas[datasetName] = dict()

            for index, clas in enumerate(self.model.model.classes_):
                fpr[datasetName][clas] = dict()
                tpr[datasetName][clas] = dict()
                areas[datasetName][clas] = dict()

                result = [self.model.model.intercept_[index] for _ in range(dataset["values"].shape[0])]
                fpr[datasetName][clas]["intercept"], tpr[datasetName][clas]["intercept"], _ = roc_curve(dataset["tags"] == clas, result)
                areas[datasetName][clas]["intercept"] = auc(fpr[datasetName][clas]["intercept"], tpr[datasetName][clas]["intercept"])

                mold = np.zeros((self.model.model.coef_.shape[1]), dtype = bool)

                for column in order[clas]:
                    mold[self.model.data.valuesName.index(column)] = True


                    base = self.model.model.coef_[index, :] * mold

                    result = [(sum(dataset["values"][i, :] * base) + self.model.model.intercept_[index]) for i in range(dataset["values"].shape[0])]


                    fpr[datasetName][clas][column], tpr[datasetName][clas][column], _ = roc_curve(dataset["tags"] == clas, result)
                    areas[datasetName][clas][column] = auc(fpr[datasetName][clas][column], tpr[datasetName][clas][column])

        return fpr, tpr, areas





    def coefficientsOrder(self):
        self.upgradeInfo("Calculating coefficients order")

        order = dict()

        for index, clas in enumerate(self.model.model.classes_):
            nonZeroValues = []
            nonZeroNames = []

            for name, element in zip(self.model.data.valuesName, self.model.model.coef_[index, :]):

                if element != 0:
                    nonZeroNames.append(name)
                    nonZeroValues.append(abs(element))

            nonZeroValues, nonZeroNames = zip(*sorted(zip(nonZeroValues, nonZeroNames)))
            order[clas] = nonZeroNames[::-1]

        return order