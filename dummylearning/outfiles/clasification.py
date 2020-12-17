from dummylearning.outfiles.base import OutfileBase
from dummylearning.analysis.clasification import Analysis

class Outfile(OutfileBase):

    def __init__(self, model, verbose = True):
        super().__init__(verbose)
        self.model = model
        self.analysis = Analysis(self.model)




    def coefficients(self, outfile):
        coefs = self.analysis.coefficients()

        with open(f"{outfile}coefficients.csv", "w") as file:
            row = "Coefficients;" + ";".join([clas for clas in coefs]) + "\n"
            file.write(row)
            for coef in coefs[list(coefs.keys())[0]].keys():
                row = f"{coef};" + ";".join([str(coefs[clas][coef]) for clas in coefs]) + "\n"
                file.write(row)




    def oddsRatios(self, outfile):
        odds = self.analysis.oddsRatios()

        with open(f"{outfile}oddsRatios.csv", "w") as file:
            row = "Odds Ratios;" + ";".join([clas for clas in odds]) + "\n"
            file.write(row)
            for odd in odds[list(odds.keys())[0]].keys():
                row = f"{odd};" + ";".join([str(odds[clas][odd]) for clas in odds]) + "\n"
                file.write(row)




    def log2oddsRatios(self, outfile):
        log2odds = self.analysis.log2oddsRatios()

        with open(f"{outfile}log2oddsRatios.csv", "w") as file:
            row = "Log2 Odds Ratios;" + ";".join([clas for clas in log2odds]) + "\n"
            file.write(row)
            for log2odd in log2odds[list(log2odds.keys())[0]].keys():
                row = f"{log2odd};" + ";".join([str(log2odds[clas][log2odd]) for clas in log2odds]) + "\n"
                file.write(row)


    def predicts(self, outfile):
        for dataset in self.model.dataset:
            with open(f"{outfile}{dataset}_predict.csv", "w") as file:
                file.write("True;" + ";".join(list(self.model.model.classes_)) + "\n")
                for tag, row in zip(self.model.dataset[dataset]["tags"], self.model.dataset[dataset]["values"]):
                    row = row.reshape(1, -1)
                    if len(self.model.model.classes_) == 2:
                        file.write(f"{tag};" + f"{list(-self.model.model.decision_function(row))[0]};" + f"{list(self.model.model.decision_function(row))[0]}" + "\n")
                    else:
                        file.write(f"{tag};" + ";".join(list(map(str, list(self.model.model.decision_function(row))))) + "\n")



    def rocInfo(self, outfile):
        fprs, tprs, _ = self.analysis.rocInfo()

        for dataset in fprs:
            for clas in fprs[dataset]:

                with open(f"{outfile}{clas}_{dataset}_roc.csv", "w", encoding = "utf8") as file:
                    file.write("FPR (x);TPR (y)\n")
                    for fpr, tpr in zip(fprs[dataset][clas], tprs[dataset][clas]):
                        file.write(f"{fpr};{tpr}\n")

    def prcInfo(self, outfile):
        pass
