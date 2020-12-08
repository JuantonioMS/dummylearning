from dummylearning.files.classification import FileCsv

file = FileCsv("./dummylearning/datasets/classification/iris.csv", sep = ",")
data = file.selectData("Class", "Sepal_length", "Petal_width")
data.purge()
data.clean()
data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()

from dummylearning.models.classification.logisticElasticNet import LogisticElasticNet
from dummylearning.models.classification.randomForest import RandomForest
from dummylearning.models.classification.logisticLasso import LogisticLasso
from dummylearning.models.classification.logisticRidge import LogisticRidge

for test in [LogisticElasticNet, RandomForest, LogisticLasso, LogisticRidge]:
    model = test(data)
    model.runClassicModel()
    model.runProductionModel()

from dummylearning.analysis.clasification import Analysis
analysis = Analysis(model)
from dummylearning.plots.classification import Plots
plots = Plots(model)
from dummylearning.reports.classification import Report
report = Report(model)
report.generate("../prueba")





