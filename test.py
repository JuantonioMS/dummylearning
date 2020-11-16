from dummylearning.fileCsv import FileCsv
from dummylearning.data import Data
from dummylearning.plots import Plots
from dummylearning.analysis import Analysis
from dummylearning.report import Report

from dummylearning.model.logisticLasso import LogisticLasso


# Loading File
file = FileCsv("./datasets/iris.csv", sep = ",")

process = [("Class", "Sepal_length", "Petal_width", "test")]

for tag, start, end, name in process:
    data = file.selectData(tag, start, end)
    print("Preprocessing", name)
    data.purge()
    data.clean()
    data.encodeCategorical()
    data.imputeEmptyValues()
    data.scaleStandard()

    print("Processing", name)
    model = LogisticLasso(data)
    
    model.runClassicModel()

    report = Report(model)
    report.generate("out_dir")


    #analysis = Analysis(model)
    #print(analysis.metrics())

    #plots = Plots(model)
    #plots.coefficients("test_coefficients")
    #plots.confussionMatrix("test_confussionMatrix")

   


