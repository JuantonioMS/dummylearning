from dummylearning.fileCsv import FileCsv
from dummylearning.data import Data
from dummylearning.plots import Plots
from dummylearning.analysis import Analysis
from dummylearning.report import Report

from dummylearning.model.logisticRidge import LogisticRidge


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
    model = LogisticRidge(data)
    model.model.set_params(**{"multi_class" : "ovr"})

    model.runClassicModel()

    report = Report(model)
    report.generate("out_dir")





