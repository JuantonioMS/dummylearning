from dummylearning.fileCsvClasification import FileCsv
from dummylearning.dataClasification import Data
from dummylearning.plotsClasification import Plots
from dummylearning.analysisClasification import Analysis
from dummylearning.reportClasification import Report

from dummylearning.model.logisticRidge import LogisticRidge


# Loading File
file = FileCsv("./datasetClasification/iris.csv", sep = ",")
process = [("Class", "Sepal_length", "Petal_width", "test")]

#file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
#process = [("Respuesta", "hsalet7a_000377", "rnomiR7#_001338", "prueba")]

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
    report.generate("../out_dir")





