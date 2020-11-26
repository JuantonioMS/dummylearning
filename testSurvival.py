from dummylearning.fileCsvSurvival import FileCsv
from dummylearning.dataSurvival import Data
from dummylearning.model.coxLasso import SurvivalLasso
from dummylearning.plotsSurvival import Plots
from dummylearning.reportSurvival import Report


# Loading File
file = FileCsv("./datasetSurvival/all.csv", sep = ";", decimal = ",")
data = file.selectData("Exitus",
                       "Supervivencia",
                       "Sexo",
                       "hsa-miR-98")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("all_supervivencia")

file = FileCsv("./datasetSurvival/all.csv", sep = ";", decimal = ",")
data = file.selectData("Exitus",
                       "Progresión",
                       "Sexo",
                       "hsa-miR-98")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("all_progresion")

file = FileCsv("./datasetSurvival/extended.csv", sep = ";", decimal = ",")
data = file.selectData("Exitus",
                       "Supervivencia",
                       "Sexo",
                       "rnomiR7#_001338")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("extended_supervivencia")

file = FileCsv("./datasetSurvival/extended.csv", sep = ";", decimal = ",")
data = file.selectData("Exitus",
                       "Progresión",
                       "Sexo",
                       "rnomiR7#_001338")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("extended_progresion")

