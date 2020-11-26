from dummylearning.fileCsvSurvival import FileCsv
from dummylearning.dataSurvival import Data
from dummylearning.model.coxLasso import SurvivalLasso
from dummylearning.plotsSurvival import Plots
from dummylearning.reportSurvival import Report


# Loading File
file = FileCsv("./datasetSurvival/all.csv", sep = "\t", decimal = ",")
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
report.generate("all_todo_supervivencia")


file = FileCsv("./datasetSurvival/all.csv", sep = "\t", decimal = ",")
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
report.generate("all_todo_progresion")



file = FileCsv("./datasetSurvival/all.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Supervivencia",
                       "Sexo",
                       "Galectin-9")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("all_clinic_supervivencia")



file = FileCsv("./datasetSurvival/all.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Progresión",
                       "Sexo",
                       "Galectin-9")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("all_clinic_progresion")




file = FileCsv("./datasetSurvival/all.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Supervivencia",
                       "hsa-let-7c",
                       "hsa-miR-98")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("all_mirna_supervivencia")




file = FileCsv("./datasetSurvival/all.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Progresión",
                       "hsa-let-7c",
                       "hsa-miR-98")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("all_mirna_progresion")



file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
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
report.generate("extended_todo_supervivencia")



file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
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
report.generate("extended_todo_progresion")



file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Supervivencia",
                       "Sexo",
                       "Galectin-9")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("extended_clinic_supervivencia")



file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Progresión",
                       "Sexo",
                       "Galectin-9")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("extended_clinic_progresion")



file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Supervivencia",
                       "hsalet7a_000377",
                       "rnomiR7#_001338")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("extended_mirna_supervivencia")



file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")
data = file.selectData("Exitus",
                       "Progresión",
                       "hsalet7a_000377",
                       "rnomiR7#_001338")

data.encodeCategorical()
data.imputeEmptyValues()
data.scaleStandard()


model = SurvivalLasso(data)
model.optimizePenalty()
model.runClassicModel()

report = Report(model)
report.generate("extended_mirna_progresion")

