from dummylearning.fileCsvClasification import FileCsv
from dummylearning.dataClasification import Data
from dummylearning.plotsClasification import Plots
from dummylearning.analysisClasification import Analysis
from dummylearning.reportClasification import Report

from dummylearning.model.logisticLasso import LogisticLasso

tags = ["Exitus", "Sexo", "Localización Primario (Derecho, Izquierdo, Recto)",
        "ECOG", "Localización Metastásica", "Número Localizaciones Metastásicas",
        "Tipo Histológico", "CIA de Rescate", "CIA anterios a BL",
        "Tto 1ª Línea", "Respuesta", "Resultado Biopsia Tejido",
        "Resultado Biopsia Líquida"]
# Loading File

file = FileCsv("./datasetSurvival/extended.csv", sep = "\t", decimal = ",")

for tag in tags:

    data = file.selectData(tag, "hsalet7a_000377", "rnomiR7#_001338")
    print("Preprocessing", tag)
    data.purge()
    data.clean()
    data.encodeCategorical()
    data.imputeEmptyValues()
    data.scaleStandard()

    print("Processing", tag)
    model = LogisticLasso(data)
    model.model.set_params(**{"multi_class" : "ovr"})

    model.runClassicModel()

    report = Report(model)
    report.generate(f"class_{tag}")