from dummylearning.fileCsvSurvival import FileCsv
from dummylearning.dataSurvival import Data
from dummylearning.model.coxLasso import SurvivalLasso
from dummylearning.plotsSurvival import Plots
from dummylearning.reportSurvival import Report
from dummylearning.analysisSurvival import Analysis

#fileNames = ("./datasetSurvival/all.csv", "./datasetSurvival/extended.csv")
times = ("Supervivencia", "Progresión")

fileNames = ["./datasetSurvival/extended.csv"]
#times = ["Progresión"]

for fileName in fileNames:

    if fileName == "./datasetSurvival/all.csv":
        blocks = {"coplete"   : ("Sexo", "hsa-miR-98"),
                  "molecular" : ("hsa-let-7c", "hsa-miR-98"),
                  "clinical"  : ("Sexo", "Galectin-9")}
    else:
        blocks = {"coplete"   : ("Sexo", "rnomiR7#_001338"),
                  "molecular" : ("hsalet7a_000377", "rnomiR7#_001338"),
                  "clinical"  : ("Sexo", "Galectin-9")}

    for time in times:
        for block in blocks:
            processName = fileName.split("/")[-1].split(".")[0] + "_" + time + "_" + block
            print(processName)

            file = FileCsv(fileName, sep = "\t", decimal = ",")
            data = file.selectData("Exitus",
                                time,
                                blocks[block][0],
                                blocks[block][1])

            data.purge()
            data.clean()
            data.encodeCategorical()
            data.imputeEmptyValues()
            data.scaleStandard()

            model = SurvivalLasso(data)
            #model.optimizePenalty()
            model.model.set_params(**{"alphas" : [0.1]})
            model.runClassicModel()

            report = Report(model)
            report.generate(f"../{processName}")

            values, names = Analysis(model).coefficients()

            with open(f"../{processName}/coef.csv", "w") as outFile:
                outFile.write("VARIABLE\tVALUE\n")

                for name, value in zip(names, values):
                    outFile.write(f"{name}\t{value}\n")