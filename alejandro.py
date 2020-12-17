from dummylearning.files.classification import FileCsv

file = FileCsv("./datos.csv", sep = ";", decimal = ",")

classes = ["Cured", "Controled", "Macro_micro", "Suprasellar", "Infrasellar",
           "Sinus_invasion", "pretreat_KETO"]

blocks = {"complete"  : ("Sexo", "PTTG1_HPRT1"),
          "clinical"  : ("Sexo", "Cortisol"),
          "molecular" : ("GH1_HPRT1", "PTTG1_HPRT1")}

for clas in classes:
    for block in blocks:
        start, end = blocks[block][0], blocks[block][1]
        file = FileCsv("./datos.csv", sep = ";", decimal = ",")

        data = file.selectData(clas, start, end)
        data.purge()
        data.clean()
        data.encodeCategorical()
        data.imputeEmptyValues()
        data.scaleStandard()

        from dummylearning.models.classification.logisticLasso import LogisticLasso
        
        model = LogisticLasso(data)
        model.runProductionModel()

        from dummylearning.reports.classification import Report
        report = Report(model)
        report.generate(f"../alejandro/{clas}_{block}")





