import numpy as np
import pandas as pd

from dummylearning.datas.base import DataBase



class Data(DataBase):

    """
    Data class

    Parameters
    ---------------------------------------------------------------------------
        values <pd.DataFrame> (positional) => Dataframe containing X values
        tags   <pd.Series>    (positional) => Dataframe column containing
                                              Y values

    Attributes
    ---------------------------------------------------------------------------

        *Main__________________________________________________________________
            __values  <pd.DataFrame> => Dataframe containing X data
            __tags    <pd.Series>    => Dataframe containing Y data

            values    <np.array>     => Numpy array containing X data
            tags      <np.array>     => Numpy array containing Y data

            dataframe <pd.DataFrame> => Dataframe containing X and Y data

        *Info__________________________________________________________________
            tagName    <str>       => Field name of Y
            valuesName <list<str>> => Fields names of X

    Methods
    ---------------------------------------------------------------------------

        *Cleaning______________________________________________________________
            purge => Remove empy tag sample
            clean => Clean rows and columns with too much empty data

        *Encoding______________________________________________________________
            encodeCategorical => Perform One Hot Encoding method

        *Imputing______________________________________________________________
            imputeEmptyValues => Impute numerical and categorical empty values
                                 Perform encodeCategorical before!

        *Scaling_______________________________________________________________
            scaleMinMax   => Scale self.__values columns between 0 and 1
                             by column
            scaleStandard => Scale self.__values columns with mean = 0 and
                             std = 1 locally
                             !WARNING Solve categorical scaling
                             TODO Solve negative and positive coefs effect
    """

    def __init__(self, values, tags, verbose = True):

        super().__init__(verbose)

        self.__values = values
        self.__tags = tags




    #________________________________Getter Section________________________________


    @property
    def values(self):
        """Getter of self.values"""
        return self.__values.to_numpy()

    @property
    def tags(self):
        """Getter of self.tags"""
        return self.__tags

    @property
    def valuesName(self):
        """Getter of self.valuesName"""
        return list(self.__values.columns)

    @property
    def dataframe(self):
        """Getter self.__values"""
        return self.__values


    #_______________________________Cleaning Section_______________________________

    def purge(self):

        boolean = np.isnan(self.tags["Survival_in_days"]) + np.isnan(self.tags["Status"])

        indexToRemove = []
        aux = np.array([], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])

        for index, empty in enumerate(boolean):
            if not empty:
                aux = np.append(aux, self.tags[index])
            else:
                indexToRemove.append(index)

        self.__tags = aux

        self.__values = self.__values.drop(indexToRemove, axis = 0)
        self.__values.index = list(range(0, self.__values.shape[0]))




    def clean(self, sampleRatio = 0.4, columnRatio = 0.5):

        """
        Function -> clean
        Eliminate rows with more empty data ratio than <sampleRatio>
        Eliminate columns with more empty data ratio than <columnRatio>
        First rows, then columns

        Parameters
        ---------------------------------------------------------------------------
            sampleRatio <float> (default: 0.4) => Row empty data ratio
            columnRatio <float> (default: 0.5) => Column empty data ratio

        Return
        ---------------------------------------------------------------------------
            None => Modify self.__values
        """

        self.upgradeInfo("Cleaning dataset of too empty columns and rows")

        self.upgradeInfo(f"Cleaning rows with empty ratio greater than {sampleRatio}")
        # Cleaning rows!
        # Auxiliar list for saving indexes to remove
        indexToRemove = []
        for index in self.__values.index:
            row = self.__values.loc[index] # Selecting row by index
            if sum(row.isna()) / row.shape[0] > sampleRatio: # If empty data ratio is greater than setted ratio
                indexToRemove.append(index) # Adding index to auxiliar list

        self.upgradeInfo(f"Detected {len(indexToRemove)} samples too empty")

        # Removing indexes from self.__tags and self.__values
        self.__values = self.__values.drop(indexToRemove, axis = 0) # axis setted as 0 means rows
        self.__tags = np.delete(self.__tags, indexToRemove)

        self.upgradeInfo(f"Cleaning columns with empty ratio greater than {columnRatio}")

        # Cleaning columns!
        # Auxiliar list for saving column names to remove
        columnToRemove = []
        for columnName in self.__values.columns:
            column = self.__values[columnName]

            if sum(column.isna()) / self.__values.shape[0] > columnRatio: # If empty data ratio is greater than setted ratio
                columnToRemove.append(columnName) # Adding column name to auxiliar list

        self.upgradeInfo(f"Detected {len(columnToRemove)} columns too empty")

        # Removing columns from self.__values
        self.__values = self.__values.drop(columnToRemove, axis = 1) # axis setted as 1 means rows
        self.__values.index = list(range(0, self.__values.shape[0]))

        self.upgradeInfo(f"Dataset purged\n\tInitial {len(self.__values.index) + len(indexToRemove)} -> Final {len(self.__values.index)}")
        self.upgradeInfo("\n\t".join(["Dataset cleaned",
                                      f"   Rows: Initial {len(self.__values.index) + len(indexToRemove)} -> Final {len(self.__values.index)}",
                                      f"Columns: Initial {len(list(self.__values.columns)) + len(columnToRemove)} -> Final {len(list(self.__values.columns))}"]))
