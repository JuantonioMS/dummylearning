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
            _values  <pd.DataFrame> => Dataframe containing X data
            _tags    <pd.Series>    => Dataframe containing Y data

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

        super().__init__(values, tags, verbose)

        self.upgradeInfo(f"Creating Data instance for tag => {self.tagName}")

    #_______________________________Cleaning Section_______________________________




    def purge(self):

        """
        Function -> purge
        Remove samples with empty tag. We can not impute a tag.

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self._values
        """

        self.upgradeInfo("Purging dataset from empty tags")

        # Auxiliar list for saving indexes to remove
        indexToRemove = []
        for index, value in zip(self._tags.index, self._tags.isna()): # (index <int> , value <bool>)

            if value: # If there is some empty value
                indexToRemove.append(index) # Adding index to auxiliar list

        self.upgradeInfo(f"Detected {len(indexToRemove)} empty tags")

        # We save removed data because we are going to use this variable for categorical imputation
        #     Only used in categorical imputation!!
        self._valuesRemoved = self._values.loc[indexToRemove]

        # Removing indexes from self.__tags and self.__values
        self._tags = self._tags.drop(indexToRemove, axis = 0) # axis setted as 0 means rows
        self._values = self._values.drop(indexToRemove, axis = 0)

        # Changing dataframe index for self.__tags and self.__values
        #     Important because if we transform a dataframe with numerical indexes to numpy array
        #     numpy will write an empty row if there is a lost index
        self._values.index = list(range(0, self._values.shape[0]))
        self._tags.index = list(range(0, self._tags.shape[0]))

        self.upgradeInfo(f"Dataset purged\n\tInitial {len(self._values.index) + len(indexToRemove)} -> Final {len(self._values.index)}")




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
            None => Modify self._values
        """

        self.upgradeInfo("Cleaning dataset of too empty columns and rows")

        self.upgradeInfo(f"Cleaning rows with empty ratio greater than {sampleRatio}")
        # Cleaning rows!
        # Auxiliar list for saving indexes to remove
        indexToRemove = []
        for index in self._values.index:
            row = self._values.loc[index] # Selecting row by index
            if sum(row.isna()) / row.shape[0] > sampleRatio: # If empty data ratio is greater than setted ratio
                indexToRemove.append(index) # Adding index to auxiliar list

        self.upgradeInfo(f"Detected {len(indexToRemove)} samples too empty")

        # Removing indexes from self.__tags and self.__values
        self._values = self._values.drop(indexToRemove, axis = 0) # axis setted as 0 means rows
        self._tags = self._tags.drop(indexToRemove, axis = 0) # ONLY THIS HAS TO BE CHANGED

        self.upgradeInfo(f"Cleaning columns with empty ratio greater than {columnRatio}")

        # Cleaning columns!
        # Auxiliar list for saving column names to remove
        columnToRemove = []
        for columnName in self._values.columns:
            column = self._values[columnName]

            if sum(column.isna()) / self._values.shape[0] > columnRatio: # If empty data ratio is greater than setted ratio
                columnToRemove.append(columnName) # Adding column name to auxiliar list

        self.upgradeInfo(f"Detected {len(columnToRemove)} columns too empty")

        # Removing columns from self.__values
        self._values = self._values.drop(columnToRemove, axis = 1) # axis setted as 1 means rows

        # Changing dataframe index for self.__tags and self.__values
        #       Important because if we transform a dataframe with numerical indexes to numpy array
        #       numpy will write an empty row if there is a lost index
        self._values.index = list(range(0, self._values.shape[0]))
        self._tags.index = list(range(0, self._tags.shape[0])) #FOR SURVIVIAL WE REMOVE THIS LINE

        self.upgradeInfo(f"Dataset purged\n\tInitial {len(self._values.index) + len(indexToRemove)} -> Final {len(self._values.index)}")
        self.upgradeInfo("\n\t".join(["Dataset cleaned",
                                      f"   Rows: Initial {len(self._values.index) + len(indexToRemove)} -> Final {len(self._values.index)}",
                                      f"Columns: Initial {len(list(self._values.columns)) + len(columnToRemove)} -> Final {len(list(self._values.columns))}"]))




if __name__ == "__main__":

    NaN = np.nan
    values = {"tag"     : ["Cat", "Cat", "Dog", "Dog", "Horse", "Horse", NaN],
              "column1" : [12, NaN, 13, 43, 53, 43, 3],
              "column2" : [6, NaN, 7, 2, 8, 3, 5],
              "column3" : [4, 4, NaN, 12, NaN, 6, 6],
              "column4" : ["Yes", NaN, "Yes", "No", "Yes", "No", "Yes"],
              "column5" : ["A", "B", "C", "A", "B", "C", "A"],
              "column6" : ["Yes", "No", "Yes", "No", NaN, NaN, "No"],
              "column7" : ["A", "B", "C", "A", "B", NaN, "A"],
              "column8" : [NaN, NaN, NaN, NaN, "B", NaN, "A"]}

    values = pd.DataFrame(data = values)
    tags = values["tag"]
    del values["tag"]

    test = Data(values, tags)
    test.purge()
    test.clean()
    test.encodeCategorical()
    test.imputeEmptyValues()
    test.scaleStandard()


