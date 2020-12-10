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

        super().__init__(values, tags, verbose)





    #________________________________Getter Section________________________________




    @property
    def tags(self):
        """Getter of self.tags"""

        tag = list(zip(self._tags[list(self._tags.columns)[0]],
                       self._tags[list(self._tags.columns)[1]]))

        tag = [(True, float(days)) if event == "Yes" else (False, float(days)) for event, days in tag]
        tag = np.array(tag, dtype = [("Status", "?"), ("Time_in_days", "<f8")])

        return tag


    @property
    def tagName(self):
        """Getter of self.tagName"""
        return list(self._tags.columns)


    @property
    def dataframe(self):
        """Getter self.__values"""
        data = self._values
        for column in self._tags.columns:
            data[column] = self._tags[column]

        return data
