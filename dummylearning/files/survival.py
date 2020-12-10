import pandas as pd
import numpy as np
from dummylearning.files.base import FileCsvBase

class FileCsv(FileCsvBase):

    """
    Csv File class

    Parameters
    ---------------------------------------------------------------------------
        file    <str>         (positional)  => File name
        sep     <str>         (default: ;)  => Field separator character
        decimal <str>         (default: .)  => Decimal separator character

    Attributes
    ---------------------------------------------------------------------------
        dataset <pandas.DataFrame> => Dataframe that contains data

    Methods
    ---------------------------------------------------------------------------
        removeFields => Modify self.dataset removing selected fields
        selectData   => Return a Data class structure. Split selected fields
                        into Data.values (X) and Data.tags (Y)
    """

    def __init__(self, file: str,
                       sep: str = ";",
                       decimal: str = ".",
                       verbose = True) -> None:

        super().__init__(verbose)

        self.dataset = pd.read_csv(file, sep = sep, decimal = decimal)


    def remove(self, remove: list) -> None:

        """
        Function -> removeFields
        Remove fields setted in 'remove' parameter

        Parameters
        ---------------------------------------------------------------------------
            remove <list<str>> (positional) => List contains fields's name
                                               to remove

        Return
        ---------------------------------------------------------------------------
            None => Modify self.dataset
        """

        for element in remove:
            del self.dataset[element]

    def selectData(self, eventTagName, eventTagTime, startColumn, endColumn):

        """
        Function -> selectData
        Create a Data instance with 'tagName' as tag (Y) and dataframe contained
        between 'startColumn' and 'endColumn' as (X)

        Parameters
        ---------------------------------------------------------------------------
            tagName     <str> (positional) => Y field name
            startColumn <str> (positional) => X first field name
            endColumn   <str> (positional) => X last field name

        Return
        ---------------------------------------------------------------------------
            Data instance
        """

        from dummylearning.datas.survival import Data

        tags = pd.DataFrame()
        tags[eventTagName] = self.dataset[eventTagName]
        tags[eventTagTime] = self.dataset[eventTagTime]
        values = self.dataset.iloc[:, list(self.dataset.columns).index(startColumn) :\
                                      list(self.dataset.columns).index(endColumn) + 1] # +1 to get this field included

        return Data(values, tags)