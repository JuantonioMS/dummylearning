import numpy as np
import pandas as pd
from dummylearning.info import Info



class Data(Info):

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

    def purge(data):
    
        boolean = np.isnan(data.tags["Survival_in_days"]) + np.isnan(data.tags["Status"])
        
        aux = np.array([], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])
        for index, empty in enumerate(boolean):
            if not empty:
                aux = np.append(aux, data.tags[index])
        
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
            column = self.__values[columnNameif there is a lost index
        self.__values.index = list(range(0, self.__values.shape[0]))

        self.upgradeInfo(f"Dataset purged\n\tInitial {len(self.__values.index) + len(indexToRemove)} -> Final {len(self.__values.index)}")
        self.upgradeInfo("\n\t".join(["Dataset cleaned",
                                      f"   Rows: Initial {len(self.__values.index) + len(indexToRemove)} -> Final {len(self.__values.index)}",
                                      f"Columns: Initial {len(list(self.__values.columns)) + len(columnToRemove)} -> Final {len(list(self.__values.columns))}"]))






    #_______________________________Encoding Section_______________________________




    def encodeCategorical(self):

        """
        Function -> encodeCategorical
        Perform One Hot Encoding method for categorical variables

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self.__values
        """

        from sklearn.preprocessing import OneHotEncoder

        self.upgradeInfo("Encoding categorical variables with One-Hot-Encoding method")

        # Auxiliar dataframe for saving results
        auxData = pd.DataFrame()

        categoricalCounter = 0
        categoricalEmptyCounter = 0
        for column in self.__values.columns:

            if self.__values[column].dtype != "O": # If column has not "str" data types
                auxData[column] = self.__values[column] # Do nothing, only saved into auxiliar dataframe

            else: # If the column has "str" data types

                categoricalCounter += 1
                if not self.__values[column].hasnans: # If there are not empty values

                    # Get column (vector form)
                    array = self.__values[column].to_numpy()

                    # Tranform vector to array with n rows and one column
                    array = array.reshape(array.shape[0], 1)

                    # Encoder things =)
                    encoder = OneHotEncoder()
                    encoder.fit(array)

                    # Get array with n rows and m column (m = number of classes)
                    transformed = encoder.transform(array).toarray()

                    # Loop iterating caregories (saved into encoder.categories_[0] (first, only one becuase we are doing column by column))
                    for index, category in enumerate(encoder.categories_[0]):

                        # Creating new column called variables_category
                        #     with correct transformed array column
                        auxData[f"{column}_{category}"] = transformed[:, index]

                else: # If the column has some empty value
                    categoricalEmptyCounter += 1
                     # Do nothing, only saved into auxiliar dataframe
                     #     We need to conserve "str" data type,
                     #     then we can detect the column and apply some model imputation
                    auxData[column] = self.__values[column]

        self.upgradeInfo(f"Detected {categoricalCounter} categorical variables\n\t{categoricalCounter - categoricalEmptyCounter} variables converted to One Hot Encoding")
        self.upgradeInfo(f"Detected {categoricalEmptyCounter} categorical variables with empty values. Use imputeEmptyValues()")


        self.__values = auxData # Overwriting self.__values with auxiliar dataframe




    #_______________________________Imputing Section_______________________________




    def imputeEmptyValues(self):

        """
        Function -> imputeEmptyValues
        Impute numerical and categorical empty values
        First numerical, then categorical
        Perform One HoT Encoding at the end
        Important to perform self.encodingCategorical() before!!

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self.__values
        """

        def imputeNumericalVariables():
            # First step, impute numerical empty values

            from sklearn.impute import KNNImputer

            # Auxiliar dataframe for saving results
            auxData = pd.DataFrame()

            # Loop iterates each column
            for column in self.__values.columns:

                if self.__values[column].dtype != "O": # If column has not "str" data type
                    auxData[column] = self.__values[column] # Only saving

            # auxData has only numerical columns
            #     We perform de imputation (bad looking but this works) and
            #     saving results into a dataframe (overwriting)
            auxData = pd.DataFrame(data = KNNImputer(n_neighbors = 6).fit_transform(auxData.to_numpy()),
                                   columns = list(auxData.columns))

            # We change self.__values with auxData imputed columns
            for column in auxData.columns: # Only numerical columns could change
                self.__values[column] = auxData[column]


        def imputeCategoricalVariables():
            # Second step, impute categorical empty values
            from dummylearning.dataClasification import Data as DataClas
            from dummylearning.model.randomForest import RandomForest

            # Auxiliar dataframe for saving columns without empty values
            moldData = pd.DataFrame()

            # Auxiliar dataframe for saving columns with empty values (numerical already imputed)
            catData = pd.DataFrame()

            for column in self.__values.columns:
                
                # Filled categorical variables are already One Hot Enconde,
                #     so they are 0 or 1 (float64 data type)
                if self.__values[column].dtype != "O":
                    moldData[column] = self.__values[column]

                # After One Hot Encoding, if column still "O" data type
                #     means that column is still having empty categorical values
                else:
                    catData[column] = self.__values[column]

            # Iterating over empty categorical columns
            for column in catData.columns:
                # Creating Data instance. Amazing process
                dataset = DataClas(moldData, catData[column], verbose = False)
                dataset.purge() # Removing empty categorical values. But remember that we saved
                                #     removed samples into self.__removedValues. Magic!

                # Developing fast model (Random Forest is the best option)
                
                
                model = RandomForest(dataset, verbose = False)
                model.runProductionModel()


                for index in dataset.__valuesRemoved.index: # Tachan! We have used self.__removedValues!!

                    # Making predictions for each empty categorical value. Bad looking but it works
                    predicted = model.model.predict(dataset.__valuesRemoved.loc[index].to_numpy().reshape(1, dataset.__valuesRemoved.loc[index].to_numpy().shape[0]))
                    
                    # Setting imputed value into main self.__values.
                    #     Question. We have made a reindexation process with dataset.purge.
                    #     Did we lose our index for assignment imputed categorical value??
                    
                    # Assign imputed categorical value like a surgeon
                    self.__values.iloc[index, self.__values.columns.get_loc(column)] = predicted[0]

        # Core function pipe
        imputeNumericalVariables() # Frist Numerical!
        imputeCategoricalVariables() # Second Categorical!
        self.encodeCategorical() # At the end, we perform One Hot Enconding, all variables will be float64




    #_______________________________Scaling  Section_______________________________




    def scaleMinMax(self):

        """
        Function -> scaleMinMax
        Scale self.__values columns between 0 and 1 by column

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self.__values
        """

        from sklearn.preprocessing import MinMaxScaler

        # Transform to array
        auxData = self.__values.to_numpy()

        # Scaler things =)
        scaler = MinMaxScaler().fit(auxData)

        # Transform data
        auxData = scaler.transform(auxData)

        # Saving again as dataframe
        self.__values = pd.DataFrame(data = auxData,
                                     columns = self.__values.columns)

        self.upgradeInfo("Min-Max scaling method performed")




    def scaleStandard(self):

        """
        Function -> scaleStandard
        Scale self.__values columns with mean = 0 and std = 1 locally

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self.__values
        """

        from sklearn.preprocessing import StandardScaler

        # Transform to array
        auxData = self.__values.to_numpy()

        # Scaler things =)
        scaler = StandardScaler().fit(auxData)

        # Transform data
        auxData = scaler.transform(auxData)

        # Rescaling to 1 and 0 categorical columns
        for column in range(auxData.shape[1]): # Iterating over columns

            # If there are only 2 differents values in the column
            if len(set(auxData[:, column])) == 2:
                newColumn = [] # auxiliar list

                negative = np.min(auxData[:, column]) # Minimum value is 0

                for element in auxData[:, column]: #Iterating over elements
                    if element == negative:
                        newColumn.append(0)
                    else:
                        newColumn.append(1)

        # Saving again as dataframe
        self.__values = pd.DataFrame(data = auxData,
                                     columns = self.__values.columns)

        self.upgradeInfo("Standard scaling method performed")


