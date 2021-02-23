from dummylearning.utilities.info import Info
import pandas as pd
import numpy as np

class DataBase(Info):


    def __init__(self, values, tags, verbose):
        super().__init__(verbose)

        self.originalValues = values
        self.originalTags = tags

        self._values = values
        self._tags = tags





    @property
    def values(self):
        """Getter of self.values"""
        return self._values.to_numpy()


    @property
    def tags(self):
        """Getter of self.tags"""
        return self._tags.to_numpy()


    @property
    def tagName(self):
        """Getter of self.tagName"""
        return self._tags.name


    @property
    def valuesName(self):
        """Getter of self.valuesName"""
        return list(self._values.columns)


    @property
    def dataframe(self):
        """Getter self.__values"""
        aux = self._values
        aux[self._tags.name] = self._tags
        return aux




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
        for column in self._values.columns:

            if self._values[column].dtype != "O": # If column has not "str" data types
                auxData[column] = self._values[column] # Do nothing, only saved into auxiliar dataframe

            else: # If the column has "str" data types
                categoricalCounter += 1
                if not self._values[column].hasnans: # If there are not empty values

                    # Get column (vector form)
                    array = self._values[column].to_numpy()

                    # Tranform vector to array with n rows and one column
                    array = array.reshape(array.shape[0], 1)

                    # Encoder things =)
                    encoder = OneHotEncoder()
                    encoder.fit(array)

                    # Get array with n rows and m column (m = number of classes)
                    transformed = encoder.transform(array).toarray()

                    # Loop iterating caregories (saved into encoder.categories_[0] (first, only one becuase we are doing column by column))
                    for index, category in enumerate(encoder.categories_[0]):

                        if len(encoder.categories_[0]) == 2 and index == 1:
                            pass
                        # Creating new column called variables_category
                        #     with correct transformed array column
                        else:
                            auxData[f"{column}_{category}"] = transformed[:, index]

                else: # If the column has some empty value
                    categoricalEmptyCounter += 1
                     # Do nothing, only saved into auxiliar dataframe
                     #     We need to conserve "str" data type,
                     #     then we can detect the column and apply some model imputation
                    auxData[column] = self._values[column]

        self.upgradeInfo(f"Detected {categoricalCounter} categorical variables\n\t{categoricalCounter - categoricalEmptyCounter} variables converted to One Hot Encoding")
        self.upgradeInfo(f"Detected {categoricalEmptyCounter} categorical variables with empty values. Use imputeEmptyValues()")

        self._values = auxData # Overwriting self.__values with auxiliar dataframe




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
        indexToRemove = set()

        if isinstance(self._tags, pd.Series):

            for index, value in zip(self._tags.index, self._tags.isna()): # (index <int> , value <bool>)

                if value: indexToRemove.add(index) # Adding index to auxiliar list

        else:

            for column in self._tags.columns:
                for index, value in zip(self._tags. index, self._tags[column].isna()):

                    if value: indexToRemove.add(index)

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

        self.upgradeInfo("\n\t".join(["Dataset cleaned",
                                      f"   Rows: Initial {len(self._values.index) + len(indexToRemove)} -> Final {len(self._values.index)}",
                                      f"Columns: Initial {len(list(self._values.columns)) + len(columnToRemove)} -> Final {len(list(self._values.columns))}"]))




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

            self.upgradeInfo("Imputing numerical empty values with KNN method")

            # Auxiliar dataframe for saving results
            auxData = pd.DataFrame()

            # Loop iterates each column
            for column in self._values.columns:

                if self._values[column].dtype != "O": # If column has not "str" data type
                    auxData[column] = self._values[column] # Only saving

            # auxData has only numerical columns
            #     We perform de imputation (bad looking but this works) and
            #     saving results into a dataframe (overwriting)
            auxData = pd.DataFrame(data = KNNImputer(n_neighbors = 6).fit_transform(auxData.to_numpy()),
                                   columns = list(auxData.columns))

            # We change self.__values with auxData imputed columns
            for column in auxData.columns: # Only numerical columns could change
                self._values[column] = auxData[column]

            self.upgradeInfo("Numerical empty values imputed")

        def imputeCategoricalVariables():
            # Second step, impute categorical empty values
            self.upgradeInfo("Imputing empty categorical values")

            # Auxiliar dataframe for saving columns without empty values
            moldData = pd.DataFrame()

            # Auxiliar dataframe for saving columns with empty values (numerical already imputed)
            catData = pd.DataFrame()

            for column in self._values.columns:

                # Filled categorical variables are already One Hot Enconde,
                #     so they are 0 or 1 (float64 data type)
                if self._values[column].dtype != "O":
                    moldData[column] = self._values[column]

                # After One Hot Encoding, if column still "O" data type
                #     means that column is still having empty categorical values
                else:
                    catData[column] = self._values[column]

            # Iterating over empty categorical columns
            for column in catData.columns:

                self.upgradeInfo(f"\tImputing {column} empty values")
                # Creating Data instance. Amazing process
                dataset = DataBase(moldData, catData[column], verbose = False)
                dataset.purge() # Removing empty categorical values. But remember that we saved
                                #     removed samples into self._removedValues. Magic!

                # Developing fast model (Random Forest is the best option)

                from dummylearning.models.classification.randomForest import RandomForest
                model = RandomForest(dataset, verbose = False)
                model.runProductionModel()


                for index in dataset._valuesRemoved.index: # Tachan! We have used self.__removedValues!!

                    # Making predictions for each empty categorical value. Bad looking but it works
                    predicted = model.model.predict(dataset._valuesRemoved.loc[index].to_numpy().reshape(1, dataset._valuesRemoved.loc[index].to_numpy().shape[0]))

                    # Setting imputed value into main self.__values.
                    #     Question. We have made a reindexation process with dataset.purge.
                    #     Did we lose our index for assignment imputed categorical value??

                    # Assign imputed categorical value like a surgeon
                    self._values.iloc[index, self._values.columns.get_loc(column)] = predicted[0]

            self.upgradeInfo("Empty categorical values imputed")

        # Core function pipe
        imputeNumericalVariables() # Frist Numerical!
        imputeCategoricalVariables() # Second Categorical!
        self.encodeCategorical() # At the end, we perform One Hot Enconding, all variables will be float64
        self.valuesImputed = self._values
        self.tagsImputed = self._tags



    #_______________________________Scaling  Section_______________________________




    def scaleMinMax(self):

        """
        Function -> scaleMinMax
        Scale self._values columns between 0 and 1 by column

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self._values
        """

        from sklearn.preprocessing import MinMaxScaler

        # Transform to array
        auxData = self._values.to_numpy()

        # Scaler things =)
        scaler = MinMaxScaler().fit(auxData)

        # Transform data
        auxData = scaler.transform(auxData)

        # Saving again as dataframe
        self._values = pd.DataFrame(data = auxData,
                                    columns = self._values.columns)

        self.upgradeInfo("Min-Max scaling method performed")


    def scaleStandard(self):

        """
        Function -> scaleStandard
        Scale self._values columns with mean = 0 and std = 1 locally

        Parameters
        ---------------------------------------------------------------------------
            None

        Return
        ---------------------------------------------------------------------------
            None => Modify self._values
        """

        from sklearn.preprocessing import StandardScaler

        # Transform to array
        auxData = self._values.to_numpy()

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
        self._values = pd.DataFrame(data = auxData,
                                     columns = self._values.columns)

        self.upgradeInfo("Standard scaling method performed")