from dummylearning.utilities.info import Info
from dummylearning.datas.classification import Data
import pandas as pd
import numpy as np

class DataBase(Info):


    def __init__(self, verbose):
        super().__init__(verbose)





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

            self.upgradeInfo("Imputing numerical empty values with KNN method")

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

            self.upgradeInfo("Numerical empty values imputed")

        def imputeCategoricalVariables():
            # Second step, impute categorical empty values
            self.upgradeInfo("Imputing empty categorical values")

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

                self.upgradeInfo(f"\tImputing {column} empty values")
                # Creating Data instance. Amazing process
                dataset = Data(moldData, catData[column], verbose = False)
                dataset.purge() # Removing empty categorical values. But remember that we saved
                                #     removed samples into self.__removedValues. Magic!

                # Developing fast model (Random Forest is the best option)

                from dummylearning.models.classification.randomForest import RandomForest
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

            self.upgradeInfo("Empty categorical values imputed")

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