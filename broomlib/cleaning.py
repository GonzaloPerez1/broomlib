from .utils import *
import pandas as pd

def show_all(df, big_desc = False):
    """
    -----------
       Function description:
        The function show, the info, number of rows, columns, and a big or small descriptive with a lot of info, mode, mean median missings, categorical, numerical columns ... .
        
    -----------
        Parameters:
        <df> (Pandas DataFrame type): 
                mandatory parameter. Dataframe  we want to check .
        
        <big_desc> (boolean): Default value = False.
                True if you want see:
                    count, nulls, type, unique, numeric, mode, mean, min, quantile, median, max, std, skew and kurt of each column .

                
        
    -----------
        Returns: If big_desc = True , a new Dataframe with a big describe, 
                else, the info, shape of row and columns and the describe of pandas.
    
    """
    
    
    if big_desc == True:
        output_df = pd.DataFrame(columns=['Count','Missing','Unique','Dtype','Numeric','Mode','Mean','Min','25%','Median','75%','Max','Std','Skew','Kurt'])
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype !='bool' and pd.isnull(df[col]).all()!=True:
                output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype,pd.api.types.is_numeric_dtype(df[col]),
                                      df[col].mode().values[0], df[col].mean(),df[col].min(), df[col].quantile(0.25), df[col].median(),
                                      df[col].quantile(0.75), df[col].max(), df[col].std(), df[col].skew(), df[col].kurt()]
            else:
                output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype,pd.api.types.is_numeric_dtype(df[col]),
                                       '-' if pd.isnull(df[col]).all() else df[col].mode().values[0],'-','-','-','-','-','-','-','-','-']
        print(df.info())
        return output_df.sort_values(by=['Numeric', 'Skew','Unique'],ascending=False)
    
    else:
        print(df.info())
        print("Rows:",df.shape[0])
        print("Columns:", df.shape[1])
        print(df.describe(include='all'))
        
 
        
def data_report(df):
    """
    -----------
       Function description:
        The function shows a small  summary of the type , percentage of missings, unique values and cardinality of each column.
        
    -----------
        Parameters:
        <df> (Pandas DataFrame type): 
                mandatory parameter. Dataframe  we want to check.
           
    -----------
        Returns: Dataframe
                From the original Dataframe, return a new dataframe with the type, percentage of missing , unique values and cadinality of each column.
    """
    # Take the NAMES of columns
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Take the TYPE
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Take  MISSINGS and his percentage
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Take UNIQUE values
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T



def how_missing(df,threshold = 0.0, drop = False):
    """
    -----------
       Function description:
        The function search for missing values, count them and show % of it.
        Optionally display only from a certain percentage of missing, and if the user wants,
        the function deletes columns exceeding threshold from the df .
        
    -----------
        Parameters:
        <df> (Pandas DataFrame type): 
                mandatory parameter. Dataframe  we want to check percentage of missing and maybe delete columns that exceed threshold.
        
        <threshold> (float): {0.0 to 100.0}. Default value = 0.
                Percentage from which we want to see how much missing the column has.
                
        <drop>(boolean): True or False. Default value = False.
                True if you want delete columns exceeding threshold.
                ¡¡¡Warning, delete from the original df!!
                
        
    -----------
        Returns: Dataframe
                From the original Dataframe, return a new dataframe corresponding to the selected threshold,
                that have the name of the columns and each percentage of missing.
    """
    
    if 0 <= threshold <= 100:
    
        #Take the names of columns
        cols = pd.DataFrame(df.columns.values, columns=["COL_N"])


        # Take the missing
        percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
        percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

        concatenado = pd.concat([cols, percent_missing_df], axis=1, sort=False)
        concatenado.set_index('COL_N', drop=True, inplace=True)

        dfaux = concatenado.T
        dfaux = dfaux[:][dfaux[:] > threshold]

        columns_out =  list(dfaux.dropna(axis = 'columns').columns)
        df.drop(columns_out, axis = 1, inplace = drop)

        return dfaux.dropna(axis = 'columns')
    
    else:
        return print('ValueError: Threshold must be >= 0, and <= 100, try it again')