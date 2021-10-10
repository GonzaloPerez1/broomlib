import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

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
    
    


def checking_duplicates(df, col):
    """
    This function's objective is to check if there is some duplicated values due to capital/small letters in a column made of strings. 
    
    For example: 'Spain' and 'spain'.
    
    Parameters:
        - df: it is the name of the database,
        - col: this is the name of the column that will be checked. The elements of this columns should be strings.
        
    The function will return which values are duplicated in the column. If there are no duplicate, it will return this information.  
    """
    
    lower_transformation = pd.DataFrame(df[col].str.lower().value_counts().reset_index())
    
    duplicates = False
    
    for i in range(0, len(lower_transformation.index)):
        for j in range(1,2):
            if lower_transformation.iloc[i, j] > 1:
                duplicates = True
                print(lower_transformation.iloc[i, 0], "is present", lower_transformation.iloc[i,j], "times in the column", col,".")
    
    if duplicates == True:
        print("There are no other duplicate values.")
        
    elif duplicates == False:
        print("There are no duplicates in the column", col,".")
        

def building_date(year, month, day, df):
    """
    It creates a single column with the date from 3 distinct columns where are the years, months and days.
    
    Arguments:
        - year: it is the name of the column where the years are showed. The values are to be integers,
        - month: it is the name of the column where the months are showed. The values can be strings with the name of the month written in English or integers between 1 and 12,
        - day: it is the name of the column where the days are showed. The values are to be integers between 1 and 31 according to the month,
        - df: it is the name of the dataframe.
    
    The function will return the dataframe with a new column 'Parsing date' in the datetime64 type. It will remove the columns with the years, months and days.
    """
    
    df["Parsing date"] = pd.to_datetime(df[year].astype(str) + '/' + df[month].astype(str) + '/' + df[day].astype(str))
    
    df = df.drop(columns=[year, month, day])
    
    return df



def search_corr(df, threshold):
    """
    Search for correlations within the indicated dataframe based on the value that we put in the threshold
    
    Arguments:
    
        - df (Pandas DataFrame type): it is the name of the dataframe we want to search correlations
        - threshold (float): it is the value of correlation from which we want to obtain results
        
    
    If the function finds elements that exceed the indicated threshold, it indicates the total number of elements that exceed the threshold 
    and their name in the form of a list
    
    """
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    corr_list = [column for column in corr_matrix.columns if any(upper[column] > threshold)]
    return print("Total Elements:", len(corr_list), corr_list)



def suggested_merge(df1, df2):
    """
    Compare a dataframe with another dataframe in search of columns in common and recommend which type of union according to its composition is the most appropriate
    
    Arguments:
    
        - df1 (Pandas DataFrame type): it is the name of the first dataframe we want to compare
        - df2 (Pandas DataFrame type): it is the name of the second dataframe we want to compare
        
    If the function finds common columns, it indicates what they are and the type of union recommended.
    Otherwise, indicate the type of union recommended for that case.
    
    """
    list1=set(df1)
    list2=set(df2)
 
    final= list1 & list2
    if len(final) > 0 :
        print("There is {} matching items. Merge type recommended between DF: Inner join".format(len(final)))
        print("Columns matching:", final)
    else:
        print("There is not Columns matching. Merge type recommended between DF: Left join or Right join ")
        
    
    

def detrending(df, value, frac):
    """       
    Function description:,
            The function calculates the trend of an ordered and continuous property extracted from a column  
            of a dataframe using the Lowess algorithm. It also calculates the residuals.
            The function displays the three curves (original, trend, residual) and outputs them in a new dataframe 
            Input data does not not have to be time-series but needs to be continuous and ordered. 
            It can also contain missing values.,
        -----------,
            Parameters:
            <df> (Pandas DataFrame type). 
                    mandatory parameter. Dataframe with the column containing the property to be processed
            
            <value> (string). 
                    mandatory parameter. Name of the column containing the property to be processed.
                    
            <frac>(float): {0.0 to 1.0}. 
                    mandatory parameter. Smoothness of the trend. The higher its values the smoother the trend.
   
        -----------,
            Returns: Dataframe,
                    From the original dataframe, it returns a new dataframe containing the three following columns: 
                    the original curve, the trend curve and the residual curve.
  
    """

    df = df.reset_index()
    df_c = df[[value]]
    df_cf=df_c.bfill().ffill()
    length=df_c.shape[0]
    trend = pd.DataFrame(lowess(df_cf[value], np.arange(length), frac=frac)[:, 1], index=df_cf.index, columns=[value])

    for i in range(0,length):
        if df[value][i]*0==0:
            df_c.loc[i, "TREND"]=trend.loc[i, value]
        else:
            df_c.loc[i, "TREND"]=None
            
    res=df_c[[value]]-trend

    df_c[["RESID"]]=res

    fig, axes = plt.subplots(3,1, figsize=(7, 7), sharex=True, dpi=120)
    df[value].plot(ax=axes[0], color='k', title='ORIGINAL '+str(value), ylim=(df[value].min(),df[value].max()))
    df_c["TREND"].plot(ax=axes[1],  color='k',title=str(value)+' TREND (with smoothness parameter ' + str(frac)+")", ylim=(df[value].min(),df[value].max()))
    df_c["RESID"].plot(ax=axes[2],  color='k',title=str(value)+' RESIDUAL (with smoothness parameter ' + str(frac) +")")
    fig.suptitle('DETRENDING FUNCTION', y=0.95, fontsize=14)
    plt.show()
    return(df_c)



def fill_missing(df, column, dropna=False, type_treatment=None):
    """
    -----------
       Function description:
        The function pick one DataFrame and the name of one column with mssings
        , the user can decide how to treat that column. 
        
    -----------
        Parameters:
        <df> (Pandas DataFrame type): 
                mandatory parameter. Dataframe  we want to check .
        <column>(string):
                name of the column that will be droped or filled
        
        <dropna> (boolean): Default value = False.
                True if you want drop that colum.
                
        <type_treatment>(string):
        
                Can be: mean  --> for mean treatment
                        mode  --> for mode treatment
                        value --> for fill with that value
    
    -----------
        Returns: Dataframe
    
    """
    if dropna == True:
        df = df.dropna(subset=[column])
    else:
        if type_treatment == 'mean':
            df[column] = df[column].fillna(df[column].type_treatment())
        elif type_treatment == 'mode':
            df[column] = df[column].fillna(df[column].type_treatment()[0])
        else:
            df[column] = df[column].fillna(type_treatment)
    return df
