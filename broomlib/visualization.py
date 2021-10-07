import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import chi2
from .utils import *
from sklearn import datasets
from scipy.stats import chi2
import statsmodels.api as sm
from adjustText import adjust_text
from sklearn import datasets


def missing_bars(data,figsize=(10, 3), style='ggplot'):
    """
    -----------
    Function description:
    Presents a ‘pandas’ barh plot with the percentage of missings for each feature
    -----------
    Parameters:
    param df(DataFrame): The DataFrame
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (10, 3)
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    -----------
    Returns:
    figure
    -----------
    Example:
    titanic = sns.load_dataset("titanic")
    missing_bars(titanic, figsize=(10, 3), style='ggplot')
    """
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        null_count =data.isnull().sum()
        null_perc_null = null_count.div(len(data))
        null_perc_notnull = 1 - null_perc_null 
        null_perc = pd.DataFrame({'Perc_Null': null_perc_null, 
                                  'Perc_not_Null': null_perc_notnull})
        ax = null_perc.plot(kind='barh', figsize=(10, 8),
                            stacked=True, #cmap='OrRd_r',
                            title='% Missings per column', grid=False)
        sns.despine(right=True, top=True, bottom=True)
        ax.legend(loc='upper right',bbox_to_anchor=(1.25, 1), frameon=False);

        for p in ax.patches:
            width = p.get_width()
            if width > 0.10:
                x = p.get_x()
                y = p.get_y()
                height = p.get_height()
                ax.text(x + width/2., y + height/2., str(round((width) * 100, 2)) + '%',
                            fontsize=10, va="center", ha='center', color='white', fontweight='bold')
    return plt.show()



def missings_heatmap(df, figsize=(12, 10), style='ggplot', cmap='RdYlBu'):   
    """
    -----------
    Function description:
    Presents a ‘seaborn’ heatmap visualization of nullity correlation in the given DataFrame
    -----------
    Parameters:
    param df(DataFrame): The DataFrame>
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (12, 12)
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    param cmap(str): What `matplotlib` colormap to use. Defaults to ‘RdYlBu’
    -----------
    Returns:
    figure
    -----------
    Example:
    titanic = sns.load_dataset("titanic")
    missings_heatmap(titanic, figsize=(6, 4), style='ggplot', cmap='RdYlBu')
    """
    
    df = df.iloc[:, [i for i, n in enumerate(np.var(df.isnull(), axis='rows')) if n > 0]]
    corr_mat = df.isnull().corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax = sns.heatmap(corr_mat, mask=mask, ax=ax, annot=True, fmt=".2f", cmap=cmap, cbar=True, vmin=-1, vmax=1)
        
    return plt.show()



def grid_displots(df, figsize=(12, 4), cols=3, bins=20, style='ggplot', fontsize=12, y_space=0.35):
    """
    -----------
    Function description:
    <>
    -----------
    Parameters:
    param df: The DataFrame
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (12, 4)
    param cols(int): number of plots displayed in parallel. 3 by default
    param bins(int): defines the number of equal-width bins in the range. 20 by default
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    param fontsize(int): The figure's font size. This default to 12
    param y_space(float):space between rows. 0.35 by default
    -----------
    Returns:
    figure
    -----------
    Example:        
    tips = sns.load_dataset('tips')
    grid_displots(tips, figsize=(15, 3), cols=3, bins=20, fontsize=15, y_space=0.5, style='ggplot',)
    """
    
    df = df.loc[:, (df.dtypes == 'int64') | (df.dtypes == 'float64') | (df.dtypes == 'int32')]
    rows = int(np.ceil(float(df.shape[1]) / cols))

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        for i, column in enumerate(df.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column, fontsize=fontsize, pad=10)
            df[column].hist(bins=bins, grid=False)
            plt.xticks(rotation="vertical")
        sns.despine(left=True, bottom=False)
        plt.subplots_adjust(hspace=y_space, wspace=0.2)
        
    return plt.show()


def grid_boxplots(df, figsize=(15, 15), cols=3, bins=20, style='ggplot', fontsize=12, y_space=0.35, whis=1.5):
    """
    -----------
    Function description:
    Presents a ‘seaborn’ boxplot visualization of each numeric column in the given DataFrame
    -----------
    Parameters:
    param df: The DataFrame
    param figsize(int): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (20, 12)
    param cols(int): number of plots displayed in parallel. 3 by default
    param bins(int): defines the number of equal-width bins in the range. 20 by default
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    param fontsize(int): The figure's font size. This default to 12
    param y_space(float):space between rows
    param whis(float): The position of the whiskers
    The lower whisker is at the lowest datum above Q1 - whis*(Q3-Q1), and the upper whisker at the highest datum below Q3 + whis*(Q3-Q1), where Q1 and Q3 are the first and third quartiles. The default value of whis = 1.5 corresponds to Tukey's original definition of boxplots
     -----------
    Returns:
    figure
    -----------
    Example:
    tips = sns.load_dataset('tips')
    grid_boxplots(tips, figsize=(15, 3), cols=3, bins=20, fontsize=15, y_space=0.5, style='ggplot')
    """
    df= df.loc[:, (df.dtypes == 'int64') | (df.dtypes == 'float64') | (df.dtypes == 'int32')]
    rows = int(np.ceil(float(df.shape[1]) / cols))

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        for i, column in enumerate(df.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column, fontsize=fontsize, pad=10)
            sns.boxplot(data=df[column], ax=ax, whis=whis)
        plt.subplots_adjust(hspace=y_space, wspace=0.2)
        
    return plt.show()




def grid_cat_bars(df, n_categories=10 ,figsize=(12, 4), cols=3, bins=20, style='ggplot', fontsize=12, y_space=0.35):
    """
    -----------
    Function description:
    <>
    -----------
    Parameters:
    param df: The DataFrame
    param n_categories(int):
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (20, 12)
    param cols(int): number of plots displayed in parallel. 3 by deafult
    param bins(int): defines the number of equal-width bins in the range. 20 by default
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    param fontsize(int): The figure's font size. This default to 12
    param y_space(float):space between rows. 0.35 by default
    -----------
    Returns:
    figure
    -----------
    Example:        
    tips = sns.load_dataset('tips')
    grid_cat_bars(tips, figsize=(15, 10), cols=3, fontsize=15, y_space=0.35, style='ggplot',)
    """
    
    df = df.loc[:, (df.dtypes == 'object') | (df.dtypes == 'category')]
    df = df[df.columns[df.nunique() < n_categories]]
    rows = int(np.ceil(float(df.shape[1]) / cols))

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        for i, column in enumerate(df.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column, fontsize=fontsize, pad=10)
            ax = sns.countplot(data= df, x=column)
            plt.xticks(rotation="vertical")
            plt.xlabel('')
        sns.despine(left=True, bottom=False)
        plt.subplots_adjust(hspace=y_space, wspace=0.2)
        
    return plt.show()



def grid_cat_target_bars(df, target, n_categories=10, figsize=(12, 4), cols=3, bins=20, style='ggplot', fontsize=12, y_space=0.35):
    """
    -----------
    Function description:
    <>
    -----------
    Parameters:
    param df: The DataFrame
    param n_categories(int):
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (12, 4)
    param cols(int): number of plots displayed in parallel. 3 by default
    param bins(int): defines the number of equal-width bins in the range. 20 by default
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    param fontsize(int): The figure's font size. This default to 12
    param y_space(float):space between rows. 0.35 by default
    -----------
    Returns:
    figure
    -----------
    Example:       
    titanic = sns.load_dataset("titanic")
    grid_cat_target_bars(titanic, target=titanic['survived'], figsize=(15, 10), cols=3, fontsize=15, y_space=0.55, style='ggplot')
    """
    
    df = df.loc[:, (df.dtypes == 'object') | (df.dtypes == 'category')]
    df = df[df.columns[df.nunique() < n_categories]]
    rows = int(np.ceil(float(df.shape[1]) / cols))

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        for i, column in enumerate(df.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column, fontsize=fontsize, pad=10)
            ax = sns.countplot(data= df, x=column, hue = target)
            plt.xticks(rotation="vertical")
            plt.xlabel('')
        sns.despine(left=True, bottom=False)
        plt.subplots_adjust(hspace=y_space, wspace=0.2)
         
    return plt.show()



def corr_bars(data, threshold, figsize=(10, 3), style='ggplot'):
    """
    -----------
    Function description:
    Presents a Pandas horizontal bar plot of the most correlated feature pairs and their correlation coefficient
    Function works with numerical features.
    -----------
    Parameters:
    param data: The DataFrame
    param threshold(float): cut off point for the value of the correlation coefficient which points out that there is a significant correlation between two features.
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (12, 4)
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    -----------
    Returns:
    figure
    -----------
    Example:       
    mpg = sns.load_dataset('mpg')
    corr_bars(mpg, 0.6, figsize= (13, 6))
    """

    threshold = threshold
    corr_orden = pd.DataFrame(data.corr().unstack().sort_values(ascending=False).drop_duplicates())
    corr_orden.rename({0 : 'Correlation'}, axis=1, inplace=True)
    corr_orden.reset_index(inplace=True)
    corr_orden['label'] = corr_orden['level_0'] + ' + ' + corr_orden['level_0']
    corr_orden
    corr_orden['abs'] = abs(corr_orden['Correlation'])
    corr_orden.sort_values(by='abs')
    corr_orden.drop(columns=['level_0', 'level_1'], inplace=True)
    corr_orden = corr_orden[corr_orden['abs'] >= threshold]
    
    with plt.style.context(style):
        ax = corr_orden[['label', 'Correlation']].sort_index(ascending=False).plot(kind='barh', x= 'label', figsize = figsize)
        plt.ylabel('')
        for p in ax.patches:
            width = p.get_width()
            x = p.get_x()
            y = p.get_y()
            height = p.get_height()
            ax.text(x + width/2., y + height/2., str(round((width) * 100, 2)) + '%',
                        fontsize=10, va="center", ha='center', color='white', fontweight='bold')
    return plt.show()



def outliers_mahalanobis_plot(x = None, extreme_points = 10, style = 'ggplot', figsize = (15,7)):
    """
    -----------
    Function description:
    Shows outliers of dataset. It compares Mahalanobis Distance of each point to Chi Square Distribution. 
    Points with index are the most extreme points (outliers) in the dataset. 
    Function works with numerical features.
    -----------
    Parameters:
    param x(DataFrame): The DataFrame (doesn't work fine with too many rows, 25000 or more)
    param extreme_points(int): Number of outliers that user can visualize (with index). A parameter which defaults to 10.
    param figsize(tuple): The size of the figure to display. This is a  'matplotlib' parameter which defaults to (15, 7)
    param style(str): The style of the figure to display. A 'matplotlib' parameter which defaults to 'ggplot'
    -----------
    Returns:
    figure
    -----------
    Example:
    diabetes = datasets.load_diabetes()
    df = pd.DataFrame(diabetes.data)
    outliers_mahalanobis(x = df)
    """

    dif = x - np.mean(x)
    cov = np.cov(x.T)
    inv = sp.linalg.inv(cov)
    izda = np.dot(dif, inv)
	
    dist = np.dot(izda, dif.T).diagonal()
    
    ppoints = np.linspace(0.001,0.999, x.shape[0])
    x['maha'] = dist
    maha_order = -np.sort(-x['maha'])
    extreme = x[x['maha'].isin(maha_order[:extreme_points])].sort_values(by = 'maha', ascending = False)
    extreme['chi'] = -np.sort(-chi2.ppf(ppoints, x.shape[1]))[:extreme_points]
    
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        ax = sns.scatterplot(x = chi2.ppf(ppoints, x.shape[1]), y = np.sort(dist));
        texto_puntos = [ax.text(extreme.chi.iloc[i], extreme.maha.iloc[i], txt, fontsize = 15)
                       for i, txt in enumerate(extreme.index)]
        adjust_text(texto_puntos)
        
        plt.title(r'QQPlot: Mahalanobis $D^2$ vs Quantiles $\chi^2(number \ of \ variables)$')
        plt.xlabel(r'Quantiles of $\chi^2$')
        plt.ylabel('Mahalanobis Distance')
    
    return plt.show()
