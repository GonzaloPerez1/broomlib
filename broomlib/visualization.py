from .utils import *

def comentarios():
  
  pass


def missing_bars(data,figsize=(10, 3), style='ggplot'):
    """
    -----------
    Function description:
    Barplt with the frecuency of missing for each feature
    -----------
    Parameters:
    param df(DataFrame): The DataFrame
    param figsize(tuple): The size of the figure to display. This is a  ‘matplotlib’ parameter which defaults to (10, 3)
    param style(str): The style of the figure to display. A ‘matplotlib’ parameter which defaults to ‘ggplot’
    -----------
    Returns:
    figure
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