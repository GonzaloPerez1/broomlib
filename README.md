[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


# Broomlib 
## Description 
This package probably does stuff
***
## Installation
To install this package use `pip install broomlib`
***
## Dependencies
This package has needs the following in order to work: 
* matplotlib 
* tensorflow
* seaborn
* scikit-learn
* plotly
* whatever else just add it here
***

<a name = 'index'> </a>
## Functions
Here is a brief overview of the functions available in this package. To further explore any of them, click on the link.

### Data Cleaning
* <a href = #show_all>show_all</a> 
* <a href = #data_report>data_report</a> 
* <a href = #how_missing>how_missing</a>
* <a href = #checking_duplicates>checking_duplicates</a>
* <a href = #building_date>building_date</a>

### Visualization 

* <a href = #missing_bars>missing_bars</a>
* <a href = #missings_heatmap>missings_heatmap</a>
* <a href = #grid_displots>grid_displots</a>
* <a href = #grid_boxplots>grid_boxplots</a>
* <a href = #grid_cat_bars>grid_cat_bars</a>
* <a href = #grid_cat_target_bars>grid_cat_target_bars></a>
* <a href = #corr_bars>corr_bars</a>
* <a href = #outliers_mahalanobis_plot>outliers_mahanalobis_plot</a>
***
## Documentation
### **Data Cleaning**
<a href="#index"><p align="right" href="#index">Back to index</p></a>

<a name = 'show_all'></a>

#### **show_all** *(`df = None`, `big_desc = False`)*
Shows the info, number of rows and a big or small description with mode, mean, median, missings, categorical and numerical columns...

**Parameters**
* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **big_desc** (*bool*) (`default = False`): If True displays count, nulls, type, unique, numeric, mode, mean, min, quantile, median, max, std, skew and kurt of each column.

**Returns**
If `big_desc = True` returns Pandas Dataframe with a big description

Otherwise returns the info, shape of each row and the pandas [describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) method.
****
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'data_report'></a>

#### **data_report** *(`df = None`)*
Shows a small summary of the type, percentage of missings, unique values ad cardinality of each column.

**Parameters**

* **df** (*Pandas Dataframe*): Dataframe we want to check.

**Returns** 

Datafrane with the types, percentage of missings, unique values and cardinality of each column.
***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'how_missing'></a>

#### **how_missing** *(`df = None`, `threshold = 0.0`, `drop = False`)*
Shows the percentage of missing values. Optional: establish a lower threshold for the values to be displayed and delete the columns meeting this criteria. 

**Parameters** 
* **df** (*Pandas dataframe*): Dataframe we want to check.
* **threshold** (*float*) (`default = 0`): Minimum percentage of missings in a column that will be displayed. Goes from 0 to 100
* **drop** (*bool*) (`default = False`): If True, deletes the columns exceding the threshold. Warning: deletes from the original df.  

**Returns** 

Dataframe with the name of the columns and the percentage of missings in each. 

***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'checking_duplicates'></a>

#### **checking_duplicates** *(`df = None`, `col = None`)*
Checks if there are duplicate values due to capitalization in a column of strings. (E.g: 'Spain' and 'spain')

**Parameters** 

* **df** (*Pandas Dataframe*): dataframe we want to check.
* **col** (*string*): name of the column. Should be made up of strings. 

**Returns**

The duplicate values. If none exist, it will display a message stating so. 

***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'building_date'></a>

#### **building_date** *(`year = None`, `month = None`, `day = None`, `df = None`)*

Combines separate year, month and day columns into a single date column. 

**Params** 

* **year** (*string*): name of the colum in which the years are. 
* **month** (*string* or *int*): name of the column in which the years are. Values can be integrers [0,12] or strings with the name of the month in English. 
* **day** (*string* or *int*): name of the column with the days. Values need to be integers [0,31]. 

**Returns** 

Original dataframe with a new column `'Parsing date'` in the datetime64 numpy type. It will drop the original columns with years, months and days. 

***

### **Visualization**
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'missing_bars'></a>

#### **missing_bars** *(`data = None`, `figsize = (10,3)`, `style = 'ggplot'`)*

Presents a horizontal barplot with the percentage of missings in each feature.

**Parameters**
* df (Pandas Dataframe): Dataframe we want to check.
* figsize (tuple) (`default = (10, 3)`): A matplotlib parameter to establish the size of the figure. 
* style (str): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

**Returns**

A figure
****
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'missings_heatmaps'></a>

#### **missings_heatmaps** *(`df = None`, `figsize = (12, 10)`, `style = 'ggplot'`, `cmap = 'RdYlBu'`)*
Presents a heatmap visualizationof nullity correlation in the give. Dataframe. 

**Parameters** 
* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **figsize** (*tuple*) (`default = (12,10)`): A matplotlib parameter to establish the size of the figure.
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
* **cmap** (*str*): What matplotlib colormap to use. Full list available [here](https://matplotlib.org/stable/tutorials/colors/colormaps.html).

**Retuns**

A figure.
***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'grid_displots'></a>
#### **grid_displots** *(`df = None`, `figsize=(12, 4)`,` cols=3`, `bins=20`, `style='ggplot'`, `fontsize=12`, `y_space=0.35`)*

Presents the distribution of each numerical variable. Will not work with categorical variables.

**Parameters** 

* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **figsize** (*tuple*) (`default = (12,4)`): A matplotlib parameter to establish the size of the figure.
* **cols** (*int*) (`default = 3`): number of plots displayed in parallel.
* **bins** (*int*) (`default = 20`): number of equal-width bins in the range. 
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
* **fontsize** (*int*) (`default = 12`): The figure's font size.
* **y_space** (*float*) (`default = 0.35`): Space between rows.

**Returns**

A figure.
***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'grid_boxplots'></a>

#### **grid_boxplots** *(`df = None`, `figsize=(15, 15)`, `cols=3`, `bins=20`, `style='ggplot'`, `fontsize=12`, `y_space=0.35`, `whis=1.5`)*:

Displays a seaborn boxplot visualization of each numeric column in the given dataframe. 

**Parameters** 
* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **figsize** (*tuple*) (`default = (15,15)`): A matplotlib parameter to establish the size of the figure.
* **cols** (*int*) (`default = 3`): number of plots displayed in parallel.
* **bins** (*int*) (`default = 20`): number of equal-width bins in the range.
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
* **fontsize** (*int*) (`default = 12`): The figure's font size.
* **y_space** (*float*) (`default = 0.35`): Space between rows.
* **whis** (*float*) (`default = 1.5`): The position of the whiskers. 1.5 corresponds to Tukey's original definition of boxplots.

**Returns**

A figure.
*** 
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'grid_cat_bars'></a>

#### **grid_cat_bars** *(`df = None`, `n_categories=10` ,`figsize=(12, 4)`, `cols=3`, `bins=20`, `style='ggplot'`, `fontsize=12`, `y_space=0.35`)*

Shows the distribution of each categorical variable. It works with categorical features. 

**Parameters** 
* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **n_categories** (*int*): Number of categorical columns.
* **figsize** (*tuple*) (`default = (12,4)`): A matplotlib parameter to establish the size of the figure.
* **cols** (*int*) (`default = 3`): number of plots displayed in parallel.
* **bins** (*int*) (`default = 20`): number of equal-width bins in the range.
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
* **fontsize** (*int*) (`default = 12`): The figure's font size.
* **y_space** (*float*) (`default = 0.35`): Space between rows.

**Returns**

A Figure
***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'grid_cat_target_bars'></a>

#### **grid_cat_target_bars** *(`df = None`, `target = None`, `n_categories=10`, `figsize=(12, 4)`, `cols=3`, `bins=20`, `style='ggplot'`, `fontsize=12`, `y_space=0.35`)*

Shows the distribution of each categorical variable compared wiith the categorical target. 

**Parameters** 
* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **target** (*Pandas Dataframe column*): The categorical target.
* **n_categories** (*int*): Number of categorical columns.
* **figsize** (*tuple*) (`default = (12,4)`): A matplotlib parameter to establish the size of the figure.
* **cols** (*int*) (`default = 3`): number of plots displayed in parallel.
* **bins** (*int*) (`default = 20`): number of equal-width bins in the range.
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).
* **fontsize** (*int*) (`default = 12`): The figure's font size.
* **y_space** (*float*) (`default = 0.35`): Space between rows.

**Returns**

A Figure

***
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'corr_bars'></a>

#### **corr_bars** *(`data = None`, `threshold = None`, `figsize=(10, 3)`, `style='ggplot'`)*

Displays a horizontal bar plor of the most correlated feature pairs and their correlation coeffcient. Works with numerical features. 

**Parameters** 
* **df** (*Pandas Dataframe*): Dataframe we want to check.
* **threshold** (*float*): cut off point for the value of the correlation coefficient which points out that there is a significant correlation between two features.
* **figsize** (*tuple*) (`default = (10,3)`): A matplotlib parameter to establish the size of the figure.
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

**Returns**

A Figure
*** 
<a href="#index"><p align="right" href="#index">Back to index</p></a>
<a name = 'outliers_mahalanobis_plot'></a>

#### **outliers_mahalanobis_plot** *(`x = None`, `extreme_points = 10`, `style = 'ggplot'`, `figsize = (15,7)`)*:

Shows outliers of the dataset. It compares Mahalanobis Distance of each point to Chi Square Distribution. Points with index are the most extreme ones (outliers) in the dataset. Works with numerical features. 

**Parameters**
* **x** (*Pandas Dataframe*): Dataframe we want to check. Note: performance is affected if rows > 2500.
* **extreme_points** (*int*) (`default = 10`): Number of outliers the user can visualize (with its index).
* **figsize** (*tuple*) (`default = (15,7)`): A matplotlib parameter to establish the size of the figure.
* **style** (*str*): Style of the figure to display. Full list available [here](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).

**Returns**

A figure



# Contributors
List of people that participated in this project:

* [Juan Manuel Cornejo](https://www.linkedin.com/in/juanmanuelcornejociruelo/)
* [Ana Blanco Delgado](https://www.linkedin.com/in/anablancodelgado/)
* [Gonzalo Pérez Díez](https://www.linkedin.com/in/gonzalo-pérez-díez/)
* [Antonio Pol Fuentes](https://www.linkedin.com/in/antoniopolfuentes)
* [Miguel Anguita Ruiz](https://www.linkedin.com/in/miguel-anguita-ruiz-73205b107/)
* [Laurent Jacquet](https://www.linkedin.com/in/laurent-jacquet-61b513102/)
* [Nieves Noha Pascual](https://www.linkedin.com/in/nieves-noha-pascual)
* [José Antonio Suárez Roig](https://www.linkedin.com/in/josé-antonio-suárez-roig)
* [Olivier Kirstetter](https://www.linkedin.com/in/olivier-kirstetter)





[license-shield]: https://img.shields.io/github/license/GonzaloPerez1/broomlib?style=for-the-badge
[license-url]: https://github.com/GonzaloPerez1/broomlib/blob/main/LICENSE
[contributors-shield]: https://img.shields.io/github/contributors/GonzaloPerez1/broomlib?style=for-the-badge
[contributors-url]: https://github.com/GonzaloPerez1/broomlib/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/GonzaloPerez1/broomlib?style=for-the-badge
[forks-url]: https://github.com/GonzaloPerez1/broomlib/network/members
[stars-shield]: https://img.shields.io/github/stars/GonzaloPerez1/broomlib?style=for-the-badge
[stars-url]: https://github.com/GonzaloPerez1/broomlib/stargazers
[issues-shield]: https://img.shields.io/github/issues/GonzaloPerez1/broomlib?style=for-the-badge
[issues-url]: https://github.com/GonzaloPerez1/broomlib/issues
[license-shield]: https://img.shields.io/github/license/GonzaloPerez1/broomlib?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt

