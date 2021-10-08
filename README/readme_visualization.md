# `broomlib visualization`

There are 2 common parameters for broomlib visualizations:
- `figsize`: a numeric tuple. This is usefull to adjust the size and proportion of the image.
- `style`: a matplotlib style. Use `plt.style.available` to find out available styles.

<p><br></p>

## `missing_bars`
Presents a ‘pandas’ barh plot with the percentage of missings for each feature.

```
from broomlib import visualization as vis
import seaborn as sns
titanic = sns.load_dataset("titanic")
vis.missing_bars(titanic, figsize=(10, 3), style='ggplot')
```

![](images/missing_bars.png)

<p><br></p>


## `missing_matrix`
Presents a ‘seaborn’ visualization of the nulls in the given DataFrame.

```
from broomlib import visualization as vis
import seaborn as sns
titanic = sns.load_dataset("titanic")
vis.missing_matrix(titanic, figsize=(10, 3), style='ggplot', cmap='PuBu')
```

![](images/missing_matrix.png)

<p><br></p>


## `missing_heatmap`
Presents a ‘seaborn’ heatmap visualization of nullity correlation in the given DataFrame.

```
from broomlib import visualization as vis
import seaborn as sns
titanic = sns.load_dataset("titanic")
vis.missing_heatmap(titanic, figsize=(6, 4), style='ggplot', cmap='RdYlBu_r')
```

![](images/missing_heatmap.png)

<p><br></p>


## `grid_displots`
Presents the distribution of each numerical variable.
Function works with numerical features.

```
from broomlib import visualization as vis
import seaborn as sns
tips = sns.load_dataset('tips')
vis.grid_displots(tips, figsize=(15, 3), cols=3, bins=20, fontsize=15, y_space=0.5, style='ggplot')
```

![](images/grid_displots.png)

<p><br></p>


## `grid_boxplots`
Presents a ‘seaborn’ boxplot visualization of each numeric column in the given DataFrame.

```
from broomlib import visualization as vis
import seaborn as sns
tips = sns.load_dataset('tips')
vis.grid_boxplots(tips, figsize=(12, 3), cols=3, fontsize=15, y_space=0.5, style='ggplot')
```

![](images/grid_boxplots.png)

<p><br></p>


## `grid_cat_bars`
Shows the distribution of each categorical variable compared with the categorical target.
Function works with categorical features.

```
from broomlib import visualization as vis
import seaborn as sns
tips = sns.load_dataset('tips')
vis.grid_cat_bars(tips, figsize=(12, 8), cols=2, fontsize=15, y_space=0.5, style='ggplot')
```

![](images/grid_cat_bars.png)

<p><br></p>


## `grid_cat_target_bars`
Shows the distribution of each categorical variable compared with the categorical target.
Function works with categorical features.

```
from broomlib import visualization as vis
import seaborn as sns
titanic = sns.load_dataset("titanic")
vis.grid_cat_target_bars(titanic, target=titanic['survived'], figsize=(15, 12), cols=3, fontsize=15, y_space=0.55, style='ggplot')
```

![](images/grid_cat_target_bars.png)

<p><br></p>


## `corr_bars`
Presents a Pandas horizontal bar plot of the most correlated feature pairs and their correlation coefficient.
Function works with numerical features.

```
from broomlib import visualization as vis
import seaborn as sns
mpg = sns.load_dataset('mpg')
vis.corr_bars(mpg, threshold=0.6, figsize=(13, 6))
```

![](images/corr_bars.png)

<p><br></p>


## `outliers_mahalanobis_plot`
Shows outliers of dataset. It compares Mahalanobis Distance of each point to Chi Square Distribution.
Points with index are the most extreme ones (outliers) in the dataset.
Function works with numerical features.

```
from broomlib import visualization as vis
from sklearn import datasets
import pandas as pd
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data)
vis.outliers_mahalanobis_plot(df, extreme_points=10, figsize=(15,7), style='ggplot')
```

![](images/outliers_mahalanobis_plot.png)

<p><br></p>


## `accuracy_time_ML`
Presents a visualization of accuracy and time data from machine learning models.
```
from broomlib import visualization as vis
df = pd.DataFrame({'Models': ['Modelo 1', 'Modelo 2', 'Modelo 3'], 
      'Accuracy': [90, 85, 95],
      'Time taken': [25, 30, 50]})

vis.accuracy_time_ML(df, figsize=(12, 6), cmap='RdYlBu', style='ggplot')
```

![](images/accuracy_time_ML.png)

<p><br></p>


## `accuracy_ML`
Presents a visualization of accuracy given from machine learning models.
```
from broomlib import visualization as vis
df = pd.DataFrame({'Models': ['Model 1', 'Model 2', 'Model 3'], 
      'Accuracy': [90, 85, 95]})

vis.accuracy_ML(df, figsize=(6, 4), cmap='RdYlBu', style='ggplot')
```

![](images/accuracy_ML.png)

<p><br></p>
