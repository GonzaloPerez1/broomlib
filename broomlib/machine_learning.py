import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
import math
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np


def transformdict(df, tansformation):
    '''
    -----------
        Function description:

    This function apply and plot a transformation to the data searching normality returning the new data,
    the transformation is choosen by the user and the options will be below in the parameters section.


    -----------
        Parameters:


    'log' : This will be the traditional logaritmic transformation, is a fancy option for most of the
        problems, ir compress the data a lot and make the variables be clooser together, reduce
        the variability of the problem but make it more considerable.

    'std' : This will make a standard scaler based on the StandarScaler function of SKLearn, this
        put all the variables in the same units but deletes the variability of the sample. Dont
        use it if you want a Principal Components Analisis of the data.

    'inv' : This will reverse the data, is usefull if you have a long right tail in the data.

    'Box_Cox' : This will adapt the data macking a translation (That dont affects the analisis)
            just to make the traditional box cox transformation based on logaritmic function
            (Cant be used for ceros or negative values). Is the more agressive one but at the
            same time one of the more effective one if u search normality. Is not recommended
            if you dont want an agressive non lineal transformation.

    'sqrt' : This will make the square root transformation to the data, is very usefull if you have
        long right tails or long left tails. In negative numbers it use the python language.

    'cuad' : This will make the cuadratic transformation and is useful if u want to gain variability.


    'data': A DataFrame with the data that you want to transform.

    'tansformation' : The transformation that you want, by deafult we use the logaritmic function ('log').

    -----------
        Returns: [transformed data]
    '''

    print('Unmodified Data')
    fig = plt.figure(figsize=(20, 30))
    for i in range(len(df.columns)):
        ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
        sns.distplot(df[df.columns.values[i]]).set_title(df.columns.values[i])
    plt.show()
    if tansformation == 'log':
        print('Logarithmic Data')
        df1 = df.apply(lambda x: np.log(x + 1))
        fig = plt.figure(figsize=(20, 30))
        for i in range(len(df.columns)):
            ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
            sns.distplot(df1[df1.columns.values[i]]).set_title(df1.columns.values[i])
        plt.show()
    elif tansformation == 'std':
        print('Standarized Data')
        scaler = StandardScaler()
        scaler.fit(df)
        df_standard_scaler = scaler.transform(df)
        df1 = pd.DataFrame(df_standard_scaler)
        fig = plt.figure(figsize=(20, 30))
        for i in range(len(df.columns)):
            ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
            sns.distplot(df1[df1.columns.values[i]]).set_title(df.columns.values[i])
        plt.show()
    elif tansformation == 'inv':
        print('Inverted Data')
        df1 = df.apply(lambda x: 1 / (x + 1))
        fig = plt.figure(figsize=(20, 30))
        for i in range(len(df.columns)):
            ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
            sns.distplot(df1[df1.columns.values[i]]).set_title(df.columns.values[i])
        plt.show()
    elif tansformation == 'box_cox':
        print('Data Box Cox')
        df1 = df
        fig = plt.figure(figsize=(20, 30))
        for i in range(len(df.columns)):
            fitted_data, fitted_lambda = stats.boxcox(df[df.columns.values[i]] + abs(df[df.columns.values[i]]) + 1)
            ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
            sns.distplot(fitted_data).set_title(df.columns.values[i] + ' ' + f"Lambda Transformation: {fitted_lambda}")
            df1[df1.columns[i]] = fitted_data
        plt.show()
    elif tansformation == 'sqrt':
        print('Root Transformed Data')
        df1 = df.apply(lambda x: x ** 0.5)
        fig = plt.figure(figsize=(20, 30))
        for i in range(len(df.columns)):
            ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
            sns.distplot(df1[df1.columns.values[i]]).set_title(df.columns.values[i])
        plt.show()
    elif tansformation == 'cuad':
        print('Squared Transformed Data')
        df1 = df.apply(lambda x: x ** 2)
        fig = plt.figure(figsize=(20, 30))
        for i in range(len(df.columns)):
            ax = plt.subplot(math.ceil(len(df.columns) / 3), 3, i + 1)
            sns.distplot(df1[df1.columns.values[i]]).set_title(df.columns.values[i])
        plt.show()
    else:
        print('Keyword not found in dict')
    return df1


def broomResample(X, y, sampling_strategy_o=0.1,
                  sampling_strategy_u='auto',
                  random_state=None,
                  k_neighbors=5,
                  n_jobs=None,
                  replacement=False):
    '''
    -----------
        Function description:


          A problem with imbalanced classification is that there are too few examples of the minority class for
          a model to effectively learn the decision boundary.
          One way to solve this problem is to oversample the examples in the minority class. This can be achieved
          by simply duplicating examples from the minority class in the training dataset prior to fitting a model.
          This can balance the class distribution but does not provide any additional information to the model.

          Knowing this, the function first uses a random undersampling to trim the number of examples
          in the majority class, then it uses Synthetic Minority Oversampling Technique also called
          SMOTE to oversample the minority class to balance the class distribution.
          The combination of SMOTE and under-sampling performs better than plain under-sampling.

     Parameters
    ----------

        This object next parameter its use in an implementation of SMOTE - Synthetic  Minority Over-sampling Technique

        >SMOTE first selects a minority class instance a at random and finds its k nearest minority class neighbors.
        The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting
        a and b to form a line segment in the feature space.
        The synthetic instances are generated as a convex combination of the two chosen instances a and b. [SMOTE]
        (https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)`

        SMOTE specific formula variables

        Random oversampling involves randomly selecting examples from the minority class, with replacement,
        and adding them to the training dataset. Random undersampling involves randomly selecting examples from
        the majority class and deleting them from the training dataset.

        sampling_strategy_o : float, str, dict or callable, default='auto'
            Sampling information to resample the data set.
            - When ``float``, it corresponds to the desired ratio of the number of
              samples in the minority class over the number of samples in the
              majority class after resampling. Therefore, the ratio is expressed as
              :math:`\\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the
              number of samples in the minority class after resampling and
              :math:`N_{M}` is the number of samples in the majority class.
                .. warning::
                   ``float`` is only available for **binary** classification. An
                   error is raised for multi-class classification.
            - When ``str``, specify the class targeted by the resampling. The
              number of samples in the different classes will be equalized.
              Possible choices are:
                ``'minority'``: resample only the minority class;
                ``'not minority'``: resample all classes but the minority class;
                ``'not majority'``: resample all classes but the majority class;
                ``'all'``: resample all classes;
                ``'auto'``: equivalent to ``'not majority'``.
            - When ``dict``, the keys correspond to the targeted classes. The
              values correspond to the desired number of samples for each targeted
              class.
            - When callable, function taking ``y`` and returns a ``dict``. The keys
              correspond to the targeted classes. The values correspond to the
              desired number of samples for each class.


        random_state : int, RandomState instance, default=None
            Control the randomization of the algorithm.
                - If int, ``random_state`` is the seed used by the random number
                  generator;
                - If ``RandomState`` instance, random_state is the random number
                  generator;
                - If ``None``, the random number generator is the ``RandomState``
                  instance used by ``np.random``.

        k_neighbors : int or object, default=5
            If ``int``, number of nearest neighbours to used to construct synthetic
            samples.  If object, an estimator that inherits from
            :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
            find the k_neighbors.

        n_jobs : int, default=None
            Number of CPU cores used during the cross-validation loop.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See
            `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
            for more details.

        Random undesampling specific formula variables

         Random undersampling involves randomly selecting examples from the majority class
         and deleting them from the training dataset. In the random under-sampling, the majority class instances are discarded at random until
         a more balanced distribution is reached.

         **Under-sample the majority by randomly picking samples
            with or without replacement.
            sampling_strategy_u : float, str, dict, list or callable,
            Sampling information to sample the data set.

            - When ``float``:
                For **under-sampling methods**, it corresponds to the ratio
                :math:`\\alpha_{us}` defined by :math:`N_{rM} = \\alpha_{us}
                \\times N_{m}` where :math:`N_{rM}` and :math:`N_{m}` are the
                number of samples in the majority class after resampling and the
                number of samples in the minority class, respectively;
                For **over-sampling methods**, it correspond to the ratio
                :math:`\\alpha_{os}` defined by :math:`N_{rm} = \\alpha_{os}
                \\times N_{m}` where :math:`N_{rm}` and :math:`N_{M}` are the
                number of samples in the minority class after resampling and the
                number of samples in the majority class, respectively.
                .. warning::
                   ``float`` is only available for **binary** classification. An
                   error is raised for multi-class classification and with cleaning
                   samplers.

            - When ``str``, specify the class targeted by the resampling. For
              **under- and over-sampling methods**, the number of samples in the
              different classes will be equalized. For **cleaning methods**, the
              number of samples will not be equal. Possible choices are:
                ``'minority'``: resample only the minority class;
                ``'majority'``: resample only the majority class;
                ``'not minority'``: resample all classes but the minority class;
                ``'not majority'``: resample all classes but the majority class;
                ``'all'``: resample all classes;
                ``'auto'``: for under-sampling methods, equivalent to ``'not
                minority'`` and for over-sampling methods, equivalent to ``'not
                majority'``.

            - When ``dict``, the keys correspond to the targeted classes. The
              values correspond to the desired number of samples for each targeted
              class.
              .. warning::
                 ``dict`` is available for both **under- and over-sampling
                 methods**. An error is raised with **cleaning methods**. Use a
                 ``list`` instead.

            - When ``list``, the list contains the targeted classes. It used only
              for **cleaning methods**.
              .. warning::
                 ``list`` is available for **cleaning methods**. An error is raised
                 with **under- and over-sampling methods**.

            - When callable, function taking ``y`` and returns a ``dict``. The keys
              correspond to the targeted classes. The values correspond to the
              desired number of samples for each class.

            y : ndarray of shape (n_samples,)
            The target array.
            sampling_type : {{'over-sampling', 'under-sampling', 'clean-sampling'}}
            The type of sampling. Can be either ``'over-sampling'``,
            ``'under-sampling'``, or ``'clean-sampling'``.
            kwargs : dict
                Dictionary of additional keyword arguments to pass to
                ``sampling_strategy`` when this is a callable.

        random_state : int, RandomState instance, default=None
            Control the randomization of the algorithm.
                - If int, ``random_state`` is the seed used by the random number
                  generator;
                - If ``RandomState`` instance, random_state is the random number
                  generator;
                - If ``None``, the random number generator is the ``RandomState``
                  instance used by ``np.random``.
        replacement : boolean, optional (default=False)
            Whether the sample is with (default) or without replacement.


        Returns
        -------
            X_resampled : ndarray, shape (n_samples_new, n_features)
                The array containing the resampled data.
            y_resampled : ndarray, shape (n_samples_new)
                The corresponding label of `X_resampled`


        Examples

        --------
            from broomlib.MachineLearning import broomResample
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.pipeline import Pipeline
            X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	        ...n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
            counter = Counter(y)
            print(counter)
            for label, _ in counter.items():
	        row_ix = np.where(y == label)[0]
	        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
            plt.legend()
            plt.show()
            X_resampled, y_resampled = broomResample(X,y)
            print(f'Resampled dataset samples per class {{Counter(y_resampled)}}')
        Resampled dataset samples per class Counter({{0: 900, 1: 900}})
            for label, _ in counter.items():
	        row_ix = np.where(y_resampled == label)[0]
	        plt.scatter(X_resampled[row_ix, 0], X_resampled[row_ix, 1], label=str(label))
            plt.legend()
            plt.show()
    '''

    over = SMOTE(sampling_strategy=sampling_strategy_o,
                 random_state=random_state,
                 k_neighbors=k_neighbors,
                 n_jobs=n_jobs)

    under = RandomUnderSampler(sampling_strategy=sampling_strategy_u,
                               random_state=random_state,
                               replacement=False)

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    return [X_resampled, y_resampled]


def pca_analisis(data):
    '''
    -----------
        Function description:
    This function makes a feature importance analysis based on variability, for this we must have a full balanced data, all the units of
    the different variables must be the same. If its not the case we can transform de data compressing the variables using any transformation maybe one
    of the function diccionario_de_transformaciones in the library (std is not recommended, it crush the variability). What we do when we have the ideal
    data is to find the ways of more variability, the PCA method based on the SKLearn PCA function. Then we used the coordinates in every principal
    component to evaluate the importance of any variable. It makes a plot of the variability explained in every direction and a plot with the importance
    of any variable. To finish the function process, it asked to the user the variables he wants to delete and return the data without this variables.

    -----------
        Parameters:

    data: Ideal data to use a principal components method, all variables in the same units.

    inputs: The function wants the final opinion of the user cause at the end we can introduce a feature selection tactive but there are a lot so the
    objective of the function is to look for the user to investigate more about variability study of the data.

    -----------
        Return:

    The final data without variables of little variability importance.



    '''
    pca_pipe = make_pipeline(PCA())
    pca_pipe.fit(data[data.columns[:-1]])
    modelo_pca = pca_pipe.named_steps['pca']
    pca = pd.DataFrame(
        data=modelo_pca.components_,
        columns=data.columns[:-1])
    # Porcentaje de varianza explicada acumulada
    # ==============================================================================
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
    print('------------------------------------------')
    print('Explained accumulated variance percentage')
    print('------------------------------------------')
    print(prop_varianza_acum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(
        np.arange(len(data.columns[:-1])) + 1,
        prop_varianza_acum,
        marker='o'
    )

    for x, y in zip(np.arange(len(data.columns[:-1])) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_title('Explained accumulated variance percentage')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('% accumulated variance');
    plt.show()
    fig_2, ax_2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    componentes = modelo_pca.components_
    plt.imshow(componentes.T, cmap='PuOr', aspect='auto', vmin=-1, vmax=1)
    plt.yticks(range(len(data.columns[:-1])), data.columns[:-1])
    plt.xticks(range(6))
    plt.grid(False)
    plt.colorbar();
    plt.show()
    numero = input('¿How many variables to drop?')
    variables = []
    for i in range(int(numero)):
        variables.append(input('Dropping the variable'))
    data = data.drop(variables, 1)
    return data
