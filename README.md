# CRYPTO CRYSTAL BALL
![logo](https://github.com/charlieburd/crypto_crystal_ball/blob/emmanuel_branch/Resources/images/Header.jpg?raw=true)


## Machine Learning Model - Final Project 2021

In our final project, our team used Machine Learning Support Vector Regression (SVR) model as a Data Science and Analytics cryptocurrency prediction. Bitcoin and other Cryptocurrencies Price Prediction with Machine Learning for the Next 30 Days.

## Team Members - Roles
 * Charlie Burd - Information Manager (Square)
 * Emmanuel Martinez - Tool Creation (Triangle)
 * George Quintanilla - Framework Creation (Circle)

## Resources and DataSets:
This new assignment consists of three technical analysis deliverables and a proposal for further statistical study:

* Data Source: `Bitcoin_1_Min_Historical_Data_2012_to_2020.csv` from [Kaggle - Bitcoin Historical Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)
* Data Tools:  `ml_yahoo_fianance_stock.py`, `MyBitcoinPredictionApp.ipynb`.
* Software: `CoLab - Google`, `Python 3.9`, `Visual Studio Code 1.50.0`, `Anaconda 4.8.5`, `Jupyter Notebook 6.1.4` and `Pandas` 

## Machine Learning Model (sklearn.svm.SVR)
> Let's have a quick review from Support Vector Regression Model (SVR Model or SVM), below you may find Regression notes and insight information. 

### Support Vector Machines (SVM)

**Support vector machines (SVMs)** are a set of supervised learning methods used for [classification](https://sklearn.org/modules/svm.html#svm-classification), [regression](https://sklearn.org/modules/svm.html#svm-regression) and [outliers detection](https://sklearn.org/modules/svm.html#svm-outlier-detection).

The advantages of support vector machines are:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different [Kernel functions](https://sklearn.org/modules/svm.html#svm-kernels) can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoid over-fitting in choosing [Kernel functions](https://sklearn.org/modules/svm.html#svm-kernels) and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see [Scores and probabilities](https://sklearn.org/modules/svm.html#scores-probabilities), below).

The support vector machines in scikit-learn support both dense (`numpy.ndarray` and convertible to that by `numpy.asarray`) and sparse (any `scipy.sparse`) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered `numpy.ndarray` (dense) or `scipy.sparse.csr_matrix` (sparse) with `dtype=float64`.

### Classification
`SVC`, `NuSVC` and `LinearSVC` are classes capable of performing multi-class classification on a dataset.

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/emmanuel_branch/Resources/images/SVC_Class.png?raw=true)


`SVC` and `NuSVC` are similar methods, but accept slightly different sets of parameters and have different mathematical formulations (see section Mathematical formulation). On the other hand, `LinearSVC` is another implementation of Support Vector Classification for the case of a linear kernel. Note that `LinearSVC` does not accept keyword `kernel`, as this is assumed to be linear. It also lacks some of the members of `SVC` and `NuSVC`, like `support_`.

As other classifiers, `SVC`, `NuSVC` and `LinearSVC` take as input two arrays: an array X of size `[n_samples, n_features]` holding the training samples, and an array y of class labels (strings or integers), size `[n_samples]`:

````Python
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
````

After being fitted, the model can then be used to predict new values:

````Python
>>> clf.predict([[2., 2.]])
array([1])
````

SVMs decision function depends on some subset of the training data, called the support vectors. Some properties of these support vectors can be found in members `support_vectors_`, `support_` and `n_support`:

````Python
>>> # get support vectors
>>> clf.support_vectors_
array([[ 0.,  0.],
       [ 1.,  1.]])
>>> # get indices of support vectors
>>> clf.support_ 
array([0, 1]...)
>>> # get number of support vectors for each class
>>> clf.n_support_ 
array([1, 1]...)
````

### Multi-Class Classification¶
`SVC` and `NuSVC` implement the “one-against-one” approach (Knerr et al., 1990) for multi- class classification. If `n_class` is the number of classes, then `n_class * (n_class - 1) / 2` classifiers are constructed and each one trains data from two classes. To provide a consistent interface with other classifiers, the `decision_function_shape` option allows to aggregate the results of the “one-against-one” classifiers to a decision function of shape `(n_samples, n_classes)`:

````Python
>>> X = [[0], [1], [2], [3]]
>>> Y = [0, 1, 2, 3]
>>> clf = svm.SVC(decision_function_shape='ovo')
>>> clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes: 4*3/2 = 6
6
>>> clf.decision_function_shape = "ovr"
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes
4
````

On the other hand, `LinearSVC` implements “one-vs-the-rest” multi-class strategy, thus training n_class models. If there are only two classes, only one model is trained:

````Python
>>> lin_clf = svm.LinearSVC()
>>> lin_clf.fit(X, Y) 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
>>> dec = lin_clf.decision_function([[1]])
>>> dec.shape[1]
4
````

See [Mathematical formulation](https://sklearn.org/modules/svm.html#svm-mathematical-formulation) for a complete description of the decision function.

Note that the LinearSVC also implements an alternative multi-class strategy, the so-called multi-class SVM formulated by Crammer and Singer, by using the option multi_class='crammer_singer'. This method is consistent, which is not true for one-vs-rest classification. In practice, one-vs-rest classification is usually preferred, since the results are mostly similar, but the runtime is significantly less.

### Regression

The method of Support Vector Classification can be extended to solve regression problems. This method is called Support Vector Regression.

The model produced by support vector classification (as described above) depends only on a subset of the training data, because the cost function for building the model does not care about training points that lie beyond the margin. Analogously, the model produced by Support Vector Regression depends only on a subset of the training data, because the cost function for building the model ignores any training data close to the model prediction.

There are three different implementations of Support Vector Regression: `SVR`, `NuSVR` and `LinearSVR`. `LinearSVR` provides a faster implementation than `SVR` but only considers linear kernels, while `NuSVR` implements a slightly different formulation than `SVR` and `LinearSVR`. See Implementation details for further details.

As with classification classes, the fit method will take as argument vectors X, y, only that in this case y is expected to have floating point values instead of integer values:

````Python
>>> from sklearn import svm
>>> X = [[0, 0], [2, 2]]
>>> y = [0.5, 2.5]
>>> clf = svm.SVR()
>>> clf.fit(X, y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
>>> clf.predict([[1, 1]])
array([ 1.5])
```` 
Examples:
[Support Vector Regression (SVR) using linear and non-linear kernels](https://sklearn.org/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py)

### Mathematical Formulation

A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/emmanuel_branch/Resources/images/mf1.png?raw=true)



## Yahoo! Finance as Machine Learning Prediction over the Most US Volatile Stocks 

In our final project, our team inegrated several Jupyter Notebook Modules, Packages and installed multiples Python Libraries to integrate our SVR Module to predict in our 2021 volatile exchange marcket in US. 

Please find below our Machine Learning Solution:

## CORE CODE:

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%201.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%202.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%203.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%204.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%205.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%206.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20Part%207.JPG?raw=true)

## MACHINE LEARNING RESULTS:

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%231.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%232.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%233.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%234.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%235.JPG?raw=true)


### Theming and templates in Python

The Plotly Python library comes pre-loaded with several themes that you can get started using right away, and it also provides support for creating and registering your own themes.

> Note on terminology: Theming generally refers to the process of defining default styles for visual elements. Themes in plotly are implemented using objects called templates. Templates are slightly more general than traditional themes because in addition to defining default styles, templates can pre-populate a figure with visual elements like annotations, shapes, images, and more. In the documentation we will refer to the overall process of defining default styles as theming, and when in comes to the plotly API we will talk about how themes are implemented using templates.

### Using built-in themes
##### View available themes
To see information about the available themes and the current default theme, display the plotly.io.templates configuration object like this.

````python
import plotly.io as pio
pio.templates
````

Templates configuration
-----------------------
````python
    Default template: 'plotly'
    Available templates:
        ['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']
````

From this, we can see that the default theme is "plotly", and we can see the names of several additional themes that we can choose from.

### Specifying themes in Plotly Express
All Plotly Express functions accept a template argument that can be set to the name of a registered theme (or to a Template object as discussed later in this section). Here is an example of using Plotly Express to build and display the same scatter plot with six different themes.

````python
import plotly.express as px

df = px.data.gapminder()
df_2007 = df.query("year==2007")

for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
    fig = px.scatter(df_2007,
                     x="gdpPercap", y="lifeExp", size="pop", color="continent",
                     log_x=True, size_max=60,
                     template=template, title="Gapminder 2007: '%s' theme" % template)
    fig.show()
````


![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl1.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl2.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl3.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl4.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl5.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl6.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl7.JPG?raw=true)



### Specifying themes in graph object figures
The theme for a particular graph object figure can be specified by setting the template property of the figure's layout to the name of a registered theme (or to a Template object as discussed later in this section). Here is an example of constructing a surface plot and then displaying it with each of six themes.

````python
import plotly.graph_objects as go
import pandas as pd

z_data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv")

fig = go.Figure(
    data=go.Surface(z=z_data.values),
    layout=go.Layout(
        title="Mt Bruno Elevation",
        width=500,
        height=500,
    ))

for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
    fig.update_layout(template=template, title="Mt Bruno Elevation: '%s' theme" % template)
    fig.show()
    
````


![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl8.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl9.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl10.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl11.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl12.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl13.JPG?raw=true)

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl14.JPG?raw=true)



### Specifying a default themes
If a theme is not provided to a Plotly Express function or to a graph object figure, then the default theme is used. The default theme starts out as "plotly", but it can be changed by setting the plotly.io.templates.default property to the name of a registered theme.

Here is an example of changing to default theme to "plotly_white" and then constructing a scatter plot with Plotly Express without providing a template.

Note: Default themes persist for the duration of a single session, but they do not persist across sessions. If you are working in an IPython kernel, this means that default themes will persist for the life of the kernel, but they will not persist across kernel restarts.

````python
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

df = px.data.gapminder()
df_2007 = df.query("year==2007")

fig = px.scatter(df_2007,
                 x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 log_x=True, size_max=60,
                 title="Gapminder 2007: current default theme")
fig.show()
````

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/pgl15.JPG?raw=true)


### Disable default theming
If you do not wish to use any of the new themes by default, or you want your figures to look exactly the way they did prior to plotly.py version 4, you can disable default theming by setting the default theme to "none".

````python
import plotly.io as pio
pio.templates.default = "none"
````

#### Reference:  [Plotly | Graphing Libraries (Fundamentals and Theming Templates)](https://plotly.com/python/templates/)


#### By Emmanuel Martinez, George Quintanilla and Charlie Burd.
> FINAL PROJECT - Columbia Engineering 2021
