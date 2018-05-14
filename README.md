# House-Prices-Advanced-Regression-Techniques

Predict sales prices and practice feature engineering, RFs, and gradient boosting

<b><h1>Introduction:</h1></b>

This project is based on Kaggel Challage i.e to predict the house prices based of lots of features  using advance techniques of <b>Machine Learning </b> algorithms.

![alt_tag](https://www.kdnuggets.com/wp-content/uploads/kaggle.jpg)

<b>Link: </b> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

<b> <h1>Advanced Learning Regression Algorithm</h1> </b>

In This project we are going to use some of the advanced learning algorithm to predict the housing prices. Because it is not easy task to fit the model from training data who has more than 80 features of single house, so we need to work with more complexity such as <b>cleaning</b> the data, <b>label encoding</b>, <b>one hot encoding</b> , <b>dimensionality reduction</b>, <b>features scaling</b> etc.

Hence, based on there Accuracy and the r2_score we will be desciding which algorithm is best for fitting the model in this case.

there are few algorithms which we are going to discuss here such as:
		
		1-XGBOOST
		2-GRADIENT BOOSTING REGRESSION
		3-RANDOM FOREST REGRESSOR

## XGBOOST

XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

It is an implementation of gradient boosting machines created by Tianqi Chen, now with contributions from many developers. It belongs to a broader collection of tools under the umbrella of the Distributed Machine Learning Community or DMLC who are also the creators of the popular mxnet deep learning library.

### Model Features

The implementation of the model supports the features of the scikit-learn and R implementations, with new additions like regularization. Three main forms of gradient boosting are supported:

	
		1-Gradient Boosting algorithm also called gradient boosting machine including the learning rate.
		2-Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels.
		3-Regularized Gradient Boosting with both L1 and L2 regularization.

## Gradient Boosting Regression

Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

Like other boosting methods, gradient boosting combines weak "learners" into a single strong learner in an iterative fashion. It is easiest to explain in the least-squares regression setting, where the goal is to "teach" a model F {\displaystyle F} F to predict values of the form y ^ = F ( x ) {\displaystyle {\hat {y}}=F(x)} \hat{y} = F(x) by minimizing the mean squared error ( y ^ − y ) 2 {\displaystyle ({\hat {y}}-y)^{2}} (\hat{y} - y)^2, averaged over some training set of actual values of the output variable y {\displaystyle y} y.

At each stage m {\displaystyle m} m, 1 ≤ m ≤ M {\displaystyle 1\leq m\leq M} 1 \le m \le M, of gradient boosting, it may be assumed that there is some imperfect model F m {\displaystyle F_{m}} F_m (at the outset, a very weak model that just predicts the mean y in the training set could be used). The gradient boosting algorithm improves on F m {\displaystyle F_{m}} F_m by constructing a new model that adds an estimator h to provide a better model: F m + 1 ( x ) = F m ( x ) + h ( x ) {\displaystyle F_{m+1}(x)=F_{m}(x)+h(x)} F_{m+1}(x) = F_m(x) + h(x). To find h {\displaystyle h} h, the gradient boosting solution starts with the observation that a perfect h would imply

    F m + 1 ( x ) = F m ( x ) + h ( x ) = y {\displaystyle F_{m+1}(x)=F_{m}(x)+h(x)=y} {\displaystyle F_{m+1}(x)=F_{m}(x)+h(x)=y}

or, equivalently,

    h ( x ) = y − F m ( x ) {\displaystyle h(x)=y-F_{m}(x)} h(x) = y - F_m(x) .

Therefore, gradient boosting will fit h to the residual y − F m ( x ) {\displaystyle y-F_{m}(x)} y - F_m(x). Like in other boosting variants, each F m + 1 {\displaystyle F_{m+1}} F_{m+1} learns to correct its predecessor F m {\displaystyle F_{m}} F_m. A generalization of this idea to loss functions other than squared error — and to classification and ranking problems — follows from the observation that residuals y − F ( x ) {\displaystyle y-F(x)} y - F(x) for a given model are the negative gradients (with respect to F(x)) of the squared error loss function 1 2 ( y − F ( x ) ) 2 {\displaystyle {\frac {1}{2}}(y-F(x))^{2}} \frac{1}{2}(y - F(x))^2. So, gradient boosting is a gradient descent algorithm; and generalizing it entails "plugging in" a different loss and its gradient.



	



