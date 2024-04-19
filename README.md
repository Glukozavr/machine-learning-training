# Basic Python Library for Machine Learning

> Created from the learning materials of DeepLearning.AI

This is a trainig repository to store my experiments with algorithms and approaches learned from the DeepLeraning.AI course in Python

# How to build

0. Create virtual environment ```python3 -m venv venv```
1. Activate virutal environment ```source venv/bin/activate```
2. Run tests ```python setup.py pytest```
3. Build library ```python setup.py bdist_wheel```
4. Install library ```pip install /path/to/wheelfile.whl```

# How to use

```python
    import learn_regress
    from learn_regress import linear_regression
    from learn_regress import logistic_regression

    # LINEAR
    initial_w = 0.
    initial_b = 0.

    # some gradient descent settings
    iterations = 1500
    alpha = 0.01

    w,b,_,_ = linear_regression.gradient_descent(
                        # training data to build learning
                        x_train ,y_train,
                        # initial values for matching the function props
                        initial_w, initial_b, 
                        linear_regression.compute_cost, linear_regression.compute_gradient,
                        alpha, iterations)

    # LOGISTIC
    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    # Some gradient descent settings
    iterations = 10000
    alpha = 0.001

    w,b, J_history,_ = logistic_regression.gradient_descent(
                                    # training data to build learning
                                    X_train ,y_train,
                                    # initial values to matching the function props
                                    initial_w, initial_b, 
                                    logistic_regression.compute_cost, logistic_regression.compute_gradient, alpha, iterations, 0)
                        
```

# How to visualise

The results of testing data can be checked in the "simple_app":

1. Go to simple_app: ```cd ./simple_app```
2. Launch the app ```python ./linear_regression_plot.py```

You will see the visualised data with linear regression.

Logistic data visualisation is not yet ready.