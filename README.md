# Basic Python Library for Machine Learning

> Created from the learning materials of DeepLearning.AI

This is a trainig repository to store my experiments with algorithms and approaches learned from the DeepLeraning.AI course in Python

# How to build

1. Activate virutal environment ```source venv/bin/activate```
2. Run tests ```python setup.py pytest```
3. Build library ```python setup.py bdist_wheel```
4. Install library ```pip install /path/to/wheelfile.whl```

# How to use

```python
    import learn_regress
    from learn_regress import linear_regression
```