# Cyclic Coordinate Descent for Logistic Regression with Lasso regularization

This notebook presents the implementation of Cyclic Coordinate Descent (CCD) algorithm for parameter 
estimation in regularized logistic regression with l1 (lasso) penalty and compares it with standard 
logistic regression model without regularization. 

In particular it involves the implementation of such algorithm from scratch and comparison to the Logistic Regression available in scikit-learn package.

# Requirements

All the scripts have been executed with [Python 3.13.2](https://www.python.org/). [Pyenv](https://github.com/pyenv/pyenv) can be used to easily manage your python installations. It is recommended to create a python environment in the cloned repository:

```
python -m venv --prompt venv .\venv
```

Afterwards activate the environment (./venv/Scripts) and install the requirements present in requirements.txt:

```
pip install -r requirements.txt
```

All the scripts are formatted with the Black formatter.

# Directory structure

```  
.  
├── datasets/  
├── report/  
├── results/  
└── solution.ipynb  
```  

TODO:

1. Place here resulting graphics of comparison and performance.
2. Add high-level overview of the algorithm
3. Add description of the datasets used
4. Add information about reproducibility, general clean code pricnicples used, and docstring documentation
5. Add information about directory structure