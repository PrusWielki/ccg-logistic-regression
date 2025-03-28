# Cyclic Coordinate Descent for Logistic Regression with Lasso regularization

This notebook presents the implementation of Cyclic Coordinate Descent (CCD) algorithm for parameter
estimation in regularized logistic regression with l1 (lasso) penalty and compares it with standard
logistic regression model without regularization.

In particular it involves the implementation of such algorithm from scratch and comparison to the Logistic Regression available in scikit-learn package.

The details of the data and algorithms used are in the solution.ipynb and in the ./report/report.pdf

# Using a different dataset

There are 3 ways in case you want to use a separate dataset:

1. Wrapping the new dataset in the Dataset class:

   1. Prepare the dataset: create a Pandas dataframe where the last column is the target and the rest of the columns are features. If your dataset is in the .arff format you should be able to load it with load_dataset() function.
   2. Create a new instance of the Dataset class. As arguments pass the name, source df, and optionally preprocessing steps (available as methods of the Dataset class)
   3. As a result you will receive an object with X and y properties that are numpy arrays ready for further processing.
   4. Create train and test sets if needed with scikit's train_test_split
   5. Create an instance of LogRegCCD, use the fit() method on the train set and then validate() with chosen metric (callable function) to receive a score.

---

If all your new datasets are in the .arff format. Place them in a single folder and use the below code snippet:

```{python}
datasets = load_datasets(path_to_datasets)

preprocessing_steps = [
    Dataset.fill_missing_values,
    Dataset.remove_colinear_features,
    Dataset.normalize,
]

for i in range(len(datasets)):
    datasets[i] = Dataset(datasets[i]["name"], datasets[i]["data"], preprocessing_steps)
```

---

2. Creating a class inheriting from the Dataset class

   1. Create a new class that inherits from the Dataset class.
   2. In the **init**() load the dataset (you may use the load_dataset() class if it's in the .arff format)
   3. Make sure your loaded data is a Pandas dataframe where the last column is the target and the rest of the columns are features
   4. Call super().**init**() with name, source df, and optionally preprocessing steps (available as methods of the Dataset class)
   5. Once you create an instance of the new class you will receive an object with X and y properties that are numpy arrays ready for further processing.
   6. Create train and test sets if needed with scikit's train_test_split
   7. Create an instance of LogRegCCD, use the fit() method on the train set and then validate() with chosen metric (callable function) to receive a score

3. Manually creating a new dataframe

   1. Load and preprocess your data to a pandas Dataframe (if your data is in the .arff format then use the load_dataset() function)
   2. Convert the features and the response variable to numpy arrays. Make sure the response classes are numeric.
   3. Create train and test sets if needed with scikit's train_test_split
   4. Create an instance of LogRegCCD, use the fit() method on the train set and then validate() with chosen metric (callable function) to receive a score

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

# Reproducibility

To ensure reproducibility: download the datasets from the provided links and place them in the datasets folder (look dir structure below), use the same Python version (Requirements) and don't change the seeds. The experiments have been performed on a Windows 11 Intel i5 machine.

To reproduce all results simply execute the notebook cells one by one. Alternatively execute the _Imports & Consts_ and _Classes & Functions_ sections and then execute sections of interest or just use the defined functions.

# Directory structure

```
.
├── datasets/
├── report/
├── results/
└── solution.ipynb
```

![Parameter Facet Grid](./results/parameter_facet_grid.png)

![Comparison on synthetic dataset](./results/comparison-synthetic-dataset.png)

![Comparison of the coefficients with redundant feautures](./results/logistic_regression_l1_logregccd_coefficients_redundant_features.png)

TODO: Add here 1-2 more images, preferably of comparison on real dataset and maybe how the weights change with lambda or similar
