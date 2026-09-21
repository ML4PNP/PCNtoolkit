Command line interface
======================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/09_command_line_interface.ipynb>`

The PCNtoolkit is a python package, but it can also be used from the
command line.

Here we show how to use the PCNtoolkit from the command line.

Furthermore, you can use this script to generate commands for the
command line interface. (Although if you are able to run this notebook,
why not just use it as a python package?)

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import sys
    import pickle

BLR Example
-----------

Data preparation
~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Download and split data first
    # If you are running this notebook for the first time, you need to download the dataset from github.
    # If you have already downloaded the dataset, you can comment out the following line
    os.makedirs("resources/data", exist_ok=True)
    pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    ).to_csv("resources/data/fcon1000.csv", index=False)


.. code:: ipython3

    data = pd.read_csv("resources/data/fcon1000.csv")

.. code:: ipython3

    # Inspect the data
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=data, x=("age"), y=("rh_MeanThickness_thickness"), hue=("site"), ax=ax[1])
    ax[1].legend([], [])
    ax[1].set_title("Scatter plot of age vs rh_MeanThickness_thickness")
    ax[1].set_xlabel("Age")
    ax[1].set_ylabel("rh_MeanThickness_thickness")
    sns.countplot(data=data, y="site", hue="sex", ax=ax[0], orient="h")
    ax[0].legend(title="Sex")
    ax[0].set_title("Count of sites")
    ax[0].set_xlabel("Site")
    ax[0].set_ylabel("Count")
    plt.show()




.. image:: 09_command_line_interface_files/09_command_line_interface_6_0.png


.. code:: ipython3

    # Split into X, y, and batch effects
    covariate_columns = ["age"]
    batch_effect_columns = ["sex", "site"]
    response_columns = ["rh_MeanThickness_thickness", "WM-hypointensities"]
    
    X = data[covariate_columns]
    Y = data[response_columns]
    batch_effects = data[batch_effect_columns]
    
    batch_effects_strings = [str(b[0]) + " " + str(b[1]) for b in batch_effects.values]
    
    # Split into train and test set
    trainidx, testidx = train_test_split(data.index, test_size=0.2, random_state=42, stratify=batch_effects_strings)
    train_X = X.loc[trainidx]
    train_Y = Y.loc[trainidx]
    train_batch_effects = batch_effects.loc[trainidx]
    
    test_X = X.loc[testidx]
    test_Y = Y.loc[testidx]
    test_batch_effects = batch_effects.loc[testidx]

.. code:: ipython3

    # Save stuff
    root_dir = os.path.join("resources", "cli_example")
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    resp = os.path.abspath(os.path.join(data_dir, "responses.csv"))
    cov = os.path.abspath(os.path.join(data_dir, "covariates.csv"))
    be = os.path.abspath(os.path.join(data_dir, "batch_effects.csv"))
    
    t_resp = os.path.abspath(os.path.join(data_dir, "test_responses.csv"))
    t_cov = os.path.abspath(os.path.join(data_dir, "test_covariates.csv"))
    t_be = os.path.abspath(os.path.join(data_dir, "test_batch_effects.csv"))
    
    
    with open(cov, "wb") as f:
        pickle.dump(train_X, f)
    with open(resp, "wb") as f:
        pickle.dump(train_Y, f)
    with open(be, "wb") as f:
        pickle.dump(train_batch_effects, f)
    with open(t_cov, "wb") as f:
        pickle.dump(test_X, f)
    with open(t_resp, "wb") as f:
        pickle.dump(test_Y, f)
    with open(t_be, "wb") as f:
        pickle.dump(test_batch_effects, f)

BLR configuration
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    alg = "blr"
    func = "fit_predict"
    
    # normative model configuration
    save_dir = os.path.join(root_dir, "blr_cli", "save_dir")
    savemodel = True
    saveresults = True
    basis_function = "linear"
    inscaler = "standardize"
    outscaler = "standardize"
    
    # Regression model configuration
    optimizer = "l-bfgs-b"
    n_iter = 200
    heteroskedastic = True
    fixed_effect = True
    warp = "WarpSinhArcsinh"
    warp_reparam = True
    
    # runner configuration
    cross_validate = True
    cv_folds = 5
    parallelize = False
    job_type = "local"
    n_jobs = 2
    temp_dir = os.path.join(root_dir, "temp")
    log_dir = os.path.join(root_dir, "log")
    python_env = os.path.join(os.path.dirname(os.path.dirname(sys.executable)))

Constructing command
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    command = "normative"
    args = f"-a {alg} -f {func} -c {cov} -r {resp} -t {t_resp} -e {t_cov} -k {cv_folds}"
    kwargs = f"be={be} t_be={t_be}"
    normative_model_kwargs = f"save_dir={save_dir} savemodel={savemodel} saveresults={saveresults} basis_function={basis_function} inscaler={inscaler} outscaler={outscaler}"
    runner_kwargs = f"cross_validate={cross_validate} parallelize={parallelize} job_type={job_type} n_jobs={n_jobs} temp_dir={temp_dir} log_dir={log_dir} environment={python_env}"
    blr_kwargs = f"optimizer={optimizer} n_iter={n_iter} heteroskedastic={heteroskedastic} fixed_effect={fixed_effect} warp={warp} warp_reparam={warp_reparam}"
    full_command = f"{command} {args} {kwargs} {runner_kwargs} {normative_model_kwargs} {blr_kwargs}"


.. code:: ipython3

    print(full_command)


.. code:: text

    normative -a blr -f fit_predict -c /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/covariates.csv -r /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/responses.csv -t /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_responses.csv -e /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_covariates.csv -k 5 be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/batch_effects.csv t_be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_batch_effects.csv cross_validate=True parallelize=False job_type=local n_jobs=2 temp_dir=resources/cli_example/temp log_dir=resources/cli_example/log environment=/opt/hostedtoolcache/Python/3.13.15/x64 save_dir=resources/cli_example/blr_cli/save_dir savemodel=True saveresults=True basis_function=linear inscaler=standardize outscaler=standardize optimizer=l-bfgs-b n_iter=200 heteroskedastic=True fixed_effect=True warp=WarpSinhArcsinh warp_reparam=True


Running command
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{full_command}


.. code:: text

    Process: 3127 - 2026-09-21 12:17:00 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3127 - 2026-09-21 12:17:01 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3127 - 2026-09-21 12:17:01 - Task ID created: fit_predict_fit_data__2026-09-21_12:17:01_3.468750
    Process: 3127 - 2026-09-21 12:17:01 - Temporary directory created:
    	/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/temp/fit_predict_fit_data__2026-09-21_12:17:01_3.468750
    Process: 3127 - 2026-09-21 12:17:01 - Log directory created:
    	/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/log/fit_predict_fit_data__2026-09-21_12:17:01_3.468750
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:01 - Predict data not used in k-fold cross-validation
      warnings.warn(message, category)
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/sklearn/model_selection/_split.py:812: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
      warnings.warn(
    Process: 3127 - 2026-09-21 12:17:01 - Fitting models on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Fitting model for response_var_0.
    Process: 3127 - 2026-09-21 12:17:01 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.523723946631791e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:01 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.375496563279125e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.9980721138137064e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.746691663731469e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.104659065295494e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 3127 - 2026-09-21 12:17:01 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_0.
    Process: 3127 - 2026-09-21 12:17:01 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:01 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:01 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:01 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:01 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:01 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:01 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:01 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:01 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:02 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:02 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:02 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:02 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:02 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:02 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:02 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:03 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:03 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:03 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:03 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:03 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:03 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:03 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:03 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:03 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:03 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:03 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:03 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:03 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:03 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:04 - Fitting models on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:04 - Fitting model for response_var_0.
    Process: 3127 - 2026-09-21 12:17:04 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.2455126268504203e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:04 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.910057133212752e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.8415638861183265e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.185259614266013e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.691929377086917e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:711: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    Process: 3127 - 2026-09-21 12:17:04 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_1.
    Process: 3127 - 2026-09-21 12:17:04 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:04 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:04 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:04 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:04 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:04 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:04 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:04 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:05 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:05 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:05 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:05 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:05 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:05 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:05 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:05 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:05 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:05 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:05 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:05 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:05 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:06 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:06 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:06 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:06 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:06 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:06 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:06 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:06 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:06 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:06 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:06 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:06 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:06 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:06 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:07 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:07 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:07 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:07 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:07 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:07 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:07 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:07 - Fitting models on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:07 - Fitting model for response_var_0.
    Process: 3127 - 2026-09-21 12:17:08 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.701875421761123e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:08 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.6521548448675865e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.3656485817707845e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.502124577440806e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 3127 - 2026-09-21 12:17:08 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_2.
    Process: 3127 - 2026-09-21 12:17:08 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:08 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:08 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:08 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:08 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:08 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:08 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:08 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:08 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:08 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:08 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:08 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:08 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:08 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:09 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:09 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:09 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:09 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:09 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:09 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:09 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:09 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:09 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:09 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:09 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:09 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:09 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:09 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:10 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:10 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:10 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:10 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:10 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:10 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:10 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:11 - Fitting models on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Fitting model for response_var_0.
    Process: 3127 - 2026-09-21 12:17:11 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.345402012863007e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:11 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.416128457233547e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.01776121710389e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.4551614365857112e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:711: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    Process: 3127 - 2026-09-21 12:17:11 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_3.
    Process: 3127 - 2026-09-21 12:17:11 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:11 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:11 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:11 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:11 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:11 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:11 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:11 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:11 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:12 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:12 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:12 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:12 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:12 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:12 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:12 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:13 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:13 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:13 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:13 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:13 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:13 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:13 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:13 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:13 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:13 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:13 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:13 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:13 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:13 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:14 - Fitting models on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:14 - Fitting model for response_var_0.
    Process: 3127 - 2026-09-21 12:17:14 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204862148507314e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:14 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.1618189276041684e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204835921010135e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:632: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204861445939724e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 3127 - 2026-09-21 12:17:14 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_4.
    Process: 3127 - 2026-09-21 12:17:14 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:14 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:14 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:14 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:14 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:14 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:14 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:14 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:14 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:14 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:14 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:15 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:15 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:15 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:15 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:15 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:15 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:15 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:15 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:15 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:15 - Harmonizing data for response_var_0.
    Process: 3127 - 2026-09-21 12:17:16 - Making predictions on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:16 - Computing z-scores for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:16 - Computing z-scores for response_var_1.
    Process: 3127 - 2026-09-21 12:17:16 - Computing z-scores for response_var_0.
    Process: 3127 - 2026-09-21 12:17:16 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:16 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:16 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:16 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:16 - Computing log-probabilities for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:16 - Computing log-probabilities for response_var_1.
    Process: 3127 - 2026-09-21 12:17:16 - Computing log-probabilities for response_var_0.
    Process: 3127 - 2026-09-21 12:17:16 - Computing yhat for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:16 - Computing yhat for response_var_1.
    Process: 3127 - 2026-09-21 12:17:16 - Computing yhat for response_var_0.
    Process: 3127 - 2026-09-21 12:17:17 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3127 - 2026-09-21 12:17:17 - Computing centiles for 2 response variables.
    Process: 3127 - 2026-09-21 12:17:17 - Computing centiles for response_var_1.
    Process: 3127 - 2026-09-21 12:17:17 - Computing centiles for response_var_0.
    Process: 3127 - 2026-09-21 12:17:17 - Harmonizing data on 2 response variables.
    Process: 3127 - 2026-09-21 12:17:17 - Harmonizing data for response_var_1.
    Process: 3127 - 2026-09-21 12:17:17 - Harmonizing data for response_var_0.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3127 - 2026-09-21 12:17:17 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)


You can find the results in the
``resources/cli_example/blr_cli/save_dir`` folder.

.. code:: ipython3

    results_path = os.path.join(
        save_dir,
        "folds",
        "fold_1",
        "results",
        "statistics_fit_data_fold_1_predict.csv",
    )
    a = pd.read_csv(results_path, index_col=0)
    display(a)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>response_var_0</th>
          <th>response_var_1</th>
        </tr>
        <tr>
          <th>statistic</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>EXPV</th>
          <td>4.678718e-01</td>
          <td>2.027784e-01</td>
        </tr>
        <tr>
          <th>Kurtosis</th>
          <td>-1.804822e-01</td>
          <td>-8.249590e-02</td>
        </tr>
        <tr>
          <th>MACE</th>
          <td>1.615801e-01</td>
          <td>1.748506e-01</td>
        </tr>
        <tr>
          <th>MAPE</th>
          <td>2.359962e-02</td>
          <td>3.068768e-01</td>
        </tr>
        <tr>
          <th>MLL</th>
          <td>1.128491e+00</td>
          <td>8.335007e-01</td>
        </tr>
        <tr>
          <th>MSLL</th>
          <td>-3.151540e-01</td>
          <td>-8.652691e-01</td>
        </tr>
        <tr>
          <th>R2</th>
          <td>4.678505e-01</td>
          <td>2.008805e-01</td>
        </tr>
        <tr>
          <th>RMSE</th>
          <td>7.323381e-02</td>
          <td>9.002539e+02</td>
        </tr>
        <tr>
          <th>Rho</th>
          <td>6.422068e-01</td>
          <td>4.885517e-01</td>
        </tr>
        <tr>
          <th>Rho_p</th>
          <td>1.691404e-21</td>
          <td>9.164062e-12</td>
        </tr>
        <tr>
          <th>SMSE</th>
          <td>5.321495e-01</td>
          <td>7.991195e-01</td>
        </tr>
        <tr>
          <th>ShapiroW</th>
          <td>9.839388e-01</td>
          <td>9.563680e-01</td>
        </tr>
        <tr>
          <th>Skewness</th>
          <td>-2.081598e-01</td>
          <td>6.883883e-01</td>
        </tr>
      </tbody>
    </table>
    </div>


HBR example
-----------

.. code:: ipython3

    alg = "hbr"
    func = "fit_predict"
    
    # normative model configuration
    save_dir = os.path.join(root_dir, "hbr", "save_dir")
    savemodel = True
    saveresults = True
    basis_function = "bspline"
    inscaler = "standardize"
    outscaler = "standardize"
    
    
    # Regression model configuration
    draws = 1000
    tune = 500
    chains = 4
    nuts_sampler = "nutpie"
    
    likelihood = "Normal"
    linear_mu = "True"
    random_intercept_mu = "True"
    random_slope_mu = "False"
    linear_sigma = "True"
    random_intercept_sigma = "False"
    random_slope_sigma = "False"

Constructing command
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    command = "normative"
    args = f"-a {alg} -f {func} -c {cov} -r {resp} -t {t_resp} -e {t_cov}"
    kwargs = f"be={be} t_be={t_be}"
    normative_model_kwargs = f"save_dir={save_dir} savemodel={savemodel} saveresults={saveresults} basis_function={basis_function} inscaler={inscaler} outscaler={outscaler}"
    hbr_kwargs = f"draws={draws} tune={tune} chains={chains} nuts_sampler={nuts_sampler} likelihood={likelihood} linear_mu={linear_mu} random_intercept_mu={random_intercept_mu} random_slope_mu={random_slope_mu} linear_sigma={linear_sigma} random_intercept_sigma={random_intercept_sigma} random_slope_sigma={random_slope_sigma}"
    full_command = f"{command} {args} {kwargs} {normative_model_kwargs} {hbr_kwargs}"
    print(full_command)


.. code:: text

    normative -a hbr -f fit_predict -c /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/covariates.csv -r /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/responses.csv -t /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_responses.csv -e /home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_covariates.csv be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/batch_effects.csv t_be=/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/data/test_batch_effects.csv save_dir=resources/cli_example/hbr/save_dir savemodel=True saveresults=True basis_function=bspline inscaler=standardize outscaler=standardize draws=1000 tune=500 chains=4 nuts_sampler=nutpie likelihood=Normal linear_mu=True random_intercept_mu=True random_slope_mu=False linear_sigma=True random_intercept_sigma=False random_slope_sigma=False


Running command
~~~~~~~~~~~~~~~

.. code:: ipython3

    !{full_command}


.. code:: text

    Process: 3141 - 2026-09-21 12:17:21 - No log directory specified. Using default log directory: /home/runner/.pcntoolkit/logs
    Process: 3141 - 2026-09-21 12:17:21 - No temporary directory specified. Using default temporary directory: /home/runner/.pcntoolkit/temp
    Process: 3141 - 2026-09-21 12:17:21 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3141 - 2026-09-21 12:17:21 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 3141 - 2026-09-21 12:17:21 - Task ID created: fit_predict_fit_data__2026-09-21_12:17:21_167.589844
    Process: 3141 - 2026-09-21 12:17:21 - Temporary directory created:
    	/home/runner/.pcntoolkit/temp/fit_predict_fit_data__2026-09-21_12:17:21_167.589844
    Process: 3141 - 2026-09-21 12:17:21 - Log directory created:
    	/home/runner/.pcntoolkit/logs/fit_predict_fit_data__2026-09-21_12:17:21_167.589844
    Process: 3141 - 2026-09-21 12:17:21 - Fitting models on 2 response variables.
    Process: 3141 - 2026-09-21 12:17:21 - Fitting model for response_var_0.
    NUTS[nutpie]: [slope_mu, mu_intercept_mu, batch_effect_0_sigma_intercept_mu, normalized_batch_effect_0_offset_intercept_mu, batch_effect_1_sigma_intercept_mu, normalized_batch_effect_1_offset_intercept_mu, slope_sigma, intercept_sigma]
    [?25l                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
    [2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━━━━[0m [90m━━━━━━━━[0m   0       0       0.000   0       0.00 draws/s   0:00…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m╸[0m[90m━━━[0m [90m━━━━━━━━[0m   111     0       0.228   7       0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━[0m[90m╺[0m[90m━[0m [90m━━━━━━━━[0m   233     0       0.738   7       0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   367     0       0.374   15      1235.61 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   103     0       0.324   15      1208.34 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   210     1       0.340   15      1185.24 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━[0m[90m╺[0m[90m━━━━━[0m   331     3       0.350   31      1190.48 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   450     5       0.360   15      1190.42 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   566     5       0.315   15      1188.36 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   674     6       0.330   19      1178.67 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━[0m[90m╺[0m[90m━[0m   783     8       0.366   7       1170.58 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━[0m[90m╺[0m   894     9       0.334   23      1165.52 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   10      0       0.453   15      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m╸[0m[90m━━━[0m [90m━━━━━━━━[0m   97      0       0.402   7       0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   210     0       0.363   15      0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   355     0       0.419   15      1148.76 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   484     0       0.309   23      1183.28 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   113     0       0.362   15      1206.62 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━━━━[0m [36m━[0m[90m╺[0m[90m━━━━━━[0m   241     0       0.372   15      1220.70 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━━━━[0m [36m━━[0m[36m╸[0m[90m━━━━━[0m   372     0       0.396   7       1235.08 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [36m━━━━[0m [36m━━━[0m[36m╸[0m[90m━━━━[0m   490     0       0.353   15      1228.25 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[31m╸[0m[90m━━━[0m   599     1       0.387   15      1214.33 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   708     1       0.356   15      1203.12 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   835     1       0.391   7       1210.31 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━[0m[31m╸[0m   952     1       0.365   7       1206.96 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [36m╸[0m[90m━━━[0m [90m━━━━━━━━[0m   58      0       0.466   7       0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   160     0       0.113   31      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   365     0       0.406   31      997.17 draw…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   496     0       0.544   7       1064.30 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   128     5       0.334   7       1113.40 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[31m╸[0m[90m━━━━━━[0m   264     7       0.337   15      1152.27 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[31m╸[0m[90m━━━━━[0m   401     11      0.340   15      1180.80 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   539     17      0.348   15      1205.28 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[31m╸[0m[90m━━━[0m   647     20      0.359   7       1193.50 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[31m╸[0m[90m━━[0m   765     26      0.397   7       1192.23 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━[0m[90m╺[0m   895     30      0.389   15      1202.55 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   13      0       0.119   7       0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [36m╸[0m[90m━━━[0m [90m━━━━━━━━[0m   74      0       0.342   15      0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   189     0       0.378   15      0.00 draws/s   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [36m━━[0m[90m╺[0m[90m━[0m [90m━━━━━━━━[0m   281     0       0.148   31      883.57 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   369     0       0.332   15      880.61 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   476     0       0.366   15      917.08 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   91      2       0.297   15      956.26 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   196     4       0.276   7       970.67 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[31m╸[0m[90m━━━━━━[0m   307     4       0.271   15      986.52 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[31m╸[0m[90m━━━━━[0m   408     4       0.290   15      990.16 draw…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[31m╸[0m[90m━━━━[0m   517     8       0.261   7       1001.94 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[31m╸[0m[90m━━━[0m   633     11      0.265   15      1015.21 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[31m╸[0m[90m━━[0m   732     14      0.285   15      1013.97 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   839     15      0.263   15      1019.01 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    16      0.268   15      1020.49 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    16      0.268   15      1020.49 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    11      0.357   15      1165.52 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    1       0.357   7       1206.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    32      0.399   15      1202.55 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    16      0.268   15      1020.49 dra…   0:00…   0:00…  
                                                                                    
    [?25hProcess: 3141 - 2026-09-21 12:17:40 - Fitting model for response_var_1.
    NUTS[nutpie]: [slope_mu, mu_intercept_mu, batch_effect_0_sigma_intercept_mu, normalized_batch_effect_0_offset_intercept_mu, batch_effect_1_sigma_intercept_mu, normalized_batch_effect_1_offset_intercept_mu, slope_sigma, intercept_sigma]
    [?25l                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
    [2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [90m━━━━[0m [90m━━━━━━━━[0m   0       0       0.000   0       0.00 draws/s   0:00…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m╸[0m[90m━━━[0m [90m━━━━━━━━[0m   90      0       0.401   15      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   213     0       0.461   15      0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━[0m[36m╸[0m[90m━[0m [90m━━━━━━━━[0m   339     0       0.331   15      1141.34 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   91      2       0.337   15      1184.29 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   219     14      0.358   15      1200.26 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━[0m[90m╺[0m[90m━━━━━[0m   342     16      0.370   7       1207.97 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   479     22      0.343   15      1226.75 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━[0m[31m╸[0m[90m━━━[0m   600     24      0.337   15      1226.26 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━[0m[31m╸[0m[90m━━[0m   719     27      0.375   11      1222.62 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   847     30      0.363   7       1230.10 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━[0m[31m╸[0m   983     54      0.378   15      1240.96 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   51      0       0.160   63      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   162     0       0.400   21      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [36m━━[0m[36m╸[0m[90m━[0m [90m━━━━━━━━[0m   302     0       0.488   15      1048.51 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [36m━━━[0m[36m╸[0m [90m━━━━━━━━[0m   432     0       0.524   7       1113.32 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   77      1       0.364   7       1182.31 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   227     13      0.399   7       1240.55 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[90m╺[0m[90m━━━━━[0m   364     15      0.411   15      1259.42 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[31m╸[0m[90m━━━━[0m   489     19      0.391   15      1259.83 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[31m╸[0m[90m━━━[0m   616     20      0.418   15      1260.98 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[31m╸[0m[90m━━[0m   756     23      0.368   15      1276.36 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━[0m[90m╺[0m   895     28      0.373   7       1286.87 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   17      0       0.165   31      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   124     0       0.404   15      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [36m━━[0m[90m╺[0m[90m━[0m [90m━━━━━━━━[0m   237     0       0.439   15      0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [36m━━[0m[36m╸[0m[90m━[0m [90m━━━━━━━━[0m   340     0       0.253   31      1055.84 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m━━━━━━━━[0m   42      2       0.278   15      1038.23 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m╸[0m[90m━━━━━━━[0m   155     5       0.304   15      1054.68 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[31m╸[0m[90m━━━━━━[0m   269     8       0.284   15      1066.51 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[90m╺[0m[90m━━━━━[0m   349     13      0.283   15      1035.27 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   460     29      0.266   5       1044.56 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   571     31      0.264   15      1050.98 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   678     35      0.295   15      1053.62 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[90m╺[0m[90m━[0m   811     53      0.299   15      1076.31 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━[0m[90m╺[0m   919     54      0.277   15      1077.41 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   22      0       0.089   3       0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   144     0       0.540   15      0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [36m━━[0m[90m╺[0m[90m━[0m [90m━━━━━━━━[0m   255     0       0.338   15      0.00 draws/s   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   362     0       0.269   31      1040.11 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   469     0       0.324   15      1049.13 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   82      1       0.285   15      1063.91 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   198     7       0.295   15      1080.40 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[90m╺[0m[90m━━━━━[0m   317     11      0.323   15      1095.13 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   430     12      0.321   15      1100.55 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   548     13      0.305   15      1108.96 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   669     19      0.291   31      1120.77 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[90m╺[0m[90m━[0m   783     26      0.291   15      1121.48 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.303   15      1125.48 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.303   15      1125.48 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    54      0.377   15      1240.96 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    28      0.368   7       1286.87 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    66      0.300   3       1077.41 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.303   15      1125.48 dra…   0:00…   0:00…  
                                                                                    
    [?25hProcess: 3141 - 2026-09-21 12:17:51 - Saving model to:
    	resources/cli_example/hbr/save_dir.
    Process: 3141 - 2026-09-21 12:17:51 - Making predictions on 2 response variables.
    Process: 3141 - 2026-09-21 12:17:51 - Computing z-scores for 2 response variables.
    Process: 3141 - 2026-09-21 12:17:51 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 3141 - 2026-09-21 12:17:52 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 3141 - 2026-09-21 12:17:53 - Computing centiles for 2 response variables.
    Process: 3141 - 2026-09-21 12:17:53 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:17:54 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:17:56 - Computing log-probabilities for 2 response variables.
    Process: 3141 - 2026-09-21 12:17:56 - Computing log-probabilities for 2 response variables.
    Process: 3141 - 2026-09-21 12:17:56 - Computing log-probabilities for response_var_1.
    Process: 3141 - 2026-09-21 12:17:58 - Computing log-probabilities for response_var_0.
    Process: 3141 - 2026-09-21 12:17:58 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:17:59 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3141 - 2026-09-21 12:17:59 - Computing centiles for 2 response variables.
    Process: 3141 - 2026-09-21 12:17:59 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:01 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:02 - Harmonizing data on 2 response variables.
    Process: 3141 - 2026-09-21 12:18:02 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:03 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:04 - Making predictions on 2 response variables.
    Process: 3141 - 2026-09-21 12:18:04 - Computing z-scores for 2 response variables.
    Process: 3141 - 2026-09-21 12:18:04 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:04 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:05 - Computing centiles for 2 response variables.
    Process: 3141 - 2026-09-21 12:18:05 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:06 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:08 - Computing log-probabilities for 2 response variables.
    Process: 3141 - 2026-09-21 12:18:08 - Computing log-probabilities for 2 response variables.
    Process: 3141 - 2026-09-21 12:18:08 - Computing log-probabilities for response_var_1.
    Process: 3141 - 2026-09-21 12:18:08 - Computing log-probabilities for response_var_0.
    Process: 3141 - 2026-09-21 12:18:08 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:09 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 3141 - 2026-09-21 12:18:09 - Computing centiles for 2 response variables.
    Process: 3141 - 2026-09-21 12:18:09 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:11 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:12 - Harmonizing data on 2 response variables.
    Process: 3141 - 2026-09-21 12:18:12 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 3141 - 2026-09-21 12:18:13 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3141 - 2026-09-21 12:18:14 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3141 - 2026-09-21 12:18:14 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3141 - 2026-09-21 12:18:14 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)


