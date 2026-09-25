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

    Process: 2873 - 2026-09-25 15:39:14 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 2873 - 2026-09-25 15:39:14 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 2873 - 2026-09-25 15:39:14 - Task ID created: fit_predict_fit_data__2026-09-25_15:39:14_787.181641
    Process: 2873 - 2026-09-25 15:39:14 - Temporary directory created:
    	/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/temp/fit_predict_fit_data__2026-09-25_15:39:14_787.181641
    Process: 2873 - 2026-09-25 15:39:14 - Log directory created:
    	/home/runner/work/PCNtoolkit/PCNtoolkit/resources/cli_example/log/fit_predict_fit_data__2026-09-25_15:39:14_787.181641
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:14 - Predict data not used in k-fold cross-validation
      warnings.warn(message, category)
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/sklearn/model_selection/_split.py:812: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
      warnings.warn(
    Process: 2873 - 2026-09-25 15:39:14 - Fitting models on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:14 - Fitting model for response_var_0.
    Process: 2873 - 2026-09-25 15:39:14 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.523723942747041e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:15 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.3754965587953245e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.998072110298649e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.746691658986639e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 6.104659061002182e-27.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 2873 - 2026-09-25 15:39:15 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_0.
    Process: 2873 - 2026-09-25 15:39:15 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:15 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:15 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:15 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:15 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:15 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:15 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:15 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:15 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:15 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:15 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:15 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:15 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:15 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:16 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:16 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:16 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:16 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:16 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:16 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:16 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:16 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:16 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:16 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:16 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:16 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:16 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:16 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:17 - Fitting models on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Fitting model for response_var_0.
    Process: 2873 - 2026-09-25 15:39:17 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.245468051093369e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:17 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.910073793103063e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.841582165217427e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.1853021087340614e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 4.6918797202409685e-20.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 2873 - 2026-09-25 15:39:17 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_1.
    Process: 2873 - 2026-09-25 15:39:17 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:17 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:17 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:17 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:17 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:17 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:17 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:17 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:17 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:18 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:18 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:18 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:18 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:18 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:18 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:18 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:18 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:18 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:18 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:18 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:18 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:18 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:18 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:19 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:19 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:19 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Fitting models on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Fitting model for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6102023850473133e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:19 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.1284984478117152e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.6283034262774061e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.263698600219996e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 2873 - 2026-09-25 15:39:19 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_2.
    Process: 2873 - 2026-09-25 15:39:19 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:19 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:19 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:19 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:19 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:19 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:19 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:20 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:20 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:20 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:20 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:20 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:20 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:20 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:20 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:20 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:20 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:20 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:20 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:20 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:21 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:21 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:21 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:21 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:21 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:21 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:21 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:21 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:21 - Fitting models on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:21 - Fitting model for response_var_0.
    Process: 2873 - 2026-09-25 15:39:22 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.779980244090766e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:22 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.5707959418056206e-19.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.9880477203002863e-17.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.911155537663099e-18.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/scipy/optimize/_numdiff.py:711: RuntimeWarning: overflow encountered in divide
      df_dx = [delf / delx for delf, delx in zip(df, dx)]
    Process: 2873 - 2026-09-25 15:39:22 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_3.
    Process: 2873 - 2026-09-25 15:39:22 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:22 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:22 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:22 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:22 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:22 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:22 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:22 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:22 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:22 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:22 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:22 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:22 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:22 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:23 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:23 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:23 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:23 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:23 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:23 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:23 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:23 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:23 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:23 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:23 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:23 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:23 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:23 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:24 - Fitting models on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Fitting model for response_var_0.
    Process: 2873 - 2026-09-25 15:39:24 - Fitting model for response_var_1.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204861165843142e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:24 - Posterior estimation failed: 
    Matrix is not positive definite. 
    The optimizer could not find a stable solution. Retrying optimization.
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 5.161817486031716e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204859649933689e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/regression_model/blr.py:640: LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 8.204862023055994e-55.
      invAXt: np.ndarray = linalg.solve(self.A, X.T, check_finite=False)
    Process: 2873 - 2026-09-25 15:39:24 - Saving model to:
    	resources/cli_example/blr_cli/save_dir/folds/fold_4.
    Process: 2873 - 2026-09-25 15:39:24 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:24 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:24 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:24 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:24 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:24 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:24 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:24 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:24 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:25 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:25 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:25 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:25 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:25 - Harmonizing data for response_var_0.
    Process: 2873 - 2026-09-25 15:39:25 - Making predictions on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing z-scores for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing z-scores for response_var_1.
    Process: 2873 - 2026-09-25 15:39:25 - Computing z-scores for response_var_0.
    Process: 2873 - 2026-09-25 15:39:25 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:25 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:25 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing log-probabilities for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing log-probabilities for response_var_1.
    Process: 2873 - 2026-09-25 15:39:25 - Computing log-probabilities for response_var_0.
    Process: 2873 - 2026-09-25 15:39:25 - Computing yhat for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:25 - Computing yhat for response_var_1.
    Process: 2873 - 2026-09-25 15:39:25 - Computing yhat for response_var_0.
    Process: 2873 - 2026-09-25 15:39:26 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2873 - 2026-09-25 15:39:26 - Computing centiles for 2 response variables.
    Process: 2873 - 2026-09-25 15:39:26 - Computing centiles for response_var_1.
    Process: 2873 - 2026-09-25 15:39:26 - Computing centiles for response_var_0.
    Process: 2873 - 2026-09-25 15:39:26 - Harmonizing data on 2 response variables.
    Process: 2873 - 2026-09-25 15:39:26 - Harmonizing data for response_var_1.
    Process: 2873 - 2026-09-25 15:39:26 - Harmonizing data for response_var_0.
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2873 - 2026-09-25 15:39:26 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
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

    Process: 2883 - 2026-09-25 15:39:28 - No log directory specified. Using default log directory: /home/runner/.pcntoolkit/logs
    Process: 2883 - 2026-09-25 15:39:28 - No temporary directory specified. Using default temporary directory: /home/runner/.pcntoolkit/temp
    Process: 2883 - 2026-09-25 15:39:28 - Dataset "fit_data" created.
        - 862 observations
        - 862 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 2883 - 2026-09-25 15:39:28 - Dataset "predict_data" created.
        - 216 observations
        - 216 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (2)
    	batch_effect_1 (23)
        
    Process: 2883 - 2026-09-25 15:39:28 - Task ID created: fit_predict_fit_data__2026-09-25_15:39:28_956.078369
    Process: 2883 - 2026-09-25 15:39:28 - Temporary directory created:
    	/home/runner/.pcntoolkit/temp/fit_predict_fit_data__2026-09-25_15:39:28_956.078369
    Process: 2883 - 2026-09-25 15:39:28 - Log directory created:
    	/home/runner/.pcntoolkit/logs/fit_predict_fit_data__2026-09-25_15:39:28_956.078369
    Process: 2883 - 2026-09-25 15:39:28 - Fitting models on 2 response variables.
    Process: 2883 - 2026-09-25 15:39:28 - Fitting model for response_var_0.
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
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   119     0       0.172   7       0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━[0m[90m╺[0m[90m━[0m [90m━━━━━━━━[0m   267     0       0.292   15      0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━━[0m[36m╸[0m [90m━━━━━━━━[0m   415     0       0.410   7       1397.18 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   242     7       0.312   15      1483.83 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━[0m[31m╸[0m[90m━━━━━[0m   406     12      0.355   15      1512.39 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   546     17      0.337   7       1498.48 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   690     23      0.307   15      1493.03 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   848     27      0.341   31      1504.40 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   16      0       0.115   47      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   194     0       0.384   15      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   355     0       0.559   7       0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m━━━━━━━━[0m   6       1       0.327   15      1581.03 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m╸[0m[90m━━━━━━━[0m   144     4       0.308   15      1536.83 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[31m╸[0m[90m━━━━━━[0m   301     6       0.316   15      1546.21 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   444     8       0.293   15      1529.88 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[31m╸[0m[90m━━━[0m   607     15      0.302   15      1543.83 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[90m╺[0m[90m━[0m   776     23      0.344   15      1563.63 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━[0m[90m╺[0m   942     29      0.316   15      1575.86 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   35      0       0.060   11      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   190     0       0.278   31      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   358     0       0.326   7       1355.87 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   18      0       0.296   15      1430.76 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [36m━━━━[0m [36m╸[0m[90m━━━━━━━[0m   186     0       0.317   15      1484.72 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [36m━━━━[0m [36m━━[0m[90m╺[0m[90m━━━━━[0m   353     0       0.324   15      1520.39 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[31m╸[0m[90m━━━━[0m   515     1       0.309   7       1535.45 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   686     1       0.329   15      1560.45 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   848     1       0.335   15      1569.20 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   16      0       0.208   7       0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   157     0       0.276   15      0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [36m━━[0m[36m╸[0m[90m━[0m [90m━━━━━━━━[0m   312     0       0.196   15      0.00 draws/s   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   465     0       0.323   7       1471.39 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   126     4       0.279   31      1508.28 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[31m╸[0m[90m━━━━━━[0m   281     7       0.322   15      1516.40 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   427     12      0.317   15      1509.68 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   561     18      0.274   15      1488.00 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   870     30      0.302   63      1502.12 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.300   15      1502.12 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.300   15      1502.12 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    34      0.323   31      1504.40 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.302   15      1575.86 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    2       0.319   15      1569.20 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    33      0.300   15      1502.12 dra…   0:00…   0:00…  
                                                                                    
    [?25hProcess: 2883 - 2026-09-25 15:39:42 - Fitting model for response_var_1.
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
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   123     0       0.153   15      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━[0m[90m╺[0m[90m━[0m [90m━━━━━━━━[0m   274     0       0.560   7       0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [36m━━━[0m[36m╸[0m [90m━━━━━━━━[0m   431     0       0.343   15      1455.96 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [90m╺[0m[90m━━━━━━━[0m   118     4       0.389   15      1552.65 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━[0m[31m╸[0m[90m━━━━━━[0m   281     7       0.380   15      1555.71 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━[0m[90m╺[0m[90m━━━━[0m   480     16      0.386   15      1617.09 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   667     22      0.386   7       1645.92 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   852     28      0.357   15      1664.98 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   13      0       0.056   79      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [36m━[0m[90m╺[0m[90m━━[0m [90m━━━━━━━━[0m   143     0       0.350   31      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   482     0       0.254   15      1464.92 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m╸[0m[90m━━━━━━━[0m   154     2       0.386   15      1527.93 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[90m╺[0m[90m━━━━━[0m   325     4       0.363   15      1562.42 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━[0m[31m╸[0m[90m━━━━[0m   511     18      0.346   15      1612.36 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[90m╺[0m[90m━━[0m   684     20      0.334   15      1630.80 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━[0m[31m╸[0m[90m━[0m   854     23      0.357   15      1641.16 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [90m━━━━[0m [90m━━━━━━━━[0m   16      0       0.118   63      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   193     0       0.408   15      0.00 draws/s   0:00…   -:--…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   374     0       0.524   7       0.00 draws/s   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [90m━━━━━━━━[0m   52      1       0.400   7       1752.22 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   231     9       0.387   7       1765.63 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[31m╸[0m[90m━━━━━[0m   408     13      0.360   7       1766.46 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   593     35      0.358   7       1785.89 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[31m╸[0m[90m━━[0m   760     37      0.375   15      1769.57 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━[0m[90m╺[0m   938     42      0.330   15      1773.06 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m[90m━[0m   0       0       0.000   0       0.00 draws/s   -:--…          
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [36m╸[0m[90m━━━[0m [90m━━━━━━━━[0m   58      0       0.139   3       0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [36m━[0m[36m╸[0m[90m━━[0m [90m━━━━━━━━[0m   213     0       0.288   15      0.00 draws/s   0:00…   -:--…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [36m━━━[0m[90m╺[0m [90m━━━━━━━━[0m   375     0       0.245   15      1414.93 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [36m━━━━[0m [90m━━━━━━━━[0m   34      0       0.334   15      1466.96 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━[0m[90m╺[0m[90m━━━━━━[0m   202     2       0.320   15      1512.85 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━[0m[31m╸[0m[90m━━━━━[0m   373     4       0.318   15      1550.56 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━[0m[90m╺[0m[90m━━━[0m   546     8       0.305   15      1580.00 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━[0m[31m╸[0m[90m━━[0m   712     10      0.300   15      1592.58 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    15      0.330   31      1599.95 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    15      0.330   31      1599.95 dra…   0:00…   0:00…  
    [2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K[1A[2K                                                                                
     [1m               [0m [1m       [0m [1m       [0m [1m [0m[1mStep [0m[1m [0m [1m [0m[1mGrad [0m[1m [0m [1m              [0m [1m       [0m [1m       [0m 
     [1m [0m[1mProgress     [0m[1m [0m [1m [0m[1mDraw [0m[1m [0m [1m [0m[1mDive…[0m[1m [0m [1m [0m[1msize [0m[1m [0m [1m [0m[1mevals[0m[1m [0m [1m [0m[1mSpeed       [0m[1m [0m [1m [0m[1mElap…[0m[1m [0m [1m [0m[1mRema…[0m[1m [0m 
     ────────────────────────────────────────────────────────────────────────────── 
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    35      0.356   15      1664.98 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    26      0.387   15      1641.16 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    43      0.370   15      1773.06 dra…   0:00…   0:00…  
      [31m━━━━[0m [31m━━━━━━━━[0m   1000    15      0.330   31      1599.95 dra…   0:00…   0:00…  
                                                                                    
    [?25hProcess: 2883 - 2026-09-25 15:39:49 - Saving model to:
    	resources/cli_example/hbr/save_dir.
    Process: 2883 - 2026-09-25 15:39:49 - Making predictions on 2 response variables.
    Process: 2883 - 2026-09-25 15:39:49 - Computing z-scores for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:49 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:50 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:50 - Computing centiles for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:50 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:52 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:53 - Computing log-probabilities for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:53 - Computing log-probabilities for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:53 - Computing log-probabilities for response_var_1.
    Process: 2883 - 2026-09-25 15:39:54 - Computing log-probabilities for response_var_0.
    Process: 2883 - 2026-09-25 15:39:54 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:55 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2883 - 2026-09-25 15:39:55 - Computing centiles for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:55 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:56 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:57 - Harmonizing data on 2 response variables.
    Process: 2883 - 2026-09-25 15:39:57 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:57 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:58 - Making predictions on 2 response variables.
    Process: 2883 - 2026-09-25 15:39:58 - Computing z-scores for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:58 - Computing z-scores for response_var_1.
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:59 - Computing z-scores for response_var_0.
    Sampling: []
    Process: 2883 - 2026-09-25 15:39:59 - Computing centiles for 2 response variables.
    Process: 2883 - 2026-09-25 15:39:59 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:40:00 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:40:01 - Computing log-probabilities for 2 response variables.
    Process: 2883 - 2026-09-25 15:40:01 - Computing log-probabilities for 2 response variables.
    Process: 2883 - 2026-09-25 15:40:01 - Computing log-probabilities for response_var_1.
    Process: 2883 - 2026-09-25 15:40:01 - Computing log-probabilities for response_var_0.
    Process: 2883 - 2026-09-25 15:40:01 - Computing yhat for 2 response variables.
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:40:02 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	batch_effect_0 (1)
    	batch_effect_1 (1)
        
    Process: 2883 - 2026-09-25 15:40:02 - Computing centiles for 2 response variables.
    Process: 2883 - 2026-09-25 15:40:02 - Computing centiles for response_var_1.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:40:03 - Computing centiles for response_var_0.
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:40:04 - Harmonizing data on 2 response variables.
    Process: 2883 - 2026-09-25 15:40:04 - Harmonizing data for response_var_1.
    Sampling: []
    Sampling: []
    Process: 2883 - 2026-09-25 15:40:04 - Harmonizing data for response_var_0.
    Sampling: []
    Sampling: []
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2883 - 2026-09-25 15:40:05 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2883 - 2026-09-25 15:40:05 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 2883 - 2026-09-25 15:40:05 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)


