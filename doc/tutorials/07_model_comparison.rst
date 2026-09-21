Compare normative models
========================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/07_model_comparison.ipynb>`

.. code:: ipython3

    import logging
    import warnings
    
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import seaborn as sns
    
    import pcntoolkit.util.output
    from pcntoolkit import (
        HBR,
        BsplineBasisFunction,
        NormalLikelihood,
        NormativeModel,
        NormData,
        load_fcon1000,
        make_prior,
    )
    from pcntoolkit.util.model_comparison import compare_hbr_models
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(True)

.. code:: ipython3

    # Download an example dataset
    norm_data: NormData = load_fcon1000()
    
    # Select only a few features
    features_to_model = [
        "WM-hypointensities",
        "Right-Lateral-Ventricle",
        # "Right-Amygdala",
        # "CortexVol",
    ]
    norm_data = norm_data.sel({"response_vars": features_to_model})
    
    # Split into train and test sets
    train, test = norm_data.train_test_split()


.. code:: text

    Process: 3005 - 2026-09-21 12:14:52 - Removed 0 NANs
    Process: 3005 - 2026-09-21 12:14:52 - Dataset "fcon1000" created.
        - 1078 observations
        - 1078 unique subjects
        - 1 covariates
        - 217 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. code:: ipython3

    mu1 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is not random, because we want to compare to a model with random intercept
        intercept=make_prior(
            dist_name="Normal",
            dist_params=(0.0, 2.0),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma1 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )
    # Set the likelihood with the priors we just created.
    likelihood1 = NormalLikelihood(mu1, sigma1)
    
    template_hbr_1 = HBR(
        name="template",
        # The number of cores to use for sampling.
        cores=16,
        # Whether to show a progress bar during the model fitting.
        progressbar=False,
        # The number of draws to sample from the posterior per chain.
        draws=1500,
        # The number of tuning steps to run.
        tune=500,
        # The number of MCMC chains to run.
        chains=4,
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood1,
    )
    model1 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_1,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model1",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    mu2 = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we have to map it to the positive domain.
            sigma=make_prior(dist_name="Normal", dist_params=(1.0, 0.5), mapping="softplus", mapping_params=(0.0, 2.0)),
        ),
        # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
    )
    sigma2 = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
        # The intercept is not random, because we assume the intercept of the variance to be the same for all sites and sexes.
        intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
        # We use a B-spline basis function to allow for non-linearity in the standard deviation.
        basis_function=BsplineBasisFunction(basis_column=0, nknots=5, degree=3),
        # We use a softplus mapping to ensure that sigma is strictly positive.
        mapping="softplus",
        # We scale the softplus mapping by a factor of 3, to avoid spikes in the resulting density.
        # The parameters (a, b, c) provided to a mapping f are used as: f_abc(x) = f((x - a) / b) * b + c
        # This basically provides an affine transformation of the softplus function.
        # a -> horizontal shift
        # b -> scaling
        # c -> vertical shift
        # You can leave c out, and it will default to 0.
        mapping_params=(0.0, 3.0),
    )
    # Set the likelihood with the priors we just created.
    likelihood2 = NormalLikelihood(mu2, sigma2)
    
    template_hbr_2 = HBR(
        name="template",
        # The number of cores to use for sampling.
        cores=16,
        # Whether to show a progress bar during the model fitting.
        progressbar=False,
        # The number of draws to sample from the posterior per chain.
        draws=1500,
        # The number of tuning steps to run.
        tune=500,
        # The number of MCMC chains to run.
        chains=4,
        # The sampler to use for the model.
        nuts_sampler="nutpie",
        # The likelihood function to use for the model.
        likelihood=likelihood2,
    )
    model2 = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr_2,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir="resources/compare_hbr/model2",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    model1.fit_predict(train, test)
    model2.fit_predict(train, test)


.. code:: text

    Process: 3005 - 2026-09-21 12:14:52 - Fitting models on 2 response variables.
    Process: 3005 - 2026-09-21 12:14:52 - Fitting model for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:17 - Fitting model for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:32 - Saving model to:
    	resources/compare_hbr/model1.
    Process: 3005 - 2026-09-21 12:15:32 - Making predictions on 2 response variables.
    Process: 3005 - 2026-09-21 12:15:32 - Computing z-scores for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:32 - Computing z-scores for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:33 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:34 - Computing centiles for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:34 - Computing centiles for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:35 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:37 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:37 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:37 - Computing log-probabilities for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:38 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:39 - Computing yhat for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:39 - Making predictions on 2 response variables.
    Process: 3005 - 2026-09-21 12:15:39 - Computing z-scores for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:39 - Computing z-scores for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:40 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:40 - Computing centiles for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:40 - Computing centiles for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:41 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:42 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:42 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:42 - Computing log-probabilities for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:15:42 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:15:43 - Computing yhat for 2 response variables.
    Process: 3005 - 2026-09-21 12:15:43 - Fitting models on 2 response variables.
    Process: 3005 - 2026-09-21 12:15:43 - Fitting model for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:09 - Fitting model for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:23 - Saving model to:
    	resources/compare_hbr/model2.
    Process: 3005 - 2026-09-21 12:16:24 - Making predictions on 2 response variables.
    Process: 3005 - 2026-09-21 12:16:24 - Computing z-scores for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:24 - Computing z-scores for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:25 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:25 - Computing centiles for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:25 - Computing centiles for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:28 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:30 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:30 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:30 - Computing log-probabilities for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:31 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:32 - Computing yhat for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:33 - Making predictions on 2 response variables.
    Process: 3005 - 2026-09-21 12:16:33 - Computing z-scores for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:33 - Computing z-scores for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:33 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:34 - Computing centiles for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:34 - Computing centiles for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:35 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:37 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:37 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:37 - Computing log-probabilities for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:38 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:38 - Computing yhat for 2 response variables.




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in notebooks */
    
    :root {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base rgba(0, 0, 0, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(0, 0, 0, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, white)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, white) h s calc(l - 15))
      );
    }
    
    html[theme="dark"],
    html[data-theme="dark"],
    body[data-theme="dark"],
    body.vscode-dark {
      --xr-font-color0: var(
        --jp-content-font-color0,
        var(--pst-color-text-base, rgba(255, 255, 255, 1))
      );
      --xr-font-color2: var(
        --jp-content-font-color2,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.54))
      );
      --xr-font-color3: var(
        --jp-content-font-color3,
        var(--pst-color-text-base, rgba(255, 255, 255, 0.38))
      );
      --xr-border-color: var(
        --jp-border-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 10))
      );
      --xr-disabled-color: var(
        --jp-layout-color3,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 40))
      );
      --xr-background-color: var(
        --jp-layout-color0,
        var(--pst-color-on-background, #111111)
      );
      --xr-background-color-row-even: var(
        --jp-layout-color1,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 5))
      );
      --xr-background-color-row-odd: var(
        --jp-layout-color2,
        hsl(from var(--pst-color-on-background, #111111) h s calc(l + 15))
      );
    }
    
    .xr-wrap {
      display: block !important;
      min-width: 300px;
      max-width: 700px;
      line-height: 1.6;
      padding-bottom: 4px;
    }
    
    .xr-text-repr-fallback {
      /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
      display: none;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
    }
    
    .xr-header {
      border-bottom: solid 1px var(--xr-border-color);
      margin-bottom: 4px;
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-obj-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type,
    .xr-group-box-contents > label {
      color: var(--xr-font-color2);
      display: block;
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;
      margin-block-start: 0;
      margin-block-end: 0;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item > input,
    .xr-group-box-contents > input,
    .xr-array-wrap > input {
      display: block;
      opacity: 0;
      height: 0;
      margin: 0;
    }
    
    .xr-section-item > input + label,
    .xr-var-item > input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item > input:enabled + label,
    .xr-var-item > input:enabled + label,
    .xr-array-wrap > input:enabled + label,
    .xr-group-box-contents > input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item > input:focus-visible + label,
    .xr-var-item > input:focus-visible + label,
    .xr-array-wrap > input:focus-visible + label,
    .xr-group-box-contents > input:focus-visible + label {
      outline: auto;
    }
    
    .xr-section-item > input:enabled + label:hover,
    .xr-var-item > input:enabled + label:hover,
    .xr-array-wrap > input:enabled + label:hover,
    .xr-group-box-contents > input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
      white-space: nowrap;
    }
    
    .xr-section-summary > em {
      font-weight: normal;
    }
    
    .xr-span-grid {
      grid-column-end: -1;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.3em;
    }
    
    .xr-group-box-contents > input:checked + label > span {
      display: inline-block;
      padding-left: 0.6em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: "►";
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: "▼";
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details,
    .xr-group-box-contents > label {
      padding-top: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      grid-column: 1 / -1;
      margin-top: 4px;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in ~ .xr-section-details {
      display: none;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-children {
      display: inline-grid;
      grid-template-columns: 100%;
      grid-column: 1 / -1;
      padding-top: 4px;
    }
    
    .xr-group-box {
      display: inline-grid;
      grid-template-columns: 0px 30px auto;
    }
    
    .xr-group-box-vline {
      grid-column-start: 1;
      border-right: 0.2em solid;
      border-color: var(--xr-border-color);
      width: 0px;
    }
    
    .xr-group-box-hline {
      grid-column-start: 2;
      grid-row-start: 1;
      height: 1em;
      width: 26px;
      border-bottom: 0.2em solid;
      border-color: var(--xr-border-color);
    }
    
    .xr-group-box-contents {
      grid-column-start: 3;
      padding-bottom: 4px;
    }
    
    .xr-group-box-contents > label::before {
      content: "📂";
      padding-right: 0.3em;
    }
    
    .xr-group-box-contents > input:checked + label::before {
      content: "📁";
    }
    
    .xr-group-box-contents > input:checked + label {
      padding-bottom: 0px;
    }
    
    .xr-group-box-contents > input:checked ~ .xr-sections {
      display: none;
    }
    
    .xr-group-box-contents > input + label > span {
      display: none;
    }
    
    .xr-group-box-ellipsis {
      font-size: 1.4em;
      font-weight: 900;
      color: var(--xr-font-color2);
      letter-spacing: 0.15em;
      cursor: default;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: "(";
    }
    
    .xr-dim-list:after {
      content: ")";
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ",";
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      border-color: var(--xr-background-color-row-odd);
      margin-bottom: 0;
      padding-top: 2px;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
      border-color: var(--xr-background-color-row-even);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-index-preview {
      grid-column: 2 / 5;
      color: var(--xr-font-color2);
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      display: none;
      border-top: 2px dotted var(--xr-background-color);
      padding-bottom: 20px !important;
      padding-top: 10px !important;
    }
    
    .xr-var-attrs-in + label,
    .xr-var-data-in + label,
    .xr-index-data-in + label {
      padding: 0 1px;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data,
    .xr-index-data-in:checked ~ .xr-index-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-data > pre,
    .xr-index-data > pre,
    .xr-var-data > table > tbody > tr {
      background-color: transparent !important;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-index-name div,
    .xr-index-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data,
    .xr-index-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt,
    .xr-attrs dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2,
    .xr-no-icon {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    
    .xr-var-attrs-in:checked + label > .xr-icon-file-text2,
    .xr-var-data-in:checked + label > .xr-icon-database,
    .xr-index-data-in:checked + label > .xr-icon-database {
      color: var(--xr-font-color0);
      filter: drop-shadow(1px 1px 5px var(--xr-font-color2));
      stroke-width: 0.8px;
    }
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 70kB
    Dimensions:            (observations: 216, response_vars: 2, covariates: 1,
                            batch_effect_dims: 2, statistic: 13, centile: 5)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U23 184B &#x27;WM-hypointensities&#x27; &#x27;Right-...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * statistic          (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
    Data variables:
        subject_ids        (observations) object 2kB &#x27;Munchen_sub96752&#x27; ... &#x27;Quee...
        Y                  (observations, response_vars) float64 3kB 2.721e+03 .....
        X                  (observations, covariates) float64 2kB 63.0 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 3kB 0.529 ... 1.175
        baseline_logp      (observations, response_vars) float64 3kB -3.66 ... -1...
        logp               (observations, response_vars) float64 3kB -1.695 ... -...
        Yhat               (observations, response_vars) float64 3kB 1.952e+03 .....
        statistics         (response_vars, statistic) float64 208B 0.3666 ... 1.444
        centiles           (centile, observations, response_vars) float64 17kB -4...
    Attributes:
        real_ids:                       True
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-4243f5cd-de4c-4150-9643-a0841e741ac0' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-4243f5cd-de4c-4150-9643-a0841e741ac0' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 2</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>statistic</span>: 13</li><li><span class='xr-has-index'>centile</span>: 5</li></ul></div></li><li class='xr-section-item'><input id='section-8b59c200-90b1-43ae-8c36-da5ab383825d' class='xr-section-summary-in' type='checkbox' checked /><label for='section-8b59c200-90b1-43ae-8c36-da5ab383825d' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-56938e0a-641b-4662-b165-82cb67e01475' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-56938e0a-641b-4662-b165-82cb67e01475' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b5d18cef-66d9-4db1-9f92-59e75df3b638' class='xr-var-data-in' type='checkbox'><label for='data-b5d18cef-66d9-4db1-9f92-59e75df3b638' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U23</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; &#x27;Right-Late...</div><input id='attrs-6ce7a0f5-f381-4f78-80ab-f6b4975d10e1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6ce7a0f5-f381-4f78-80ab-f6b4975d10e1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9e3e777a-9be4-430d-ac1f-56014896b247' class='xr-var-data-in' type='checkbox'><label for='data-9e3e777a-9be4-430d-ac1f-56014896b247' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Right-Lateral-Ventricle&#x27;], dtype=&#x27;&lt;U23&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-5c1dabe9-cae2-4a6e-977a-9322cbf6a457' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5c1dabe9-cae2-4a6e-977a-9322cbf6a457' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3bbe4ed8-30e7-4b7b-bb8f-ec54b0351918' class='xr-var-data-in' type='checkbox'><label for='data-3bbe4ed8-30e7-4b7b-bb8f-ec54b0351918' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-be67b1f9-ef98-4acd-ac33-66a1d8f2be4a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-be67b1f9-ef98-4acd-ac33-66a1d8f2be4a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-74db6e0e-e160-426e-b982-afbb1b04e8a2' class='xr-var-data-in' type='checkbox'><label for='data-74db6e0e-e160-426e-b982-afbb1b04e8a2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-5bcc1d98-831b-415a-8a62-67ff7619d66f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5bcc1d98-831b-415a-8a62-67ff7619d66f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3a7d1658-b527-40b5-8772-c03df960ea30' class='xr-var-data-in' type='checkbox'><label for='data-3a7d1658-b527-40b5-8772-c03df960ea30' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-29d4c7a6-6cdb-4ce4-9371-9eec178eb45d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-29d4c7a6-6cdb-4ce4-9371-9eec178eb45d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7272a2ff-4bcf-4648-acdc-a4febcbbcf61' class='xr-var-data-in' type='checkbox'><label for='data-7272a2ff-4bcf-4648-acdc-a4febcbbcf61' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-2377ed0c-37d8-403d-a60b-ef1072ddb622' class='xr-section-summary-in' type='checkbox' checked /><label for='section-2377ed0c-37d8-403d-a60b-ef1072ddb622' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>object</div><div class='xr-var-preview xr-preview'>&#x27;Munchen_sub96752&#x27; ... &#x27;Queensla...</div><input id='attrs-c536ffe1-aa99-4703-8ec3-0365aaa691ef' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c536ffe1-aa99-4703-8ec3-0365aaa691ef' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8a67911c-bac9-4b56-991f-b8c3d10d618e' class='xr-var-data-in' type='checkbox'><label for='data-8a67911c-bac9-4b56-991f-b8c3d10d618e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;Munchen_sub96752&#x27;, &#x27;NewYork_a_sub18638&#x27;, &#x27;Leiden_2200_sub87320&#x27;,
           &#x27;ICBM_sub47658&#x27;, &#x27;AnnArbor_b_sub45569&#x27;, &#x27;Beijing_Zang_sub18960&#x27;,
           &#x27;Leiden_2200_sub18456&#x27;, &#x27;Berlin_Margulies_sub27711&#x27;,
           &#x27;Beijing_Zang_sub87776&#x27;, &#x27;Milwaukee_b_sub63196&#x27;,
           &#x27;Beijing_Zang_sub07144&#x27;, &#x27;Atlanta_sub76280&#x27;,
           &#x27;Beijing_Zang_sub40037&#x27;, &#x27;Cambridge_Buckner_sub17737&#x27;,
           &#x27;ICBM_sub89049&#x27;, &#x27;ICBM_sub55656&#x27;, &#x27;Oulu_sub45566&#x27;,
           &#x27;Beijing_Zang_sub89088&#x27;, &#x27;Atlanta_sub16563&#x27;,
           &#x27;Cambridge_Buckner_sub51172&#x27;, &#x27;Oulu_sub98739&#x27;,
           &#x27;Queensland_sub49845&#x27;, &#x27;Cambridge_Buckner_sub84256&#x27;,
           &#x27;Cleveland_sub80263&#x27;, &#x27;ICBM_sub16607&#x27;, &#x27;Newark_sub46570&#x27;,
           &#x27;NewYork_a_sub88286&#x27;, &#x27;Cambridge_Buckner_sub02591&#x27;,
           &#x27;Oulu_sub66467&#x27;, &#x27;Beijing_Zang_sub74386&#x27;, &#x27;Newark_sub55760&#x27;,
           &#x27;ICBM_sub30623&#x27;, &#x27;Oulu_sub68752&#x27;, &#x27;Leiden_2180_sub19281&#x27;,
           &#x27;Beijing_Zang_sub50972&#x27;, &#x27;Beijing_Zang_sub85030&#x27;,
           &#x27;Milwaukee_b_sub36386&#x27;, &#x27;Baltimore_sub31837&#x27;, &#x27;PaloAlto_sub84978&#x27;,
           &#x27;Oulu_sub01077&#x27;, &#x27;NewYork_a_ADHD_sub54828&#x27;, &#x27;PaloAlto_sub96705&#x27;,
           &#x27;Cambridge_Buckner_sub40635&#x27;, &#x27;ICBM_sub66794&#x27;,
           &#x27;Beijing_Zang_sub46541&#x27;, &#x27;Beijing_Zang_sub87089&#x27;,
           &#x27;Pittsburgh_sub97823&#x27;, &#x27;Beijing_Zang_sub98617&#x27;, &#x27;ICBM_sub92028&#x27;,
    ...
           &#x27;Leiden_2200_sub04484&#x27;, &#x27;Beijing_Zang_sub80163&#x27;, &#x27;ICBM_sub02382&#x27;,
           &#x27;Cambridge_Buckner_sub77435&#x27;, &#x27;NewYork_a_sub54887&#x27;,
           &#x27;Oulu_sub85532&#x27;, &#x27;Baltimore_sub73823&#x27;, &#x27;Beijing_Zang_sub29590&#x27;,
           &#x27;Oulu_sub99718&#x27;, &#x27;Beijing_Zang_sub08455&#x27;, &#x27;Beijing_Zang_sub85543&#x27;,
           &#x27;Cambridge_Buckner_sub45354&#x27;, &#x27;Beijing_Zang_sub07717&#x27;,
           &#x27;Baltimore_sub76160&#x27;, &#x27;Beijing_Zang_sub17093&#x27;,
           &#x27;AnnArbor_b_sub90127&#x27;, &#x27;SaintLouis_sub73002&#x27;,
           &#x27;Queensland_sub93238&#x27;, &#x27;Cleveland_sub34189&#x27;,
           &#x27;Cambridge_Buckner_sub89107&#x27;, &#x27;Atlanta_sub75153&#x27;,
           &#x27;NewYork_a_ADHD_sub73035&#x27;, &#x27;Cambridge_Buckner_sub59434&#x27;,
           &#x27;Milwaukee_b_sub44912&#x27;, &#x27;Cleveland_sub46739&#x27;, &#x27;Oulu_sub20495&#x27;,
           &#x27;SaintLouis_sub28304&#x27;, &#x27;Cambridge_Buckner_sub35430&#x27;,
           &#x27;Oulu_sub86362&#x27;, &#x27;Newark_sub58526&#x27;, &#x27;Leiden_2180_sub12255&#x27;,
           &#x27;ICBM_sub48210&#x27;, &#x27;Cambridge_Buckner_sub77989&#x27;,
           &#x27;Berlin_Margulies_sub75506&#x27;, &#x27;NewYork_a_sub29216&#x27;,
           &#x27;Beijing_Zang_sub05267&#x27;, &#x27;AnnArbor_b_sub18546&#x27;, &#x27;Oulu_sub75620&#x27;,
           &#x27;AnnArbor_b_sub30250&#x27;, &#x27;Berlin_Margulies_sub86111&#x27;,
           &#x27;Beijing_Zang_sub89592&#x27;, &#x27;Beijing_Zang_sub68012&#x27;,
           &#x27;NewYork_a_sub50559&#x27;, &#x27;Munchen_sub66933&#x27;,
           &#x27;Cambridge_Buckner_sub59729&#x27;, &#x27;Queensland_sub86245&#x27;], dtype=object)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>2.721e+03 1.289e+04 ... 1.07e+04</div><input id='attrs-4183fe3c-3e78-4117-a603-13e2841e09b8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4183fe3c-3e78-4117-a603-13e2841e09b8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-86e9f15b-142e-440c-a0e2-cbcf5224020e' class='xr-var-data-in' type='checkbox'><label for='data-86e9f15b-142e-440c-a0e2-cbcf5224020e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 2721.4, 12891.6],
           [ 1143.1,  9919.1],
           [  955.8,  7477.3],
           [ 1473.9, 14302.1],
           [  757.8,  4119.3],
           [  871.1,  5030.9],
           [ 1207.3, 17866.4],
           [  595. ,  5007.9],
           [  682.4,  7286.6],
           [  445.1,  5742.9],
           [ 1620. ,  3713.7],
           [  602.8,  5301.2],
           [ 1432.5,  4429.7],
           [ 1908.2,  3578.1],
           [ 1834. ,  3271.9],
           [  459.6,  3985.8],
           [ 1210. ,  8721.3],
           [  845.9,  6593.1],
           [  995.2,  7040.2],
           [ 1734.7,  4014.8],
    ...
           [  785.8,  5709. ],
           [ 2240.1,  4366.6],
           [  758.1,  6529.8],
           [ 1440.5,  6705.3],
           [  818.6,  9383.3],
           [ 3769.9, 15864.4],
           [  880.2,  4370.2],
           [  823.9,  6379. ],
           [ 2113.9, 10722.5],
           [  741.9,  8801.7],
           [ 1333.9,  6980. ],
           [  707.3,  5680.7],
           [ 1134.1,  5592.2],
           [  438.6,  6330. ],
           [  966.3,  9215.5],
           [  424.3,  4511.1],
           [  604.7,  7590.8],
           [ 2343.2, 17192.3],
           [ 2721.7,  6086. ],
           [  703.5, 10700.3]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 23.27 22.0 ... 72.0 23.0 23.0</div><input id='attrs-7da1e0a4-8bcf-47ac-9f20-1d886387741b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7da1e0a4-8bcf-47ac-9f20-1d886387741b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e9919888-ba54-42ca-9c17-ad8927a937ce' class='xr-var-data-in' type='checkbox'><label for='data-e9919888-ba54-42ca-9c17-ad8927a937ce' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[63.  ],
           [23.27],
           [22.  ],
           [42.  ],
           [63.  ],
           [23.  ],
           [21.  ],
           [26.  ],
           [21.  ],
           [49.  ],
           [20.  ],
           [23.  ],
           [20.  ],
           [26.  ],
           [35.  ],
           [21.  ],
           [22.  ],
           [19.  ],
           [34.  ],
           [18.  ],
    ...
           [21.  ],
           [20.  ],
           [22.  ],
           [25.  ],
           [25.  ],
           [73.  ],
           [22.  ],
           [28.  ],
           [29.06],
           [19.  ],
           [20.  ],
           [22.  ],
           [19.  ],
           [24.  ],
           [21.  ],
           [24.  ],
           [22.79],
           [72.  ],
           [23.  ],
           [23.  ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-8169d5da-7ed1-4c37-9238-3d8ae4962a43' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8169d5da-7ed1-4c37-9238-3d8ae4962a43' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-84726c9a-5efc-41ab-942b-1ef1b4379a61' class='xr-var-data-in' type='checkbox'><label for='data-84726c9a-5efc-41ab-942b-1ef1b4379a61' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2200&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Milwaukee_b&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;ICBM&#x27;],
           [&#x27;M&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;Atlanta&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
    ...
           [&#x27;F&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;Newark&#x27;],
           [&#x27;M&#x27;, &#x27;Leiden_2180&#x27;],
           [&#x27;M&#x27;, &#x27;ICBM&#x27;],
           [&#x27;F&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Oulu&#x27;],
           [&#x27;F&#x27;, &#x27;AnnArbor_b&#x27;],
           [&#x27;F&#x27;, &#x27;Berlin_Margulies&#x27;],
           [&#x27;M&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;F&#x27;, &#x27;Beijing_Zang&#x27;],
           [&#x27;M&#x27;, &#x27;NewYork_a&#x27;],
           [&#x27;M&#x27;, &#x27;Munchen&#x27;],
           [&#x27;M&#x27;, &#x27;Cambridge_Buckner&#x27;],
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.529 0.1916 ... -1.062 1.175</div><input id='attrs-08876b17-6788-4e78-a269-682e3a659080' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-08876b17-6788-4e78-a269-682e3a659080' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9750e56e-4578-4c84-8f14-34b4a2380ad2' class='xr-var-data-in' type='checkbox'><label for='data-9750e56e-4578-4c84-8f14-34b4a2380ad2' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 5.28980251e-01,  1.91576473e-01],
           [ 3.04294566e-02,  9.79931377e-01],
           [ 2.94070189e-01,  2.59706519e-01],
           [ 4.42177956e-02,  1.50304404e+00],
           [-8.78506952e-01, -1.12055512e+00],
           [-8.08193110e-01, -6.45847437e-01],
           [ 5.19667306e-01,  3.58717505e+00],
           [ 6.76126476e-02, -5.87448978e-01],
           [-8.55181842e-01,  6.33970768e-01],
           [-9.50880145e-01, -4.54531921e-01],
           [ 1.12710441e+00, -1.07392332e+00],
           [-8.09618425e-01, -3.37119647e-01],
           [ 1.06474134e+00, -3.58832388e-01],
           [ 1.81233983e+00, -6.83764607e-01],
           [ 1.05279830e+00, -1.18842649e+00],
           [-1.43416124e+00, -7.65057560e-01],
           [-6.38828330e-01,  4.32411911e-01],
           [-4.28710147e-01,  4.95043007e-01],
           [-4.45697930e-01, -2.05578712e-01],
           [ 1.26438647e+00, -3.84926869e-01],
    ...
           [ 2.15038676e-01, -1.46992229e-01],
           [ 2.23983298e+00, -7.84191245e-01],
           [-1.39735766e+00,  9.62861975e-02],
           [-6.44284352e-01,  1.47999042e-01],
           [-9.17464266e-01,  4.24978208e-01],
           [ 2.48318963e-01,  1.00636476e-01],
           [-8.26613256e-01, -3.88667817e-01],
           [ 1.79470641e-01, -5.37160445e-01],
           [ 2.31250967e+00,  1.00449828e+00],
           [-6.86818114e-01,  1.32824528e+00],
           [-3.13044269e-01,  2.77921581e-01],
           [-1.53075937e+00, -1.99687420e-01],
           [-4.07333980e-01,  2.70824230e-01],
           [-3.08948305e-01, -1.46131079e-01],
           [-5.41639244e-01,  8.85525340e-01],
           [-1.56988032e+00, -4.20350369e-01],
           [-1.38743092e+00,  2.13568176e-01],
           [-4.32350493e-01,  2.43730787e-01],
           [ 3.59881716e+00, -2.42810437e-01],
           [-1.06210431e+00,  1.17524857e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-3.66 -2.043 ... -0.9959 -1.366</div><input id='attrs-580d4c53-3e2e-4c47-b7c4-982c89c4381c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-580d4c53-3e2e-4c47-b7c4-982c89c4381c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f46830b4-cfb6-40ee-a220-c27bd028f1ec' class='xr-var-data-in' type='checkbox'><label for='data-f46830b4-cfb6-40ee-a220-c27bd028f1ec' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -3.66025491,  -2.04288507],
           [ -0.62929369,  -1.20018059],
           [ -0.72099829,  -0.94007561],
           [ -0.70127013,  -2.64484182],
           [ -0.9220753 ,  -1.21898282],
           [ -0.79391552,  -1.07037055],
           [ -0.61989989,  -4.7455716 ],
           [ -1.16758184,  -1.07345203],
           [ -1.02678525,  -0.93617147],
           [ -1.45761657,  -0.99208431],
           [ -0.82816414,  -1.30256853],
           [ -1.15416898,  -1.03674779],
           [ -0.67590815,  -1.16228047],
           [ -1.24932613,  -1.33291168],
           [ -1.11921947,  -1.4058518 ],
           [ -1.42688131,  -1.24530702],
           [ -0.61975138,  -1.02388273],
           [ -0.81939019,  -0.94201722],
           [ -0.69375343,  -0.93464746],
           [ -0.96861762,  -1.23948959],
    ...
           [ -0.88714286,  -0.9950603 ],
           [ -2.01527752,  -1.17329719],
           [ -0.92168968,  -0.94411669],
           [ -0.68044429,  -0.93893952],
           [ -0.8489441 ,  -1.10972487],
           [ -9.4332193 ,  -3.46339473],
           [ -0.78514238,  -1.17266166],
           [ -0.84304737,  -0.95017362],
           [ -1.68860199,  -1.37098214],
           [ -0.94286471,  -1.03277979],
           [ -0.63434716,  -0.93487843],
           [ -0.99048957,  -0.99760222],
           [ -0.63150983,  -1.00588922],
           [ -1.47158077,  -0.95246174],
           [ -0.7133234 ,  -1.08525544],
           [ -1.502708  ,  -1.14845319],
           [ -1.1509269 ,  -0.94352785],
           [ -2.31442671,  -4.28458605],
           [ -3.66147746,  -0.96619251],
           [ -0.99591923,  -1.36569557]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.695 -1.344 ... -0.7515 -1.386</div><input id='attrs-0a991068-e7e8-4b26-94c2-b47714527a17' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0a991068-e7e8-4b26-94c2-b47714527a17' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e3800cec-d9e2-4ecc-bfd1-d3f26dda509b' class='xr-var-data-in' type='checkbox'><label for='data-e3800cec-d9e2-4ecc-bfd1-d3f26dda509b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ -1.69530549,  -1.3436187 ],
           [ -0.16381847,  -1.1537423 ],
           [ -0.23404669,  -0.69165654],
           [ -0.69847552,  -2.15714615],
           [ -1.90618657,  -1.92893879],
           [ -0.48219679,  -0.8668296 ],
           [ -0.33729862,  -7.07000175],
           [ -0.19502349,  -0.93755835],
           [ -0.53844947,  -0.80195548],
           [ -1.45344568,  -1.20746572],
           [ -0.82797068,  -1.15284449],
           [ -0.50658662,  -0.73919704],
           [ -0.75868325,  -0.63928112],
           [ -1.81470051,  -0.9756343 ],
           [ -0.96415339,  -1.63659449],
           [ -1.21859822,  -0.90992994],
           [ -0.36803544,  -0.72599458],
           [ -0.30946407,  -0.6736645 ],
           [ -0.47955032,  -0.94174233],
           [ -1.05189708,  -0.60495138],
    ...
           [ -0.21518111,  -0.63488297],
           [ -2.70364825,  -0.88348283],
           [ -1.14020277,  -0.63622381],
           [ -0.39728581,  -0.74990109],
           [ -0.62722337,  -0.84315678],
           [ -1.95716301,  -1.52858017],
           [ -0.50220467,  -0.7044141 ],
           [ -0.23675597,  -0.95529983],
           [ -2.90446745,  -1.32315167],
           [ -0.45366568,  -1.43426701],
           [ -0.2696538 ,  -0.64245551],
           [ -1.33567551,  -0.65156364],
           [ -0.32859872,  -0.61793486],
           [ -0.229465  ,  -0.72515937],
           [ -0.31989507,  -0.99417796],
           [ -1.3883194 ,  -0.77472483],
           [ -1.12874769,  -0.68268856],
           [ -1.98872922,  -1.53888322],
           [ -6.63744946,  -0.68771735],
           [ -0.7514804 ,  -1.38563207]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.952e+03 1.184e+04 ... 7.232e+03</div><input id='attrs-395e85e1-de96-4799-bc64-d97120106ae9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-395e85e1-de96-4799-bc64-d97120106ae9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-398c6f70-c3bc-45d6-82b5-91a71b68d1b7' class='xr-var-data-in' type='checkbox'><label for='data-398c6f70-c3bc-45d6-82b5-91a71b68d1b7' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1951.63161723, 11843.39559077],
           [ 1131.49188467,  7004.3629308 ],
           [  843.78961163,  6731.57894552],
           [ 1444.81424052,  7975.46458092],
           [ 2038.24331579, 10248.66349191],
           [ 1176.91592603,  6937.27568145],
           [ 1006.99703652,  7857.01423473],
           [  569.1981165 ,  6889.14386664],
           [ 1012.05775457,  5517.44874397],
           [ 1267.66184013,  7804.05416106],
           [ 1177.34885729,  6631.66082651],
           [  909.16218025,  6295.32966338],
           [ 1014.27392735,  5404.15681574],
           [ 1214.58318408,  5769.61239101],
           [ 1325.3670646 ,  7832.42129997],
           [ 1012.61993649,  6119.95975176],
           [ 1453.29799589,  7480.83439916],
           [ 1018.6092244 ,  5280.23299058],
           [ 1202.47378511,  7817.25501968],
           [ 1208.16981973,  5011.97853966],
    ...
           [  702.86526993,  6119.30803504],
           [ 1360.41958488,  6497.37078794],
           [ 1290.22306595,  6253.33038839],
           [ 1685.00911681,  6243.64248421],
           [ 1166.91509888,  8056.09587102],
           [ 3236.4835068 , 15168.78702857],
           [ 1194.99598721,  5485.22742695],
           [  753.16342223,  8183.75590443],
           [ 1182.69392413,  7263.16495954],
           [ 1018.6092244 ,  5280.23299058],
           [ 1456.81098385,  6225.34626564],
           [ 1290.22306595,  6253.33038839],
           [ 1298.07135096,  4873.91842971],
           [  555.45517751,  6772.86148671],
           [ 1175.13268451,  6744.95275475],
           [ 1017.76951751,  5787.62004966],
           [ 1130.05761396,  6964.15250221],
           [ 3247.9120191 , 15575.3431525 ],
           [ 1359.98665361,  6802.98564287],
           [ 1105.25258388,  7231.81132331]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.3666 1.085 ... 0.8927 1.444</div><input id='attrs-e4d3af33-6e35-404d-b40a-3e11411d7c5d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e4d3af33-6e35-404d-b40a-3e11411d7c5d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7eca90d0-1e7d-4cac-ba32-3e4eb501bb62' class='xr-var-data-in' type='checkbox'><label for='data-7eca90d0-1e7d-4cac-ba32-3e4eb501bb62' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 3.66568244e-01,  1.08450248e+00,  1.60835397e-01,
             3.41922476e-01,  7.96127705e-01, -3.23141335e-01,
             3.63599451e-01,  4.82821999e+02,  4.94144676e-01,
             1.06459699e-14,  6.36400549e-01,  9.69468153e-01,
             7.25234629e-01],
           [ 1.96247586e-01,  2.63102803e+00,  1.77431043e-01,
             4.23487252e-01,  1.35168985e+00, -8.29308850e-02,
             1.95509249e-01,  3.50797523e+03,  2.62175613e-01,
             9.64922326e-05,  8.04490751e-01,  8.92737001e-01,
             1.44414709e+00]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-466.3 2.769e+03 ... 1.209e+04</div><input id='attrs-f07468c3-a006-4e50-87f5-2a4ed6527cff' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f07468c3-a006-4e50-87f5-2a4ed6527cff' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-88b8d044-6078-4a91-a9d5-ef987d306cf6' class='xr-var-data-in' type='checkbox'><label for='data-88b8d044-6078-4a91-a9d5-ef987d306cf6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ -466.34744074,  2768.78027338],
            [  508.98039918,  2107.80186042],
            [  216.88896283,  2008.84003962],
            ...,
            [ -227.41094613,  4508.98953027],
            [  737.02110089,  1944.03001834],
            [  482.28703116,  2372.85569877]],
    
           [[  960.11354895,  8122.25291992],
            [  876.22441925,  4996.47590726],
            [  586.72232304,  4794.96955202],
            ...,
            [ 1822.81881916, 11037.46697438],
            [ 1104.53299309,  4810.51913296],
            [  849.79892335,  5239.3448134 ]],
    
           [[ 1951.63161723, 11843.39559077],
            [ 1131.49188467,  7004.3629308 ],
            [  843.78961163,  6731.57894552],
            ...,
            [ 3247.9120191 , 15575.3431525 ],
            [ 1359.98665361,  6802.98564287],
            [ 1105.25258388,  7231.81132331]],
    
           [[ 2943.1496855 , 15564.53826162],
            [ 1386.75935009,  9012.24995434],
            [ 1100.85690022,  8668.18833902],
            ...,
            [ 4673.00521904, 20113.21933062],
            [ 1615.44031414,  8795.45215279],
            [ 1360.70624441,  9224.27783322]],
    
           [[ 4369.6106752 , 20918.01090817],
            [ 1754.00337016, 11900.92400118],
            [ 1470.69026043, 11454.31785142],
            ...,
            [ 6723.23498433, 26641.69677473],
            [ 1982.95220633, 11661.94126741],
            [ 1728.2181366 , 12090.76694784]]], shape=(5, 216, 2))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-c893d0b6-ff8a-402f-8a6e-43f1e1c88d36' class='xr-section-summary-in' type='checkbox' checked /><label for='section-c893d0b6-ff8a-402f-8a6e-43f1e1c88d36' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>True</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7efccc20c220&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: 489, &#x27;F&#x27;: 589}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;F&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    # Delete references to model objects to ensure what follows will work for models saved to disk too
    del model1
    del model2

.. code:: ipython3

    dct = {"model1": "resources/compare_hbr/model1", "model2": "resources/compare_hbr/model2"}
    comparison = compare_hbr_models(dct)


.. code:: text

    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3005 - 2026-09-21 12:16:39 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3005 - 2026-09-21 12:16:39 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3005 - 2026-09-21 12:16:39 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)


.. code:: text

    Process: 3005 - 2026-09-21 12:16:39 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (19)
        
    Process: 3005 - 2026-09-21 12:16:39 - Synthesizing data for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:39 - Synthesizing data for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:40 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:40 - Making predictions on 2 response variables.
    Process: 3005 - 2026-09-21 12:16:40 - Computing z-scores for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:40 - Computing z-scores for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:40 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:40 - Computing centiles for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:40 - Computing centiles for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:41 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:42 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:42 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:42 - Computing log-probabilities for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:43 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:43 - Computing yhat for 2 response variables.


.. code:: text

    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3005 - 2026-09-21 12:16:43 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3005 - 2026-09-21 12:16:43 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)
    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/util/output.py:309: UserWarning: Process: 3005 - 2026-09-21 12:16:44 - This model was saved with PCNtoolkit v1.3.0, but you are running v1.3.0. Loading this model in v1.3.0...
      warnings.warn(message, category)


.. code:: text

    Process: 3005 - 2026-09-21 12:16:44 - Dataset "synthesized" created.
        - 92 observations
        - 92 unique subjects
        - 1 covariates
        - 2 response variables
        - 2 batch effects:
        	sex (2)
    	site (19)
        
    Process: 3005 - 2026-09-21 12:16:44 - Synthesizing data for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:44 - Synthesizing data for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:44 - Synthesizing data for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:44 - Making predictions on 2 response variables.
    Process: 3005 - 2026-09-21 12:16:44 - Computing z-scores for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:44 - Computing z-scores for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:45 - Computing z-scores for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:45 - Computing centiles for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:45 - Computing centiles for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:47 - Computing centiles for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:49 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:49 - Computing log-probabilities for 2 response variables.
    Process: 3005 - 2026-09-21 12:16:49 - Computing log-probabilities for WM-hypointensities.
    Process: 3005 - 2026-09-21 12:16:49 - Computing log-probabilities for Right-Lateral-Ventricle.
    Process: 3005 - 2026-09-21 12:16:50 - Computing yhat for 2 response variables.



.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. code:: text

    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1170: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1170: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(



.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. code:: text

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



.. code:: text

    /opt/hostedtoolcache/Python/3.13.15/x64/lib/python3.13/site-packages/arviz_stats/loo/loo_helper.py:1170: UserWarning: Estimated shape parameter of Pareto distribution is greater than 0.70 for one or more samples. You should consider using a more robust model, this is because importance sampling is less likely to work well if the marginal posterior and LOO posterior are very different. This is more likely to happen with a non-robust model and highly influential observations.
      warnings.warn(


.. code:: ipython3

    for k, v in comparison.items():
        print(k)
        display(v)


.. code:: text

    WM-hypointensities



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
          <th>rank</th>
          <th>elpd_diff</th>
          <th>dse</th>
          <th>p_worse</th>
          <th>diag_diff</th>
          <th>diag_elpd</th>
          <th>p</th>
          <th>elpd</th>
          <th>se</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model1</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td>2 k̂ &gt; 0.70</td>
          <td>13.0</td>
          <td>-150.0</td>
          <td>15.0</td>
          <td>0.64</td>
        </tr>
        <tr>
          <th>model2</th>
          <td>1</td>
          <td>-10.0</td>
          <td>18.0</td>
          <td>0.76</td>
          <td>N &lt; 100</td>
          <td>3 k̂ &gt; 0.70</td>
          <td>18.6</td>
          <td>-160.0</td>
          <td>12.0</td>
          <td>0.36</td>
        </tr>
      </tbody>
    </table>
    </div>


.. code:: text

    Right-Lateral-Ventricle



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
          <th>rank</th>
          <th>elpd_diff</th>
          <th>dse</th>
          <th>p_worse</th>
          <th>diag_diff</th>
          <th>diag_elpd</th>
          <th>p</th>
          <th>elpd</th>
          <th>se</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>model1</th>
          <td>0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>NaN</td>
          <td></td>
          <td></td>
          <td>3.0</td>
          <td>-130.0</td>
          <td>6.2</td>
          <td>0.75</td>
        </tr>
        <tr>
          <th>model2</th>
          <td>1</td>
          <td>-30.0</td>
          <td>12.0</td>
          <td>0.98</td>
          <td>N &lt; 100</td>
          <td>3 k̂ &gt; 0.70</td>
          <td>15.6</td>
          <td>-160.0</td>
          <td>11.0</td>
          <td>0.25</td>
        </tr>
      </tbody>
    </table>
    </div>

