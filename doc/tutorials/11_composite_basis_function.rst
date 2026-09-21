.. code:: ipython3

    import logging
    import os
    import warnings
    
    import pandas as pd
    import seaborn as sns
    
    import pcntoolkit.util.output
    from pcntoolkit import (
        HBR,
        BsplineBasisFunction,
        CompositeBasisFunction,
        NormalLikelihood,
        NormativeModel,
        NormData,
        make_prior,
        plot_centiles_advanced,
    )
    
    sns.set_style("darkgrid")
    
    # Suppress some annoying warnings and logs
    pymc_logger = logging.getLogger("pymc")
    
    pymc_logger.setLevel(logging.WARNING)
    
    pymc_logger.propagate = False
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    pd.options.mode.chained_assignment = None  # default='warn'
    pcntoolkit.util.output.Output.set_show_messages(False)


.. code:: ipython3

    save_path = os.path.join("pcntoolkit_resources", "data")
    os.makedirs(save_path, exist_ok=True)
    data_path = os.path.join(save_path, "fcon1000.csv")
    if not os.path.exists(data_path):
        data = pd.read_csv(
            "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
        )
        data.to_csv(data_path, index=False)
    else:
        data = pd.read_csv(data_path)
    
    # Define the variables
    sex_map = {0: "F", 1: "M"}
    data["sex"] = data["sex"].map(sex_map)
    subject_ids = "sub_id"
    covariates = ["age", "EstimatedTotalIntraCranialVol"]
    batch_effects = ["sex", "site"]
    response_vars = ["CortexVol"]
    
    data = NormData.from_dataframe("fcon1000", data, covariates, batch_effects, response_vars)
    train, test = data.train_test_split()

.. code:: ipython3

    CompositeBasisFunction(
            (BsplineBasisFunction(basis_column=0, degree=3, nknots=5), BsplineBasisFunction(basis_column=1, degree=3, nknots=5))
        ),





.. code:: text

    (<pcntoolkit.math_functions.basis_function.CompositeBasisFunction at 0x7f54d0befcb0>,)



.. code:: ipython3

    mu = make_prior(
        # Mu is linear because we want to allow the mean to vary as a function of the covariates.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 10.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 5.0)),
        # The intercept is random, because we expect the intercept to vary between sites and sexes.
    
        intercept=make_prior(
            random=True,
            # Mu is the mean of the intercept, which is normally distributed with a mean of 0 and a standard deviation of 1.
            mu=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
            # Sigma is the scale at which the intercepts vary. It is a positive parameter, so we have to map it to the positive domain.
            sigma=make_prior(dist_name="Gamma", dist_params=(1.0, 1.0))
        ),
            # We use a B-spline basis function to allow for non-linearity in the mean.
        basis_function=CompositeBasisFunction(
            (BsplineBasisFunction(basis_column=0, degree=3, nknots=5), BsplineBasisFunction(basis_column=1, degree=3, nknots=5))
        ),
    )
    sigma = make_prior(
        # Sigma is also linear, because we want to allow the standard deviation to vary as a function of the covariates: heteroskedasticity.
        linear=True,
        # The slope coefficients are assumed to be normally distributed, with a mean of 0 and a standard deviation of 2.
        slope=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
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
        mapping_params=(0.0, 2.0),
    )
    
    
    # Set the likelihood with the priors we just created.
    likelihood = NormalLikelihood(mu, sigma)
    
    template_hbr = HBR(
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
        likelihood=likelihood,
    )

.. code:: ipython3

    model = NormativeModel(
        # The regression model to use for the normative model.
        template_regression_model=template_hbr,
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=True,
        # The directory to save the model, results, and plots.
        save_dir="resources/composite_basis/save_dir",
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
    )

.. code:: ipython3

    model.fit_predict(train, test)




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 54kB
    Dimensions:            (observations: 216, response_vars: 1, covariates: 2,
                            batch_effect_dims: 2, centile: 5, statistic: 13)
    Coordinates:
      * observations       (observations) int64 2kB 756 769 692 616 ... 751 470 1043
      * response_vars      (response_vars) &lt;U9 36B &#x27;CortexVol&#x27;
      * covariates         (covariates) &lt;U29 232B &#x27;age&#x27; &#x27;EstimatedTotalIntraCrani...
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
    Data variables:
        subject_ids        (observations) int64 2kB 756 769 692 616 ... 751 470 1043
        Y                  (observations, response_vars) float64 2kB 4.579e+05 .....
        X                  (observations, covariates) float64 3kB 63.0 ... 1.603e+06
        batch_effects      (observations, batch_effect_dims) &lt;U17 29kB &#x27;F&#x27; ... &#x27;Q...
        Z                  (observations, response_vars) float64 2kB -0.1483 ... ...
        centiles           (centile, observations, response_vars) float64 9kB 4.2...
        baseline_logp      (observations, response_vars) float64 2kB -1.036 ... -...
        logp               (observations, response_vars) float64 2kB -0.06719 ......
        Yhat               (observations, response_vars) float64 2kB 4.609e+05 .....
        statistics         (response_vars, statistic) float64 104B 0.6693 ... 0.3136
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000_test
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85....
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-236bab91-55ba-43a7-9939-6e84466a6502' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-236bab91-55ba-43a7-9939-6e84466a6502' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 216</li><li><span class='xr-has-index'>response_vars</span>: 1</li><li><span class='xr-has-index'>covariates</span>: 2</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 13</li></ul></div></li><li class='xr-section-item'><input id='section-6bbce3b7-c697-418f-a455-68c7c22fb211' class='xr-section-summary-in' type='checkbox' checked /><label for='section-6bbce3b7-c697-418f-a455-68c7c22fb211' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-aedde3d2-8d77-44c0-b6dd-69a1d82ff36c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-aedde3d2-8d77-44c0-b6dd-69a1d82ff36c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-dda5ab0d-734f-4cd3-86bd-812b977eb111' class='xr-var-data-in' type='checkbox'><label for='data-dda5ab0d-734f-4cd3-86bd-812b977eb111' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692, ...,  751,  470, 1043], shape=(216,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U9</div><div class='xr-var-preview xr-preview'>&#x27;CortexVol&#x27;</div><input id='attrs-63e530f2-a27c-425f-a3c9-b34243130ec0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-63e530f2-a27c-425f-a3c9-b34243130ec0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a6da0698-9fe7-4e4d-9370-c1e4415a2d74' class='xr-var-data-in' type='checkbox'><label for='data-a6da0698-9fe7-4e4d-9370-c1e4415a2d74' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;CortexVol&#x27;], dtype=&#x27;&lt;U9&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U29</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27; &#x27;EstimatedTotalIntraCrania...</div><input id='attrs-e6589c7f-23ad-4bd8-b38f-39d2471484a4' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e6589c7f-23ad-4bd8-b38f-39d2471484a4' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1c5ede8c-9120-458b-bb45-e57a16a9f6a4' class='xr-var-data-in' type='checkbox'><label for='data-1c5ede8c-9120-458b-bb45-e57a16a9f6a4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;, &#x27;EstimatedTotalIntraCranialVol&#x27;], dtype=&#x27;&lt;U29&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-a17d87fd-c46e-4667-8fe8-26a6d662c91c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a17d87fd-c46e-4667-8fe8-26a6d662c91c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-85c8ed03-f132-432d-90f3-342d6add0a13' class='xr-var-data-in' type='checkbox'><label for='data-85c8ed03-f132-432d-90f3-342d6add0a13' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-b92937fc-2cd8-47ab-a09b-d6a3ae58d292' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-b92937fc-2cd8-47ab-a09b-d6a3ae58d292' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e6bf2bf7-ea72-4901-a49e-2b7871d83a1c' class='xr-var-data-in' type='checkbox'><label for='data-e6bf2bf7-ea72-4901-a49e-2b7871d83a1c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-5aa473c5-3900-494e-8949-fc70335692da' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5aa473c5-3900-494e-8949-fc70335692da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ede8b28c-7041-48ba-93da-37d4bc567770' class='xr-var-data-in' type='checkbox'><label for='data-ede8b28c-7041-48ba-93da-37d4bc567770' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-f630358d-5dd6-4288-b969-426ad2fc04fd' class='xr-section-summary-in' type='checkbox' checked /><label for='section-f630358d-5dd6-4288-b969-426ad2fc04fd' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>756 769 692 616 ... 751 470 1043</div><input id='attrs-ea196e02-0ed9-4b1b-bc2a-c74446212c62' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ea196e02-0ed9-4b1b-bc2a-c74446212c62' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7bb19159-03c6-48d1-b6d8-14d8a06763f3' class='xr-var-data-in' type='checkbox'><label for='data-7bb19159-03c6-48d1-b6d8-14d8a06763f3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 756,  769,  692,  616,   35,  164,  680,  331,  299,  727,  136,
             80,  209,  394,  653,  626,  935,  302,   61,  449,  984, 1036,
            518,  574,  593,  870,  828,  354,  947,  275,  874,  604,  948,
            670,  228,  294,  708,   90, 1020,  884,  856, 1023,  428,  635,
            221,  298, 1027,  324,  654,  844, 1003,  390,  259,  384,  205,
            881,   63,  681,   34,   16,  219,  214,  738,  501,  242, 1051,
            465,  553,  269,  757,  339,  826,  640,  925,  120,  717,  548,
            248,  726,  334,  556,  186,  822,  761,  411,  783,  109,  960,
            982,  424,  405,  800,  361,  938,  714,  993,  413,  279,  392,
            517,  357,  129,  277,  198,  608, 1033,   19,  660, 1060,  600,
            113,  539,  900,  823,  824,  436,   78,  462,  446, 1021,  435,
            973,  241,  955,  192,  519,  940,   23,  332,  378,  549,  515,
            137,  937,  936,  111,   18,  855,  853,  628,  201,  814,  698,
            366, 1063,   93,  134,  225,  423,  476,   71,  807,  142,  801,
              2,  220,  656,   98,  722,  603,  989,  754,  474,  545,  487,
            538,  646, 1000, 1053,   54,  678,  280,  582,  502,  804,  967,
             95,  185,  985,  141,  295,  437,  138,   96,  155,   51, 1064,
           1046,  562,  527,   79,  861,  469,  710,  564,  907, 1054,  421,
            968,  875,  669,  618,  504,  343,  777,  133,   27,  959,   29,
            346,  304,  264,  798,  751,  470, 1043])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.579e+05 5.268e+05 ... 5.035e+05</div><input id='attrs-1c0a1ad2-e0d3-4349-be2f-cd47d7aa6439' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1c0a1ad2-e0d3-4349-be2f-cd47d7aa6439' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c78d7d7e-fe19-4ed7-8a55-c8637323b83c' class='xr-var-data-in' type='checkbox'><label for='data-c78d7d7e-fe19-4ed7-8a55-c8637323b83c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[457858.327875],
           [526780.362454],
           [495744.470654],
           [585303.839185],
           [333111.551539],
           [510794.940093],
           [550533.324588],
           [467673.976819],
           [460129.533137],
           [444494.81748 ],
           [559424.623684],
           [421551.233862],
           [519842.763049],
           [506679.262498],
           [535569.986908],
           [467607.554967],
           [530904.612455],
           [509371.867477],
           [460068.379043],
           [487269.373272],
    ...
           [453982.166201],
           [558453.1234  ],
           [473575.183228],
           [382788.490644],
           [502713.911273],
           [512490.347519],
           [437300.068601],
           [567331.907771],
           [512273.764245],
           [491973.561824],
           [478907.15396 ],
           [474077.083308],
           [454163.909225],
           [468067.037499],
           [509199.707778],
           [526635.257997],
           [520499.662889],
           [486680.791077],
           [610402.005701],
           [503535.771203]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>63.0 1.533e+06 ... 23.0 1.603e+06</div><input id='attrs-51b68b3b-0317-4224-aa21-da606b88d6a1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-51b68b3b-0317-4224-aa21-da606b88d6a1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c8746328-05a7-49d5-90ee-4f348606693a' class='xr-var-data-in' type='checkbox'><label for='data-c8746328-05a7-49d5-90ee-4f348606693a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[6.30000000e+01, 1.53274143e+06],
           [2.32700000e+01, 1.40047223e+06],
           [2.20000000e+01, 1.48954279e+06],
           [4.20000000e+01, 1.86413298e+06],
           [6.30000000e+01, 1.09596196e+06],
           [2.30000000e+01, 1.68477930e+06],
           [2.10000000e+01, 1.86815080e+06],
           [2.60000000e+01, 1.44257370e+06],
           [2.10000000e+01, 1.29684168e+06],
           [4.90000000e+01, 1.35607954e+06],
           [2.00000000e+01, 1.70498340e+06],
           [2.30000000e+01, 1.15830591e+06],
           [2.00000000e+01, 1.57575448e+06],
           [2.60000000e+01, 1.67796587e+06],
           [3.50000000e+01, 1.87317033e+06],
           [2.10000000e+01, 1.53007303e+06],
           [2.20000000e+01, 1.48131494e+06],
           [1.90000000e+01, 1.60557346e+06],
           [3.40000000e+01, 1.43249051e+06],
           [1.80000000e+01, 1.58241871e+06],
    ...
           [2.10000000e+01, 1.41064754e+06],
           [2.00000000e+01, 1.91714030e+06],
           [2.20000000e+01, 1.30791128e+06],
           [2.50000000e+01, 8.93525339e+05],
           [2.50000000e+01, 1.74430270e+06],
           [7.30000000e+01, 1.65369057e+06],
           [2.20000000e+01, 1.48584426e+06],
           [2.80000000e+01, 1.79437919e+06],
           [2.90600000e+01, 1.84599685e+06],
           [1.90000000e+01, 1.57800703e+06],
           [2.00000000e+01, 1.46577039e+06],
           [2.20000000e+01, 1.30793200e+06],
           [1.90000000e+01, 1.33464236e+06],
           [2.40000000e+01, 1.31818656e+06],
           [2.10000000e+01, 1.64204629e+06],
           [2.40000000e+01, 1.54616694e+06],
           [2.27900000e+01, 1.45689873e+06],
           [7.20000000e+01, 1.57223354e+06],
           [2.30000000e+01, 1.98747750e+06],
           [2.30000000e+01, 1.60306371e+06]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;F&#x27; &#x27;Munchen&#x27; ... &#x27;M&#x27; &#x27;Queensland&#x27;</div><input id='attrs-fe426c69-17d7-47ce-a906-b82bf4474723' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-fe426c69-17d7-47ce-a906-b82bf4474723' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e520d3af-36ff-4497-b23b-be8c72111ec9' class='xr-var-data-in' type='checkbox'><label for='data-e520d3af-36ff-4497-b23b-be8c72111ec9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;F&#x27;, &#x27;Munchen&#x27;],
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
           [&#x27;M&#x27;, &#x27;Queensland&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.1483 1.023 ... 2.334 -0.845</div><input id='attrs-4b35afef-9222-489a-826b-f1b39de06506' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4b35afef-9222-489a-826b-f1b39de06506' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cc358d83-12f7-4d08-80d0-b200744817e3' class='xr-var-data-in' type='checkbox'><label for='data-cc358d83-12f7-4d08-80d0-b200744817e3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.48322711e-01],
           [ 1.02328630e+00],
           [ 6.27153712e-01],
           [ 1.52152687e+00],
           [-1.45827707e+00],
           [-1.32049009e+00],
           [-1.25918001e+00],
           [-9.88406220e-01],
           [ 4.24007035e-02],
           [-1.09466142e-01],
           [ 2.49989850e-01],
           [-6.97232363e-01],
           [ 2.87651641e-01],
           [ 1.38790893e+00],
           [-1.32069425e+00],
           [-1.14853758e+00],
           [-5.52899492e-02],
           [-5.30266842e-01],
           [-8.07549885e-01],
           [ 3.38320476e-01],
    ...
           [-1.15497262e+00],
           [ 5.57375822e-02],
           [-6.13275158e-01],
           [-5.26658593e-01],
           [-1.53977196e+00],
           [ 1.99249858e+00],
           [-3.14296420e-01],
           [-3.70125930e-01],
           [-3.65172768e+00],
           [-9.79410386e-01],
           [-3.73155256e-01],
           [-5.94331203e-01],
           [-9.68513098e-02],
           [-2.23349725e-01],
           [-1.14382972e+00],
           [ 1.32314408e+00],
           [ 3.22989360e-01],
           [ 9.41609900e-01],
           [ 2.33385968e+00],
           [-8.45004470e-01]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.264e+05 4.604e+05 ... 5.612e+05</div><input id='attrs-a0c618c1-d717-41ed-a00e-0322b7795c89' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a0c618c1-d717-41ed-a00e-0322b7795c89' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-aa7281bf-1c95-47ad-886d-597abbb26bbd' class='xr-var-data-in' type='checkbox'><label for='data-aa7281bf-1c95-47ad-886d-597abbb26bbd' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[426444.54292573],
            [460372.9703569 ],
            [439827.24549258],
            ...,
            [427535.82682282],
            [530825.20053395],
            [484981.67980236]],
    
           [[446790.51437415],
            [484532.26460791],
            [463715.34532811],
            ...,
            [449822.50079095],
            [550280.54331798],
            [507467.80567618]],
    
           [[460932.78566894],
            [501325.13657544],
            [480319.71290267],
            ...,
            [465313.73404639],
            [563803.74797215],
            [523097.67585873]],
    
           [[475075.05696373],
            [518118.00854298],
            [496924.08047724],
            ...,
            [480804.96730183],
            [577326.95262632],
            [538727.54604127]],
    
           [[495421.02841216],
            [542277.30279399],
            [520812.18031277],
            ...,
            [503091.64126996],
            [596782.29541034],
            [561213.6719151 ]]], shape=(5, 216, 1))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.036 -1.182 ... -4.021 -0.9114</div><input id='attrs-931fcfe7-93ab-41cf-b1d7-91cc98d91da2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-931fcfe7-93ab-41cf-b1d7-91cc98d91da2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bdbaff51-0fc4-403c-bd26-8d3b9f6d66bb' class='xr-var-data-in' type='checkbox'><label for='data-bdbaff51-0fc4-403c-bd26-8d3b9f6d66bb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.03593217],
           [-1.18228146],
           [-0.8710495 ],
           [-2.86273506],
           [-5.81179816],
           [-0.97178023],
           [-1.6920084 ],
           [-0.93572612],
           [-1.00917076],
           [-1.23698624],
           [-1.94337031],
           [-1.75597632],
           [-1.07782826],
           [-0.93484472],
           [-1.34346116],
           [-0.9362691 ],
           [-1.25389532],
           [-0.95820961],
           [-1.00986314],
           [-0.85592663],
    ...
           [-1.08657625],
           [-1.91430013],
           [-0.89483328],
           [-3.13185767],
           [-0.90594411],
           [-0.989051  ],
           [-1.37609215],
           [-2.19462803],
           [-0.98677789],
           [-0.86061958],
           [-0.87038174],
           [-0.89202585],
           [-1.08406159],
           [-0.93255066],
           [-0.95662517],
           [-1.17989111],
           [-1.08685779],
           [-0.85598943],
           [-4.02130046],
           [-0.91139503]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-0.06719 -0.7156 ... -2.757 -0.496</div><input id='attrs-d93b12fa-1b8a-4ba1-bead-e91f8b009cd8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d93b12fa-1b8a-4ba1-bead-e91f8b009cd8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-79b2172a-ece4-4c44-b825-ba4a3058d20e' class='xr-var-data-in' type='checkbox'><label for='data-79b2172a-ece4-4c44-b825-ba4a3058d20e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-0.06719125],
           [-0.71564529],
           [-0.39312262],
           [-1.15713134],
           [-1.2891764 ],
           [-0.95545517],
           [-0.87993161],
           [-0.63990588],
           [-0.26609821],
           [-0.17688366],
           [-0.16647469],
           [-0.53908063],
           [-0.22463567],
           [-1.01441114],
           [-0.8415011 ],
           [-0.84589112],
           [-0.18138822],
           [-0.3377576 ],
           [-0.46576539],
           [-0.28928908],
    ...
           [-0.90325337],
           [-0.06761755],
           [-0.43136622],
           [-0.68812924],
           [-1.26431866],
           [-2.15695588],
           [-0.22316729],
           [-0.0811972 ],
           [-6.67008699],
           [-0.68731458],
           [-0.31269925],
           [-0.41990683],
           [-0.31727263],
           [-0.2460947 ],
           [-0.79062521],
           [-0.9966246 ],
           [-0.23016342],
           [-0.58696976],
           [-2.75749716],
           [-0.49597778]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>4.609e+05 5.013e+05 ... 5.231e+05</div><input id='attrs-ee562a8b-14f3-4fc0-bb30-322cf2a42ffc' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ee562a8b-14f3-4fc0-bb30-322cf2a42ffc' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c5d8f108-ba47-4fc1-8dda-12489e5f35d8' class='xr-var-data-in' type='checkbox'><label for='data-c5d8f108-ba47-4fc1-8dda-12489e5f35d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[460932.78566894],
           [501325.13657544],
           [480319.71290267],
           [554627.1173806 ],
           [368831.62311043],
           [540433.04303315],
           [577906.35855688],
           [491030.72020256],
           [458977.708082  ],
           [447149.98690339],
           [553515.19945446],
           [440585.31841601],
           [512698.60846205],
           [476691.27698739],
           [561493.00755893],
           [495990.76344408],
           [532271.29285816],
           [522715.81137202],
           [478766.78570424],
           [478454.24670877],
    ...
           [483826.46936402],
           [557239.33105649],
           [489717.56761864],
           [398158.94018062],
           [535539.25352106],
           [467906.89409231],
           [445043.69926534],
           [574866.14026909],
           [584705.43862745],
           [516876.66789068],
           [488573.66643951],
           [489720.7777097 ],
           [456870.87459645],
           [473729.53968049],
           [536305.70777973],
           [495828.92799329],
           [512569.62837114],
           [465313.73404639],
           [563803.74797215],
           [523097.67585873]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.6693 1.069 ... 0.9797 0.3136</div><input id='attrs-3673e255-154d-4cc3-a63d-699f4251357a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3673e255-154d-4cc3-a63d-699f4251357a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6bfc5aa9-074e-4513-80e3-793722aa29f5' class='xr-var-data-in' type='checkbox'><label for='data-6bfc5aa9-074e-4513-80e3-793722aa29f5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 6.69326919e-01,  1.06923352e+00,  1.73853276e-01,
             4.44782146e-02,  8.63320653e-01, -4.92605666e-01,
             6.64758343e-01,  2.83439752e+04,  8.06247991e-01,
             1.09447456e-50,  3.35241657e-01,  9.79712184e-01,
             3.13628008e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-1b6e10a7-7f6f-4c0e-8125-7f977f699f62' class='xr-section-summary-in' type='checkbox' checked /><label for='section-1b6e10a7-7f6f-4c0e-8125-7f977f699f62' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000_test</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;M&#x27;, &#x27;F&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7f54d098eca0&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: 489, &#x27;F&#x27;: 589}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 85, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 803895.003258, &#x27;max&#x27;: 2213930.77819}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;M&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 803895.003258, &#x27;max&#x27;: 2213930.77819}}, &#x27;F&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 85.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 854269.795819, &#x27;max&#x27;: 1839480.7792}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1239237.74772, &#x27;max&#x27;: 1797270.09178}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1095961.96155, &#x27;max&#x27;: 1785422.9362}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 991144.994738, &#x27;max&#x27;: 1961041.2011400005}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1277794.4776, &#x27;max&#x27;: 1858687.83646}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1209177.6117399998, &#x27;max&#x27;: 1839143.13269}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1002619.98087, &#x27;max&#x27;: 1869137.32932}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1283953.04581, &#x27;max&#x27;: 2034930.18739}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1226746.2365, &#x27;max&#x27;: 1987477.49695}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1195065.98729, &#x27;max&#x27;: 1727397.1725}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 85.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1315469.72766, &#x27;max&#x27;: 2156696.50099}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1508119.55881, &#x27;max&#x27;: 1968074.24155}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1309818.45623, &#x27;max&#x27;: 1868150.79879}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 961761.166078, &#x27;max&#x27;: 1749317.40938}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1205099.3215, &#x27;max&#x27;: 1807706.83688}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 945590.839808, &#x27;max&#x27;: 1980091.79537}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1182852.33915, &#x27;max&#x27;: 1836931.72887}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 803895.003258, &#x27;max&#x27;: 1483045.87846}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1038404.46752, &#x27;max&#x27;: 1807616.45732}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 989444.241903, &#x27;max&#x27;: 1689557.92241}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 854269.795819, &#x27;max&#x27;: 1673520.84453}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 823904.532171, &#x27;max&#x27;: 1026186.09328}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1335132.49541, &#x27;max&#x27;: 1845048.33956}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}, np.str_(&#x27;EstimatedTotalIntraCranialVol&#x27;): {&#x27;min&#x27;: 1374160.52253, &#x27;max&#x27;: 2213930.77819}}}}</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    model.covariates




.. code:: text

    ['age', 'EstimatedTotalIntraCranialVol']



.. code:: ipython3

    plot_centiles_advanced(
        model,
        covariate="age",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [20, 100], "EstimatedTotalIntraCranialVol": [0.75e6, 1.5e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )
    plot_centiles_advanced(
        model,
        covariate="age",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [0, 50], "EstimatedTotalIntraCranialVol": [0.75e6, 1.5e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )
    plot_centiles_advanced(
        model,
        covariate="age",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [20, 100], "EstimatedTotalIntraCranialVol": [1.5e6, 2.0e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )
    plot_centiles_advanced(
        model,
        covariate="age",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [0, 50], "EstimatedTotalIntraCranialVol": [1.5e6, 2.0e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )




.. image:: 11_composite_basis_function_files/11_composite_basis_function_7_0.png



.. image:: 11_composite_basis_function_files/11_composite_basis_function_7_1.png



.. image:: 11_composite_basis_function_files/11_composite_basis_function_7_2.png



.. image:: 11_composite_basis_function_files/11_composite_basis_function_7_3.png




.. code:: text

    [<Figure size 400x300 with 1 Axes>]



.. code:: ipython3

    plot_centiles_advanced(
        model,
        covariate="EstimatedTotalIntraCranialVol",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [20, 100], "EstimatedTotalIntraCranialVol": [0.75e6, 1.5e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )
    plot_centiles_advanced(
        model,
        covariate="EstimatedTotalIntraCranialVol",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [0, 50], "EstimatedTotalIntraCranialVol": [0.75e6, 1.5e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )
    plot_centiles_advanced(
        model,
        covariate="EstimatedTotalIntraCranialVol",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [20, 100], "EstimatedTotalIntraCranialVol": [1.5e6, 2.0e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )
    plot_centiles_advanced(
        model,
        covariate="EstimatedTotalIntraCranialVol",
        response_vars=["CortexVol"],
        scatter_data=data,
        covariate_ranges={"age": [0, 50], "EstimatedTotalIntraCranialVol": [1.5e6, 2.0e6]},
        batch_effects="all",
        show_legend=False,
        plt_kwargs={"figsize": (4, 3)},
    )



.. image:: 11_composite_basis_function_files/11_composite_basis_function_8_0.png



.. image:: 11_composite_basis_function_files/11_composite_basis_function_8_1.png



.. image:: 11_composite_basis_function_files/11_composite_basis_function_8_2.png



.. image:: 11_composite_basis_function_files/11_composite_basis_function_8_3.png




.. code:: text

    [<Figure size 400x300 with 1 Axes>]






