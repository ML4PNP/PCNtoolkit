The NormData class
==================

.. container:: notebook-download

   :download:`Download Jupyter notebook <notebooks/01_loading_data.ipynb>`

A key component of the PCNtoolkit is the NormData object. It is a
container for the data that will be used to fit the normative model. The
NormData object keeps track of the all the dimensions of your data, the
features and response variables, batch effects, preprocessing steps, and
more.

.. code:: ipython3

    import copy
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    from pcntoolkit import NormData

Creating a NormData object
--------------------------

There are currently two easy ways to create a NormData object. 1. Load
from a pandas dataframe 2. Load from numpy arrays

Here are examples of both.

.. code:: ipython3

    # Creating a NormData object from a pandas dataframe
    
    # Download an example dataset:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    )
    
    # specify the column names to use
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = ["WM-hypointensities", "Left-Lateral-Ventricle", "Brain-Stem"]
    
    # create a NormData object
    norm_data = NormData.from_dataframe(
        name="fcon1000",
        dataframe=data,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        remove_outliers=True,
        z_threshold=10,
    )
    norm_data.coords


.. code:: text

    Process: 2356 - 2026-09-25 15:32:07 - Removed 2 outliers for WM-hypointensities
    Process: 2356 - 2026-09-25 15:32:07 - Removed 2 outliers
    Process: 2356 - 2026-09-25 15:32:07 - Dataset "fcon1000" created.
        - 1076 observations
        - 1076 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        


.. code:: text

    /home/runner/work/PCNtoolkit/PCNtoolkit/pcntoolkit/dataio/norm_data.py:427: UserWarning: No grouping was set for outlier removal, so thresholds are computed across all rows and site differences are ignored. Consider remove_outliers_group_by=['site'] (if available) or your site column.
      dataframe = cls.remove_outliers(




.. code:: text

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'



.. code:: ipython3

    # Creating a NormData object from numpy arrays
    import numpy as np
    
    from pcntoolkit import NormData
    
    # synthesize some data
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)
    batch_effects = np.random.randint(0, 2, 100)[:,None]
    subject_ids = np.arange(100)
    
    # Create a NormData object
    np_norm_data = NormData.from_ndarrays("fcon1000", X=X, Y=Y, batch_effects=batch_effects, subject_ids=subject_ids)
    np_norm_data.coords


.. code:: text

    Process: 2356 - 2026-09-25 15:32:07 - Dataset "fcon1000" created.
        - 100 observations
        - 100 unique subjects
        - 10 covariates
        - 10 response variables
        - 1 batch effects:
        	batch_effect_0 (2)
        




.. code:: text

    Coordinates:
      * observations       (observations) int64 800B 0 1 2 3 4 5 ... 95 96 97 98 99
      * response_vars      (response_vars) <U14 560B 'response_var_0' ... 'respon...
      * covariates         (covariates) <U11 440B 'covariate_0' ... 'covariate_9'
      * batch_effect_dims  (batch_effect_dims) <U14 56B 'batch_effect_0'



As you can see, it is very simple to create a NormData object.

There is an important difference though: the coordinates of the NormData
object that was created with ``from_dataframe`` have the name of the
column in the dataframe, but the ``from_ndarrays`` method creates
coordinates with generic names. This is why the from_dataframe method is
favorable.

Casting back to a pandas dataframe
----------------------------------

The NormData object can be cast back to a pandas dataframe using the
``to_dataframe`` method. This will return a pandas dataframe with a
columnar multi-index.

.. code:: ipython3

    df = norm_data.to_dataframe()
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    
        .dataframe thead tr:last-of-type th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>X</th>
          <th colspan="3" halign="left">Y</th>
          <th colspan="2" halign="left">batch_effects</th>
          <th>subject_ids</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>Brain-Stem</th>
          <th>Left-Lateral-Ventricle</th>
          <th>WM-hypointensities</th>
          <th>sex</th>
          <th>site</th>
          <th>subject_ids</th>
        </tr>
        <tr>
          <th>observations</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.63</td>
          <td>20663.2</td>
          <td>4049.4</td>
          <td>1686.7</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>19954.0</td>
          <td>9312.6</td>
          <td>1371.1</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>21645.2</td>
          <td>8972.6</td>
          <td>1414.8</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>20790.6</td>
          <td>6798.6</td>
          <td>1830.6</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>17692.6</td>
          <td>6112.5</td>
          <td>1642.4</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
    </div>



Inspecting the NormData
-----------------------

So let’s go over the attributes of the NormData object. Because it is a
subclass of xarray.Dataset, it has all the attributes of a
xarray.Dataset, but it has some additional attributes that are specific
to normative modelling.

The data variables
~~~~~~~~~~~~~~~~~~

The data variables of the NormData object are: - ``X``: The covariates -
``Y``: The response variables - ``batch_effects``: The batch effects -
``subjects``: The subject ids

And all these data variables are xarray.DataArrays, with corresponding
dimensions, stored in the ``data_vars`` attribute of the NormData
object.

.. code:: ipython3

    norm_data.data_vars




.. code:: text

    Data variables:
        subject_ids    (observations) int64 9kB 0 1 2 3 4 ... 1072 1073 1074 1075
        Y              (observations, response_vars) float64 26kB 1.687e+03 ... 1...
        X              (observations, covariates) float64 9kB 25.63 18.34 ... 23.0
        batch_effects  (observations, batch_effect_dims) <U17 146kB '1' ... 'Sain...



The coordinates
~~~~~~~~~~~~~~~

Because it is a subclass of xarray.Dataset, the NormData object also
holds all the coordinates of the data, found under the ``coords``
attribute.

The coordinates are: - ``observations``: The index of the observations -
``response_vars``: The names of the response variables - ``covariates``:
The names of the covariates - ``batch_effect_dims``: The names of the
batch effect dimensions

.. code:: ipython3

    norm_data.coords




.. code:: text

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'



Indexing using the coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarrays powerful indexing methods can also be used on NormData.

Selecting a response variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example, to select the data for a specific response variable, you
can use the ``response_vars`` coordinate:

.. code:: python

   norm_data.sel(response_vars="WM-hypointensities")

This will return a new NormData object with only the data for the
response variable “WM-hypointensities”.

.. code:: ipython3

    norm_data.sel(response_vars="WM-hypointensities")




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 181kB
    Dimensions:            (observations: 1076, covariates: 1, batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
        response_vars      &lt;U22 88B &#x27;WM-hypointensities&#x27;
    Data variables:
        subject_ids        (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
        Y                  (observations) float64 9kB 1.687e+03 1.371e+03 ... 509.1
        X                  (observations, covariates) float64 9kB 25.63 ... 23.0
        batch_effects      (observations, batch_effect_dims) &lt;U17 146kB &#x27;1&#x27; ... &#x27;...
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-d77e9caa-845c-466d-bc63-2bd85f5205e6' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-d77e9caa-845c-466d-bc63-2bd85f5205e6' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 1076</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div></li><li class='xr-section-item'><input id='section-5bb2d94a-a7c1-4d21-8927-2ff560b99e99' class='xr-section-summary-in' type='checkbox' checked /><label for='section-5bb2d94a-a7c1-4d21-8927-2ff560b99e99' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1072 1073 1074 1075</div><input id='attrs-f8361e25-8407-4d5a-bf28-efe6d79de731' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f8361e25-8407-4d5a-bf28-efe6d79de731' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4906f077-f918-4995-9616-da73b3fa1b63' class='xr-var-data-in' type='checkbox'><label for='data-4906f077-f918-4995-9616-da73b3fa1b63' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1073, 1074, 1075], shape=(1076,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-8a92dfb0-81d4-48f8-b9a8-b5f7b142e4b8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8a92dfb0-81d4-48f8-b9a8-b5f7b142e4b8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-89e4afcb-a4e8-4054-9eca-8f48e17db7e6' class='xr-var-data-in' type='checkbox'><label for='data-89e4afcb-a4e8-4054-9eca-8f48e17db7e6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-d617e28c-4ceb-4e67-8a72-adf038aceb1d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d617e28c-4ceb-4e67-8a72-adf038aceb1d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-55140b77-1915-42bd-b7ba-016c3e57ec79' class='xr-var-data-in' type='checkbox'><label for='data-55140b77-1915-42bd-b7ba-016c3e57ec79' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-d7b8b785-1fcd-4e02-a68d-3391b8a151ef' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d7b8b785-1fcd-4e02-a68d-3391b8a151ef' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9eba0e12-d86a-4121-bf86-58602974fc37' class='xr-var-data-in' type='checkbox'><label for='data-9eba0e12-d86a-4121-bf86-58602974fc37' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-4d4892a2-fe00-41d4-9661-d23eee56cadb' class='xr-section-summary-in' type='checkbox' checked /><label for='section-4d4892a2-fe00-41d4-9661-d23eee56cadb' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1072 1073 1074 1075</div><input id='attrs-142bdc6f-7622-46d3-a721-15ce7cddc1d2' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-142bdc6f-7622-46d3-a721-15ce7cddc1d2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-31dcd4a2-77bc-4d13-8c2f-ffe76c701a1c' class='xr-var-data-in' type='checkbox'><label for='data-31dcd4a2-77bc-4d13-8c2f-ffe76c701a1c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1073, 1074, 1075], shape=(1076,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 1.371e+03 ... 448.3 509.1</div><input id='attrs-f1af3ad7-2c35-478a-8aa1-925f364289cd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f1af3ad7-2c35-478a-8aa1-925f364289cd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-07f19889-9bcf-4198-9b89-8c960f6637cb' class='xr-var-data-in' type='checkbox'><label for='data-07f19889-9bcf-4198-9b89-8c960f6637cb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1686.7, 1371.1, 1414.8, ..., 1061. ,  448.3,  509.1], shape=(1076,))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 27.0 29.0 23.0</div><input id='attrs-8268420e-2837-4517-9daa-df9518ff8f2f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8268420e-2837-4517-9daa-df9518ff8f2f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a438a121-fc39-46a4-9821-a95315bdfb8b' class='xr-var-data-in' type='checkbox'><label for='data-a438a121-fc39-46a4-9821-a95315bdfb8b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           ...,
           [27.  ],
           [29.  ],
           [23.  ]], shape=(1076, 1))</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;SaintLouis&#x27;</div><input id='attrs-358a1777-d222-49c8-943d-7e7548e5f70b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-358a1777-d222-49c8-943d-7e7548e5f70b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b95f4145-3f2d-4cfb-82c5-4c9e1f0dc3c9' class='xr-var-data-in' type='checkbox'><label for='data-b95f4145-3f2d-4cfb-82c5-4c9e1f0dc3c9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           ...,
           [&#x27;1&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;],
           [&#x27;0&#x27;, &#x27;SaintLouis&#x27;]], shape=(1076, 2), dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-5183a114-f32d-42aa-8fa1-2d905f81c58e' class='xr-section-summary-in' type='checkbox' checked /><label for='section-5183a114-f32d-42aa-8fa1-2d905f81c58e' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7fac63812b60&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: 489, &#x27;0&#x27;: 587}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 83, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;0&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



Selecting a number of observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

But we can also filter out a slice of the data. For example, to select
the first 10 observations, you can use the ``observations`` coordinate:

.. code:: python

   norm_data.sel(observations=slice(0, 9))

This will return a new NormData object with only the first 10
observations.

.. code:: ipython3

    norm_data.sel(observations=slice(0, 9))




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 2kB
    Dimensions:            (observations: 10, response_vars: 3, covariates: 1,
                            batch_effect_dims: 2)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
    Data variables:
        subject_ids        (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-1d3fc385-d310-4ce2-a3e1-27b4c6a6e989' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-1d3fc385-d310-4ce2-a3e1-27b4c6a6e989' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li></ul></div></li><li class='xr-section-item'><input id='section-b12fdb1f-1885-481c-85e8-c33f272f4e04' class='xr-section-summary-in' type='checkbox' checked /><label for='section-b12fdb1f-1885-481c-85e8-c33f272f4e04' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-868e24df-f255-450f-86fd-82791fefed7c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-868e24df-f255-450f-86fd-82791fefed7c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7480b9bb-54e7-41c6-bcd1-d6647b7038c6' class='xr-var-data-in' type='checkbox'><label for='data-7480b9bb-54e7-41c6-bcd1-d6647b7038c6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-e279c865-29d1-4949-ba13-589b783b035e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e279c865-29d1-4949-ba13-589b783b035e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8fae7fc8-7798-467a-90b0-168412d0f88e' class='xr-var-data-in' type='checkbox'><label for='data-8fae7fc8-7798-467a-90b0-168412d0f88e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-238959ad-a371-4d42-86fa-36a7f4cb2fb3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-238959ad-a371-4d42-86fa-36a7f4cb2fb3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-de40581e-29cb-4e17-b872-36358e866a4a' class='xr-var-data-in' type='checkbox'><label for='data-de40581e-29cb-4e17-b872-36358e866a4a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-c62a5d09-5180-443e-9203-0e718463faa5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c62a5d09-5180-443e-9203-0e718463faa5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d4c2f0af-74fb-478a-8cc8-944072c97c75' class='xr-var-data-in' type='checkbox'><label for='data-d4c2f0af-74fb-478a-8cc8-944072c97c75' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-adba0337-a37b-4118-8ff8-22dc64d25b10' class='xr-section-summary-in' type='checkbox' checked /><label for='section-adba0337-a37b-4118-8ff8-22dc64d25b10' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-f5edccda-3554-42b3-85be-46ed63edd61d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f5edccda-3554-42b3-85be-46ed63edd61d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-53e5d308-c999-43bd-9304-43f7f3150a73' class='xr-var-data-in' type='checkbox'><label for='data-53e5d308-c999-43bd-9304-43f7f3150a73' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-d6b805d6-9a3b-4082-8e53-7f6ecc160878' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d6b805d6-9a3b-4082-8e53-7f6ecc160878' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4f8d566e-3677-4ec6-aea9-05433be9f8d8' class='xr-var-data-in' type='checkbox'><label for='data-4f8d566e-3677-4ec6-aea9-05433be9f8d8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-5c8d96bf-7537-422e-83f0-493cd5b93b88' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5c8d96bf-7537-422e-83f0-493cd5b93b88' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-95f97358-261b-4710-90cb-927ee8dfdd69' class='xr-var-data-in' type='checkbox'><label for='data-95f97358-261b-4710-90cb-927ee8dfdd69' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-f27a4db1-e000-4314-8d86-977b491b2774' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f27a4db1-e000-4314-8d86-977b491b2774' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-960eb8b2-194e-47b5-a724-6a92f5165309' class='xr-var-data-in' type='checkbox'><label for='data-960eb8b2-194e-47b5-a724-6a92f5165309' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-dea43499-2397-4aaf-99d2-d4c57ad9b65d' class='xr-section-summary-in' type='checkbox' checked /><label for='section-dea43499-2397-4aaf-99d2-d4c57ad9b65d' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7fac63812b60&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: 489, &#x27;0&#x27;: 587}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 83, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;0&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



NormData with predictions
-------------------------

After fitting a model and predicting on NormData, the NormData object
will have new attributes holding the predictions.

Specifically, the NormData object will be extended with new data
variables:

-  ``Z``: The predicted Z scores for each response variable
-  ``centiles``: The predicted centiles
-  ``logp``: The predicted log-p-values for each response variable
-  ``Yhat``: The predicted mean of the response variable
-  ``Y_harmonized``: The harmonized response variables
-  ``statistics``: An array of statistics for each response variable

And the following new coordinates: - ``centile``: The specific centile
values - ``statistic``: The name of the computed statistics

.. code:: ipython3

    from pcntoolkit import BLR, NormativeModel
    
    # We create a very simple BLR model because it is fast to fit
    model = NormativeModel(BLR())
    model.fit(norm_data)  # Fitting on the data also makes predictions for that data


.. code:: text

    Process: 2356 - 2026-09-25 15:32:07 - Fitting models on 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Fitting model for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:07 - Fitting model for Left-Lateral-Ventricle.
    Process: 2356 - 2026-09-25 15:32:07 - Fitting model for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:07 - Saving model to:
    	/home/runner/.pcntoolkit/saves.
    Process: 2356 - 2026-09-25 15:32:07 - Making predictions on 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Computing z-scores for 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Computing z-scores for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:07 - Computing z-scores for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:07 - Computing z-scores for Left-Lateral-Ventricle.
    Process: 2356 - 2026-09-25 15:32:07 - Computing centiles for 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Computing centiles for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:07 - Computing centiles for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:07 - Computing centiles for Left-Lateral-Ventricle.
    Process: 2356 - 2026-09-25 15:32:07 - Computing log-probabilities for 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Computing log-probabilities for 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Computing log-probabilities for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:07 - Computing log-probabilities for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:07 - Computing log-probabilities for Left-Lateral-Ventricle.
    Process: 2356 - 2026-09-25 15:32:07 - Computing yhat for 3 response variables.
    Process: 2356 - 2026-09-25 15:32:07 - Computing yhat for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:07 - Computing yhat for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:07 - Computing yhat for Left-Lateral-Ventricle.
    Process: 2356 - 2026-09-25 15:32:08 - Dataset "centile" created.
        - 150 observations
        - 150 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (1)
    	site (1)
        
    Process: 2356 - 2026-09-25 15:32:08 - Computing centiles for 3 response variables.
    Process: 2356 - 2026-09-25 15:32:08 - Computing centiles for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:08 - Computing centiles for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:08 - Computing centiles for Left-Lateral-Ventricle.
    Process: 2356 - 2026-09-25 15:32:08 - Harmonizing data on 3 response variables.
    Process: 2356 - 2026-09-25 15:32:08 - Harmonizing data for Brain-Stem.
    Process: 2356 - 2026-09-25 15:32:08 - Harmonizing data for WM-hypointensities.
    Process: 2356 - 2026-09-25 15:32:08 - Harmonizing data for Left-Lateral-Ventricle.


.. code:: ipython3

    norm_data.data_vars




.. code:: text

    Data variables:
        subject_ids    (observations) int64 9kB 0 1 2 3 4 ... 1072 1073 1074 1075
        Y              (observations, response_vars) float64 26kB 1.687e+03 ... 1...
        X              (observations, covariates) float64 9kB 25.63 18.34 ... 23.0
        batch_effects  (observations, batch_effect_dims) <U17 146kB '1' ... 'Sain...
        Z              (observations, response_vars) float64 26kB 0.7423 ... -0.2588
        centiles       (centile, observations, response_vars) float64 129kB 152.3...
        baseline_logp  (observations, response_vars) float64 26kB -1.131 ... -0.9549
        logp           (observations, response_vars) float64 26kB -1.158 ... -0.9537
        Yhat           (observations, response_vars) float64 26kB 1.21e+03 ... 2....
        statistics     (response_vars, statistic) float64 312B 0.1172 ... 0.1246



.. code:: ipython3

    norm_data.coords




.. code:: text

    Coordinates:
      * observations       (observations) int64 9kB 0 1 2 3 ... 1072 1073 1074 1075
      * response_vars      (response_vars) <U22 264B 'WM-hypointensities' ... 'Br...
      * covariates         (covariates) <U3 12B 'age'
      * batch_effect_dims  (batch_effect_dims) <U4 32B 'sex' 'site'
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) <U8 416B 'EXPV' 'Kurtosis' ... 'Skewness'



Indexing of predicted data
~~~~~~~~~~~~~~~~~~~~~~~~~~

All the indexing methods can still be used, and they will also slice
through the newly added data variables. So for example, to select the
first 10 observations, you can use:

.. code:: python

   norm_data.sel(observations=slice(0, 9))

This will return a new NormData object with only the first 10
observations.

.. code:: ipython3

    norm_data.sel(observations=slice(0, 9))




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.NormData&gt; Size: 5kB
    Dimensions:            (observations: 10, response_vars: 3, covariates: 1,
                            batch_effect_dims: 2, centile: 5, statistic: 13)
    Coordinates:
      * observations       (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
      * response_vars      (response_vars) &lt;U22 264B &#x27;WM-hypointensities&#x27; ... &#x27;Br...
      * covariates         (covariates) &lt;U3 12B &#x27;age&#x27;
      * batch_effect_dims  (batch_effect_dims) &lt;U4 32B &#x27;sex&#x27; &#x27;site&#x27;
      * centile            (centile) float64 40B 0.05 0.25 0.5 0.75 0.95
      * statistic          (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
    Data variables:
        subject_ids        (observations) int64 80B 0 1 2 3 4 5 6 7 8 9
        Y                  (observations, response_vars) float64 240B 1.687e+03 ....
        X                  (observations, covariates) float64 80B 25.63 ... 19.88
        batch_effects      (observations, batch_effect_dims) &lt;U17 1kB &#x27;1&#x27; ... &#x27;An...
        Z                  (observations, response_vars) float64 240B 0.7423 ... ...
        centiles           (centile, observations, response_vars) float64 1kB 152...
        baseline_logp      (observations, response_vars) float64 240B -1.131 ... ...
        logp               (observations, response_vars) float64 240B -1.158 ... ...
        Yhat               (observations, response_vars) float64 240B 1.21e+03 .....
        statistics         (response_vars, statistic) float64 312B 0.1172 ... 0.1246
    Attributes:
        real_ids:                       False
        is_scaled:                      False
        name:                           fcon1000
        unique_batch_effects:           {np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;sit...
        batch_effect_counts:            defaultdict(&lt;function NormData.register_b...
        covariate_ranges:               {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}
        batch_effect_covariate_ranges:  {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {...</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.NormData</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-d1d1f2a2-ac63-491e-8734-efc6ad955af3' class='xr-section-summary-in' type='checkbox' disabled /><label for='section-d1d1f2a2-ac63-491e-8734-efc6ad955af3' class='xr-section-summary'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>observations</span>: 10</li><li><span class='xr-has-index'>response_vars</span>: 3</li><li><span class='xr-has-index'>covariates</span>: 1</li><li><span class='xr-has-index'>batch_effect_dims</span>: 2</li><li><span class='xr-has-index'>centile</span>: 5</li><li><span class='xr-has-index'>statistic</span>: 13</li></ul></div></li><li class='xr-section-item'><input id='section-76eb22df-974e-4645-b181-4b3c0dc296cb' class='xr-section-summary-in' type='checkbox' checked /><label for='section-76eb22df-974e-4645-b181-4b3c0dc296cb' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>observations</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-1004c8e9-785f-445d-a417-3fbfc33f627a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1004c8e9-785f-445d-a417-3fbfc33f627a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-df8115f3-01ad-4b49-839a-1ea31a706e9e' class='xr-var-data-in' type='checkbox'><label for='data-df8115f3-01ad-4b49-839a-1ea31a706e9e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>response_vars</span></div><div class='xr-var-dims'>(response_vars)</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27; ... &#x27;Brain-...</div><input id='attrs-5fb7d532-533e-4020-9119-c2fb1bc6768d' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5fb7d532-533e-4020-9119-c2fb1bc6768d' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-22d7494e-10f0-4287-91c3-fb3b071b2534' class='xr-var-data-in' type='checkbox'><label for='data-22d7494e-10f0-4287-91c3-fb3b071b2534' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;WM-hypointensities&#x27;, &#x27;Left-Lateral-Ventricle&#x27;, &#x27;Brain-Stem&#x27;],
          dtype=&#x27;&lt;U22&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>covariates</span></div><div class='xr-var-dims'>(covariates)</div><div class='xr-var-dtype'>&lt;U3</div><div class='xr-var-preview xr-preview'>&#x27;age&#x27;</div><input id='attrs-0a3d13fc-5070-448d-a273-e85ecdecfc7b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0a3d13fc-5070-448d-a273-e85ecdecfc7b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0ec25c4d-0358-4925-a82e-d5c2d12392a8' class='xr-var-data-in' type='checkbox'><label for='data-0ec25c4d-0358-4925-a82e-d5c2d12392a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;age&#x27;], dtype=&#x27;&lt;U3&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>batch_effect_dims</span></div><div class='xr-var-dims'>(batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U4</div><div class='xr-var-preview xr-preview'>&#x27;sex&#x27; &#x27;site&#x27;</div><input id='attrs-c85826e1-ba72-4d1d-809d-fee50a9eac67' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c85826e1-ba72-4d1d-809d-fee50a9eac67' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6d7412b3-5a53-4390-96e3-62657f5ea990' class='xr-var-data-in' type='checkbox'><label for='data-6d7412b3-5a53-4390-96e3-62657f5ea990' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;sex&#x27;, &#x27;site&#x27;], dtype=&#x27;&lt;U4&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>centile</span></div><div class='xr-var-dims'>(centile)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.05 0.25 0.5 0.75 0.95</div><input id='attrs-f9717e31-eafa-4945-9a61-b5ee913a5aca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f9717e31-eafa-4945-9a61-b5ee913a5aca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-39fbcb4f-b5c2-41ab-80ce-c7688a880179' class='xr-var-data-in' type='checkbox'><label for='data-39fbcb4f-b5c2-41ab-80ce-c7688a880179' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0.05, 0.25, 0.5 , 0.75, 0.95])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-1f68c8d7-131b-4923-9c94-c925b887c014' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1f68c8d7-131b-4923-9c94-c925b887c014' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-aeabb5f5-72bf-4712-a1a5-0b2420a258c1' class='xr-var-data-in' type='checkbox'><label for='data-aeabb5f5-72bf-4712-a1a5-0b2420a258c1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8c7df195-4933-4313-895d-92d12cc61734' class='xr-section-summary-in' type='checkbox' checked /><label for='section-8c7df195-4933-4313-895d-92d12cc61734' class='xr-section-summary' title='Expand/collapse section'>Data variables: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>subject_ids</span></div><div class='xr-var-dims'>(observations)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-7b6231a0-63fc-43cb-b96b-ac3ff399a964' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7b6231a0-63fc-43cb-b96b-ac3ff399a964' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f3287619-83bf-414b-9a9f-dc2db3c341b5' class='xr-var-data-in' type='checkbox'><label for='data-f3287619-83bf-414b-9a9f-dc2db3c341b5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Y</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.687e+03 4.049e+03 ... 2.092e+04</div><input id='attrs-9915a318-cb98-403c-b4a1-57b43ea2e92f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9915a318-cb98-403c-b4a1-57b43ea2e92f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5d8229ca-b3bd-4f11-8f1e-33926a2f488c' class='xr-var-data-in' type='checkbox'><label for='data-5d8229ca-b3bd-4f11-8f1e-33926a2f488c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1686.7,  4049.4, 20663.2],
           [ 1371.1,  9312.6, 19954. ],
           [ 1414.8,  8972.6, 21645.2],
           [ 1830.6,  6798.6, 20790.6],
           [ 1642.4,  6112.5, 17692.6],
           [ 2108.4,  7076.4, 20996.8],
           [ 2023.1,  4862.2, 20964.9],
           [ 2193.4,  9931.7, 21339.8],
           [ 1086. ,  6479.5, 18517.9],
           [ 1604.9,  5890.9, 20919.9]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>X</span></div><div class='xr-var-dims'>(observations, covariates)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>25.63 18.34 29.2 ... 17.58 19.88</div><input id='attrs-5c5e1db7-cbb6-418f-9ec7-e7f5a4aa3bb9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5c5e1db7-cbb6-418f-9ec7-e7f5a4aa3bb9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a95fb6f2-ea7f-4984-8099-0903468945c0' class='xr-var-data-in' type='checkbox'><label for='data-a95fb6f2-ea7f-4984-8099-0903468945c0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[25.63],
           [18.34],
           [29.2 ],
           [31.39],
           [13.58],
           [29.21],
           [15.92],
           [21.46],
           [17.58],
           [19.88]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>batch_effects</span></div><div class='xr-var-dims'>(observations, batch_effect_dims)</div><div class='xr-var-dtype'>&lt;U17</div><div class='xr-var-preview xr-preview'>&#x27;1&#x27; &#x27;AnnArbor_a&#x27; ... &#x27;AnnArbor_a&#x27;</div><input id='attrs-1505b8fe-4c61-49cd-86e5-248d920d21d9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-1505b8fe-4c61-49cd-86e5-248d920d21d9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2f1a806f-26f8-414c-9dad-91c1452511cb' class='xr-var-data-in' type='checkbox'><label for='data-2f1a806f-26f8-414c-9dad-91c1452511cb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;],
           [&#x27;1&#x27;, &#x27;AnnArbor_a&#x27;]], dtype=&#x27;&lt;U17&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Z</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.7423 -0.8588 ... -0.2164 0.1348</div><input id='attrs-75902b83-1eee-417d-af3f-dd4e83995077' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-75902b83-1eee-417d-af3f-dd4e83995077' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-551fbf42-4f3f-45d7-92f9-d4a6e86e8dd1' class='xr-var-data-in' type='checkbox'><label for='data-551fbf42-4f3f-45d7-92f9-d4a6e86e8dd1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 0.74231137, -0.85877919,  0.02189509],
           [ 0.44522032,  0.68748038, -0.24940649],
           [ 0.22427048,  0.25876175,  0.40908844],
           [ 0.81287777, -0.35399385,  0.06283337],
           [ 0.99357247,  0.03846796, -1.14655481],
           [ 1.30313186, -0.21552947,  0.14927745],
           [ 1.52355424, -0.34800134,  0.15975065],
           [ 1.64146459,  0.74351599,  0.30026933],
           [ 0.02198455,  0.00355557, -0.82329652],
           [ 0.76796136, -0.21636589,  0.13481129]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>centiles</span></div><div class='xr-var-dims'>(centile, observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>152.3 904.6 ... 1.334e+04 2.469e+04</div><input id='attrs-3f668b93-5858-4d89-8431-095b15793d08' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3f668b93-5858-4d89-8431-095b15793d08' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-2b71151f-c353-4b49-b8b7-825fc2bbf7fa' class='xr-var-data-in' type='checkbox'><label for='data-2b71151f-c353-4b49-b8b7-825fc2bbf7fa' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[ 1.52348837e+02,  9.04557999e+02,  1.65032223e+04],
            [ 2.73876824e+01, -2.05497135e+01,  1.64703336e+04],
            [ 2.13436546e+02,  1.35692658e+03,  1.65189118e+04],
            [ 2.50875616e+02,  1.63421273e+03,  1.65284010e+04],
            [-5.43642089e+01, -6.25585397e+02,  1.64482432e+04],
            [ 2.13607561e+02,  1.35819311e+03,  1.65189554e+04],
            [-1.41596036e+01, -3.28054464e+02,  1.64591635e+04],
            [ 8.09049868e+01,  3.75605185e+02,  1.64845491e+04],
            [ 1.43432619e+01, -1.17099743e+02,  1.64668392e+04],
            [ 5.38100039e+01,  1.75030170e+02,  1.64773764e+04]],
    
           [[ 7.76050569e+02,  4.78668536e+03,  1.89251183e+04],
            [ 6.51236372e+02,  3.86249247e+03,  1.88928002e+04],
            [ 8.37129588e+02,  5.23899985e+03,  1.89407741e+04],
            [ 8.74583918e+02,  5.51638098e+03,  1.89503226e+04],
            [ 5.69674000e+02,  3.25863657e+03,  1.88714456e+04],
            [ 8.37300637e+02,  5.24026658e+03,  1.89408178e+04],
            [ 6.09776206e+02,  3.55553005e+03,  1.88819683e+04],
            [ 7.04669551e+02,  4.25812368e+03,  1.89066891e+04],
            [ 6.38217254e+02,  3.76609995e+03,  1.88894040e+04],
    ...
            [ 1.51849773e+03,  9.26061624e+03,  2.22604624e+04],
            [ 1.70417457e+03,  1.06357766e+04,  2.23075963e+04],
            [ 1.74165011e+03,  1.09132898e+04,  2.23172271e+04],
            [ 1.43719882e+03,  8.65840044e+03,  2.22401307e+04],
            [ 1.70434567e+03,  1.06370437e+04,  2.23076402e+04],
            [ 1.47715868e+03,  8.95440775e+03,  2.22501007e+04],
            [ 1.57181396e+03,  9.65551942e+03,  2.22738973e+04],
            [ 1.50551379e+03,  9.16444269e+03,  2.22572028e+04],
            [ 1.54481131e+03,  9.45551922e+03,  2.22670830e+04]],
    
           [[ 2.26680936e+03,  1.40656647e+04,  2.47138834e+04],
            [ 2.14234642e+03,  1.31436584e+04,  2.46829290e+04],
            [ 2.32786761e+03,  1.45178499e+04,  2.47294586e+04],
            [ 2.36535841e+03,  1.47954581e+04,  2.47391486e+04],
            [ 2.06123703e+03,  1.25426224e+04,  2.46633330e+04],
            [ 2.32803874e+03,  1.45191172e+04,  2.47295026e+04],
            [ 2.10109448e+03,  1.28379923e+04,  2.46729055e+04],
            [ 2.19557852e+03,  1.35380379e+04,  2.46960372e+04],
            [ 2.12938778e+03,  1.30476424e+04,  2.46797676e+04],
            [ 2.16861451e+03,  1.33382782e+04,  2.46893729e+04]]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>baseline_logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.131 -1.309 ... -1.02 -0.9262</div><input id='attrs-e0c1812f-4f60-40c7-9ae1-ea620371d092' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e0c1812f-4f60-40c7-9ae1-ea620371d092' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78976b62-7fec-4949-90ec-9bc1a8964d00' class='xr-var-data-in' type='checkbox'><label for='data-78976b62-7fec-4949-90ec-9bc1a8964d00' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.1308011 , -1.30932152, -0.91909123],
           [-0.93467347, -0.98176331, -0.95458511],
           [-0.94845513, -0.95661131, -1.00355878],
           [-1.29465948, -0.94700075, -0.92129015],
           [-1.08974199, -0.99825982, -1.60822936],
           [-1.74286138, -0.93365463, -0.93038344],
           [-1.58675451, -1.15865566, -0.92852933],
           [-1.91471408, -1.04399044, -0.96066507],
           [-0.9502843 , -0.96760084, -1.27432629],
           [-1.05843845, -1.02038092, -0.92619223]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>logp</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.158 -1.227 ... -0.8818 -0.9294</div><input id='attrs-8893544f-6ca4-4eeb-8da2-60fc41383964' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8893544f-6ca4-4eeb-8da2-60fc41383964' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e05631b7-1538-4073-b106-9eb867ce23c4' class='xr-var-data-in' type='checkbox'><label for='data-e05631b7-1538-4073-b106-9eb867ce23c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[-1.15822243, -1.22697976, -0.92035109],
           [-0.98205551, -1.09477916, -0.95144874],
           [-0.90784404, -0.89169379, -1.00377414],
           [-1.21310502, -0.92089526, -0.92209595],
           [-1.37684181, -0.8595082 , -1.5779446 ],
           [-1.73177179, -0.8814415 , -0.9312394 ],
           [-2.04369335, -0.91915665, -0.93324669],
           [-2.23001309, -1.13473767, -0.96529295],
           [-0.88322716, -0.85851141, -1.25929608],
           [-1.17775434, -0.88179869, -0.92936108]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>Yhat</span></div><div class='xr-var-dims'>(observations, response_vars)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>1.21e+03 7.485e+03 ... 2.058e+04</div><input id='attrs-f3572b72-5fd1-49c2-affa-4335ac6aa092' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f3572b72-5fd1-49c2-affa-4335ac6aa092' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-01ee3f02-9506-45f5-a5ac-910977346a24' class='xr-var-data-in' type='checkbox'><label for='data-01ee3f02-9506-45f5-a5ac-910977346a24' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1209.5790997 ,  7485.11135948, 20608.55283429],
           [ 1084.86705068,  6561.55435498, 20576.63127352],
           [ 1270.65207844,  7937.38824645, 20624.18520356],
           [ 1308.11701498,  8214.83541241, 20633.77480824],
           [ 1003.43641236,  5958.51850568, 20555.7881145 ],
           [ 1270.82315121,  7938.65512849, 20624.22899171],
           [ 1043.46744044,  6254.96890219, 20566.03454142],
           [ 1138.24175479,  6956.82155032, 20590.29317608],
           [ 1071.86552019,  6465.27132022, 20573.30337418],
           [ 1111.21225719,  6756.65418858, 20583.3746485 ]])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>statistics</span></div><div class='xr-var-dims'>(response_vars, statistic)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>0.1172 14.1 ... 0.9965 0.1246</div><input id='attrs-4c3c9227-83ca-4c57-880c-86acf62ed438' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4c3c9227-83ca-4c57-880c-86acf62ed438' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-541c6661-b2bf-4222-bd2f-e41c995e9562' class='xr-var-data-in' type='checkbox'><label for='data-541c6661-b2bf-4222-bd2f-e41c995e9562' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[ 1.17178773e-01,  1.40960628e+01,  1.55120188e-01,
             3.98458621e-01,  1.35593870e+00, -6.29998378e-02,
             1.17178773e-01,  6.26199688e+02,  4.38152477e-03,
             8.85849246e-01,  8.82821227e-01,  8.47148188e-01,
             2.46368107e+00],
           [ 1.57907407e-01,  5.69235133e+00,  1.06808574e-01,
             4.10913989e-01,  1.33302553e+00, -8.59130046e-02,
             1.57907407e-01,  3.90104480e+03,  2.56092410e-01,
             1.42121630e-17,  8.42092593e-01,  8.87533011e-01,
             1.70206530e+00],
           [ 5.48906457e-04,  4.45229581e-01,  1.63841143e-01,
             9.77076127e-02,  1.41864321e+00, -2.95325423e-04,
             5.48906457e-04,  2.49225388e+03,  6.47556344e-02,
             3.36782367e-02,  9.99451094e-01,  9.96466623e-01,
             1.24617337e-01]])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-13ee3449-4feb-4bae-ad76-bca006d22e24' class='xr-section-summary-in' type='checkbox' checked /><label for='section-13ee3449-4feb-4bae-ad76-bca006d22e24' class='xr-section-summary' title='Expand/collapse section'>Attributes: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>real_ids :</span></dt><dd>False</dd><dt><span>is_scaled :</span></dt><dd>False</dd><dt><span>name :</span></dt><dd>fcon1000</dd><dt><span>unique_batch_effects :</span></dt><dd>{np.str_(&#x27;sex&#x27;): [&#x27;1&#x27;, &#x27;0&#x27;], np.str_(&#x27;site&#x27;): [&#x27;AnnArbor_a&#x27;, &#x27;AnnArbor_b&#x27;, &#x27;Atlanta&#x27;, &#x27;Baltimore&#x27;, &#x27;Bangor&#x27;, &#x27;Beijing_Zang&#x27;, &#x27;Berlin_Margulies&#x27;, &#x27;Cambridge_Buckner&#x27;, &#x27;Cleveland&#x27;, &#x27;ICBM&#x27;, &#x27;Leiden_2180&#x27;, &#x27;Leiden_2200&#x27;, &#x27;Milwaukee_b&#x27;, &#x27;Munchen&#x27;, &#x27;NewYork_a&#x27;, &#x27;NewYork_a_ADHD&#x27;, &#x27;Newark&#x27;, &#x27;Oulu&#x27;, &#x27;Oxford&#x27;, &#x27;PaloAlto&#x27;, &#x27;Pittsburgh&#x27;, &#x27;Queensland&#x27;, &#x27;SaintLouis&#x27;]}</dd><dt><span>batch_effect_counts :</span></dt><dd>defaultdict(&lt;function NormData.register_batch_effects.&lt;locals&gt;.&lt;lambda&gt; at 0x7fac63812b60&gt;, {np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: 489, &#x27;0&#x27;: 587}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: 24, &#x27;AnnArbor_b&#x27;: 32, &#x27;Atlanta&#x27;: 28, &#x27;Baltimore&#x27;: 23, &#x27;Bangor&#x27;: 20, &#x27;Beijing_Zang&#x27;: 198, &#x27;Berlin_Margulies&#x27;: 26, &#x27;Cambridge_Buckner&#x27;: 198, &#x27;Cleveland&#x27;: 31, &#x27;ICBM&#x27;: 83, &#x27;Leiden_2180&#x27;: 12, &#x27;Leiden_2200&#x27;: 19, &#x27;Milwaukee_b&#x27;: 46, &#x27;Munchen&#x27;: 15, &#x27;NewYork_a&#x27;: 83, &#x27;NewYork_a_ADHD&#x27;: 25, &#x27;Newark&#x27;: 19, &#x27;Oulu&#x27;: 102, &#x27;Oxford&#x27;: 22, &#x27;PaloAlto&#x27;: 17, &#x27;Pittsburgh&#x27;: 3, &#x27;Queensland&#x27;: 19, &#x27;SaintLouis&#x27;: 31}})</dd><dt><span>covariate_ranges :</span></dt><dd>{np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}</dd><dt><span>batch_effect_covariate_ranges :</span></dt><dd>{np.str_(&#x27;sex&#x27;): {&#x27;1&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 9.21, &#x27;max&#x27;: 78.0}}, &#x27;0&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 79.0}}}, np.str_(&#x27;site&#x27;): {&#x27;AnnArbor_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 13.41, &#x27;max&#x27;: 40.98}}, &#x27;AnnArbor_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Atlanta&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 57.0}}, &#x27;Baltimore&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 40.0}}, &#x27;Bangor&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 38.0}}, &#x27;Beijing_Zang&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 26.0}}, &#x27;Berlin_Margulies&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 23.0, &#x27;max&#x27;: 44.0}}, &#x27;Cambridge_Buckner&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 30.0}}, &#x27;Cleveland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 24.0, &#x27;max&#x27;: 60.0}}, &#x27;ICBM&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 19.0, &#x27;max&#x27;: 79.0}}, &#x27;Leiden_2180&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 27.0}}, &#x27;Leiden_2200&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 18.0, &#x27;max&#x27;: 28.0}}, &#x27;Milwaukee_b&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 44.0, &#x27;max&#x27;: 65.0}}, &#x27;Munchen&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 63.0, &#x27;max&#x27;: 74.0}}, &#x27;NewYork_a&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 7.88, &#x27;max&#x27;: 49.16}}, &#x27;NewYork_a_ADHD&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.69, &#x27;max&#x27;: 50.9}}, &#x27;Newark&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 39.0}}, &#x27;Oulu&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 23.0}}, &#x27;Oxford&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 35.0}}, &#x27;PaloAlto&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 22.0, &#x27;max&#x27;: 46.0}}, &#x27;Pittsburgh&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 25.0, &#x27;max&#x27;: 47.0}}, &#x27;Queensland&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 20.0, &#x27;max&#x27;: 34.0}}, &#x27;SaintLouis&#x27;: {np.str_(&#x27;age&#x27;): {&#x27;min&#x27;: 21.0, &#x27;max&#x27;: 29.0}}}}</dd></dl></div></li></ul></div></div>



Or, if we want to select only the WM-hypointensities, we can use:

.. code:: python

   norm_data.sel(response_vars="WM-hypointensities")

This will return a new NormData object with only the WM-hypointensities.

.. code:: ipython3

    norm_data.sel(response_vars="WM-hypointensities").statistics




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
    </style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;statistics&#x27; (statistic: 13)&gt; Size: 104B
    array([ 1.17178773e-01,  1.40960628e+01,  1.55120188e-01,  3.98458621e-01,
            1.35593870e+00, -6.29998378e-02,  1.17178773e-01,  6.26199688e+02,
            4.38152477e-03,  8.85849246e-01,  8.82821227e-01,  8.47148188e-01,
            2.46368107e+00])
    Coordinates:
      * statistic      (statistic) &lt;U8 416B &#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;
        response_vars  &lt;U22 88B &#x27;WM-hypointensities&#x27;</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-obj-name'>&#x27;statistics&#x27;</div><ul class='xr-dim-list'><li><span class='xr-has-index'>statistic</span>: 13</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-ebd117be-9a30-40bc-b550-43fc589bc133' class='xr-array-in' type='checkbox' checked><label for='section-ebd117be-9a30-40bc-b550-43fc589bc133' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0.1172 14.1 0.1551 0.3985 1.356 ... 0.8858 0.8828 0.8471 2.464</span></div><div class='xr-array-data'><pre>array([ 1.17178773e-01,  1.40960628e+01,  1.55120188e-01,  3.98458621e-01,
            1.35593870e+00, -6.29998378e-02,  1.17178773e-01,  6.26199688e+02,
            4.38152477e-03,  8.85849246e-01,  8.82821227e-01,  8.47148188e-01,
            2.46368107e+00])</pre></div></div></li><li class='xr-section-item'><input id='section-91dfea68-8993-4f4a-a423-52f1d6bbf7bc' class='xr-section-summary-in' type='checkbox' checked /><label for='section-91dfea68-8993-4f4a-a423-52f1d6bbf7bc' class='xr-section-summary' title='Expand/collapse section'>Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>statistic</span></div><div class='xr-var-dims'>(statistic)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;EXPV&#x27; &#x27;Kurtosis&#x27; ... &#x27;Skewness&#x27;</div><input id='attrs-14fa77f2-77ff-4990-b642-83d043c0b31a' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-14fa77f2-77ff-4990-b642-83d043c0b31a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-67c8e6b4-fb3e-4245-bedd-c1a230024c24' class='xr-var-data-in' type='checkbox'><label for='data-67c8e6b4-fb3e-4245-bedd-c1a230024c24' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([&#x27;EXPV&#x27;, &#x27;Kurtosis&#x27;, &#x27;MACE&#x27;, &#x27;MAPE&#x27;, &#x27;MLL&#x27;, &#x27;MSLL&#x27;, &#x27;R2&#x27;, &#x27;RMSE&#x27;, &#x27;Rho&#x27;,
           &#x27;Rho_p&#x27;, &#x27;SMSE&#x27;, &#x27;ShapiroW&#x27;, &#x27;Skewness&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>response_vars</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>&lt;U22</div><div class='xr-var-preview xr-preview'>&#x27;WM-hypointensities&#x27;</div><input id='attrs-dc0c6dea-6ee3-4ccd-b23a-2f8ea594e4ec' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-dc0c6dea-6ee3-4ccd-b23a-2f8ea594e4ec' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7ce89a53-2bab-4375-8f78-0da9e68263c4' class='xr-var-data-in' type='checkbox'><label for='data-7ce89a53-2bab-4375-8f78-0da9e68263c4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(&#x27;WM-hypointensities&#x27;, dtype=&#x27;&lt;U22&#x27;)</pre></div></li></ul></div></li></ul></div></div>



Now we can use the to_dataframe method to cast that selection back to a
pandas dataframe.

.. code:: ipython3

    new_df = norm_data.sel(response_vars="WM-hypointensities").to_dataframe()
    new_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    
        .dataframe thead tr:last-of-type th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>logp</th>
          <th>Yhat</th>
          <th colspan="2" halign="left">batch_effects</th>
          <th>subject_ids</th>
          <th colspan="5" halign="left">centiles</th>
        </tr>
        <tr>
          <th></th>
          <th>age</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>WM-hypointensities</th>
          <th>sex</th>
          <th>site</th>
          <th>subject_ids</th>
          <th>(WM-hypointensities, 0.05)</th>
          <th>(WM-hypointensities, 0.25)</th>
          <th>(WM-hypointensities, 0.5)</th>
          <th>(WM-hypointensities, 0.75)</th>
          <th>(WM-hypointensities, 0.95)</th>
        </tr>
        <tr>
          <th>observations</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>25.63</td>
          <td>1686.7</td>
          <td>0.742311</td>
          <td>-1.158222</td>
          <td>1209.579100</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>0</td>
          <td>152.348837</td>
          <td>776.050569</td>
          <td>1209.579100</td>
          <td>1643.107630</td>
          <td>2266.809363</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.34</td>
          <td>1371.1</td>
          <td>0.445220</td>
          <td>-0.982056</td>
          <td>1084.867051</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>1</td>
          <td>27.387682</td>
          <td>651.236372</td>
          <td>1084.867051</td>
          <td>1518.497730</td>
          <td>2142.346419</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.20</td>
          <td>1414.8</td>
          <td>0.224270</td>
          <td>-0.907844</td>
          <td>1270.652078</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>2</td>
          <td>213.436546</td>
          <td>837.129588</td>
          <td>1270.652078</td>
          <td>1704.174569</td>
          <td>2327.867611</td>
        </tr>
        <tr>
          <th>3</th>
          <td>31.39</td>
          <td>1830.6</td>
          <td>0.812878</td>
          <td>-1.213105</td>
          <td>1308.117015</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>3</td>
          <td>250.875616</td>
          <td>874.583918</td>
          <td>1308.117015</td>
          <td>1741.650112</td>
          <td>2365.358414</td>
        </tr>
        <tr>
          <th>4</th>
          <td>13.58</td>
          <td>1642.4</td>
          <td>0.993572</td>
          <td>-1.376842</td>
          <td>1003.436412</td>
          <td>1</td>
          <td>AnnArbor_a</td>
          <td>4</td>
          <td>-54.364209</td>
          <td>569.674000</td>
          <td>1003.436412</td>
          <td>1437.198824</td>
          <td>2061.237034</td>
        </tr>
      </tbody>
    </table>
    </div>



This should give you a pretty good overview of how to work with
NormData. Most of the functionality is built on top of xarray, so if you
want to learn more about xarray, you can check out the `xarray
documentation <https://docs.xarray.dev/en/stable/>`__. However, the
Xarray.DataSet class does not officially support being extended, so the
API does not work completely as expected.

If you have any suggestions for improvements, please let us know!

Pre-processing and split datasets
---------------------------------

Sometimes we have a dataset that is pre-split into train and test, and
we want to use that exact data split to fit the model. We can then load
the data into two NormData objects, but we have to make sure that the
two datasets are compatible. This will ensure that the fitted model is
applicable to both of them.

.. code:: ipython3

    # Download an example dataset:
    data = pd.read_csv(
        "https://raw.githubusercontent.com/predictive-clinical-neuroscience/PCNtoolkit-demo/refs/heads/main/data/fcon1000.csv"
    )
    # Create an arbitrary split as a placeholder for a predefined split ()
    train, test = train_test_split(data, test_size=100)

.. code:: ipython3

    # specify the column names to use
    covariates = ["age"]
    batch_effects = ["sex", "site"]
    response_vars = ["WM-hypointensities", "Left-Lateral-Ventricle", "Brain-Stem"]
    
    # create NormData objects
    norm_train = NormData.from_dataframe(
        name="train", dataframe=train, covariates=covariates, batch_effects=batch_effects, response_vars=response_vars
    )
    norm_test = NormData.from_dataframe(
        name="test", dataframe=test, covariates=covariates, batch_effects=batch_effects, response_vars=response_vars
    )


.. code:: text

    Process: 2356 - 2026-09-25 15:32:10 - Dataset "train" created.
        - 978 observations
        - 978 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (23)
        
    Process: 2356 - 2026-09-25 15:32:10 - Dataset "test" created.
        - 100 observations
        - 100 unique subjects
        - 1 covariates
        - 3 response variables
        - 2 batch effects:
        	sex (2)
    	site (18)
        


.. code:: ipython3

    # Should print false, because the train and test split do not contain the same sites
    print(norm_train.check_compatibility(norm_test))


.. code:: text

    True


.. code:: ipython3

    norm_train.check_compatibility(norm_test)




.. code:: text

    True



.. code:: ipython3

    norm_train.make_compatible(norm_test)

.. code:: ipython3

    norm_train.check_compatibility(norm_test)




.. code:: text

    True


