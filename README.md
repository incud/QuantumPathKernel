# Path Kernel for Quantum Neural Networks

## Dependencies 

The code has been tested for Python 3.9.7.

See file `requirement.txt` for the bare minimum dependencies (generated 
using `pipreqs . --force`) and, if that does not work, use 
`requirements2.txt` (generated using `pip3 freeze > requirements2.txt`).

## Available datasets

The datasets are located into `downloaded_datasets` folder. We have used
`OpenML` platform to retrieve the dataset and its metadata. Moreover,
we have used `sklearn` to perform PCA in order to reduce the number
of features of some datasets, and we have arranged them into a `Pandas`'s 
`DataFrame` whose columns are the features plus an additional column
named `target`. All datasets are saved into `pickle` format, in order
to load them you can use `Pandas` but please *check* the version is the same
as the one we use. 

The available datasets are:
* `haberman`:
* `pima`:
* `pima-4PCA`: 4 principal components of `pima`
* `pima-6PCA`: 6 principal components of `pima`
* `wine`:
* `wine-4PCA`: 4 principal components of `wine`
* `wine-6PCA`: 6 principal components of `wine`
* `wine-8PCA`: 8 principal components of `wine`

You can re-generate the dataset by running, within the 
`downloaded_datasets` folder, the Python file `download_datasets.py`. 
Moreover, a tutorial showing how to interact with `OpenML` is 
included in `openml_tutorial.py`.

## Configurations

The configuration file for the project is `config.json` and 
contains the following properties:
* `EXPERIMENTS_FOLDER`: string, where to save the experiments.
* `DOWNLOADED_DATASET_FOLDER`: string, the directory containing the datasets. Must be set to `downloaded_datasets`.
* `CODEFILE_NAME`: the name of the file that will be used to track which version of the code was used for each experiment.

The configuration is loaded into `config_loader.py` that implements also
utilities to load json files. You can keep the default values. 

## Experiment specifications
  
In order to run an experiment, a certain specification file (in json
format) must be compiled. The file must contain the following 
fields:
* `DATASET_NAME`: "haberman",
* `DATASET_SHUFFLE_SEED`: 551,
* `DATASET_PERCENT`: 0.015,
* `DATASET_TEST_PERCENTAGE`: 0.5,
* `DATASET_TEST_SEED`: 54646,
* `MAX_LAYERS`: 2,
* `TRAINING_EPOCHS`: 3,
* `TRAINING_BATCH_SIZE`: 2,
* `OPTIMIZER_STR`: "AdamOptimizer(0.5, beta1=0.9, beta2=0.999)"

You must have *exactly* these fields. The code check its correctness.
Please save all the specification files in `experiments_specifications` 
folder. The simplest example of working specification file 
is `simple_haberman.json`.

**Warning**: the code *can* fail if the (subset of) dataset has 
one unique class. In that case you will see the error 
*"The set of labels must contain the exact two labels 0,1"*.

## How to run the code

To run the code please move into the main folder and run:
```
python main.py experiments_specifications/spec1.json   \
              [experiments_specifications/spec2.json]  \
              ...                                      \
              [experiments_specifications/specN.json]
```

## Experiment output

## Code documentation

### Kernel methods

### Quantum circuits, quantum kernels and quantum neural networks

### Main loop

### Parallel execution

## TODO

* Add jax instead of numpy