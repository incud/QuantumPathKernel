# Path Kernel for Quantum Neural Networks

## Introduction


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
* `haberman`: It has 306 instances. The data represent the survival study of patient undergone to breast cancer. 
There are three legit features: 
    * `Age_of_patient_at_time_of_operation`, 
    * `Patients_year_of_operation`,
    * `Number_of_positive_axillary_nodes_detected`. 
    * The target is `Survival_status` can be 1 (survived $\ge 5$ years) or 2 (dead $<5$ years).
* `pima`: It has 768 instances. Study of Diabetes in Pima Indians population. There are nine features: 
    * `age`,
    * `pedi`,
    * `mass`, 
    * `insu`, 
    * `skin`,
    * `pres`,
    * `plas`,
    * `preg`,
    * The target is 1 (tested negative) or 2 (tested positive). 
* `pima-4PCA`: 4 principal components of `pima`
* `pima-6PCA`: 6 principal components of `pima`
* `wine`: Wine recognition task. It has 178 instances. There are 14 features, including the three-classes target
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

The output of each experiment is saved into a sub-folder of `experiments` named with the timestamp at the moment of 
starting the process. Then there will be generated:
* a file `codefile.version` that contains the actual version of GIT code
* a copy of the specification file in `specifications.json`
* a training trace file `QNN_SHIRAI_layer_XX_training_trace.pickle` for each QNN having 1, 2, ..., N layers
* a couple of csv files `kname_train.csv`, `kname_test.csv` for each kernel, namely:
  * (classical) polynomial kernel for $d=1,2,3,4$
  * (classical) gaussian kernel for $\sigma=.01, 1, 100, 10000$
  * (classical) laplacian kernel for $\sigma=.01, 1, 100, 10000$
  * (quantum) ZZFeatureMap-based kernel
  * (quantum) NTK_SHIRAI (Neural Tangent Kernel with QNN having Shirai's ansatz)
  * (quantum) PK_SHIRAI (Path kernel with QNN having Shirai's ansatz)

## Code documentation

### Kernel methods

`kernel_helper.py` contains the functions generating the Gram Matrix (can be done in a parallel way)
and calculates the efficacy of kernels according to the SVM algorithm

### Quantum circuits, quantum kernels and quantum neural networks

`pennylane_circuits.py` define the components, namely the "ZZ" feature map and Shirai's variational 
ansatz. 

`pennylane_fixed_qubit_circuits.py` define the component `PathKernelSimulator` which must be instanciated with the number of qubits 
that the simulator will have. Then, you can access calculation of "ZZ" kernel, training (multiple) QNNs, 
calculate Neural Tangent and Path Kernels. 

### Main

`main_training.py` contains the function that will be launched as separate process in order to 
study a single QNN (training, NTK and PK). 

`main.py` read the specification file and launch all the processes. 

## Improvements

* JAX does not work on Windows, yet! Some initial support exists (remove comments in ) and the gradient calculations need to be modified too (for details see: https://pennylane.readthedocs.io/en/stable/introduction/interfaces/jax.html).
Only the files `pennylane_circuits.py` and `pennylane_fixed_qubit_circuits.py` are affected
* timestamp inizio e fine computazione