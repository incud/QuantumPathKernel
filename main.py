"""
Main file for noiseless simulation of NTK / Path Kernel
"""
from datetime import datetime
import git
from multiprocessing import Process
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import sys

from config_loader import get_config, read_json, write_json
from kernel_helper import build_gram_matrix, build_gram_matrix_of_classical_kernels, get_metrics_for_all_kernels
from pennylane_fixed_qubit_circuits import zz_kernel
from main_training import run_path_kernel_process


def check_specification_dict(s_dict):
    ALLOWED_FIELDS = set("DATASET_NAME", "DATASET_SHUFFLE_SEED", "DATASET_PERCENT", "DATASET_TEST_PERCENTAGE",
                         "DATASET_TEST_SEED", "MAX_LAYERS", "TRAINING_EPOCHS", "TRAINING_BATCH_SIZE", "OPTIMIZER_STR")
    return s_dict.keys() == ALLOWED_FIELDS


def run_experiment_file(specifications_dict):
    """
    Run the experiment whose specification are in the specified json file
    :param specifications_dict: dictionary containing the specifications
    :return: None
    """
    # create a sub-directory in "EXPERIMENTS_FOLDER" folder
    experiment_subfolder = datetime.now.strftime("%Y-%m-%d--%H-%M-%S")
    experiment_folder = get_config("EXPERIMENTS_FOLDER") + "/" + experiment_subfolder
    os.mkdir(experiment_folder)

    # save codeversion inside the experiment folver
    with open(f"{experiment_folder}/{get_config('CODEFILE_NAME')}") as f:
        repo = git.Repo(search_parent_directories=True)
        f.write(repo.head.object.hexsha)

    # save configuration file inside the experiment folder
    write_json(specifications_dict, "specifications.json")

    # load the dataset
    dataset_name = specifications_dict["DATASET_NAME"]
    dataset_path = f"{get_config('DOWNLOADED_DATASET_FOLDER')}/{dataset_name}.pickle"
    dataset_df = pd.read_pickle(dataset_path)
    y = dataset_df['target'].to_numpy()
    X = dataset_df.drop('target', axis=1).to_numpy()

    # preprocess dataset (normalize, choose a subset of items, etc...)
    X = MinMaxScaler().fit_transform(X)

    # shuffle dataset in order to not test always the same training and testing set
    X, y = shuffle(X, y, random_state=int(specifications_dict["DATASET_SHUFFLE_SEED"]))

    # split into training and testing dataset
    test_percentage = int(len(y) * float(specifications_dict["DATASET_TEST_PERCENTAGE"]))
    test_seed = int(specifications_dict["DATASET_TEST_SEED"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=test_seed)

    # test classical kernels
    build_gram_matrix_of_classical_kernels(X_train, X_test, experiment_folder)

    # test ZZFeatureMap quantum kernel
    build_gram_matrix(zz_kernel, X_train, save_path=f"{experiment_folder}/ZZFeatureMap_train.csv")
    build_gram_matrix(zz_kernel, X_train, X_test, save_path=f"{experiment_folder}/ZZFeatureMap_test.csv")

    # test QNN having Shirai ansatz and layers from 1 to 20
    MAX_LAYERS = int(specifications_dict["MAX_LAYERS"])
    OPTIMIZER_STR = str(specifications_dict["OPTIMIZER_STR"])
    BATCH_SIZE = int(specifications_dict["TRAINING_BATCH_SIZE"])
    EPOCHS = int(specifications_dict["EPOCHS"])
    processes = [Process(
        target=run_path_kernel_process,
        args=(X_train, X_test, y_train, y_test, i+1, OPTIMIZER_STR, BATCH_SIZE, EPOCHS, experiment_folder)) for i in range(MAX_LAYERS)]
    for i in range(MAX_LAYERS):
        processes[i].start()  # start all processes
    for i in range(MAX_LAYERS):
        processes[i].join()  # wait until each process is in READY state (meaning it has terminated)

    # print statistics
    get_metrics_for_all_kernels(experiment_folder, y_train, y_test)


if __name__ == '__main__':
    print("============= PATH KERNEL FOR QNN =============")

    # check input files
    experiments_file = sys.argv[1:]
    assert all(e.endswith(".json") for e in experiments_file), "All files must be JSON"
    assert all(os.path.exists(e) for e in experiments_file), "All files must exists in the current directory"
    print(f"List of experiments configuration file: {experiments_file}")

    # for each experiment file run the procedure
    experiments_dict = [run_experiment_file(read_json(fpath)) for fpath in experiments_file]
    for i, s_dict in enumerate(experiments_dict):
        assert check_specification_dict(s_dict), f"Specification file #{i} has too few or too many fields"
        run_experiment_file(read_json(s_dict))

    print("All experiments are ended successfully")
