import openml
import pandas as pd
from sklearn.decomposition import PCA


def download_dataset(the_openml_df, name):
    # retrieve id by name
    the_id = int(the_openml_df.query(f'name == "{name}"').iloc[0]["did"])
    return download_dataset_by_id(name, the_id)


def download_dataset_by_id(the_name, the_id):
    # retrieve metadata from ID
    metadata = openml.datasets.get_dataset(the_id)
    # download data
    X, y, _, attribute_names = metadata.get_data(dataset_format="array", target=metadata.default_target_attribute)
    # create dataframe
    df = pd.DataFrame(X, columns=attribute_names)
    df["target"] = y
    # save dataframe to file
    df.to_pickle(f"{the_name}.pickle")
    # return dataframe to (optionally) postprocess
    return df


def apply_pca(original_df, n_components=4, name=None):
    # remove target from dataframe
    notarget_df = original_df.drop('target', axis=1)
    # initialize pca object
    pca = PCA(n_components=n_components)
    pca.fit(notarget_df)
    # create new table
    columns = ['pca_%i' % i for i in range(n_components)]
    df_pca = pd.DataFrame(pca.transform(notarget_df), columns=columns, index=notarget_df.index)
    df_pca["target"] = original_df["target"]
    if name is not None:
        df_pca.to_pickle(f"{name}.pickle")
    return df_pca


openml_df = openml.datasets.list_datasets(output_format="dataframe")

# Download Haberman (3 feature + target)
download_dataset(openml_df, "haberman")

# Download wine-review (12 features [11 numeric + 1 categorical] + target)
wine_df = download_dataset(openml_df, "wine")
wine_df = wine_df[wine_df['target'] < 2]  # restrict to two classes only
wine_df.to_pickle("wine.pickle")
apply_pca(wine_df, n_components=4, name="wine-4PCA")
apply_pca(wine_df, n_components=6, name="wine-6PCA")
apply_pca(wine_df, n_components=8, name="wine-8PCA")

# Download PIMA (8 features + 1 target)
pima_df = download_dataset_by_id('pima', 40715)
apply_pca(pima_df, n_components=4, name="pima-4PCA")
apply_pca(pima_df, n_components=6, name="pima-6PCA")
