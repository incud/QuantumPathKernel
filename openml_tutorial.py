import openml
import pandas as pd
from openml.datasets import get_dataset

# The following code is taken from:
# https://openml.github.io/openml-python/develop/usage.html

# load the list of dataset available
openml_df = openml.datasets.list_datasets(output_format="dataframe")

# you can filter by # of samples, # of features ect
smalldatasetlist_df = openml_df.query("NumberOfFeatures <= 4")

# if you have a certain name in mind, search for that. Then take the ID
haberman_id = int(openml_df.query('name == "haberman"').iloc[0]["did"])
haberman_metadata = openml.datasets.get_dataset(haberman_id)
print(f"The chosen dataset is named '{haberman_metadata.name}'")
print(f"\tTarget feature: '{haberman_metadata.default_target_attribute}'")
print(f"\tURL: '{haberman_metadata.url}'")

# download the data then use it as you please
X, y, categorical_indicator, attribute_names = haberman_metadata.get_data(
    dataset_format="array", target=haberman_metadata.default_target_attribute)
haberman_df = pd.DataFrame(X, columns=attribute_names)
haberman_df["target"] = y
