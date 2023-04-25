# disaster-tweets

These are some notebooks associated with the Kaggle ["Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started) competition.

# Structure

**src/shared_code** - Various Python code files I use to effectively share code across notebooks

**src/notebooks/eda_and_clean** - Notebooks for exploring the data and looking at effects of various text cleaning operations

**src/notebooks/models** - Notebooks implementing various approaches to classify tweets as either "disaster tweets" or not.

# Setup

The notebooks have been submitted in a "run" state, so can largely be browsed simply through GitHub or (preferably) through a Jupyter notebook environment.

To setup the environment to run these notebooks on linux, assuming pyenv is installed, see the commands in "install_notes.sh". In addition to this, the required competition data needs to be downloaded and unzipped into the raw_data folder. For example, training data should be located at "raw_data/train.csv".

To actually run the notebooks will also sometimes require certain configuration variables (which appear in the second "code" cells) to be changed. This is because I tend to write data to disk on initial runs, to avoid expensive fitting everytime a notebook is run. The relevant variables should be reasonably obvious and called things like "RERUN_ALL = False".
