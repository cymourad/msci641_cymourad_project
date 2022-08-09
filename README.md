# Solving Recommender Cold Start Using BERT Embeddings for Product Description

### Spring 22 - MSCI-641 Project

Author: Christian Mourad
Email: cymourad@uwaterloo.ca

## Virtual Environment

We use virtual environment to manage dependencies.

To run it:

```powershell
.\msci641-env\Scripts\Activate.ps1
```

> I followed [this tutorial](https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/)

We also create a jupyter kernel to run form that virtual environment

```
ipython kernel install --user --name=msci641-env
```

We only ran that once, then we choose the kernel in VS Code or Jupyter notebook.

Once we are done the project, we can remove it form the list of jupyter kernels.

```
jupyter-kernelspec uninstall msci641-env
```

## Data

The data is too large to be uploaded onto GitHub. To get around that, I distilled the data that I used frequently into `data/distil_data.csv`. Those are the movies that have been watched and therfore rated enough times.

The entire dataset can be downloaded from [this kaggle dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). It must be downloaded and unzipped into `data/IMBD_Ratings/<data_file>.csv`.

## Code Files

This is a description of what each code file was used for. I will display them on chrnological order to match the order in which they were developped.

- `baseline_explore.ipynb`:
  - This is the file that I created. I used it to explore the raw data files and find out good threshholds for which data to keep.
  - I then used it to make functions to process the raw data into dataframes that I can use.
  - I also used it to develop my testing functions.
  - I used it to develope the different parts of the baseline model.
- `main_baseline.py`: Once I had one baseline working in `baseline_explore.ipynb`, I split the different parts of that pipeline into utility functions in different utility files (outlined below) and used them in for loops to find the best parameters of the baseline.
  - `data_prep.py`: utility functions to load and process the raw data into dataframes that has the data that meets my thresholds.
  - `overview_processing.py`: utility functions to pre-processes overviews into tf-idf.
  - `features.py`: utility functions to extract the actors (a.k.a. cast), production companies, genres and director of a movie and prepare them for a similarity matrix.
  - `similarity.py`: utility function to take matrices and associated weights and turn them into a similarity matrix.
  - `test.py`: utility functions to load raw data and process it into testing datasets and test the recommender using the hits in 100 score.
  - `recommendations.py`: utility functions to make recommendations and make look-up tables to accelerate this process.
- `get_bert_embeddings.py`: used to pre-process movie overviews into BERT tokens, create a multi-label BERT classifier where genres are the classes, and extract embeddings from the last and second last layer of the network
- `final_explore.ipynb`: developed the pipeline to test the recommendations made using BERT embeddings to represent a movie overview.
- `main_bert_embeddings.py`: used a for loop to test the recommendations made by the BERT embeddings and the features with different weights.
- `embed_visualization.ipynb`: used to visualize the embeddings created by BERT to represent the overviews in 2D, and then used to visualize the watching patter of sample users in this space.
- `explore_feature_importance.ipynb`: used to run the hits in 100 test for movie representations made with different combinations of features (the ones extracted in `features.py`) to understand which of them are most influential in movie recommendations.
- `bert_embeddings_for_director.ipynb`: used to explore whether a BERT classifier could be built with directors as the class to be predicted.

## Test Results

### Hits in 100 score using BERT Embeddings as the vector representing the movie

| BERT Layer | Overview Weight | Features Weight | Hits in 100        |
| ---------- | --------------- | --------------- | ------------------ |
| -2         | 1               | 1               | 0.6698442780162572 |
| -2         | 1               | 0               | 0.7132957347702986 |
| -2         | 0               | 1               | 0.9752032149054708 |
| -2         | 2               | 1               | 0.7068567905744817 |
| -2         | 3               | 1               | 0.7072449538770664 |
| -2         | 1               | 2               | 0.5599826468170609 |
| -1         | 1               | 0               | 0.7272810302310714 |
| -1         | 0               | 1               | 1.0020663987578775 |
| -1         | 1               | 2               | 0.4494931043930952 |
| -1         | 1               | 1               | 0.6733377477395196 |
| -1         | 2               | 1               | 0.7342679696775961 |
| -1         | 3               | 1               | 0.7184217736779615 |

### Testing Effect of Different Features on the Hits in 100 Score

| Director | Cast | Genres | Produciton Companies | Hits in 100        |
| -------- | ---- | ------ | -------------------- | ------------------ |
| ❌       | ❌   | ❌     | ✅                   | 0.840499132340853  |
| ❌       | ❌   | ✅     | ❌                   | 1.2906429810941638 |
| ❌       | ❌   | ✅     | ✅                   | 0.936010137912138  |
| ❌       | ✅   | ❌     | ❌                   | 0.9731938989862088 |
| ❌       | ✅   | ❌     | ✅                   | 0.8567334916430724 |
| ❌       | ✅   | ✅     | ❌                   | 1.127271896976893  |
| ❌       | ✅   | ✅     | ✅                   | 0.9689811854963923 |
| ✅       | ❌   | ❌     | ❌                   | 0.8685039729655676 |
| ✅       | ❌   | ❌     | ✅                   | 1.0158576125673577 |
| ✅       | ❌   | ✅     | ❌                   | 1.3216047127591561 |
| ✅       | ❌   | ✅     | ✅                   | 1.0820166225226049 |
| ✅       | ✅   | ❌     | ❌                   | 1.4103913599415472 |
| ✅       | ✅   | ❌     | ✅                   | 0.8843730021006485 |
| ✅       | ✅   | ✅     | ❌                   | 1.339620056626176  |
| ✅       | ✅   | ✅     | ✅                   | 1.0413165585898256 |
