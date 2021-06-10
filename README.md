# BigDataBotDetection
## Environment installation
This program was developped using Anaconda and Python 3.5.8. 
Several additional package are used:

  - conda install -c conda-forge pyarrow
  - conda install -c conda-forge dask-ml
  - conda install -c conda-forge graphviz
  - conda install -c conda-forge python-graphviz
  - conda install -c districtdatalabs yellowbrick
  
## Folder structure
- Data directory must contains the Parquets files to be analyzed.
- SupervizedTrainingData directory must contains the .txt and .csv that will be read to build a label dataset.
- An output dircetory will be created upon running any code, it will contains:
  - A supervized subfolder with vizualization and a csv file containing the prediction for each requests.
  - A un-supervized subfolder, similar to the last one, except that the output is in a .txt.
- The Src folder contains both the code for the supervized and un-supervized algorithms. 

## Code description
In the Src directory, the following files are present:
- <i>build_labelized_data.py</i>: </br>This code will create a CSV file with the labelized data for training (and scoring). This code is run from the others if the csv doesn't already exist. 
- <i>supervized_train_predict.py</i>: </br>This code will vectorize the domain names from the CSV and train a model with it, it will provide a score (by splitting the csv) and output the predictions in <b>partitioned CSV files</b> <b>using Dask</b>. 
- <i>supervized_plot.py</i>: </br>This code will build a identical model to the previous one and plot the most meaningful features? It also provides visualization on the labelized set, scuh as the most common tuples of charachters. 
- <i>unsupervized_kmeans.py</i>: </br>This code train a model using K-Means, decides the best number of clusters using the elbow method, plot the results and output the predictions in a txt file. 

## Data used to create the labelized dataset
https://we.tl/t-JHED1KvHiI
