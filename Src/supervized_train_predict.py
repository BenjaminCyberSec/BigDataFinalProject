# -*- coding: utf-8 -*-
"""
@author: AnicetSIÉWÉ KOUÉTA: 20172760
@author: Benjamin BERGÉ: 20182608
@author: Jolan WATHELET: 20182628

https://livebook.manning.com/book/data-science-at-scale-with-python-and-dask/chapter-7/33
sns.pairplot(iris, hue = 'variety')
https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments
"""
from dask_ml.model_selection import train_test_split
import os
import dask.dataframe as dd
import dask.array as da
from sklearn.feature_extraction.text import CountVectorizer
from dask import delayed
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame as pd
import seaborn as sns
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from dask.distributed import Client

PARQUET_PATH = "../Data"
PARQUET_FILES = PARQUET_PATH + '/*.parq'
OUTPUT_PATH = "../Output/Supervized"
TRAINING_DATASET_PATH = OUTPUT_PATH + '/training_dataset.csv'
MAX_FEATURE = 250


def read_and_vectorized_data():
    if not os.path.exists(TRAINING_DATASET_PATH):
        build_csv_dataset()
    dataset = dd.read_csv(TRAINING_DATASET_PATH, sep=',',names=('Domain', 'Bot'))
    
    # separe le 'target' et les 'features' du dataset
    y = dataset.Bot #here bot is the last column
    x = dataset.Domain  #all val except last column    
    
    # encode strings to see how many instances of each bigram in each screen name (or domain name)
    # I thik there is a bug with dask_ml vectorizer
    
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2)) #♪changer pa rapport à (,3) atteint 0.82%
    x = vectorizer.fit_transform(x)  
    
    return x,y,vectorizer, dataset


def build_and_score_clfs():
    x,y,vectorizer,dataset = read_and_vectorized_data()
    #x.visualize()
    
    y = y.compute()
    # creation des variable d'entrainement et de validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)
    
    
    clf = LogisticRegression(max_iter=500,solver="lbfgs").fit(x_train, y_train)
    
    
    print("Score: ",clf.score(x_val, y_val))
    
    return clf, vectorizer, x,y,dataset


def strip_registar_name_map(df):
    strip_registar_name
    df['qname'] = df['qname'].map(lambda x: strip_registar_name(x))
    return df



def predict(clf, vectorizer):
    
    parquet = dd.read_parquet(PARQUET_FILES, columns=['id','qname'], engine='pyarrow')
    #x = df.map_partitions(vectorizer.transform)(df['qname'])
    
    #remove top domain (tjrs .be.)
    pq_qname = parquet['qname'].apply(lambda x: x[:-4], meta=('str'))
    
    """
    parquet.map_partitions(strip_registar_name_map)
    print(pq_qname)
    """
    parquetContentVectorized = vectorizer.transform(pq_qname)
    
    result = clf.predict_proba(parquetContentVectorized)[:,1]
    
    result = da.from_array(result)
    result_df = dd.from_dask_array(result)
    result_df.columns = ['result']
    
    
    df = parquet
    arr = result
    
    # create dask frame and series
    ddf = df
    darr = dd.from_array(arr)
    # give it a name to use as a column head
    darr.name = 'result'
    parquet_with_result = ddf.merge(darr.to_frame())
    
    
    
    evil = parquet_with_result.loc[lambda parquet_with_result: parquet_with_result["result"] >= 0.85, ["qname"]]
    good = parquet_with_result.loc[lambda parquet_with_result: parquet_with_result["result"] < 0.85, ["qname"]]
    evil.to_csv('badDomains.csv')
    good.to_csv('goodDomains.csv')
    print("written to csv")
    
    
    
if __name__ == "__main__":  
    with Client(n_workers=2, threads_per_worker=3, memory_limit='4GB') as client:
        """
        local library must be uploaded to clients
        """
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        
        client.upload_file('build_labelized_data.py')  
        from build_labelized_data import build_csv_dataset
        from build_labelized_data import strip_registar_name

        if not os.path.exists(TRAINING_DATASET_PATH):
            build_csv_dataset()
        clf, vectorizer, x,y, dataset = build_and_score_clfs()

        #predict(clf, vectorizer)
