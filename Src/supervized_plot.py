# -*- coding: utf-8 -*-
"""
@author: AnicetSIÉWÉ KOUÉTA: 20172760
@author: Benjamin BERGÉ: 20182608
@author: Jolan WATHELET: 20182628
"""

from dask_ml.model_selection import train_test_split
import os
import dask.dataframe as dd
import dask.array as da
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from dask_ml.feature_extraction.text import CountVectorizer
#from sklearn.metrics import accuracy_score
from dask import delayed
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

from dask.distributed import Client
from supervized_train_predict import build_and_score_clfs
from build_labelized_data import build_csv_dataset 
import numpy as np

PARQUET_PATH = "../Data"
PARQUET_FILES = PARQUET_PATH + '/*.parq'
OUTPUT_PATH = "../Output/Supervized"
TRAINING_DATASET_PATH = OUTPUT_PATH + '/training_dataset.csv'
MAX_FEATURE = 250
VISU_PATH = OUTPUT_PATH+"/"

def gen_base_tdif_plot(x, name, ascending=True, nbr=20):
    # encode strings to see how many instances of each bigram in each screen name (or domain name)
    # I thik there is a bug with dask_ml vectorizer
    vectorizer = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='char', ngram_range=(2,2))
    txt_fitted = vectorizer.fit(x.values)
    txt_transformed = txt_fitted.transform(x.values)
    
    #feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    
    with ProgressBar():
        idf = vectorizer.idf_ #Inverse Document Frequency(idf) per token
        rr = dict(zip(txt_fitted.get_feature_names(), idf))
        token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()
        token_weight.columns=('token','weight')
        token_weight = token_weight.sort_values(by='weight', ascending=ascending).head(nbr)
        
        
        #this is to pick a color palette
        data = token_weight['weight']
        pal = sns.color_palette("Greens_d", len(data))
        if ascending:
            rank = data.argsort()[::-1]      
        else:
            rank = data.argsort()[::-1]  
               
        sns.barplot(x='token', y='weight', data=token_weight, palette=np.array(pal[::-1])[rank]) 
        if ascending:
            plt.title("Document Frequency(idf) per token")  
        else:
            plt.title("Inverse Document Frequency(idf) per token")       
        
        fig=plt.gcf()
        fig.set_size_inches(20,10)
        plt.show()
        fig.savefig(VISU_PATH+name+".pdf", bbox_inches='tight')
        


        
def most_informative_feature_for_class(vectorizer, classifier, classlabel=1, n=30):
    #label id =0 car on a pas de classe/ seuelemnt 1
    labelid = 0#list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]
    
    """
    for coef, feat in topn:
        print( classlabel," ",  feat," ", coef," ")
    """
        
    df = pd.DataFrame(topn, columns = ['Coefficient', 'Token']) 
    
    #this is to pick a color palette
    data = df['Coefficient']
    pal = sns.color_palette("Reds_d", len(data))
    rank = data.argsort()[::-1]         
    
    sns.barplot(x='Token', y='Coefficient', data=df,palette=np.array(pal[::-1])[rank])          
    plt.title("Top meaningful features")
    fig=plt.gcf()
    fig.set_size_inches(20,10)
    plt.show()
    fig.savefig(VISU_PATH+"top_coefficient.pdf", bbox_inches='tight')
    
        

"""
    TF-IDF: term frequency-inverse document frequency
"""       
def highest_numbers():
    from build_labelized_data import build_csv_dataset
    if not os.path.exists(TRAINING_DATASET_PATH):
        build_csv_dataset()
    dataset = dd.read_csv(TRAINING_DATASET_PATH, sep=',',names=('Domain', 'Bot'))
    
    x = dataset[dataset.Bot < 1].Domain.compute()
    gen_base_tdif_plot(x, "top_20_from_real_sites")
    gen_base_tdif_plot(x, "bottom_20_from_real_sites", ascending=False)
    
    x = dataset[dataset.Bot > 0].Domain.compute()
    gen_base_tdif_plot(x, "top_20_from_fake_sites")
    gen_base_tdif_plot(x, "bottom_20_from_fake_sites", ascending=False)
    

if __name__ == "__main__":  
    if not os.path.exists(VISU_PATH):
        os.makedirs(VISU_PATH)
    if not os.path.exists(TRAINING_DATASET_PATH):
        build_csv_dataset()
    
    
    highest_numbers()
    
    clf, vectorizer, x,y, dataset = build_and_score_clfs()
    most_informative_feature_for_class(vectorizer, clf)
    
    
    
    
    #https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html

    
    """
        print(x)
        
        
        with ProgressBar():
            #seaborn.scatterplot(x="Temp", y="Summons Number",data=x, ax=ax)
            dataset = dataset.compute() #becomes a pd dataframe
            print(dataset)
            sns.pairplot(data=dataset, hue = 'Bot')
            #plt.ylim(ymin=0)
            #plt.xlim(xmin=0)
            
    """
    
    
