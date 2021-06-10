
from pandas import read_parquet
import os
from unsupervized_utils.utils import *
from unsupervized_utils.tsne import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer

PARQUET_PATH = "../Data"
OUTPUT_PATH = "../Output/Unsupervized"
PARQUET_FILES = PARQUET_PATH + str(os.path) + '*.parq'
parquets = read_parquet(PARQUET_PATH, columns=['id','qname','ttl','req_len','res_len','server'])

parquets2 = parquets.copy()
parquets2 = parquets2.iloc[0:2000,:]
id_items = parquets["id"]
k=2


def chr_to_dec(strs):
    value = 0
    for i in range(len(strs)):
        value = value + (ord(strs[i]) * (i + 1))

    return value

def transforme (parquets, feature):
    i = 0
    for value1 in parquets[feature]:
        parquets.loc[i, feature] = chr_to_dec(value1)
        i += 1

def save_results(parquet, labels_, statut, filename):

    with open(filename, 'w') as fichier:
        for i in range(0, len(labels_)):
            if labels_[i] == statut:
                fichier.write("| %10s | %10s | %5s | %5s| %5s | %20s \n" % (
                parquet.iloc[i][0], parquet.iloc[i][1], parquet.iloc[i][2], parquet.iloc[i][3], parquet.iloc[i][4], parquet.iloc[i][5]))

def create_dirs():
    if not os.path.exists(PARQUET_PATH):
        os.makedirs(PARQUET_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    

if __name__ == "__main__":  
    create_dirs()
    i = 0
    for value1 in parquets2.qname:
        parquets2.loc[i, 'qname'] = chr_to_dec(value1)
        i += 1
    
    j = 0
    for value2 in parquets2.server:
        parquets2.loc[j, 'server'] = chr_to_dec(value2)
        j += 1
    
    
    data_scaled = preprocessing.scale(parquets2)
    data_scaled = np.nan_to_num(data_scaled)
    data_2D = tsne(data_scaled, perplexity = 15)
    
    #construction du graphe pour visualiser les données 2D
    plt.scatter(data_2D[:,0],data_2D[:,1], label='True Position')
    plt.title("data visualization")
    plt.savefig(OUTPUT_PATH + '/data_visualization.png',dpi=200)
    
    
    kmeans = KMeans(n_jobs=-1, n_clusters=k, init='k-means++')
    
    
    """
    Show obtimal number of cluster - elbow method
    conda install -c districtdatalabs yellowbrick
    from yellowbrick.cluster import KElbowVisualizer
    """
    visualizer = KElbowVisualizer(kmeans, k=(2,12))
    visualizer.fit(data_2D)        # Fit the data to the visualizer
    #visualizer.show()        # Finalize and render the figure
    
    
    k = visualizer.elbow_value_
    print(k)
    
    kmeans = KMeans(n_jobs=-1, n_clusters=k, init='k-means++')
    kmeans.fit(data_2D)
    show_annotated_clustering(data_2D, kmeans.labels_, id_items)
    plt.title("distribution by cluster")
    plt.savefig(OUTPUT_PATH + '/distribution_by_cluster.png',dpi=200)
    
    
    for i in range(k):
        save_results(parquets, kmeans.labels_, i, OUTPUT_PATH +'/cluster_'+str(i)+'.txt')


