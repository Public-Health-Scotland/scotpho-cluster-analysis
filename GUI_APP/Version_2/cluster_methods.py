# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:43:12 2019

@author: panagn01
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from kneed import KneeLocator
from sklearn.decomposition import PCA
import warnings
import time
from sklearn.cluster import DBSCAN
from itertools import chain
sns.set() 
warnings.filterwarnings('ignore')



## df is the dataframe, for our example Final
## results are the kmeans.labels_ after Clustering
def display_indicators_per_cluster_0(df, results):
    cluster_n = np.unique(results).size
    returner = [ [] for l in range(cluster_n) ]
    for i in range(0,len(results)):
        returner[results[i]].append(df.loc[i,'indicator'])
    return returner

def display_indicators_per_cluster_1(df, results):
    cluster_n = np.unique(results).size
    returner = [ [] for l in range(cluster_n) ]
    for i in range(0,len(results)):
        ind = df.loc[i,'indicator']
        area_name = df.loc[i,'area_name']
        area_type = df.loc[i,'area_type']
        year = df.loc[i,'year']
        dict_ = {'Cluster': results[i], 'indicator' : ind, 'area_name' : area_name, 'area_type': area_type, 'year': year}
        returner[results[i]].append(dict_)
    dfaki = pd.DataFrame(list(chain.from_iterable(returner)))
    return dfaki



def kmeans_elbow(points, range_, title):
    scaler = MinMaxScaler()
    points_scaled = scaler.fit_transform(points)

    inertia = []
    clusters_n = range(1,range_)
    for k in clusters_n:
        kmeans = KMeans(n_clusters = k, random_state= 5221)
        kmeans.fit(points_scaled)
        y_km = kmeans.predict(points)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10,6))
    plt.plot(clusters_n, inertia,)
    plt.scatter(clusters_n,inertia, marker = 'x', c='r', s = 100, label = 'Inertia')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k ('+ title+' )')
    plt.show()
    kn = KneeLocator(clusters_n, inertia, S=2.0,  curve='convex', direction='decreasing')
    return kn.knee

def PCA_for_kmeans(points,n_components_ ):
    
    pca = PCA(n_components = n_components_ )
    pca.fit(points)  
    #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    #print('Explained variation for all principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    
    pca_results = pca.transform(points)
    Pca_df = pd.DataFrame(
    {'first feature': pca_results[:,0],
     'second feature': pca_results[:,1]
    })
    return Pca_df

def Visualise_after_reduction(df,data_df, title):
    
    rndperm = np.random.permutation(data_df.shape[0])
    plt.figure(figsize=(10,6))
    plt.title('Data After Reduction ( '+title+' )')
    sns.scatterplot(
    x="first feature", y="second feature",
    
    palette=sns.color_palette("muted", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
    
    

def tsne_on_data(datapoints,n_components_):   
    time_start = time.time()
    tsne = TSNE(n_components= n_components_, verbose=1, perplexity=40, n_iter=300, random_state= 5432)
    tsne_results = tsne.fit_transform(datapoints)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    tsne_df = pd.DataFrame(
    {'first feature': tsne_results[:,0],
     'second feature': tsne_results[:,1]
    })
    return tsne_df    
    

def PCA_Kmeans_elbow(points, range_, title):
    points_ = PCA_for_kmeans(points,2)
    return kmeans_elbow(points_.values,range_, title)



#INPUT:  the Scaled whole dataset
#output the labels and the centroids
#how: uses the function of PCA_for_Kmeans in order to produce the dimensional reduction
#      then we use those data to perform a kmeans
# we should know the number of Clusters, we can choose that by the Elbow Method
def PCA_Kmeans(points, clusters_):
 
    points_ = PCA_for_kmeans(points,2)
    #print(points_)
    
    kmeans = KMeans(n_clusters = clusters_, random_state= 5221)
    kmeans.fit(points_)
    y_km = kmeans.predict(points_)
    points_['labels'] = y_km
    return { 'centroids': kmeans.cluster_centers_ , 'DF': points_}


def TSNE_Kmeans_elbow(points, range_, title):
    points_ = tsne_on_data(points,2)
    return kmeans_elbow(points_.values,range_, title)
    


def TSNE_Kmeans(points, clusters_):
    
    points_ = tsne_on_data(points,2)
    kmeans = KMeans(n_clusters = clusters_, random_state= 5221)
    kmeans.fit(points_)
    y_km = kmeans.predict(points_)
    points_['labels'] = y_km
    
    return { 'centroids': kmeans.cluster_centers_ , 'DF': points_}
    
    
def Visualise_Clusters(points, center, labels, title,subplot):
    subplot.cla()
    subplot.scatter(points[:, 0], points[:, 1], c = labels, s=60, cmap='viridis')
    subplot.scatter(center[:, 0], center[:, 1], c='black', s=250, alpha=0.6)
    subplot.set_title('Clustering Result ( '+title+' )', fontsize = 12.0,fontweight="bold")
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_xlabel('First Feature', fontsize = 12.0)
    subplot.set_ylabel('Second Feature', fontsize = 12.0)
    
    
def Visualise_Clusters_dbscan(points,clusters, title, subplot):
    subplot.scatter(points[:, 0], points[:, 1], c = clusters, s=60, cmap = "plasma")
    subplot.set_title('Clustering Result ( '+title+' )', fontsize = 12.0,fontweight="bold" )
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_xlabel('First Feature', fontsize = 12.0)
    subplot.set_ylabel('Second Feature', fontsize = 12.0)
    


def dbscan_clustering(points):
    dbscan = DBSCAN(eps=0.30, min_samples = 2)
    #res_pca.values
    clusters = dbscan.fit_predict(points)
    return clusters

def kmeans_kneed(points, range_):
    inertia = []
    # maximum 20 clusters. There is no reason for more. We would not be able to evaluate.
    clusters_n = range(1,min(20, range_))
    for k in clusters_n:
        kmeans = KMeans(n_clusters = k, random_state= 5221)
        kmeans.fit(points)
        y_km = kmeans.predict(points)
        inertia.append(kmeans.inertia_)
    kn = KneeLocator(clusters_n, inertia, S=2.0,  curve='convex', direction='decreasing')
    if (kn.knee == None):
        return range_//2
    return kn.knee


#dataset = With_Intervals = (Final.loc[(Final['has_intervals']== True)])
def select_df(datset, area_type, area_name, year):
    df_ = datset.loc[(datset['year']==year) & ( datset['area_name'] == area_name) & (datset['area_type'] == area_type)]
    return df_

def selective_clustering_kmeans(dataset,area_type, area_name, year,option,subplot):
    input_ = select_df(dataset,area_type, area_name, year)
    print(input_.shape)
    input_ = input_.reset_index(drop=True)
    points_ = input_.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
    scaler = MinMaxScaler()
    points_scaled_all = scaler.fit_transform(points_)
    if(input_.shape[0] > 2 ):
            points_scaled_all =np.nan_to_num(points_scaled_all)
            res_pca_all = PCA_for_kmeans(points_scaled_all,2)
            clusters = kmeans_kneed(res_pca_all.values, input_.shape[0])
            results_kmeans_pca = PCA_Kmeans(points_scaled_all,clusters)
            df_pca = results_kmeans_pca['DF']
            points_pca = df_pca[['first feature','second feature']]
            labels_pca = df_pca['labels']
            centroids_pca = results_kmeans_pca['centroids']
            to_Export = display_indicators_per_cluster_1(input_, labels_pca)
            Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA and Kmeans at '+ area_type+ ' '+area_name+' '+str(year),subplot )
            #save clusters?
            if(option == 'yes'):
                to_Export = display_indicators_per_cluster_1(input_, labels_pca)
                to_Export.to_excel('Kmeans_'+area_type+' '+area_name+' '+str(year)+'.xlsx')
    else:
        print('sorry not enough data')
            
            
def selective_clustering_dbscan(dataset,area_type, area_name, year,option,subplot):
            #DBSCAN
            input_ = select_df(dataset,area_type, area_name, year)
            print(input_.shape)
            input_ = input_.reset_index(drop=True)
            points_ = input_.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
            scaler = MinMaxScaler()
            points_scaled_all = scaler.fit_transform(points_)
            if(input_.shape[0] > 2 ):
                points_scaled_all =np.nan_to_num(points_scaled_all)
                res_pca_all = PCA_for_kmeans(points_scaled_all,2)
                dbscan = DBSCAN(eps=0.123, min_samples = 2)
                clusters = dbscan.fit_predict(res_pca_all.values)
                Visualise_Clusters_dbscan(res_pca_all.values, clusters , 'PCA and DBSCAN at '+ area_type+ ' '+area_name+' '+str(year),subplot )
                if(option == 'yes'):
                    to_Export = display_indicators_per_cluster_1(input_, clusters)
                    to_Export.to_excel('DBSCAN_'+area_type+' '+area_name+' '+str(year)+'.xlsx')

            else:
                print('sorry not enough Data')

   