# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:06:30 2019

@author: 2410873n
"""
import pandas as pd
import numpy as np

dataset_all = pd.read_csv("scotpho_data_extract.csv")
prof_ind = pd.read_excel('Profiles to indicators.xlsx')
#prof_ind.drop(columns=['Unnamed: 0'],inplace= True)
unique_profiles = list(prof_ind['Profile'].unique())

dataset_full = pd.read_csv('dataset_all.csv')



def select_df(df, area_type, area_name, year):
    df_ = df.loc[(df['year'] == year) & ( df['area_name'] == area_name) & (df['area_type'] == area_type)]
    return df_

def top_k(df,k,option):
    #sorting
    #ascending order
    df = df.sort_values(by=['actual_diff'])
    if option == 'better':
        return df[['indicator', 'area_name', 'area_type', 'year', 'actual_diff','definition']].head(k)
    elif option == 'worse':
        return df[['indicator', 'area_name', 'area_type', 'year', 'actual_diff', 'definition']].tail(k)[::-1]
    else:
        print('wrong option')
        


def rank_by_profiles(df,df_profiles,option):
    #sorting
    #ascending order
    
    if option == 'better':
        #the less positive distance the worse
        df = df.sort_values(by=['actual_diff'])
        
    elif option == 'worse':
        # the more positive distance the worst
         df = df.sort_values(by= 'actual_diff' , ascending=False)
    else:
        print('wrong option')
        return
        
    dict_list = [[] for i in range(len(unique_profiles))]
    #main_matrix = df.to_numpy()
    #profiles_ = df_profiles.to_numpy()
    main_matrix = df.as_matrix()
    profiles_ = df_profiles.as_matrix()

    #for every row at dataset check if the indicator match with the profile

    for row in main_matrix:
        for row_prof in profiles_:
            if(row[0]== row_prof[2]):
                index_ = unique_profiles.index(row_prof[1])
                dict_ = {'Indicator': row[0], 'definition': row[10],'area_name': row[1], 'area_type': row[3], 'year':row[4], 'actual_diff': row[12], 'Profile':row_prof[1]}
                dict_list[index_].append(dict_)
    
    return dict_list

def find_the_most_Similar(instance, k):
    #got all the other areas and the indicators for the instance that i check
    checker = dataset_full.loc[dataset_full['year'].isin(instance['year']) & dataset_full['area_type'].isin(instance['area_type']) & dataset_full['indicator'].isin(instance['indicator']) & (~dataset_full['area_name'].isin(instance['area_name'])) ]


    # for every same indicator subtract the value
    merged_checker = checker[['indicator','area_name','area_type','year','measure','Scotland','actual_diff']].merge(instance[['indicator','actual_diff']],on=['indicator'],how='left')

    # I calculate the absolute distance between the whole dataset versus the area. Absolute distance gives us the correct result.
    merged_checker['sim_diff'] = np.abs(merged_checker['actual_diff_x'] - merged_checker['actual_diff_y'])

    group_indicators = merged_checker.groupby('indicator')

    list_of_df_general = []
    list_of_df_indicator = []

    for pair, pair_df in group_indicators:
        list_of_df_general.append((pair_df.sort_values(by=['sim_diff']).head(k)))
        list_of_df_indicator.append((pair_df.sort_values(by=['sim_diff']).head(1)))
        
    top_similar_df = pd.concat(list_of_df_general)
    top_indicator_df = pd.concat(list_of_df_indicator)
    

    
    
    final_top_k_similar_general = top_similar_df.sort_values(by= ['sim_diff'])
    final_top_k_per_indicator = top_indicator_df.sort_values(by= ['sim_diff'])
    
    #make it pretier, set index in asceding order
    val_gen = list(range(1 ,min(k+1,final_top_k_similar_general.shape[0]+1)))
    val_ind = list(range(1 ,min(k+1,final_top_k_per_indicator.shape[0]+1)))
    
    indexaki_gen =  pd.Index(val_gen)
    indexaki_ind =  pd.Index(val_ind)
    
    return final_top_k_per_indicator.head(k).set_index(indexaki_ind)