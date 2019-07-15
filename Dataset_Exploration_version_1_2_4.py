#!/usr/bin/env python
# coding: utf-8

# # Data Exploration of Indicators Dataset

# #### In this notebook, I will try to reshape the dataset in order to be easily interpeted

# #### importing the necessary libraries and load the dataset 

# In[3]:


import pandas as pd
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)


# In[ ]:


dataset_all = pd.read_excel("scotpho_data_extract.xlsx")


# In[2]:


pd.set_option('display.max_colwidth', 500)


# In[ ]:





# #### Indicators

# In[85]:



indicator_names = dataset_all['indicator'].unique()
print('Number of Indicators: '+str(len(indicator_names)))
print('Some Indicators are described bellow')
indicator_names_df = pd.DataFrame(indicator_names, columns=['indicator'])
indicator_names_df.head()


# #### Area Name

# In[67]:


area_names = dataset_all['area_name'].unique()
print('Number of area names: '+str(len(area_names)))
print('Some area names are described bellow')
area_names_df = pd.DataFrame(area_names, columns=['area_name'])
area_names_df.head()


# #### Area Type

# In[70]:


area_types = dataset_all['area_type'].unique()[:-1]
print('Number of area types: '+str(len(area_types)))
print(' Area types are described bellow')
area_types_df = pd.DataFrame(area_types, columns=['area_type'])
area_types_df


# #### Area Code

# In[73]:


area_codes = dataset_all['area_code'].unique()
print('Number of area codes: '+str(len(area_codes)))
print(' some area codes are described bellow')
area_codes_df = pd.DataFrame(area_codes, columns=['area_code'])
area_codes_df.head()


# #### Definition

# In[75]:


definitions = dataset_all['definition'].unique()
print('Number of definitions or metrics: '+str(len(definitions)))
print(' Definitions are described bellow')
area_definitions_df = pd.DataFrame(definitions, columns=['definition/metric'])
area_definitions_df


# ## Grouping

# In[93]:


grouped_by_indicator = dataset_all.groupby('indicator')
for indicator, indicator_df in grouped_by_indicator:
    print(indicator)


# In[102]:


grouped_by_indicator_area_type = dataset_all.groupby(['indicator','area_type'])
for pair, pair_df in grouped_by_indicator_area_type:
    print(pair)


# In[94]:


grouped_by_indicator.get_group('COPD deaths')


# In[131]:


grouped_by_indicator_area_type = dataset_all.groupby(['indicator','area_type'])
for pair, pair_df in grouped_by_indicator_area_type:
    print(" ".join(pair))


# In[116]:


grouped_by_indicator_area_type.ngroups


# ### Representation grouped by indicator, area type, and indexed by year

# In[135]:


#Exporting
i=2
with pd.ExcelWriter("new_representation_extract.xlsx") as writer:
    
    for pair, pair_df in grouped_by_indicator_area_type:
        df_1 =pd.DataFrame(pair_df)
        df_1.set_index('year', inplace= True)
        df_1.to_excel(writer,startrow=i)
        worksheet = writer.sheets["Sheet1"]
        worksheet.write(i-1,2," -- ".join(pair))
        i=i+df_1.shape[0]+3
    
    


# In[ ]:


df_1.shape[0]


# ## Load only the Health Board Data for 2017

# In[2]:


df_HB = pd.read_csv('test fo.csv')


# In[3]:


df_HB.head()


# ## Methods that will be used for the Scoring of Indicators

# In[89]:


#this method defines if the area value is statistical different with the comparator.
# arr: measure, lower_CI, upper_CI
def is_statistical_different(vector,comparator):
    #when it is worse, means that the lower interval is greater than the measure of the comparator.
    if(vector[1]> comparator):
        return 1
    #when it better, means the the upper interval is lower than the measure of the comparator.
    if(vector[2]< comparator):
        return 1
    #else we cannot define
    return 0

#for every indicator calculates how many areas(in percentage) are statisticall significant
def score_of_statistical_different(arr, comparator):
    counter =0
    for row in arr:
        counter = counter + is_statistical_different(row, comparator)
    return counter/(arr.shape[0])

#just counts the measures that are greater than the comparator(dont know if that's correct in terms of statistics)
def is_better_for_measure_without_cf(arr,comparator):
    return (np.sum(arr[:,0]>comparator))/(arr.shape[0])
def is_worse_for_measure_without_cf(arr,comparator):
    return (np.sum(arr[:,0]<comparator))/(arr.shape[0])
#better

def is_better(vector,comparator):
    if(vector[2]< comparator):
        return 1
    return 0
    
def is_worse(vector,comparator):
    if(vector[1]> comparator):
        return 1
    return 0

#score in detail
def score_of_statistical_different_detail(arr, comparator):
    counter_better=0
    counter_worse=0
    counter=0
    arr_len = arr.shape[0]
    for row in arr:
        #counter = counter + is_better(row, comparator) + is_worse(row, comparator)
        counter_better = counter_better + is_better(row, comparator)
        counter_worse = counter_worse + is_worse(row,comparator)
        
    #scores[general,better,worse, not difference]
    counter = counter_better + counter_worse
    scores = [counter/arr_len, counter_better/arr_len, counter_worse/arr_len, 1 - (counter/arr_len)]
    return scores
    
        
        
    
    


# In[73]:


#group the data of the healthboard 2017 by indicator
#Calculate the score of statisticall difference.
df_HB_ind = df_HB.groupby('indicator')
df_HB_ind.ngroups

for pair, pair_df in df_HB_ind:
    print(pair)
    #we remove Scotland's value
    train_data = (pair_df.iloc[:-1,7:10].to_numpy())

    a=pair_df.loc[pair_df['area_name'] == 'Scotland']['measure'].item()
    if(not np.isnan(train_data).any()):
        print(score_of_statistical_different(train_data,a))
        print(score_of_statistical_different_detail(train_data,a))
    else:
        print(is_better_for_measure_without_cf(train_data,a))
        print(is_worse_for_measure_without_cf(train_data,a))
    break
        
        


# ##  Generalise for each geographic level

# In[1]:


# Convert Jupyter Notebook to python file.

#!jupyter nbconvert --to script Dataset_Exploration_v1.ipynb


# ## We need to group firstly by indicator, then by Geography Level and Lastly By year

# In[82]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# In[83]:


Scotland_values.tail()


# In[5]:


dataset_all.head()


# In[84]:


print(Scotland_values.loc[(Scotland_values['year']==2013) & ( Scotland_values['indicator'] == 'Child healthy weight in primary 1')]['measure'].item())


# In[85]:


#Search function with input indicator and year and output the comparator value (Scotland value for this Comparator)
def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()

#returns the statistical difference score for a particular indicator,area_type,year

    


# In[12]:


grouped_by_indicator_area_type = dataset_all.groupby(['indicator','area_type','year'])
name_list = []
values_list = []
for pair, pair_df in grouped_by_indicator_area_type:
    #pair[0] indicator name
    #pair[2] year
    comparator = (find_comparator(pair[0], pair[2]))
    #we remove Scotland's value
    train_data = (pair_df.iloc[:,7:10].to_numpy())
   
    if(not np.isnan(train_data).any()):
        stat_diff = (score_of_statistical_different(train_data,comparator))
        stat_diff_detail = (score_of_statistical_different_detail(train_data,comparator))
        #printed_dict = {'indicator': pair[0], 'year': pair[2], 'Scotland Value': comparator,"area_type": pair[1], 'Statistical Different Score':stat_diff, 'better':stat_diff_detail[1] ,'worse':stat_diff_detail[2] , 'not difference':stat_diff_detail[3] }
        values_list.append({'Scotland Value': comparator, 'Statistical Different Score':stat_diff, 'better':stat_diff_detail[1] ,'worse':stat_diff_detail[2] , 'not difference':stat_diff_detail[3]})
    else:
        better = (is_better_for_measure_without_cf(train_data,comparator))
        worse = (is_worse_for_measure_without_cf(train_data,comparator))
        printed_dict = {'indicator': pair[0], 'year': pair[2], 'Scotland Value': comparator, "area_type": pair[1], 'Statistical Different Score':'Cannot say', 'better':better ,'worse':worse , 'not difference': 'Cannot say' }
        values_list.append({'Scotland Value': comparator,  'Statistical Different Score':'Cannot say', 'better':better ,'worse':worse , 'not difference': 'Cannot say'})
    #print(printed_dict)
    name_list.append({'indicator': pair[0], 'year': pair[2],"area_type": pair[1]})

# seperate the dataframes for a feauture use (eg Numpy array for clustering)
df_name = pd.DataFrame(name_list)
df_values = pd.DataFrame(values_list)
df_name.head(20)
df_values.head(20)
nice_df = pd.concat([df_name, df_values], axis=1, sort = False)
  


# In[ ]:





# In[13]:


nice_df.head(20)


# ## Exporting to excel

# In[23]:


nice_df.to_excel("Indicator Scoring.xlsx")


# In[ ]:





# ## Another Point of view Version.1.2

# #### Usually the users are from a specific Area that they are seeking for valuable information. Let's try and group the indicators by area names

# In[7]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# In[8]:


#returns the above scores for a particular area
def pick_area(df,area,year):
    df.loc[(df['area_name']==area) & ( df['year'] == year)]


# In[9]:


def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()


# #### Decide if an indicator for a specific area name and year is different from Scotland

# In[ ]:


import numpy as np

# for each indicator
# a= [Significant, better, worse, not sure]
def is_what(comparator, lower, upper):
    if(lower > comparator):
        #significant= significant+1
        #better = better +1
        a = [1,1,0,0]
        return a
    if(upper < comparator):
        #significant = significant + 1
        #worse = worse + 1
        a = [1,0,1,0]
        return a
    #not_sure = not_sure + 1
    return [0,0,0,1]
    

#for all the area
# check the indicators
# calclulate the ratio
def score(comparator_values, lower_values, upper_values):
    a = [is_what(x,y,z) for x,y,z in zip(comparator_values,lower_values,upper_values)]
    return np.sum(np.matrix(a), axis = 0)/len(comparator_values)


# #### Run the experiment, for 21993 groups, it will take almost 30 minutes

# In[ ]:


import time
grouped_by_indicator_area_type = dataset_all.groupby(['area_name','year'])
name_list = []
values_list = []
i=0
scores_per_area =[]
for pair, pair_df in grouped_by_indicator_area_type:
    #pair[0] indicator name
    #pair[2] year
    #print(pair)
    #print(pair_df.shape)
    start_time = time.time()
    comparators = [find_comparator(x,y) for x,y in zip(pair_df['indicator'], pair_df['year'])]
    area_score = score(comparators,pair_df['lower_confidence_interval'], pair_df['upper_confidence_interval'])
    elapsed_time = time.time() - start_time
    scores_per_area.append(area_score)
elapsed_time = time.time() - start_time
print(elapsed_time)

    


# In[72]:


print(len(scores_per_area))


# In[118]:


per_area = [l[0].tolist() for l in scores_per_area]


# In[150]:


# our input is a list of list, so we need to take the value of the list.
per_area_=[]
for i in range(0,len(per_area)):
    per_area_.append(per_area[i][0])


# In[167]:


#same as above
l = [row[0] for row in per_area]


# In[151]:


per_area_df = pd.DataFrame(per_area_, columns = ['Significant' , 'Better', 'Worse', 'Not Sure'])


# In[163]:


pairs=[]
for pair,b in grouped_by_indicator_area_type:
    pairs.append({'area_name': pair[0], 'year': pair[1]})

labels_df = pd.DataFrame(pairs)


# In[164]:


labels_df.head()


# In[166]:


per_area_df.head()


# In[169]:


Final_per_area = pd.concat([labels_df, per_area_df], axis= 1)
Final_per_area.head(20)


# ## Excel

# In[168]:


Final_per_area.to_excel("Scoring per area.xlsx")


# ## To do, List the important indcators

# In[173]:


#!jupyter nbconvert --to script Dataset_Exploration_version_1.ipynb


# In[1]:


import pandas as pd
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)


# In[2]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# In[3]:


def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()


# In[ ]:


import numpy as np

# for each indicator
# a= [Significant, better, worse, not sure]
def is_what(comparator, lower, upper):
    if(lower > comparator):
        #significant= significant+1
        #better = better +1
        a = [1,1,0,0]
        return a
    if(upper < comparator):
        #significant = significant + 1
        #worse = worse + 1
        a = [1,0,1,0]
        return a
    #not_sure = not_sure + 1
    return [0,0,0,1]
    

#for all the area
# check the indicators
# calclulate the ratio
def score(comparator_values, lower_values, upper_values):
    a = [is_what(x,y,z) for x,y,z in zip(comparator_values,lower_values,upper_values)]
    return np.sum(np.matrix(a), axis = 0)/len(comparator_values)


# In[113]:


#find the min distance between the comparator and the confident intervals.
def find_difference(comparator,lower, upper, measure):
    if(has_confident_intervals(lower)):
        upper_diff = abs(comparator-upper)
        lower_diff = abs(comparator-lower)
        return min(upper_diff,lower_diff)
    else:
        return abs(comparator-measure)
    


def find_differnce_all(comparator_values, lower_values, upper_values,measure):
    a = [find_difference(x,y,z,d) for x,y,z,d in zip(comparator_values,lower_values,upper_values,measure)]
    return a

def create_dictionary(arr, indicator_names, area, year, area_type, has_intervals):
    return [mini_dict(a,b,area,year,c, d) for a,b,c,d in zip(arr, indicator_names, area_type,has_intervals)]
    
    return area
def mini_dict(diff,indicator,area,year,area_type,has_intervals):
    return {'indicator': indicator, 'area': area, 'year': year, 'area_type': area_type, 'has_intervals': has_intervals,'difference': diff}

import math
def has_confident_intervals(value):
    return (np.logical_not(np.isnan(value)))
def has_intervals(arr):
    # if we have nan means that we dont have intervals.
    #np is nan return true where is NaN and False when we have intervals
    #so we use the logical not
    #as a result we have an array where it contains true at the index that we have intervals.
    return (np.logical_not(np.isnan(arr)))


# ## Testing the functions on Healthboard 2017 dataset (check bellow for the whole dataset and                                                                                                 functions details)

# In[9]:


df_HB = pd.read_csv('test fo.csv')
df_HB.head()


# In[15]:


df_HB.shape


# In[ ]:





# In[115]:


import time
grouped_by_indicator_area_type = df_HB.groupby(['area_name','year'])
name_list = []
values_list = []
i=0
scores_per_area =[]
dict_list =[]
for pair, pair_df in grouped_by_indicator_area_type:
    #pair[0] indicator name
    #pair[2] year
   
    #print(pair_df)
    #print(pair_df.shape
    start_time = time.time()
    comparators = [find_comparator(x,y) for x,y in zip(pair_df['indicator'], pair_df['year'])]
    area_diff = find_differnce_all(comparators,pair_df['lower_confidence_interval'], pair_df['upper_confidence_interval'], pair_df['measure'])
    diff_ = create_dictionary(area_diff,pair_df['indicator'],pair[0],pair[1],pair_df['area_type'],has_intervals(pair_df['lower_confidence_interval']))
    scores_per_area.append(area_score)
    dict_list.extend(diff_)
elapsed_time = time.time() - start_time
#print(please)
#print(scores_per_area)
print(elapsed_time)


# In[79]:





# In[116]:


per_area_diff = pd.DataFrame(dict_list)


# In[126]:


per_area_diff.head()


# # Lets try to run this approach to the whole Dataset
# 

# ## Load the Dataset

# In[ ]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# ## Functions

# In[142]:


import math
def has_confident_intervals(value):
    return (np.logical_not(np.isnan(value)))

def has_intervals(arr):
    # if we have nan means that we dont have intervals.
    #np is nan return true where is NaN and False when we have intervals
    #so we use the logical not
    #as a result we have an array where it contains true at the index that we have intervals.
    return (np.logical_not(np.isnan(arr)))

#Search function with input indicator and year and output the comparator value (Scotland value for this Comparator)
def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()

#calculate the distance between the comparator and the the intervals, for one indicator at a particular area
#if our instance doesn't contain intervals, we calculate the distance between the comparator and the measure.
def find_difference(comparator,lower, upper, measure):
    if(has_confident_intervals(lower)):
        upper_diff = abs(comparator-upper)
        lower_diff = abs(comparator-lower)
        return min(upper_diff,lower_diff)
    else:
        return abs(comparator-measure)
    

#calculates the differences for an area, for each individual indicator.
def find_differnce_all(comparator_values, lower_values, upper_values,measure):
    a = [find_difference(x,y,z,d) for x,y,z,d in zip(comparator_values,lower_values,upper_values,measure)]
    return a


#creates a dictionary for a particular indicator for a specific area.
def mini_dict(diff,indicator,area,year,area_type,has_intervals):
    return {'indicator': indicator, 'area': area, 'year': year, 'area_type': area_type, 'has_intervals': has_intervals,'difference': diff}

#create a list of dictionaries for every area.
def create_dictionary(arr, indicator_names, area, year, area_type, has_intervals):
    return [mini_dict(a,b,area,year,c, d) for a,b,c,d in zip(arr, indicator_names, area_type,has_intervals)]

def pick_by_year(df_,year):
    return df_.loc[(df_['year']==year)]

def pick_by_indicator(df_, indicator):
    return df_.loc[(df_['indicator']==indicator)]

def pick_by_indicator_year(df_, indicator, year):
    return df_.loc[(df_['year']==year) & ( df_['indicator'] == indicator)]['measure']

def pick_by_area(df_,area):
    return df_.loc[(df_['area']==area)]

def pick_by_area_type(df_, area_type):
    return df_.loc[(df_['area_type']== area_type)]

def pick_by_intervals(df_,boolean_interval):
    return df_.loc[(df_['has_intervals']== boolean_interval)]



    


# ## Run the experiment almost 16 minutes running time

# #### This experiment follows the same approach as the above examples. I grouped the dataset by area_name and year, then I calculated the differences and I saved them into a list of Dictionaries in order to create the Data Frame

# In[122]:


import time
grouped_by_indicator_area_type = dataset_all.groupby(['area_name','year'])

scores_per_area =[]
dict_list =[]
start_time = time.time()
for pair, pair_df in grouped_by_indicator_area_type:
    #pair[0] indicator name
    #pair[2] year
    comparators = [find_comparator(x,y) for x,y in zip(pair_df['indicator'], pair_df['year'])]
    area_diff = find_differnce_all(comparators,pair_df['lower_confidence_interval'], pair_df['upper_confidence_interval'], pair_df['measure'])
    diff_ = create_dictionary(area_diff,pair_df['indicator'],pair[0],pair[1],pair_df['area_type'],has_intervals(pair_df['lower_confidence_interval']))
    #list.extend appends only the values of a list and not a list itself.
    dict_list.extend(diff_)

elapsed_time = time.time() - start_time
print(elapsed_time)


# ## create the dataframe

# In[123]:


per_area_diff = pd.DataFrame(dict_list)


# In[129]:


per_area_diff


# ## Exporting to Excel

# In[125]:


per_area_diff.to_excel("Scoring per area with differences.xlsx")


# In[133]:


#!jupyter nbconvert --to script Dataset_Exploration_version_1_2.ipynb


# In[136]:





# In[143]:


pick_by_intervals(per_area_diff, False)


# ## 24-25/06/19 Goal: Append the general Score for each indicator and try to use previous year scores.

# ### Load the CSV data

# In[2]:


general_score_df = pd.read_excel('Indicator Scoring.xlsx')
per_area_general = pd.read_excel('Scoring per area.xlsx')
per_area_diff = pd.read_excel('Scoring per area with differences.xlsx')
dataset_all = pd.read_csv("scotpho_data_extract.csv")
#dataset_all = pd.read_excel("scotpho_data_extract.xlsx")


# In[5]:


general_score_df.head()


# In[7]:


per_area_general.head()


# In[4]:


per_area_diff.head()


# ## General Stats for Indicators

# ## We need these methods for evaluation, I will use Pandas merge function in order to Join the data frames.

# In[8]:


# This method takes as input the dataframe that we want to explore, the indicator, the area_Type and the year
# It will help me to append this columns to the dataset that I ll try to cluster
#Python knowledge: explore a dataframe and return the rows and columns for a particular if statement.

def get_general_indicator_values(df, indicator, area_type, year):
    general = df.loc[(df['year']==year) & ( df['indicator'] == indicator) & (df['area_type'] == area_type)][['Statistical Different Score','better', 'worse','not difference']]
    return general.values[0]

    
def get_better_value_indicator(df, indicator, area_type, year):
    better = df.loc[(df['year']==year) & ( df['indicator'] == indicator) & (df['area_type'] == area_type)]['better']
    return better

                                                                                                          
                                                                                                       
def get_stat_diff_value_indicator(df, indicator, area_type, year):
    statdiff = df.loc[(df['year']==year) & ( df['indicator'] == indicator) & (df['area_type'] == area_type)]['Statistical Different Score']
    return statdiff

def get_worse_value_indicator(df, indicator, area_type, year):
    worse = df.loc[(df['year']==year) & ( df['indicator'] == indicator) & (df['area_type'] == area_type)]['worse']
    return worse 

def get_not_difference_value_indicator(df, indicator, area_type, year):
    not_diff = df.loc[(df['year']==year) & ( df['indicator'] == indicator) & (df['area_type'] == area_type)]['not difference']
    return not_diff


# ## Test The Methods

# In[9]:


print(get_general_indicator_values(general_score_df, '% people perceiving rowdy behaviour very/fairly common in their neighbourhood', 'Alcohol & drug partnership', 2007))
print(get_better_value_indicator(general_score_df, '% people perceiving rowdy behaviour very/fairly common in their neighbourhood', 'Alcohol & drug partnership', 2007))
print(get_worse_value_indicator(general_score_df, '% people perceiving rowdy behaviour very/fairly common in their neighbourhood', 'Alcohol & drug partnership', 2007))
print(get_stat_diff_value_indicator(general_score_df, '% people perceiving rowdy behaviour very/fairly common in their neighbourhood', 'Alcohol & drug partnership', 2007))
print(get_not_difference_value_indicator(general_score_df, '% people perceiving rowdy behaviour very/fairly common in their neighbourhood', 'Alcohol & drug partnership', 2007))


# ## Stats per Area and Renaming the Collumns

# In[10]:


def get_stats_by_area(df, area_name, year):
    stats = df.loc[(df['year']==year) & ( df['area_name'] == area_name)][['Significant','Better', 'Worse','Not Sure']]
    return stats.values[0]


# In[11]:


get_stats_by_area(per_area_general, 'Abbeyhill', 2003)


# In[12]:


#rename columns in order to join the dataframes
per_area_diff = per_area_diff.rename(columns = {'area':'area_name'})


# In[13]:


temp_dfinal = per_area_diff.merge(general_score_df[["year",'indicator','area_type','Statistical Different Score', 'better', 'not difference', 'worse' ]], on=["year",'indicator','area_type'], how = 'left')
semi_final = temp_dfinal.merge(per_area_general[['area_name', 'year', 'Significant', 'Worse', 'Better', 'Not Sure']], on = ['area_name', 'year'], how = 'left')


# In[14]:


#rename the columns in order to join the dataframes
per_area_diff = per_area_diff.rename(columns = {'area': 'area_name'})
semi_final = semi_final.rename(columns = {'Worse': 'Percentage of Indicators that are worse at this area'})
semi_final = semi_final.rename(columns = {'Significant': 'Percentage of Indicators that are Different at this area'})
semi_final = semi_final.rename(columns = {'Not Sure': 'Percentage of Indicators that we are not sure at this area'})
semi_final = semi_final.rename(columns = {'Better': 'Percentage of Indicators that  are better at this area'})

semi_final = semi_final.rename(columns = {'better': 'Percentage of areas at this area type that are Better for this Indicator'})
semi_final = semi_final.rename(columns = {'worse': 'Percentage of areas at this area type that are Worse for this Indicator'})
semi_final = semi_final.rename(columns = {'not difference': 'Percentage of areas at this area type that we are not sure if its Stat Diff for this Indicator'})
semi_final = semi_final.rename(columns = {'Statistical Different Score': 'Percentage of areas at this area type that are Statistical Diff for this Indicator'})



# In[15]:


semi_final.head(1)


# ## Try to create a dataframe that contains the flunctuations at the measures per area_name
# 

# In[16]:


dataset_all.head(1)


# ## Group by the keys that we will join at our Final DataFrame

# In[17]:


area_type_area = dataset_all.groupby(['area_name','area_type','indicator'])


# In[18]:


#substraction of feature year from this year
#I will set default 3 years we can change it if we want.
def get_flunctuation(df):
    temp = []
    #for each measure at the dataframe (recall: this dataframe consists for a particular area, area_type, and indicator)
    # so it is the values across the years for these particular keys
    # Calculate the difference for the next one and append this list to a list.
    # Each sublist contains the differences for each particular year.
    # Eg measures = [10,5,3,5] ---> [[5-10,3-10,5-10], [3-5, 5-5], [5-3], []] --> [[-5,-7,-5], [-2,0], [-2], []]
    for i in range(0,len(df['measure'].values-1)):
        temp.append([j-df['measure'].values[i] for j in df['measure'].values[i+1:]])
    return temp


# for each sublist it creates a dict which will be used in order to create the Dataframe.
#recall you can make a  dataframe from list of dicts, or from just lists
# This function just checks if the sub lists have values for all the feature years.
def create_dict(df):
    arr = get_flunctuation(df)
    temp = []
    for l in arr:
        if len(l) >= 3:
            temp.append({'difference from next year': l[0], 'difference from next 2 years': l[1], 'difference from next 3 years': l[2]})
        elif len(l) == 2:
            temp.append({'difference from next year': l[0], 'difference from next 2 years': l[1], 'difference from next 3 years': 'no data'})
        elif len(l) == 1:
            temp.append({'difference from next year': l[0], 'difference from next 2 years': 'no data', 'difference from next 3 years': 'no data'}) 
        elif len(l) ==0:
            temp.append({'difference from next year': 'no data', 'difference from next 2 years': 'no data', 'difference from next 3 years': 'no data'})
    return temp  




# ## RUN for All the groups

# In[19]:


# I use the grouped dataframes that I have allready computed with the help of Panda's groupby function.
#After that I have a list with Dataframes. Each Dataframe represents an indicator at a specific area/area_type through years.
import time
tempaki = []
start = time.time()
for pair, pair_df in area_type_area:
    pair_df.sort_values(by=['year'])
    data = create_dict(pair_df)
    df_ = pd.DataFrame(data, columns=data[0].keys())
    tempaki.append(pd.concat([pair_df.reset_index(drop=True), df_], axis=1, sort = False))
end = time.time()

print(end-start)


# ## Concat the dataframes

# In[20]:


#Join the dataframes
df_time = pd.concat(tempaki)


# In[21]:


#Reset the index so to Concat them
df_time.reset_index(drop = True, inplace = True)
semi_final.reset_index(drop = True, inplace = True)


# In[22]:


semi_final = semi_final.sort_values(by=['indicator','area_name','year','area_type'])
semi_final.head(1)


# In[23]:


df_time = df_time.sort_values(by=['indicator','area_name','year','area_type'])
#pick the columns that we want to add
to_merge = df_time[['difference from next year', 'difference from next 2 years', 'difference from next 3 years']]
to_merge.head(1)


# ### Concat

# In[24]:


Final = pd.concat([semi_final, to_merge], axis=1, sort = False)


# In[25]:


Final.head(1)


# ## Time Fluncuation DataFrame to Excel

# In[26]:


df_time.to_excel('Time DF.xlsx')


# ## Excel

# In[2]:


Final.to_csv("25_06_full.csv")


# ## Memory Problems

# ##### I figured out that when I used  pandas Merge function to join my two final dataframes, I had a memory usage Problem.  12.6GB of Memory Usage cannot work efficiently for a comodity machine.

# In[81]:


final = semi_final.merge(df_time[['area_name', 'year','area_type', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']], on = ['area_name', 'year','area_type'], how = 'left')
final.info(memory_usage= 'deep')


# ##### Although, I knew the keys that I would need to use(area_name, area_type, year, indicator). Having that in my mind and also that every Dataframe contains the same number of rows, I sorted each dataframe by these values.  After that I concatenate the dataframes. This has lead to a tremendous reduce of memory usage. From 12.6 GB to 274.1MB. This works more efficiently at every machine.

# In[83]:


Final.info(memory_usage = 'deep')


# # CLUSTERING and Assumptions

# In[15]:


import pandas as pd
from kneed import KneeLocator

Final = pd.read_csv('13_07_19_full.csv')
Final.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'],inplace= True)


# In[16]:


#Renaming the columns

Final = Final.rename(columns = { 'Percentage of areas at this area type that are Statistical Diff for this Indicator' : 'Stat diff at indicator' })
Final = Final.rename(columns = { 'Percentage of areas at this area type that are Better for this Indicator': 'Better at Indicator' } )
Final = Final.rename(columns = { 'Percentage of areas at this area type that we are not sure if its Stat Diff for this Indicator': 'Not Sure at Indicator' } )
Final = Final.rename(columns = { 'Percentage of areas at this area type that are Worse for this Indicator': 'Worse at Indicator' } )
Final = Final.rename(columns = { 'Percentage of Indicators that are Different at this area': 'Stat diff at area' } )
Final = Final.rename(columns = { 'Percentage of Indicators that are worse at this area': 'Worse at area' } )
Final = Final.rename(columns = { 'Percentage of Indicators that  are better at this area' : 'Better at area'} )
Final = Final.rename(columns = { 'Percentage of Indicators that we are not sure at this area': 'Not Sure at area' } )


# In[17]:


Final.head(2)


# #### Assumption

# For the instances that we don't have enought data to calculate the fluctuation to the next year we replace the 'no data' value to 0. So we assume that the values didn't fluctuate at all. One other assumption would be to add the average of the fluctuations during the years

# In[19]:


#Python knowledge, Change the value of a column at a DataFrame given an if statements stands.
Final.loc[Final['difference from next year'] == 'no data', 'difference from next year'] = 0.0
Final.loc[Final['difference from next 2 years'] == 'no data', 'difference from next 2 years'] = 0.0
Final.loc[Final['difference from next 3 years'] == 'no data', 'difference from next 3 years'] = 0.0
display(Final.head(1))


# ## I will try 3 kinds of Clustering:

# Clustering : Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). [WIKI]

# ##### I ll divide the data in two groups: 1) The Instances of the dataset that contain Confident Intervals, 2) Don't Contain Confident Intervals

# In[20]:


With_Intervals = (Final.loc[(Final['has_intervals']== True)])
Without_Intervals = (Final.loc[(Final['has_intervals']== False)])
print(Without_Intervals.shape)
print(With_Intervals.shape)


# #### 1) area name | area type | year

# This Clustering will group the 'similar' indicators for a specific area, area type and year.

# ###### group the data by the keys

# In[21]:


grouped_1_yes = With_Intervals.groupby(['area_name', 'area_type', 'year'])
grouped_1_no = Without_Intervals.groupby(['area_name', 'area_type', 'year'])


# In[ ]:


# NO INTERVALS
# a Lot of Dataframes contains only 1 record, maybe have an option for data without confident intervals ?
for pair, pair_df in grouped_1_no:
    #print(pair)
    display(pair_df)
    break
    


# In[ ]:


# With Intervals


# In[22]:


for pair, pair_df in grouped_1_yes:
    print(pair)
    display(pair_df.head(1))
    test_for_clustering = pair_df
    print(test_for_clustering.shape[0])
    break


# In[23]:


test_for_clustering


# ### Libraries

# In[1]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.manifold import TSNE


# In[25]:


test_for_clustering.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].describe()


# ## K means

# K-means clustering is one of the simplest and popular unsupervised machine learning algorithms. Its goal is to group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.
# 
# In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
# 
# Drawbacks: 1)initialisation sensitivity. 2)for some very bad data input, K means can be super-polynomial in the input size
# 
# https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
# 
# https://www.youtube.com/watch?v=RD0nNK51Fp8
# 
# sklearn uses Kmeans++ alleviates the second drawback  : https://en.wikipedia.org/wiki/K-means%2B%2B , http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
# (pick centroids that are as fas as other centroids (probabilistically and not deterministically (outliers problem)))
# 
# 

# ### Index Indicators to clusters

# In[26]:


## df is the dataframe, for our example Final
## results are the kmeans.labels_ after Clustering
def display_indicators_per_cluster(df, results):
    cluster_n = np.unique(results).size
    returner = [ [] for l in range(cluster_n) ]
    for i in range(0,len(results)):
        returner[results[i]].append(df.loc[i,'indicator'])
    return returner

#example 


# ### Elbow Method

# The Elbow method is a method of interpretation and validation of consistency within cluster analysis designed to help finding the appropriate number of clusters in a dataset. Inertia : Sum of squared distances of samples to their closest cluster center.

# In[27]:


import warnings
warnings.filterwarnings('ignore')
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


# find the points
points = test_for_clustering.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
scaler = MinMaxScaler()
points_scaled = scaler.fit_transform(points)

print(kmeans_elbow(points,5, 'All columns'))


# ## PCA for visualisation

# PCA is a technique for reducing the number of dimensions in a dataset whilst retaining most information

# In[28]:


from sklearn.decomposition import PCA
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

    
res_pca = PCA_for_kmeans(points_scaled,2)


# In[29]:


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
Visualise_after_reduction(res_pca,points,'PCA')


# ### TSNE

# “t-Distributed stochastic neighbor embedding (t-SNE) minimizes the divergence between two distributions: a distribution that measures pairwise similarities of the input objects and a distribution that measures pairwise similarities of the corresponding low-dimensional points in the embedding”.
# 
# Essentially what this means is that it looks at the original data that is entered into the algorithm and looks at how to best represent this data using less dimensions by matching both distributions.
# 
# The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects have a high probability of being picked while dissimilar points have an extremely small probability of being picked. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence between the two distributions with respect to the locations of the points in the map.[WIKI]
# 
# original paper: http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

# In[30]:


import time
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
    

tsne_df = tsne_on_data(points_scaled, 2)
Visualise_after_reduction(tsne_df,points_scaled, 'TSNE')


# ## Silhouette https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

# In[31]:


clusters_n = range(2,4)
for n_clusters in clusters_n:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    #silhouette_score is between -1,1
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, points_scaled.size + (n_clusters + 1) * 10])
    
    clusterer = KMeans(n_clusters=n_clusters, random_state = 10)
    cluster_labels = clusterer.fit_predict(points_scaled)
    silhouette_avg = silhouette_score(points_scaled, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    
    sample_silhouette_values = silhouette_samples(points_scaled, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    


# ## Using Kmeans After Dimensional Reduction

# I will try to use Kmeans at the 2 dimensions. Thus I will be able to visualise the clusters with each corresponding point that belongs to them.

# #### PCA

# In[32]:


def PCA_Kmeans_elbow(points, range_, title):
    points_ = PCA_for_kmeans(points,2)
    return kmeans_elbow(points_.values,range_, title)
    
print(PCA_Kmeans_elbow(points_scaled, 5, 'PCA'))

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
    
    
(PCA_Kmeans(points_scaled,3))


# #### TSNE 

# In[33]:


def TSNE_Kmeans_elbow(points, range_, title):
    points_ = tsne_on_data(points,2)
    return kmeans_elbow(points_.values,range_, title)
    
print(TSNE_Kmeans_elbow(points_scaled,5, 'TSNE'))

def TSNE_Kmeans(points, clusters_):
    
    points_ = tsne_on_data(points,2)
    kmeans = KMeans(n_clusters = clusters_, random_state= 5221)
    kmeans.fit(points_)
    y_km = kmeans.predict(points_)
    points_['labels'] = y_km
    
    return { 'centroids': kmeans.cluster_centers_ , 'DF': points_}
    


# ### Visualise the Clusters

# different results

# In[34]:


# input all the points, the center, and the prediction of our Kmeans model
#output the plot
def Visualise_Clusters(points, center, labels, title):
    plt.figure(figsize=(12,8))
    plt.scatter(points[:, 0], points[:, 1], c = labels, s=60, cmap='viridis')
    plt.scatter(center[:, 0], center[:, 1], c='black', s=250, alpha=0.6)
    plt.title('Clustering Result ( '+title+' )')
    plt.show()


# In[35]:


#returns dictionary points and center
results_kmeans_pca = PCA_Kmeans(points_scaled,3)
df_pca = results_kmeans_pca['DF']
points_pca = df_pca[['first feature','second feature']]
labels_pca = df_pca['labels']
centroids_pca = results_kmeans_pca['centroids']

##TSNE

results_kmeans_tsne = TSNE_Kmeans(points_scaled,3)
df_tsne = results_kmeans_tsne['DF']
points_tsne = df_tsne[['first feature','second feature']]
labels_tsne = df_tsne['labels']
centroids_tsne = results_kmeans_tsne['centroids']




# In[36]:


##import mpld3


# In[37]:


import seaborn as sns; sns.set()  # for plot styling
Visualise_Clusters(points_pca.values, centroids_pca, labels_pca,'PCA' )
display_indicators_per_cluster(test_for_clustering, labels_pca)


# In[38]:


Visualise_Clusters(points_tsne.values, centroids_tsne, labels_tsne,'TSNE' )
display_indicators_per_cluster(test_for_clustering, labels_tsne)


# #### K means its not the best option for Non Linear Data. We can use DBSCAN for that problem.

# We need to scale our data first.

# In[39]:


def Visualise_Clusters_dbscan(points,clusters, title):
    plt.figure(figsize=(12,8))
    plt.scatter(points[:, 0], points[:, 1], c = clusters, s=60, cmap = "plasma")
    plt.title('Clustering Result ( '+title+' )')
    plt.show()
    


# In[40]:


from sklearn.cluster import DBSCAN


dbscan = DBSCAN(eps=0.30, min_samples = 2)
clusters = dbscan.fit_predict(res_pca.values)

Visualise_Clusters_dbscan(res_pca.values, clusters , 'PCA and DBSCAN')


# #### 2) area name | area type

# This Clustering will group the 'similar' indicators for a specific area name | area type through years

# In[41]:


grouped_2_yes = With_Intervals.groupby(['area_name', 'area_type'])
grouped_2_no = Without_Intervals.groupby(['area_name', 'area_type'])


# In[42]:


for pair,pair_df in grouped_2_yes:
    print(pair)
    pair_df = pair_df.reset_index(drop=True)
    temp_second = pair_df
    break


# In[43]:


temp_second.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].describe()


# ## ELBOW METHOD

# In[44]:


points_second = temp_second.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
scaler = MinMaxScaler()
points_scaled_second = scaler.fit_transform(points_second)

print(kmeans_elbow(points_scaled_second,20,'All columns'))


# ## Visualisation of data after reduction methods

# ### PCA

# In[45]:


res_pca = PCA_for_kmeans(points_scaled_second,2)
Visualise_after_reduction(res_pca,points_scaled_second, 'PCA')


# ### TSNE

# In[46]:



tsne_df = tsne_on_data(points_scaled_second, 2)
Visualise_after_reduction(tsne_df,points_scaled_second, 'TSNE')


# In[47]:


PCA_Kmeans_elbow(points_scaled_second,15, 'PCA')


# In[48]:


TSNE_Kmeans_elbow(points_scaled_second,15, 'TSNE')


# ### Clustering

# #### PCA

# In[49]:


results_kmeans_pca = PCA_Kmeans(points_scaled_second,4)
df_pca = results_kmeans_pca['DF']
points_pca = df_pca[['first feature','second feature']]
labels_pca = df_pca['labels']
centroids_pca = results_kmeans_pca['centroids']

Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA' )





# In[50]:


#Display the points of the clusters
#display_indicators_per_cluster(temp_second, labels_pca.values)


# In[51]:


#display_indicators_per_cluster(temp_second, labels_pca)


# #### TSNE

# In[52]:



results_kmeans_tsne = TSNE_Kmeans(points_scaled_second,2)
df_tsne = results_kmeans_tsne['DF']
points_tsne = df_tsne[['first feature','second feature']]
labels_tsne = df_tsne['labels']
centroids_tsne = results_kmeans_tsne['centroids']

Visualise_Clusters(points_tsne.values, centroids_tsne, labels_tsne,'TSNE' )


# #### Creating the dataframes for each Cluster

# In[53]:


from itertools import chain
def display_indicators_per_cluster(df, results):
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


# In[54]:


to_Export = display_indicators_per_cluster(temp_second, labels_tsne)
display(to_Export.head(1))


# ### Exporting

# In[55]:


to_Export.to_excel('clustering area_type area one example.xlsx')


# In[ ]:





# ### DBSCAN

# For PCA Reduction

# In[56]:


from sklearn.cluster import DBSCAN


dbscan = DBSCAN(eps=0.15, min_samples = 2)
clusters = dbscan.fit_predict(res_pca.values)

Visualise_Clusters_dbscan(res_pca.values, clusters , 'PCA and DBSCAN')


# #### 3)  Indicator

# This Clustering will group areas and area types through year per each Indicator

# In[57]:


grouped_3_yes = With_Intervals.groupby(['indicator'])
grouped_3_no = Without_Intervals.groupby(['indicator'])


# In[58]:


for pair, pair_df in grouped_3_yes:
    pair_df = pair_df.reset_index(drop=True)
    temp_third = pair_df
    break


# In[59]:


temp_third.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].describe()


# ## ELBOW METHOD

# In[60]:


points_third = temp_third.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
scaler = MinMaxScaler()
points_scaled_third = scaler.fit_transform(points_third)

kmeans_elbow(points_scaled_third,20,'All columns')


# In[ ]:





# ## Visualisation of data after reduction methods

# ### PCA

# In[61]:


res_pca = PCA_for_kmeans(points_scaled_third,2)
Visualise_after_reduction(res_pca,points_scaled_third, 'PCA')


# ### TSNE

# In[62]:


tsne_df = tsne_on_data(points_scaled_third, 2)
Visualise_after_reduction(tsne_df,points_scaled_third, 'TSNE')


# ### Elbow method after reduction

# In[63]:


PCA_Kmeans_elbow(points_scaled_third,15, 'PCA')


# In[64]:


TSNE_Kmeans_elbow(points_scaled_third,15, 'TSNE')


# ## Clustering Visualisation

# ### PCA

# In[65]:


results_kmeans_pca = PCA_Kmeans(points_scaled_third,4)
df_pca = results_kmeans_pca['DF']
points_pca = df_pca[['first feature','second feature']]
labels_pca = df_pca['labels']
centroids_pca = results_kmeans_pca['centroids']

Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA' )


# In[ ]:





# ### TSNE

# In[66]:


results_kmeans_tsne = TSNE_Kmeans(points_scaled_third,3)
df_tsne = results_kmeans_tsne['DF']
points_tsne = df_tsne[['first feature','second feature']]
labels_tsne = df_tsne['labels']
centroids_tsne = results_kmeans_tsne['centroids']

Visualise_Clusters(points_tsne.values, centroids_tsne, labels_tsne,'TSNE' )


# In[67]:


def display_indicators_per_cluster(df, results):
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


# In[68]:


to_Export = display_indicators_per_cluster(temp_third, labels_tsne)


# In[69]:


to_Export.head(20)


# In[70]:


to_Export.to_excel('Indicator Clustering.xlsx')


# In[71]:


from sklearn.cluster import DBSCAN


dbscan = DBSCAN(eps=0.15, min_samples = 3)
clusters = dbscan.fit_predict(res_pca.values)

Visualise_Clusters_dbscan(res_pca.values, clusters , 'PCA and DBSCAN')


# In[72]:


import sklearn
print(sklearn.__version__)


# ## All dataset

# Let's try the clustering for all the dataset.

# ### Functions

# In[98]:


import warnings
warnings.filterwarnings('ignore')
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

def PCA_Kmeans(points, clusters_):
 
    points_ = PCA_for_kmeans(points,2)
    #print(points_)
    
    kmeans = KMeans(n_clusters = clusters_, random_state= 5221)
    kmeans.fit(points_)
    y_km = kmeans.predict(points_)
    points_['labels'] = y_km
    return { 'centroids': kmeans.cluster_centers_ , 'DF': points_}
    
    


# In[2]:


import pandas as pd
from kneed import KneeLocator

Final = pd.read_csv('13_07_19_full.csv')
Final.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'],inplace= True)


# In[3]:


#Renaming the columns

Final = Final.rename(columns = { 'Percentage of areas at this area type that are Statistical Diff for this Indicator' : 'Stat diff at indicator' })
Final = Final.rename(columns = { 'Percentage of areas at this area type that are Better for this Indicator': 'Better at Indicator' } )
Final = Final.rename(columns = { 'Percentage of areas at this area type that we are not sure if its Stat Diff for this Indicator': 'Not Sure at Indicator' } )
Final = Final.rename(columns = { 'Percentage of areas at this area type that are Worse for this Indicator': 'Worse at Indicator' } )
Final = Final.rename(columns = { 'Percentage of Indicators that are Different at this area': 'Stat diff at area' } )
Final = Final.rename(columns = { 'Percentage of Indicators that are worse at this area': 'Worse at area' } )
Final = Final.rename(columns = { 'Percentage of Indicators that  are better at this area' : 'Better at area'} )
Final = Final.rename(columns = { 'Percentage of Indicators that we are not sure at this area': 'Not Sure at area' } )


# In[4]:


#Python knowledge, Change the value of a column at a DataFrame given an if statements stands.
Final.loc[Final['difference from next year'] == 'no data', 'difference from next year'] = 0.0
Final.loc[Final['difference from next 2 years'] == 'no data', 'difference from next 2 years'] = 0.0
Final.loc[Final['difference from next 3 years'] == 'no data', 'difference from next 3 years'] = 0.0
display(Final.head(1))


# In[6]:


With_Intervals = (Final.loc[(Final['has_intervals']== True)])
Without_Intervals = (Final.loc[(Final['has_intervals']== False)])
print(Without_Intervals.shape)
print(With_Intervals.shape)


# ## area_name | area type | year

# In[74]:


grouped_1_yes = With_Intervals.groupby(['area_name', 'area_type', 'year'])
grouped_1_no = Without_Intervals.groupby(['area_name', 'area_type', 'year'])


# ## The process (will take a lot of time, need to run it for hours at home)

# In[146]:


from sklearn.preprocessing import MinMaxScaler
from itertools import chain

writer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')

for pair, pair_df in grouped_1_yes:
    
    k=1
    pair_df = pair_df.reset_index(drop=True)
    test_for_clustering = pair_df
    points_ = test_for_clustering.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
    scaler = MinMaxScaler()
    points_scaled_all = scaler.fit_transform(points_)
    if(test_for_clustering.shape[0] <=2 ):
        continue
    points_scaled_all =np.nan_to_num(points_scaled_all)
    res_pca_all = PCA_for_kmeans(points_scaled_all,2)
    clusters = kmeans_kneed(res_pca_all.values, test_for_clustering.shape[0])
    results_kmeans_pca = PCA_Kmeans(points_scaled_all,clusters)
    df_pca = results_kmeans_pca['DF']
    points_pca = df_pca[['first feature','second feature']]
    labels_pca = df_pca['labels']
    centroids_pca = results_kmeans_pca['centroids']
    to_Export = display_indicators_per_cluster(test_for_clustering, labels_pca)
    to_Export.to_excel(writer, sheet_name='Sheet'+str(k))
    i = k + 1

writer.save()
    
    
    


# In[71]:


writer.save()


# ## area name | area type

# In[75]:


grouped_2_yes = With_Intervals.groupby(['area_name', 'area_type'])
grouped_2_no = Without_Intervals.groupby(['area_name', 'area_type'])


# In[76]:


from sklearn.preprocessing import MinMaxScaler
from itertools import chain

writer = pd.ExcelWriter('area_name_area_type.xlsx', engine='xlsxwriter')

for pair, pair_df in grouped_2_yes:
    
    k=1
    pair_df = pair_df.reset_index(drop=True)
    test_for_clustering = pair_df
    points_ = test_for_clustering.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
    scaler = MinMaxScaler()
    points_scaled_all = scaler.fit_transform(points_)
    if(test_for_clustering.shape[0] <=2 ):
        continue
    points_scaled_all =np.nan_to_num(points_scaled_all)
    res_pca_all = PCA_for_kmeans(points_scaled_all,2)
    clusters = kmeans_kneed(res_pca_all.values, test_for_clustering.shape[0])
    results_kmeans_pca = PCA_Kmeans(points_scaled_all,clusters)
    df_pca = results_kmeans_pca['DF']
    points_pca = df_pca[['first feature','second feature']]
    labels_pca = df_pca['labels']
    centroids_pca = results_kmeans_pca['centroids']
    to_Export = display_indicators_per_cluster(test_for_clustering, labels_pca)
    to_Export.to_excel(writer, sheet_name='Sheet'+str(k))
    i = k + 1

writer.save()
    


# ## Indicator

# In[ ]:


grouped_3_yes = With_Intervals.groupby(['indicator'])
grouped_3_no = Without_Intervals.groupby(['indicator'])


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from itertools import chain

writer = pd.ExcelWriter('indicator.xlsx', engine='xlsxwriter')

for pair, pair_df in grouped_3_yes:
    
    k=1
    pair_df = pair_df.reset_index(drop=True)
    test_for_clustering = pair_df
    points_ = test_for_clustering.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
    scaler = MinMaxScaler()
    points_scaled_all = scaler.fit_transform(points_)
    if(test_for_clustering.shape[0] <=2 ):
        continue
    points_scaled_all =np.nan_to_num(points_scaled_all)
    res_pca_all = PCA_for_kmeans(points_scaled_all,2)
    clusters = kmeans_kneed(res_pca_all.values, test_for_clustering.shape[0])
    results_kmeans_pca = PCA_Kmeans(points_scaled_all,clusters)
    df_pca = results_kmeans_pca['DF']
    points_pca = df_pca[['first feature','second feature']]
    labels_pca = df_pca['labels']
    centroids_pca = results_kmeans_pca['centroids']
    to_Export = display_indicators_per_cluster(test_for_clustering, labels_pca)
    to_Export.to_excel(writer, sheet_name='Sheet'+str(k))
    i = k + 1

writer.save()


# ## Selective Clustering
# 

# Clustering the whole dataset is really time consuming. For this reason I will make a function that the user can put the area of his/her insterest.

# ### area type | area name | year

# In[108]:


def select_df(area_type, area_name, year):
    df_ = With_Intervals.loc[(With_Intervals['year']==year) & ( With_Intervals['area_name'] == area_name) & (With_Intervals['area_type'] == area_type)]
    return df_


# In[109]:


area_type = 'Alcohol & drug partnership'
area_name = 'Lanarkshire'
year = 2017
input_ = select_df(area_type, area_name, year)

from sklearn.preprocessing import MinMaxScaler
from itertools import chain

    
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
        to_Export = display_indicators_per_cluster(input_, labels_pca)
        to_Export.to_excel('Kmeans_'+area_type+' '+area_name+' '+str(year)+'.xlsx')
        
        #DBSCAN
        dbscan = DBSCAN(eps=0.123, min_samples = 2)
        clusters = dbscan.fit_predict(res_pca_all.values)
        to_Export = display_indicators_per_cluster(input_, clusters)
        to_Export.to_excel('DBSCAN_'+area_type+' '+area_name+' '+str(year)+'.xlsx')
        Visualise_Clusters_dbscan(res_pca_all.values, clusters , 'PCA and DBSCAN')
        
        
        
else:
        print('sorry not enough Data')

writer.save()
Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA' )


# ### area type | area name

# In[110]:


def select_df(area_type, area_name ):
    df_ = With_Intervals.loc[( With_Intervals['area_name'] == area_name) & (With_Intervals['area_type'] == area_type)]
    return df_


# In[115]:


area_type = 'Alcohol & drug partnership'
area_name = 'Lanarkshire'
input_ = select_df(area_type, area_name)
from sklearn.preprocessing import MinMaxScaler
from itertools import chain

    
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
        Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA' )
        to_Export = display_indicators_per_cluster(input_, labels_pca)
        to_Export.to_excel('Kmeans_'+area_type+' '+area_name+'.xlsx')
        
        #DBSCAN
        dbscan = DBSCAN(eps=0.123, min_samples = 2)
        clusters = dbscan.fit_predict(res_pca_all.values)
        to_Export = display_indicators_per_cluster(input_, clusters)
        to_Export.to_excel('DBSCAN_'+area_type+' '+area_name+'.xlsx')
        Visualise_Clusters_dbscan(res_pca_all.values, clusters , 'PCA and DBSCAN')
        
        
else:
        print('sorry not enough Data')


# ### Indicator

# In[116]:


def select_df(indicator):
    df_ = With_Intervals.loc[( With_Intervals['indicator'] == indicator)]
    return df_


# In[117]:


print(test_for_clustering.shape)
indicator = 'Babies exclusively breastfed at 6-8 weeks'
input_ = select_df(indicator)
print(input_.shape)
from sklearn.preprocessing import MinMaxScaler
from itertools import chain


    
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
        to_Export = display_indicators_per_cluster(input_, labels_pca)
       
        to_Export.to_excel('KMEANS_'+indicator+'.xlsx')
        
        #DBSCAN
        dbscan = DBSCAN(eps=0.123, min_samples = 2)
        clusters = dbscan.fit_predict(res_pca_all.values)
        to_Export = display_indicators_per_cluster(input_, clusters)
        to_Export.to_excel('DBSCAN_'+indicator+'.xlsx')
        Visualise_Clusters_dbscan(res_pca_all.values, clusters , 'PCA and DBSCAN')
else:
        print('sorry not enough Data')

Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA' )


# ### Indicator and year

# In[119]:


def select_df(indicator, year ):
    df_ = With_Intervals.loc[( With_Intervals['indicator'] == indicator) & (With_Intervals['year'] == year)]
    return df_


# In[120]:


from sklearn.cluster import DBSCAN

indicator = 'Babies exclusively breastfed at 6-8 weeks'
year = 2016
input_ = select_df(indicator, year)
print(input_.shape)
from sklearn.preprocessing import MinMaxScaler
from itertools import chain


    
input_ = input_.reset_index(drop=True)
points_ = input_.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
scaler = MinMaxScaler()
points_scaled_all = scaler.fit_transform(points_)
if(input_.shape[0] > 2 ):
        points_scaled_all =np.nan_to_num(points_scaled_all)
        res_pca_all = PCA_for_kmeans(points_scaled_all,2)
        clusters = kmeans_kneed(res_pca_all.values, test_for_clustering.shape[0])
        results_kmeans_pca = PCA_Kmeans(points_scaled_all,clusters)
        df_pca = results_kmeans_pca['DF']
        points_pca = df_pca[['first feature','second feature']]
        labels_pca = df_pca['labels']
        centroids_pca = results_kmeans_pca['centroids']
        to_Export = display_indicators_per_cluster(input_, labels_pca)
        to_Export.to_excel('KMEANS_'+indicator+' '+str(year)+'.xlsx')
        
        #DB SCAN#
        dbscan = DBSCAN(eps=0.123, min_samples = 2)
        clusters = dbscan.fit_predict(res_pca_all.values)
        to_Export = display_indicators_per_cluster(input_, clusters)
        to_Export.to_excel('DBSCAN_'+indicator+' '+str(year)+'.xlsx')
        Visualise_Clusters_dbscan(res_pca_all.values, clusters , 'PCA and DBSCAN')
else:
        print('sorry not enough Data')

Visualise_Clusters(points_pca.values, centroids_pca, labels_pca, 'PCA' )


# In[ ]:






# ### EXTRA TOOL

# I will demonstrate the top worse indicators per area and also the top worse areas per indicator. I will use the actual difference between the indicator value of the area and Scotland's value.

# ## Datasets

# In[121]:


Final.head(1)


# In[123]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# In[124]:


dataset_all.head(1)


# In[125]:


Scotland_values.head(1)


# ## Functions

# In[127]:


#the most positiv the most the worse since the indicator has a negative meaning for an area.
def find_actual_difference(comparator, measure):
    return measure - comparator
    


def find_actual_difference_all(comparator_values, lower_values, upper_values,measure):
    a = [find_actual_difference(x,y,z,d) for x,y,z,d in zip(comparator_values,lower_values,upper_values,measure)]
    return a

def create_dictionary(arr, indicator_names, area, year, area_type, has_intervals):
    return [mini_dict(a,b,area,year,c, d) for a,b,c,d in zip(arr, indicator_names, area_type,has_intervals)]
    
    return area
def mini_dict(diff,indicator,area,year,area_type,has_intervals):
    return {'indicator': indicator, 'area': area, 'year': year, 'area_type': area_type, 'has_intervals': has_intervals,'difference': diff}

import math
def has_confident_intervals(value):
    return (np.logical_not(np.isnan(value)))
def has_intervals(arr):
    # if we have nan means that we dont have intervals.
    #np is nan return true where is NaN and False when we have intervals
    #so we use the logical not
    #as a result we have an array where it contains true at the index that we have intervals.
    return (np.logical_not(np.isnan(arr)))

def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()


# In[128]:


dataset_all['Scotland'] = dataset_all.apply(lambda row: find_comparator(row.indicator, row.year), axis=1)


# In[131]:


dataset_all.head(1)


# In[130]:


dataset_all['actual_diff'] = dataset_all['measure'] - dataset_all['Scotland']


# In[132]:


dataset_all.to_csv('dataset_all.csv')


# ### Grouping

# In[27]:


dataset_all = pd.read_csv('dataset_all.csv')
dataset_all.drop(columns=['Unnamed: 0'],inplace= True)
#remove the estimation since we can't compare the values of an area to the sum of population.
dataset_all_ = dataset_all[(dataset_all.indicator !='Mid-year population estimate - all ages') & (dataset_all.indicator != 'Quit attempts')]


# ### Function

# In[28]:


#input
#df = dataframe
#k = number of values that will returned, top5, top10 ?
#option = better/ worse
def top_k(df,k,option):
    #sorting
    #ascending order
    df = df.sort_values(by=['actual_diff'])
    if option == 'better':
        return df.head(k)
    elif option == 'worse':
        return df.tail(k)[::-1]
    else:
        print('wrong option')
    


# #### Indicatator | year | area_type

# Find the worse/better areas for each indicator per year

# In[29]:


grouped_indicator = dataset_all_.groupby(['indicator', 'year', 'area_type'])


# In[77]:


for pair, pair_df in grouped_indicator:
    k = 5
    print('\n \n \n')
    print('TOP '+str(k)+' Better')
    indexaki =  pd.Index(list(range(min(k,pair_df.shape[0]))))
    display(top_k(pair_df[['indicator','area_name','area_type','year','actual_diff']],5,'better').set_index([indexaki]))
    print('\n \n \n')
    print('TOP '+str(k)+' Worse')
    display(top_k(pair_df[['indicator','area_name','area_type','year','actual_diff']],5,'worse').set_index([indexaki]))
    break


# #### area_name | area_type | year

# Find the worse/better indicators per area and year

# In[31]:


def select_df(df, area_type, area_name, year):
    df_ = df.loc[(df['year'] == year) & ( df['area_name'] == area_name) & (df['area_type'] == area_type)]
    return df_


# In[32]:


grouped_area = dataset_all_.groupby(['area_name','area_type', 'year'])


# In[74]:


i=0
for pair, pair_df in grouped_area:
    k = 5
    print('\n \n \n')
    indexaki =  pd.Index(list(range(min(k,pair_df.shape[0]))))
    print('TOP '+str(k)+' Better')
    display(top_k(pair_df[['indicator','area_name','area_type','year','actual_diff']],k,'better').set_index([indexaki]))
    print('\n \n \n')
    print('TOP '+str(k)+' Worse')
    display(top_k(pair_df[['indicator','area_name','area_type','year','actual_diff']],k,'worse').set_index([indexaki]))
    break


# ### Interactive

# In[79]:


k=5
year = 2016

area_type = 'Council area'
area_name = 'Stirling'
df_ = select_df(dataset_all_, area_type, area_name, year)
indexaki =  pd.Index(list(range(min(k,df_.shape[0]))))

print('\n \n \n')
print('TOP '+str(k)+' Better')
display(top_k(df_[['indicator','definition','measure','area_name','area_type','year','actual_diff']],k,'better').set_index([indexaki]))

print('\n \n \n')
print('TOP '+str(k)+' Worse')
display(top_k(df_[['indicator','area_name','area_type','year','actual_diff']],k,'worse').set_index([indexaki]))


# AVERAGE DIFFERENCE ?
# WHAT SHOULD I FLAG?

# In[80]:




