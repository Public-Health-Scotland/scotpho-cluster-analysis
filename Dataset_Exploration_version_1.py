#!/usr/bin/env python
# coding: utf-8

# # Data Exploration of Indicators Dataset

# #### In this notebook, I will try to reshape the dataset in order to be easily interpeted

# #### importing the necessary libraries and load the dataset 

# In[1]:


import pandas as pd
import pandas as pd
import numpy as np
pd.set_option('display.width', 1000)


# In[ ]:


dataset_all = pd.read_excel("scotpho_data_extract.xlsx")


# In[2]:


pd.set_option('display.max_colwidth', 500)


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

# In[11]:


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

# In[3]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# In[4]:


Scotland_values.tail()


# In[5]:


dataset_all.head()


# In[7]:


print(Scotland_values.loc[(Scotland_values['year']==2013) & ( Scotland_values['indicator'] == 'Child healthy weight in primary 1')]['measure'].item())


# In[8]:


#Search function with input indicator and year and output the comparator value (Scotland value for this Comparator)
def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()


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
  


# In[13]:


nice_df.head(20)


# ## Exporting to excel

# In[23]:


nice_df.to_excel("Indicator Scoring.xlsx")


# In[ ]:





# ## Another Point of view Version.1.2

# #### Usually the users are from a specific Area that they are seeking for valuable information. Let's try and group the indicators by area names

# In[15]:


#Indicators, load whatever data you want, at ScotphoProfile tool Format.
dataset_all = pd.read_csv("scotpho_data_extract.csv")

#Scotland values, Load the coresponding Scotland Values, it doesnt need to be sorted
Scotland_values = pd.read_excel("Scotland_comparator.xlsx")


# In[14]:


#returns the above scores for a particular area
def pick_area(df,area,year):
    df.loc[(nice_df['area_name']==area) & ( nice_df['year'] == year)]


# In[16]:


def find_comparator(indicator, year):
    a = Scotland_values.loc[(Scotland_values['year']==year) & ( Scotland_values['indicator'] == indicator)]['measure']
    if(a.empty):
        return 0
    return a.item()


# #### Decide if an indicator for a specific area name and year is different from Scotland

# In[68]:


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

# In[71]:


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

# In[ ]:





# In[ ]:




