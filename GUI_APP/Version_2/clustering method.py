# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:28:24 2019

@author: panagn01
"""


from cluster_methods import *

Final = pd.read_csv('13_07_19_full.csv')
Final.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'],inplace= True)


Final = Final.rename(columns = { 'Percentage of areas at this area type that are Statistical Diff for this Indicator' : 'Stat diff at indicator' })
Final = Final.rename(columns = { 'Percentage of areas at this area type that are Better for this Indicator': 'Better at Indicator' } )
Final = Final.rename(columns = { 'Percentage of areas at this area type that we are not sure if its Stat Diff for this Indicator': 'Not Sure at Indicator' } )
Final = Final.rename(columns = { 'Percentage of areas at this area type that are Worse for this Indicator': 'Worse at Indicator' } )
Final = Final.rename(columns = { 'Percentage of Indicators that are Different at this area': 'Stat diff at area' } )
Final = Final.rename(columns = { 'Percentage of Indicators that are worse at this area': 'Worse at area' } )
Final = Final.rename(columns = { 'Percentage of Indicators that  are better at this area' : 'Better at area'} )
Final = Final.rename(columns = { 'Percentage of Indicators that we are not sure at this area': 'Not Sure at area' } )

Final.loc[Final['difference from next year'] == 'no data', 'difference from next year'] = 0.0
Final.loc[Final['difference from next 2 years'] == 'no data', 'difference from next 2 years'] = 0.0
Final.loc[Final['difference from next 3 years'] == 'no data', 'difference from next 3 years'] = 0.0


area_type = 'Alcohol & drug partnership'
area_name = 'Lanarkshire'
year = 2017

With_Intervals = (Final.loc[(Final['has_intervals']== True)])
selective_clustering(With_Intervals,area_type, area_name, year)
