# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:10:37 2019

@author: panagn01
"""


from functions_ import *
import unittest
from cluster_methods import *
from pandas.util.testing import assert_frame_equal # <-- for testing dataframes
import pandas as pd
from Scoring_Functions import *
dataset_all = pd.read_csv("scotpho_data_extract.csv")
dataset_full = pd.read_csv('dataset_all.csv')
dataset_full.drop(columns=['Unnamed: 0'],inplace= True)
test_df = select_df(dataset_full,'Health board','Greater Glasgow & Clyde', 2016)
prof_ind = pd.read_excel('Profiles to indicators.xlsx')
import numpy as np


print(is_statistical_different([0.5,1,5], 10))
class TestSelectMethod(unittest.TestCase):

    def test_score(self):
        
        x = np.array([[0.5,1,1.5], [2,6,10],[2,6,9]])
       
        check = score_of_statistical_different(x,10)
    
        self.assertEqual(check, 2/3.0)
    

    
    def test_select_df(self):
         foo = pd.DataFrame({'indicator':['All-cause mortality among the 15-44 year olds'] ,'area_name': ['Carfin North'] , 'area_code': ['S02002152'] , 'area_type': ['Intermediate zone'], 'year':[2016], 'period':['2015 to 2017 calendar years; 3-year aggregates'], 'numerator':[1.0], 'measure': [79.4], 'lower_confidence_interval': [3.3]  , 'upper_confidence_interval':[365.5], 'definition':['Age-sex standardised rate per 100,000']})
         exactly_equal = select_df(dataset_all,'Intermediate zone','Carfin North', 2016).reset_index(drop=True).head(1)
         assert_frame_equal(foo, exactly_equal)
    
    def test_topk(self):
        list_ = [{'indicator':'Mid-year population estimate - all ages' ,'area_name': 'Greater Glasgow & Clyde' , 'area_type': 'Health board', 'year':2016, 'actual_diff': -4243330.0, 'definition':'Number' },
                             {'indicator':'Quit attempts' ,'area_name': 'Greater Glasgow & Clyde' ,  'area_type': 'Health board', 'year':2016, 'actual_diff': -44454.0, 'definition':'Number' },
                             {'indicator':'ABIs delivered' ,'area_name': 'Greater Glasgow & Clyde' , 'area_type': 'Health board', 'year':2016, 'actual_diff': -24.2, 'definition':'Percentage' }]
        indexaki =  ['indicator', 'area_name', 'area_type', 'year', 'actual_diff','definition']        
        ground_df_1 = pd.DataFrame(list_, columns = indexaki)
              
        list_2  = list_ = [{'indicator':'Patients (65+) with multiple emergency hospitalisations' ,'area_name': 'Greater Glasgow & Clyde' , 'area_type': 'Health board', 'year':2016, 'actual_diff': 1003, 'definition':'Age-sex standardised rate per 100,000' },
                                {'indicator':'Patients with emergency hospitalisations' ,'area_name': 'Greater Glasgow & Clyde' ,  'area_type': 'Health board', 'year':2016, 'actual_diff': 993.5, 'definition':'Age-sex standardised rate per 100,000' },
                                {'indicator':'Alcohol-related hospital stays' ,'area_name': 'Greater Glasgow & Clyde' , 'area_type': 'Health board', 'year':2016, 'actual_diff': 374.2, 'definition':'Age-sex standardised rate per 100,000' }]
        ground_df_2 = pd.DataFrame(list_2, columns = indexaki)
        ground_df_1.reset_index(drop = True)
           
        assert_frame_equal(ground_df_1, top_k(test_df,3,'better').reset_index(drop=True))
        assert_frame_equal(ground_df_2, top_k(test_df,3,'worse').reset_index(drop=True))
        
    def test_find_similar(self):
         to_be_tested = (find_the_most_Similar(test_df,3)[['indicator','area_type','area_name','year','sim_diff']].reset_index(drop=True))
         list_  = [{'indicator':'Primary school children', 'area_type': 'Health board','area_name': 'Highland'  , 'year':2016, 'sim_diff':0.0 },{'indicator':'Mid-year population estimate - aged under 18 years' ,'area_type': 'Health board', 'area_name': 'Borders' ,   'year':2016, 'sim_diff':0.0 },{'indicator':'Mid-year population estimate - aged 1-4 years' ,'area_type': 'Health board','area_name': 'Lanarkshire' ,  'year':2016, 'sim_diff':0.0}]
         ground_df = pd.DataFrame(list_,columns = ['indicator','area_type', 'area_name', 'year','sim_diff'])
         assert_frame_equal(ground_df,to_be_tested)
    
    
    def test_sign_different(self):
         self.assertEqual(1, is_statistical_different([1,0.5,1,5], 10))
         self.assertEqual(1,is_statistical_different([5,2.5,7.5],1))
         self.assertEqual(0,is_statistical_different([22,20,24],21))
    
    def test_is_better_without_ci(self):
        x = np.array([[0.5,1,1.5], [2,6,10],[11,6,16]])
        score = is_better_for_measure_without_cf(x,10)
        self.assertEqual(score,1/3.0)
        
    def test_is_worse(self):
        x = np.array([[0.5,1,1.5], [2,6,10],[11,6,16]])
        score = is_worse_for_measure_without_cf(x,10)
        self.assertEqual(score,2/3.0)
    
    def test_is_what(self):
       a= is_what(10,5,15)
       b = is_what(6,7,10)
       c = is_what (20,5,15)
       
       self.assertEqual(a,[0,0,0,1])
       self.assertEqual(b,[1,1,0,0])
       self.assertEqual(c,[1,0,1,0])
       
    
    def test_score(self):
        a= is_what(10,5,15)
        b = is_what(6,7,10)
        c = is_what (20,5,15)
        arr = [[10,6,20], [5,7,5], [15,10,15]]
        score_ = (score(a,b,c))
        self.assertEqual([0.75,0.5,0.25,0.25],score_.tolist()[0])
    
    def test_difference(self):
        diff = find_difference(10,5, 7, 6)
        self.assertEqual(diff,3)
        diff_1 = find_difference(10,np.nan,np.nan,12)
        self.assertEqual(diff_1,2)

if __name__ == '__main__':
    unittest.main()