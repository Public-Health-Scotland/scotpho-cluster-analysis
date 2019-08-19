# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:27:40 2019

@author: Panagiotis Ntoulos
"""
import numpy as np
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

#score in detail for significantly different
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

def has_confident_intervals(value):
    return (np.logical_not(np.isnan(value)))

def find_difference(comparator,lower, upper, measure):
    if(has_confident_intervals(lower)):
        upper_diff = abs(comparator-upper)
        lower_diff = abs(comparator-lower)
        return min(upper_diff,lower_diff)
    else:
        return abs(comparator-measure)
    