# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:00:59 2022

@author: Jesús S. Alegre
"""
import numpy as np
import pandas as pd


def load(file):
    Mdata=pd.read_excel(file)
    
    return Mdata


def analysis(data, scenarios, folds):
    analytics = []
    for scenario in scenarios:
        sub_set = data[data.scenario==scenario]
        for k in folds:
            sub_set_k = sub_set[sub_set['K-folds']==k]
            std = sub_set_k.std()
            mean = sub_set_k.mean()
            #analytics.append([scenario, k, mean['Gridsearch Acc'],std['Gridsearch Acc'],mean['CV acc'],std['CV acc']])
            analytics.append([scenario, k, mean['SCV c-index'],std['SCV c-index']])
            # analytics = pd.DataFrame(analytics)
            # analytics.columns['scenario','k-folds','mean Gridsearch','std Gridsearch','mean CV','std CV']
            
    return analytics


def main():
    
    #data_results = load('results_KNN_V3_with_newVAR_V2.xlsx')
    data_results = load('RSF_results.xlsx')
    k_folds_max = data_results['K-folds'].max()
    k_folds = range(2,k_folds_max+1)
    scenario_max = data_results['scenario'].max()
    scenarios = range(1,scenario_max+1)
    
    Mdatas_results = analysis(data_results, scenarios, k_folds)
    Mdatas_results = pd.DataFrame(Mdatas_results)
    # Mdatas_results.columns=['scenario','k-folds','mean Gridsearch','std Gridsearch','mean CV','std CV']            
    # Mdatas_results.to_excel('Analytics_KNN_V3_with_newVAR_V2.xlsx') 
    Mdatas_results.columns=['scenario','k-folds','mean SCV c-index','std SCV c-index']            
    Mdatas_results.to_excel('Analytics_RSF_results.xlsx') 
    
    return Mdatas_results

if __name__ == "__main__":
    results = main()