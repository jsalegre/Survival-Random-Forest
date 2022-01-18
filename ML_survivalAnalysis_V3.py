# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:17:17 2022

@author: Jesús S. Alegre
"""

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sksurv.ensemble import RandomSurvivalForest
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

from sksurv.metrics import concordance_index_censored
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from eli5.sklearn import PermutationImportance
import eli5
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import cumulative_dynamic_auc
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

### NOTES: to be implemented
##############################################################################
# try concordance_index_ipcw to analyze better censored data - Done

# The function cumulative_dynamic_auc implements an estimator of the cumulative/dynamic 
# area under the ROC for a given list of time points. --> useful measure of performance if a specific time range is of primary interest 



### Classes definition
##############################################################################

class reciprocal_int:
    def __init__(self, a, b):
        self._distribution = reciprocal(a, b)

    def rvs(self, *args, **kwargs):
        return self._distribution.rvs(*args, **kwargs).astype(int)


## Importing data and defining variables
##############################################################################

data_results = pd.read_excel('analisis 2022 GB version 1.xlsx')  
headers = data_results.columns  

var_blanco = ['nhc','sexo','crisis','cognitivo','deficit','deficitpost','ki67',
              'ciclosrecibidos','qtxinterrupcion','captacion_tipo_i','necrosis_forma_i',
              'adc_cuali','desv_linmedia_i','flairbeyond_semi','flairbeyond65',
              'flairbeyond70','calloso']

var_rojo = ['edad','mgmt','bulktumor','voltotal_flair_i','diam_f_1','diam_f_2',
            'diam_f_thick','vol_restostumorales','eor','ftr','edad65']

var_verde = ['cefalea','karnofsky','morfologia_i','morf_bordes_i','loc_i','eor_r',
             'ciclos6','diamt1_2d','diamflair_2d','diam2000flair_2d','ftr3','ftr4',
             'ftr_3d_5','ftr_diam','Ala_r','Glu_r','p31pi','p31pdet']

var_azul = ['ait','restoqx','monitoriz','navegad','hematnoqx','hematqx','stupp_rtxrtx',
            'qtxadyuv','qtxesquem','ecog0','ecog3','ecog6','ecog9','ecog12','hemort1_t2_i',
            'flair60','bulk_3dmanual','flair_3dmanual','ftr_3dmanual','flair60_3dmanual',
            'eor2cm','ftr25','necrosis40','bulk50','bulk60','flair75','ftr_3d_4','ntr03',
            'bulk25','bulk40','H1_Ac','H1_Asp','H1_Cho','H1_Cr','H1_GABA','H1_Glc',
            'H1_Gln','H1_Gly','H1_GPC','H1_GSH','H1_Ile','H1_Lac','H1_Leu','H1_Myo',
            'H1_NAA','H1_PC','H1_PCr','H1_PE','H1_Tau','H1_Thr','H1_Val','H1_X_CrCH2',
            'H1_Gua','H1_GPC_Cho','H1_Cr_PCr','H1_Glu_Gln','H1_Lip13a','H1_Lip13b',
            'H1_Lip09','P31_XX','P31_PME1','P31_PME2','P31_PDE1','P31_PDE2','P31_PDE3']

var_naranja = ['ecog','multicent','elocuencia','rtxincompleta','qtxincompleta',
               'captacion_volumen_i','necrosis_volumen_i','vscmax','adc_cuanti',
               'satelites_multifocal','capt_ependimaria_i','swi_t2_i','diam_t1_1',
               'diam_2_t1','diam_t1_thick','ntr','kps90','ntr4','ftr5','ftr_diam_5',
               'H1_Ala','H1_Glu','P31_PMEt','P31_Pi','P31_PDEt']

var_amarillo = ['tiemposeguimiento','muerte']

var_gris = ['recurrnosi','tiemporecurrenciam']

# var_cat = ['sexo','cefalea','karnofsky','crisis','ait','cognitivo','deficit',
#            'ecog','multicent','elocuencia','restoqx','monitoriz','navegad','hematnoqx',
#            'hematqx','deficitpost','mgmt','stupp_rtxrtx','qtxadyuv','qtxesquem','rtxincompleta',
#            'qtxincompleta','qtxinterrupcion','ecog0','ecog3','ecog6','ecog9','ecog12',
#            'recurrnosi','muerte','captacion_tipo_i','necrosis_forma_i','adc_cuali',
#            'satelites_multifocal','capt_ependimaria_i','hemort1_t2_i','swi_t2_i',
#            'morfologia_i','morf_bordes_i','loc_i','eor_r','flair60','flair60_3dmanual',
#            'edad65','ciclos6','eor2cm','ftr25','necrosis40','bulk50','bulk60','flair75',
#            'kps90','diam2000flair_2d','ftr3','ntr4','ftr4','ftr_3d_4','ntr03','bulk25',
#            'bulk40','ftr5','ftr_3d_5','ftr_diam_5','flairbeyond65','flairbeyond70',
#            'calloso','H1_Cr_PCr','Ala_r','Glu_r','p31pi','p31pdet']

var_cat = ['sexo','cefalea','karnofsky','crisis','ait','cognitivo','deficit',
           'ecog','multicent','elocuencia','restoqx','monitoriz','navegad','hematnoqx',
           'hematqx','deficitpost','mgmt','stupp_rtxrtx','qtxadyuv','qtxesquem','rtxincompleta',
           'qtxincompleta','qtxinterrupcion','ecog0','ecog3','ecog6','ecog9','ecog12',
           'captacion_tipo_i','necrosis_forma_i','adc_cuali',
           'satelites_multifocal','capt_ependimaria_i','hemort1_t2_i','swi_t2_i',
           'morfologia_i','morf_bordes_i','loc_i','eor_r','flair60','flair60_3dmanual',
           'edad65','ciclos6','eor2cm','ftr25','necrosis40','bulk50','bulk60','flair75',
           'kps90','diam2000flair_2d','ftr3','ntr4','ftr4','ftr_3d_4','ntr03','bulk25',
           'bulk40','ftr5','ftr_3d_5','ftr_diam_5','flairbeyond65','flairbeyond70',
           'calloso','H1_Cr_PCr','Ala_r','Glu_r','p31pi','p31pdet']


### Selecting objective of the Phd to evaluate and its corresponding subset
##############################################################################
objetivo = 1  # 1 para objetivo principal, 2 para objetivo secundario

if objetivo == 1:
    var_objetivo = var_amarillo
    var_descartar = var_gris
    var_salida = 'muerte'
    var_2 = 'tiemposeguimiento'
elif objetivo ==2:
    var_objetivo = var_gris
    var_descartar = var_amarillo  
    var_salida ='recurrnosi'
    var_2 = ['tiemporecurrenciam']

Mdata = data_results.drop(columns = var_descartar)


## Filling missing data

# Subset of data of positive cases
Mdata_pos = Mdata[Mdata[var_salida]==1]
Mdata_neg = Mdata[Mdata[var_salida]==0]


print('\n')
for row in Mdata_pos.index:
    distances = Mdata_pos[var_2]-Mdata_pos[var_2][row]
    distances = abs(distances)
    distances = distances.drop([row])
    distances = distances.sort_values()
    closest_points = distances[0:int(len(Mdata_pos.index)/3)] # stores the index and values of the closes points
    
    subset = Mdata_pos.loc[closest_points.index,:]
    
    for header in Mdata.head():
        if math.isnan(Mdata_pos.loc[row,header]) == True:
            if header in var_cat:
                moda = subset[header].mode()
                if moda.empty == True: # if all the data points of the neightbors are empty, take the mode of the whole data set
                    moda_ = Mdata[header].mode()
                    moda_ = moda_.tolist()[0]
                    Mdata_pos.loc[row,header] = moda_
                else:
                    moda = moda.tolist()[0]
                    Mdata_pos.loc[row,header] = moda
            else:
                if subset[header].isna().sum() == int(len(Mdata_pos.index)/3): # if all the data points of the neightbors are empty, take the mean of the whole data set
                    Mdata_pos.loc[row,header] = int(Mdata[header].mean())    
                else:
                    Mdata_pos.loc[row,header] = int(subset[header].mean())
                
for row in Mdata_neg.index:
    distances = Mdata_neg[var_2]-Mdata_neg[var_2][row]
    distances = abs(distances)
    distances = distances.drop([row])
    distances = distances.sort_values()
    closest_points = distances[0:int(len(Mdata_neg.index)/3)] # stores the index and values of the closes points
    
    subset = Mdata_neg.loc[closest_points.index,:]
    
    for header in Mdata.head():
        if math.isnan(Mdata_neg.loc[row,header]) == True:
            if header in var_cat:
                moda = subset[header].mode()
                if moda.empty == True: # if all the data points of the neightbors are empty, take the mode of the whole data set
                    moda_ = Mdata[header].mode()
                    moda_ = moda_.tolist()[0]
                    Mdata_neg.loc[row,header] = moda_
                else:
                    moda = moda.tolist()[0]
                    Mdata_neg.loc[row,header] = moda
            else:
                if subset[header].isna().sum() == int(len(Mdata_neg.index)/3): # if all the data points of the neightbors are empty, take the mean of the whole data set
                    Mdata_neg.loc[row,header] = int(Mdata[header].mean())    
                else:
                    Mdata_neg.loc[row,header] = int(subset[header].mean())
    

frames = [Mdata_pos, Mdata_neg]
Mdata_filled = pd.concat(frames)



### Normalizar los datos menos los de salida y las variables continuas
##############################################################################
headers = Mdata_filled.columns
X = Mdata_filled.drop(columns=var_objetivo)
y = Mdata_filled[var_objetivo]

data_cont = X.drop(columns=var_cat)
data_cat = X[var_cat]

headers_cont = data_cont.columns
headers_cat = data_cat.columns
header_y = y.columns
headers = np.r_[headers_cont,headers_cat, header_y]

scaler = StandardScaler()
norm_data = scaler.fit_transform(data_cont)
norm_data = np.c_[norm_data,data_cat,y]
norm_data = pd.DataFrame(norm_data)
norm_data.columns=headers    
Mdata_filled_norm=norm_data


### Definir los distintos escenarios a evaluar
##############################################################################
    
# Escenario 1: Todas las variables
Mdata1 = Mdata_filled_norm

# Escenario 2: Todas las variables a excepción de las de color AZUL
Mdata2 = Mdata1.drop(columns = var_azul)

# Escenario 3: Todas a excepción de la color AZUL Y VERDE
Mdata3 = Mdata2.drop(columns = var_verde)

# Escenario 4: Todas a excepción de las AZUL VERDE Y BLANCAS
Mdata4 = Mdata3.drop(columns = var_blanco)

# Escenario 5: únicamente las variables de color ROJO
Mdata5 = Mdata1[var_rojo + var_objetivo]

escenarios = [Mdata1, Mdata2, Mdata3, Mdata4, Mdata5]
n=0


### choose random seeds: define here so they are the same for all the scenarios
##############################################################################
np.random.seed(0)
seeds = np.random.permutation(1000)[:5]


### choosing k_folds to test:
##############################################################################
k_folds = list(range(2,11))

### Bucle para analizar cada escenario
##############################################################################
data_results = []

for escenario in escenarios:
    n=n+1
    
    ### Separate covariates from output variables
    Xt_ = escenario.drop(columns=var_objetivo)
    y_ = escenario[var_objetivo]
    
    ### Preparing output variables to have necessary structure: (boolean, time)
    for head in y_.columns:
        if max(y[head])==1:
            var_1 = head
        else:
            var_2 = head
    
    y_ = y_[[var_1,var_2]]
    y_[var_1] = y_[var_1].astype(bool)
    y_split = y_[var_1]
    y = y_.to_records(index=False) # not convince with the shape of y
    
    ### Censoring descriptive data of the dataset
    print('\n\n Escenario {}'.format(n))
    print('\n')
    print(f'Number of samples: {len(y_)}')
    print(f'Number of right censored samples: {len(y_.query("{0} == False".format(var_1)))}')
    print(f'Percentage of right censored samples: {100*len(y_.query("{0} == False".format(var_1)))/len(y_):.1f}%')

    
    ### Selecting hyperparameter ranges of the model
    param_distributions1 = {'randomsurvivalforest__max_features': reciprocal_int(2, len(Xt_.columns)),      # max = max number of columns
                            'randomsurvivalforest__max_depth': reciprocal_int(5, len(Xt_)),        # find what is the max
                            'randomsurvivalforest__min_samples_leaf': reciprocal_int(1, 100), # find what is the max
                            }
    
    
    ### Probando algoritmo de rejilla para encontrar la mejor configuración
    for k_fold in k_folds:
        print('K_folds {}'.format(k_fold))
        for seed in seeds:
            print('seed {}'.format(seed))
            
            ### Create the stratified folds
            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
            
            ### Create the workflow pipeline - A bit useless as data is alreaday preprocessed, but kept for format compatibility
            rsf = make_pipeline(RandomSurvivalForest(random_state=seed))
            model_random_search = RandomizedSearchCV(rsf, param_distributions=param_distributions1, n_iter=50, n_jobs=-1, cv=skf.split(Xt_,y_split), random_state=seed)
            model_random_search.fit(Xt_, y)
                        
            print(f"The c-index of random survival forest using a {model_random_search.__class__.__name__} is "
                  f"{model_random_search.best_score_:.3f}")
            print(f"The best set of parameters is: {model_random_search.best_params_}"    )
            
                    
            ### Guardando variables optimas
            max_depth = model_random_search.best_params_['randomsurvivalforest__max_depth']
            max_features = model_random_search.best_params_['randomsurvivalforest__max_features']
            min_samples_leaf = model_random_search.best_params_['randomsurvivalforest__min_samples_leaf']
            best_score = model_random_search.best_score_
                  
            data_results.append([n,k_fold,seed,max_depth,max_features,
                                 min_samples_leaf, best_score])
            
            ### Analyze the feature importance
            # model = RandomSurvivalForest(random_state=seed, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf)
            # tr = model.fit(Xt_,y)
            # perm = PermutationImportance(tr, n_iter=50, random_state=seed, cv=k_fold)
            # perm = perm.fit(Xt_,y)
            
            # info = pd.DataFrame({'Value':[n,seed,k_fold],'name':['escenario','seed','Kfold']}, index=[0,1,2])
            # features_results = pd.DataFrame(perm.feature_importances_)
            # feature_labels = pd.DataFrame(Xt_.columns)
            # permutation_import = pd.concat([info,features_results,feature_labels],axis=1)
            # s_name = 'S{}_K{}_R{}'.format(n,seed,k_fold)
            
            # with pd.ExcelWriter('RSF_results.xlsx') as writer:  
            #     permutation_import.to_excel(writer, sheet_name=s_name)

            
data_results=pd.DataFrame(data_results)
data_results.columns=['scenario','K-folds','seed','Optimal max_depth','Optimal max_features','Optimal min_samples_leaf','SCV acc']
with pd.ExcelWriter('RSF_results.xlsx') as writer:  
    data_results.to_excel(writer, sheet_name='results')
    
            