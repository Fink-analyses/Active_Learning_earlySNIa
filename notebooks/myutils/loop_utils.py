import os,io, glob
import pickle
import requests
import numpy as np
import pandas as pd
from actsnfink import *
from copy import deepcopy
from pathlib import Path
from . import plot_utils as pu
from distutils.dir_util import copy_tree
from actsnclass.metrics import get_snpcc_metric
from sklearn.ensemble import RandomForestClassifier


def train_and_test_initial_model(train_for_loop,test_for_loop,strategy):
    """Train and test initial model in loop

    Args:
        train_for_loop (pd.DataFrame): Training data
        test_for_loop (pd.DataFrame): Testing data
        strategy (str): AL strategy used
    """
    # Stats train
    print(f"Number of training light-curves {len(train_for_loop)}")
    train_for_loop.groupby('type').count()['c_g']

    # train initial model
    clf = RandomForestClassifier(random_state=42, n_estimators=1000)
    clf.fit(train_for_loop[list(train_for_loop.keys())[:-3]], train_for_loop['type'].values == 'Ia')
    # save mode
    pickle.dump(clf, open('../data/initial_state/' + strategy + '/models/model.pkl', 'wb'))

    # make predictions
    pred = clf.predict(test_for_loop[list(test_for_loop.keys())[:-2]])
    # pred_prob = clf.predict_proba(test_for_loop[list(test_for_loop.keys())[:-2]])

    # calculate metrics
    metric_names, metric_values = get_snpcc_metric(pred, test_for_loop['type'].values == 'Ia')
    
    output = [f"{k} {round(metric_values[i],2)}" for i, k in enumerate(metric_names)]
    print(output)


def create_directory_structure(dir_output,date,strategy):
    """Create directory structure

    Args:
        dir_output (str): path output
        date (str): date of loop
    """
    dirname_output = dir_output +  date + '/'
    
    for name in [dirname_output + '/', 
                    dirname_output + '/' + strategy + '/', 
                    dirname_output + '/' + strategy + '/models/', 
                    dirname_output + '/' + strategy + '/class_prob/',
                    dirname_output + '/' + strategy + '/metrics/', 
                    dirname_output + '/' + strategy + '/queries/',
                    dirname_output + '/' + strategy + '/training_samples/', 
                    dirname_output + '/' + strategy + '/test_samples/']:
        if not os.path.isdir(name):
            os.makedirs(name)

def read_previous_state(d,date,new_labels,dir_output,strategy,verbose=False):
    """Load RF model

    Args:
        d (int): counter on loop
        date (str): date in stirng format
        new_labels (dict): labels in the AL loop
        dir_output (str): output directory
    """
    # read previous training sample and trained model
    if date == list(new_labels.keys())[0]:
        dir_training = dir_output + date + '/' + strategy  + '/training_samples/train.csv'
        is_file = os.path.isfile(dir_training)
        if is_file and d!=0: # in case no updates were done in training later on
            # if already trained
            if verbose:
                print(f'Read {date}')
            train_for_loop = pd.read_csv(dir_training, index_col=False)
            clf_before = pickle.load(open(dir_output +  date + '/' + strategy + '/models/model.pkl', 'rb'))
            train_for_loop_probas = pd.read_csv(dir_training.replace('train.csv','probabilities.csv'), index_col=False)
        else:
            print('Read initial state')
            train_for_loop = pd.read_csv(dir_output + 'initial_state/' + strategy + \
                                            '/training_samples/train_after_loop.csv', 
                                            index_col=False)
            
            train_for_loop_probas = pd.DataFrame()
            train_for_loop_probas['objectId'] = train_for_loop['objectId'].values
            train_for_loop_probas['probability'] = np.zeros(len(train_for_loop))
            
            clf_before = pickle.load(open(dir_output + 'initial_state/' + strategy + \
                                            '/models/model.pkl', 'rb'))
    else:
        is_file = False
        indx = list(new_labels.keys()).index(date) - 1
        
        while not is_file and indx >= 0:
            
            date_before = list(new_labels.keys())[indx]

            is_file = os.path.isfile(dir_output + date_before + \
                                            '/' + strategy  + \
                                            '/training_samples/train.csv')
            
            if is_file:
                train_for_loop = pd.read_csv(dir_output +  date_before + \
                                            '/' + strategy  + \
                                            '/training_samples/train.csv', 
                                            index_col=False)    
        
                clf_before = pickle.load(open(dir_output +  date_before + '/' + strategy + \
                                            '/models/model.pkl', 'rb'))
                
                train_for_loop_probas = pd.read_csv(dir_output +  date_before + \
                                            '/' + strategy  + \
                                            '/training_samples/probabilities.csv', index_col=False)
                print(f'Read {date_before} state')
            else:
                indx = indx - 1
    # print(f'Read training {len(train_for_loop)}')
    return train_for_loop, clf_before, train_for_loop_probas


def get_alert_data(new_labels, date,verbose=False):
    """Get Fink alert data

    Args:
        new_labels (dic): dictionary with AL loop labels
        date (str): date currently in the loop
        verbose (bool, optional): If print information. Defaults to False.

    Returns:
        list: list containing all alert information
    """
    alerts_list = []
    
    for i in range(new_labels[date].shape[0]):
        
        name = new_labels[date][i][0]
        # print(name)

        r = requests.post(
            'https://fink-portal.org/api/v1/objects',
            json={
                'objectId':name,
                'output-format': 'json',
                'withupperlim': 'True' # online ML uses bad quality
            }
        )

        try:
            # Format output in a DataFrame
            pdf = pd.read_json(io.BytesIO(r.content))
            pdf = pdf[pdf['d:tag'] != 'upperlim'] #using valid and badquality detections
        except Exception:
            pdf= pd.DataFrame()

        if pdf.shape[0] == 0:
            print(f"No alerts found for object: {name}")
        else:
            # add label
            pdf['type'] = new_labels[date][i][1]

            if verbose:
                print(f'Got data for {name}')
    
            alerts_list.append(pdf)
    
    return alerts_list


def AL_loop(new_labels, strategy,test_for_loop, dir_suffix ="", proba_cut = True, plot_lcs=False,verbose=False):

    dir_output= f'../data_{dir_suffix}/' if dir_suffix!="" else  '../data/'
    if not os.path.exists(f"{dir_output}/initial_state"):
        print("No path",f"{dir_output}/initial_state")
        os.makedirs(dir_output, exist_ok=True)
        # copy initial state
        copy_tree("../data/initial_state", f"{dir_output}/initial_state")
    
    metrics_all_list = []

    for d, date in enumerate(list(new_labels.keys())):
    
        if verbose:
            print(date)
        
        # read previous state
        train_for_loop, clf_before, train_for_loop_probas = read_previous_state(d,date,new_labels,dir_output,strategy,verbose=verbose)
    
        # create output directory structure
        create_directory_structure(dir_output,date,strategy)
    
        #Get alert data        
        alerts_list = get_alert_data(new_labels, date,verbose=False)
        
        # Check data from Fink
        if len(alerts_list)>0:

            # reformat alerts into a list
            alerts_format_list = []
            alerts_pd = pd.concat(alerts_list, ignore_index=True)

            for indx_obj in range(new_labels[date].shape[0]):
                # isolate one object
                flag_obj = alerts_pd['i:objectId'].values == new_labels[date][indx_obj][0]
        
                # separate only dates until the alert was sent to follow-up
                cjd = alerts_pd[flag_obj]['i:jd'].values
                flag_jd = cjd < float(new_labels[date][indx_obj][2])

                # get data until date in the loop
                # check if we have these alerts
                if len(alerts_pd[flag_obj]['i:jd'].values[flag_jd])>0: 
                    lc = pd.DataFrame([[new_labels[date][indx_obj][0], 
                                    alerts_pd[flag_obj]['i:candid'].values[flag_jd][np.argsort(alerts_pd[flag_obj]['i:jd'].values[flag_jd])[-1]],
                                    alerts_pd[flag_obj]['i:jd'].values[flag_jd],         
                                    alerts_pd[flag_obj]['i:fid'].values[flag_jd], 
                                    alerts_pd[flag_obj]['i:magpsf'].values[flag_jd],
                                    alerts_pd[flag_obj]['i:sigmapsf'].values[flag_jd],
                                    new_labels[date][indx_obj][1]]], 
                                    columns=['objectId', 'candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf', 'TNS'], index=[0], dtype=object)
                
                    alerts_format_list.append(lc)

                    
                    
            # Have data to add to the loop           
            if len(alerts_format_list)>0:
                alerts_format_pd = pd.concat(alerts_format_list, ignore_index=True)
            
                # convert from mag to FLUXCAL
                alerts_flux = convert_full_dataset(alerts_format_pd, obj_id_header='objectId')
            
                # extract features
                alerts_features = featurize_full_dataset(alerts_flux, screen=False,
                                                         ewma_window=3, 
                                                          min_rising_points=1, 
                                                          min_data_points=3, rising_criteria='ewma')
            
                # filter alerts with zero in all filters
                flag_zero = np.logical_and(alerts_features['a_g'].values == 0.0,alerts_features['a_r'].values == 0.0)
                # flag_zero = np.logical_or(alerts_features['a_g'].values == 0.0,
                #                            alerts_features['a_r'].values == 0.0)
                alerts_use = deepcopy(alerts_features[~flag_zero])

                if verbose:
                    if len(alerts_features[flag_zero])>0:
                        print(f'Missing {alerts_features[flag_zero]["id"]}')
                        print(alerts_flux[alerts_flux['id'].isin(alerts_features[flag_zero]["id"])])
            
                # go through events if they have at least one band with features
                if alerts_use.shape[0] > 0:
                    
                    alerts_use.rename(columns={'id':'objectId'}, inplace=True)
                    alerts_use['loop'] = list(new_labels.keys()).index(date) + 30
                
                    # see what current model says about queried alerts
                    pred_prob_query = clf_before.predict_proba(alerts_use[list(alerts_use.keys())[2:-1]])

                    # check classification probability
                    probabilities_list= []
                    if proba_cut:
                        keep_index = []
                        for j in range(len(pred_prob_query)):
                            if (pred_prob_query[j][0]>0.4) & (pred_prob_query[j][0]<0.6):
                                keep_index.append(j)
                                probabilities_list.append(float(pred_prob_query[j][0]))
                        new_alerts_use = alerts_use.iloc[keep_index]
                        if verbose:
                            print(probabilities_list)
                    else:
                        new_alerts_use = alerts_use
                        probabilities_list= pred_prob_query[:,0].flatten()
                    if verbose:
                        print(f"New alerts to use {len(new_alerts_use)}")
            
                    if len(new_alerts_use) > 0:

                        if plot_lcs:
                            for i, lcname in enumerate(new_alerts_use['objectId'].unique()):
                                lc = alerts_format_pd[alerts_format_pd['objectId']==lcname]
                                proba = probabilities_list[i]
                                pu.plot_lc(lc.reset_index(), proba, dir_suffix)
                                
     
                        # update training
                        train_updated = pd.concat([train_for_loop, new_alerts_use], ignore_index=True)
                        if verbose:
                            print('    train_updated.shape = ', train_updated.shape)
                            
                        # save probabilities:
                        proba_out = np.append(train_for_loop_probas['probability'].values, probabilities_list)
                        train_for_loop_probas = pd.DataFrame()
                        train_for_loop_probas['objectId'] = train_updated['objectId'].values
                        train_for_loop_probas['probability'] = proba_out
                        train_for_loop_probas.to_csv(dir_output + date +'/' + strategy  + '/training_samples/probabilities.csv', index=False)

                        # save to file
                        train_updated.to_csv(dir_output + date +'/' + strategy  + \
                                                          '/training_samples/train.csv', 
                                                           index=False)
                    
                        # train model
                        clf = RandomForestClassifier(random_state=42, n_estimators=1000)
                        clf.fit(train_updated[list(train_updated.keys())[:-3]], 
                                    train_updated['type'].values == 'Ia')
                    
                        # make predictions
                        pred = clf.predict(test_for_loop[list(test_for_loop.keys())[:-2]])
                        pred_prob = clf.predict_proba(test_for_loop[list(test_for_loop.keys())[:-2]])
                        
                        # save mode
                        pickle.dump(clf, open(dir_output + date + '/' + strategy + '/models/model.pkl', 
                                                  'wb'))
                    
                        # save predictions
                        pred_prob_pd = pd.DataFrame(np.hstack([test_for_loop['objectId'].values.reshape(-1,1), 
                                                        np.array(pred_prob[:,1]).reshape(-1,1)]), 
                                                        columns=['objectId','probIa'])
                        pred_prob_pd.to_csv(dir_output +  date + '/' + strategy + \
                                            '/class_prob/test_class_prob.csv', index=False)
                
                        # calculate metrics
                        names, res = get_snpcc_metric(pred, test_for_loop['type'].values == 'Ia')
                        # print('   res = ', res)
                        
                        metric_pd = pd.DataFrame([[date] + [list(new_labels.keys()).index(date) + 30] + res + \
                                                     [list(alerts_use['objectId'].values)]],columns=['date','loop'] + names + ['query_objectIds'])
                        metric_pd['date_plot'] = d 

                        metric_pd.to_csv(dir_output +  date +'/' + strategy  + \
                                                          '/metrics/metric.csv', 
                                                           index=False)
                        metrics_all_list.append(metric_pd)

                        
    if len(metrics_all_list)>0:        
        metrics = pd.concat(metrics_all_list, ignore_index=True)
        metrics['n spectra']= metrics['query_objectIds'].apply(lambda x: len(x)).cumsum()

    else:
        metrics = pd.DataFrame()

    metrics.to_csv(dir_output + 'metrics.csv', index=False)

    return metrics


def read_previous_metrics(dir_output,strategy):
        """"Read metrcs files for als dates
        """
        tmp = []
        dat_path = np.sort(glob.glob(f"{dir_output}/*/{strategy}/metrics/metric.csv"))
        print(f"Read {len(dat_path)} metrics")
        for dat in dat_path:
            metric_path = dat
            if Path(metric_path).exists():
                tmp.append(pd.read_csv(metric_path))
        metrics = pd.concat(tmp, ignore_index=True)
        metrics['n spectra']= metrics['query_objectIds'].apply(lambda x: x.count('ZTF')).cumsum()
        return metrics

def convert_DF_dic_labels(df):
    """TNS labels DataFrame to dictionary

    Args:
        df (pd.DataFrame): TNS classifications (Fink format)

    Returns:
        dictionary : formated labels in dictionary for AL loop
    """
    # convert to new_labels format
    dic_labels = {}
    for dat in df['discoveryjd+12'].unique()[:60]:
        sel = df[df['discoveryjd+12']==dat]
        tmp_arr = sel[['ztf_names','type AL','discoveryjd+12']].to_numpy()

        date_in_str_fmt = sel['discoveryjd+12_strfmt'].values[0]
        dic_labels[date_in_str_fmt] = tmp_arr

    return dic_labels