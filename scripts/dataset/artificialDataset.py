# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  artificialDataset script creates an artificial dataset for face similarity scores, gender, age and    #
#  height estimations, and time of interaction for evaluation of MMIBN in Recognition Memory.            #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #
#    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (under review), 'Multi-modal Open World User #
#    Identification', ACM Transactions on Human-Robot Interaction (THRI).                                #
#                                                                                                        #            
#  artificialDataset, RecognitionMemory and each script in this project is under the GNU General Public  #
#  License.                                                                                              #
#========================================================================================================#

import numpy as np
import crossValidation as cv
import math
from scipy.stats import truncnorm
import os
import csv
import shutil
import WeightOptimisation as wop
import time
import multiprocessing as mp
import itertools
import pandas

# import random

# Initial seeding
np.random.seed(1234)

def getCleanDB(num_people, num_samples, params, param_specs):
    """
    param_specs[num_param][0] = param_form_method
    param_specs[num_param][1] = param_type
    param_specs[num_param][2] = num_samples per person for db_param ( = num_people for "F"; = 1 for "G, A, H"; = num_samples (number of times a person was seen) for "T" )
    param_specs[num_param][3] = [uniform_range_min, uniform_range_max]
    param_specs[num_param][4] = [clip_range_min, clip_range_max]
    param_specs[num_param][5] = mu / num_curves
    param_specs[num_param][6] = sigma
    param_specs[num_param][7] = extras (labels for "G", period for "T")
    param_specs[num_param][8] = decimals
    """
    db_param_list = []
    
    if not isinstance(num_samples, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_samples_list = [num_samples for _ in range(0, num_people)]
    else:
        num_samples_list = num_samples[:]
    
    if "T" in params:
        # update T num_samples (to avoid compatibility errors)
        index_T = params.index("T")
        param_specs[index_T][2] = num_samples_list
        
    for num_param in range(0, len(params)):
        param = params[num_param]
        param_form_method = param_specs[num_param][0]
        db_param = []
        if param_specs[num_param][2] == 1:
            size_samples = num_people
        else:
            size_samples = [num_people, param_specs[num_param][2]]
        
        if param == "F":
            db_param = getSimilarityMatrix(num_people, param_form_method, param_specs[num_param][3][0], param_specs[num_param][3][1], param_specs[num_param][8])
            
        elif param_form_method == "uniform-range":
            db_param = getUniformRangeVar(size_samples, param_specs[num_param][3][0], param_specs[num_param][3][1], param_specs[num_param][1], param_specs[num_param][8])
        
        elif param_form_method == "uniform-label":
            db_param = getUniformLabelVar(size_samples, param_specs[num_param][7])
                
        elif param_form_method == "gaussian-range":
            db_param = getGaussianRangeVar(size_samples, param_specs[num_param][5], param_specs[num_param][6], param_specs[num_param][3][0], param_specs[num_param][3][1], 
                                           param_specs[num_param][4][0], param_specs[num_param][4][1], param_specs[num_param][1], param_specs[num_param][8])
            
        elif param_form_method == "GMM-range":
            db_param = getGMMVar(num_people, param_specs[num_param][2], param_specs[num_param][3][0], param_specs[num_param][3][1], 
                                 param_specs[num_param][5], param_specs[num_param][6], param_specs[num_param][1], param_specs[num_param][8])
        
        if param == "T":
            db_param = list(db_param)
        db_param_list.append(db_param)
    clean_db = [[] for _ in range(0, num_people)]        
    for num_param in range(0, len(params)):
        db_param = db_param_list[num_param]
        param_samples = []
        for num_person in range(1, num_people+1):
            param = params[num_param]
            if isinstance(db_param[num_person-1], list):
                person_val = []
                person_val.extend(db_param[num_person-1])
            else:
                person_val = db_param[num_person-1]
            if param == "F":
                clean_db[num_person-1].append([person_val for _ in range(0, num_samples_list[num_person-1])])
            elif param == "T":
                clean_db[num_person-1].append(person_val)
            else:
                clean_db[num_person-1].append([[person_val, 1.0] for _ in range(0, num_samples_list[num_person-1])])
            if param == "T":
                person_time_list = []
                for person_val_time in person_val:
                    person_time_list.append(getTimeFromInterval(person_val_time, param_specs[num_param][7]))
                db_param_list[num_param][num_person-1] = person_time_list
    return clean_db, db_param_list

def saveArtificialDB(num_people, num_samples, params, param_specs, db, db_file):
    
    if not isinstance(num_samples, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_samples_list = [num_samples for _ in range(0, num_people)]
    else:
        num_samples_list = num_samples[:]
        
    if os.path.isfile(db_file):
        os.remove(db_file)
    with open(db_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["I", "F", "G", "A", "H", "T"])
    t_index = params.index("T")
    with open(db_file, 'a') as outcsv:
        writer = csv.writer(outcsv)
        for num_person in range(1, num_people+1):
            for num_sample in range(0, num_samples_list[num_person-1]):
                row = [num_person]
                for num_param in range(0, len(params)):
                    val = db[num_person-1][num_param][num_sample]
                    if num_param == t_index:
                        val = getTimeFromInterval(val, param_specs[num_param][7])
                    row.append(val)
                writer.writerow(row)

def saveRecognitionFile(num_people, params, param_specs, db, fold_seq_order, new_id_list, recog_file, isInitRecog=False, isTraining=False, acc=1.0):
    if os.path.isfile(recog_file):
        os.remove(recog_file)
    with open(recog_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        if isInitRecog:
            writer.writerow(["I_est", "F", "G", "A", "H", "T", "N"])
        else:
            writer.writerow(["I", "F", "G", "A", "H", "T", "R", "N"])
    f_index = params.index("F")
    t_index = params.index("T")
    registered = []
    recog_list = []
    with open(recog_file, 'a') as outcsv:
        writer = csv.writer(outcsv)
        for num_recog in range(1, len(fold_seq_order)+1):
            num_person = fold_seq_order[num_recog-1][0]
            orig_num_sample = fold_seq_order[num_recog-1][1]
            R = 0
            
            if isTraining and len(registered) < num_people and num_person not in registered:
                R = 1
            row = [new_id_list[num_person-1]]
            
            for num_param in range(0, len(params)):
                val = db[num_person-1][num_param][orig_num_sample-1]
                if num_param == f_index:
                    val_base = []
                    val_base.extend(val)
                    val_app = []
                    if isTraining:
                        for re_id in registered:
                            val_app.append([str(new_id_list[re_id-1]), float("{0:.3f}".format(val_base[re_id-1]))])
                        if not isInitRecog:
                            val_app.append([str(new_id_list[num_person-1]), float("{0:.3f}".format(val_base[num_person-1]))])
                    else:
                        for re_id in range(1, num_people+1):
                            val_app.append([str(new_id_list[re_id-1]), float("{0:.3f}".format(val_base[re_id-1]))])
                            
                    if val_app:
                        val = sorted(val_app, key=lambda x: (x[1]), reverse=True)
                    else:
                        val = []
                    min_acc = acc - 0.1
                    max_acc = acc + 0.1
                    if min_acc < 0.0:
                        min_acc = 0.0
                    if max_acc > 1.0:
                        max_acc = 1.0
                    val = [getUniformRangeVar(1, min_acc, max_acc)[0], val]
                if num_param == t_index:
                    val = getTimeFromInterval(val, param_specs[num_param][7])
                row.append(val)
            if isInitRecog:
                row.append(num_recog)
            else:
                row.append(R)
                row.append(num_recog)
            if R == 1:
                registered.append(num_person)
            recog_list.append(row)
            writer.writerow(row)
    return recog_list
 
def saveDBFile(num_samples, params, new_id_list, db_param_list, db_file):
    if not isinstance(num_samples, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_samples_list = [num_samples for _ in range(0, num_people)]
    else:
        num_samples_list = num_samples[:]
    
    if os.path.isfile(db_file):
        os.remove(db_file)
    with open(db_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["id", "name", "gender", "age", "height", "times", "occurrence"])
    t_index = params.index("T")
    db_list = []
    with open(db_file, 'a') as outcsv:
        writer = csv.writer(outcsv)
        for num_person in range(1, num_people+1):
            row = [num_person]
            num_orig_person = new_id_list.index(num_person) + 1
            row.append(str(num_orig_person))
            for num_param in range(1, len(params)-1):
                row.append(db_param_list[num_param][num_orig_person-1])
            row.append([db_param_list[t_index][num_orig_person-1][0]])
            row.append([num_samples_list[num_person-1],1,num_samples_list[num_person-1]])
            writer.writerow(row)
            db_list.append(row)
    return db_list

def saveParams(num_people, params, db_param_list, param_file):
    if os.path.isfile(param_file):
        os.remove(param_file)
    with open(param_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["I", "F", "G", "A", "H", "T"])
    with open(param_file, 'a') as outcsv:
        writer = csv.writer(outcsv)
        for num_person in range(1, num_people+1):
            row = [num_person]
            for num_param in range(0, len(params)):
                row.append(db_param_list[num_param][num_person-1])
            writer.writerow(row)    

def getSimilarityMatrix(num_people, method_form="uniform", range_min=0.0, range_max=0.95, decimals=3):
    sim_matrix = []
    if "uniform" in method_form:
        sim = np.random.uniform(low=range_min, high=range_max, size=[num_people,num_people])
        sim = np.around(sim, decimals=decimals)
        sim_matrix = (sim + sim.T)/2
        np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix

def getSimilarity(num_person, sim_matrix):
    return sim_matrix[num_person-1][:]

def getUniformRangeVar(size_samples, range_min, range_max, param_type = "cont", decimals=3):
    # num_samples = 1 for age and height, but equal to num_samples for time
    if isinstance(size_samples, (list,)) and isinstance(size_samples[1], (list,)):
        samples = []
        for num_person in range(1, len(size_samples[1])+1):
            num_samples_person = size_samples[1][num_person-1]
            if "discrete" in param_type:
                samples.append(np.random.randint(low=range_min, high=range_max+1, size=num_samples_person))
            else:
                samples.append(np.around(np.random.uniform(low=range_min, high=range_max, size=num_samples_person), decimals=decimals))
        return samples
    
    if "discrete" in param_type:
        return np.random.randint(low=range_min, high=range_max+1, size=size_samples)
    samples = np.random.uniform(low=range_min, high=range_max, size=size_samples)
    return np.around(samples, decimals=decimals)

def getGaussianRangeVar(size_samples, mu, sigma, range_min = None, range_max = None, clip_range_min = None, clip_range_max = None, param_type = "cont", decimals=3, isClip=True):
    # num_samples = 1 for age and height, but equal to num_samples for time
    
    if isinstance(size_samples, (list,)) and isinstance(size_samples[1], (list,)):
        samples = []
        for num_person in range(1, size_samples[1]+1):
            num_samples_person = size_samples[1][num_person-1]
            if param_type == "trunc-discrete":
                samples_values = truncnorm(a=range_min, b=range_max, loc=mu, scale=sigma).rvs(size=num_samples_person)
                samples.append(samples_values.round().astype(int))
            elif param_type == "trunc-cont":
                samples_values = truncnorm(a=range_min, b=range_max, loc=mu, scale=sigma).rvs(size=num_samples_person)
                samples.append(np.around(samples_values, decimals=decimals))
            else:
                samples_values = np.random.normal(mu, sigma, size=num_samples_person)
                if isClip:
                    if clip_range_min is not None:
                        samples_values[samples_values < clip_range_min] = clip_range_min
                    if clip_range_max is not None:
                        samples_values[samples_values > clip_range_max] = clip_range_max
                samples.append(np.around(samples_values, decimals=decimals))
        return samples
                
    
    if param_type == "trunc-discrete":
        samples = truncnorm(a=range_min, b=range_max, loc=mu, scale=sigma).rvs(size=size_samples)
        return samples.round().astype(int)
    elif param_type == "trunc-cont":
        # NOTE: DON'T USE FOR HEIGHT, WOULD GIVE INF BECAUSE THE RANGE IS SO FAR OUT IN THE TAIL, USE CONT AND THEN TRUNCATE IT
        samples = truncnorm(a=range_min, b=range_max, loc=mu, scale=sigma).rvs(size=size_samples)
        return np.around(samples, decimals=decimals)        
    samples = np.random.normal(mu, sigma, size=size_samples)
    if isClip:
        if clip_range_min is not None:
            samples[samples < clip_range_min] = clip_range_min
        if clip_range_max is not None:
            samples[samples > clip_range_max] = clip_range_max
        
    return np.around(samples, decimals=decimals)

def getUniformLabelVar(size_samples, labels):
    return np.random.choice(labels, size=size_samples)
    
def getGMMVar(num_rows, num_cols, range_min, range_max, num_curves, sigma, param_type = "cont", decimals=3):
    #gaussian mixture model for time
    var_list= []
    if not isinstance(num_cols, (list,)):
        num_samples_list = [num_cols for _ in range(0, num_rows)]
    else:
        num_samples_list = num_cols[:]
        
    sample_means_list = getUniformRangeVar([num_rows, num_curves], range_min, range_max, param_type, decimals=decimals)

    for num_row in range(1, num_rows+1):
        samples_per_curve = num_samples_list[num_row-1]/num_curves
        sample_means = sample_means_list[num_row-1]
        samples_person = []

        for num_curve in range(0, num_curves):
            if param_type == "trunc-discrete":
                samples = truncnorm(a=range_min, b=range_max, loc=sample_means[num_curve], scale=sigma).rvs(size=num_samples_list[num_row-1])
                norm_curve = samples.round().astype(int)
                
            elif param_type == "trunc-cont":
                samples = truncnorm(a=range_min, b=range_max, loc=sample_means[num_curve], scale=sigma).rvs(size=num_samples_list[num_row-1])
                norm_curve = np.around(samples, decimals=decimals)
            else:
                samples = np.random.normal(sample_means[num_curve], sigma, size=num_samples_list[num_row-1])
                norm_curve = np.around(samples, decimals=decimals)
            samples_person = np.concatenate((samples_person, norm_curve), axis=0)
        var_list.append(np.random.choice(samples_person, size=num_samples_list[num_row-1]))
    return var_list
     
def getNoisyDB(num_people, num_samples, params, param_specs, clean_db):
    """
        param_specs[num_param][0] = [noise_level, param_noise_method, param_noise_add_method, isReverseLabel, isAddConfidenceScore, fixedConfScore]
        param_specs[num_param][1] = param_type
        param_specs[num_param][2] = num_samples per person for db_param ( = num_people for "F"; = 1 for "G, A, H", )
        param_specs[num_param][3] = [uniform_range_min, clip_range_min]
        param_specs[num_param][4] = [uniform_range_max, clip_range_max]
        param_specs[num_param][5] = mu / num_curves
        param_specs[num_param][6] = sigma
        param_specs[num_param][7] = extras (labels for "G", period for "T")
        param_specs[num_param][8] = decimals
    """
    if not isinstance(num_samples, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_samples_list = [num_samples for _ in range(0, num_people)]
    else:
        num_samples_list = num_samples[:]
        
    bag_of_samples = []
    noise_list = []
    
    noisy_db = [[] for _ in range(0, num_people)]            
    for num_param in range(0, len(params)):
        param = params[num_param]
        if param_specs[num_param][0][0] > 0:
            # if there is noise
            if param_specs[num_param][0][1] == "uniform-pick":
                for num_person in range(1, num_people+1):
                    to_add = []
                    to_add.extend(clean_db[num_person-1][num_param])
                    bag_of_samples = np.concatenate((bag_of_samples, to_add), axis=0)
            for num_person in range(1, num_people+1):
                original = []
                original.extend(clean_db[num_person-1][num_param])
                num_samples_person = num_samples_list[num_person-1]
                noisy_samples_param, noise = getNoisySamples(num_people, num_samples_person, param_specs[num_param][0][0], original, param, param_specs[num_param][0][1], param_specs[num_param][0][2], param_specs[num_param][1],
                                                      param_specs[num_param][3][0], param_specs[num_param][3][1], param_specs[num_param][4][0], param_specs[num_param][4][1],
                                                      param_specs[num_param][2], bag_of_samples=bag_of_samples, 
                                                      isReverseLabel=param_specs[num_param][0][3], isAddConfidenceScore=param_specs[num_param][0][4], fixedConfScore=param_specs[num_param][0][5], labels=param_specs[num_param][7], decimals=param_specs[num_param][8])
                noisy_db[num_person-1].append(noisy_samples_param)
                noise_list.append(noise)
        else:
            # use original db values
            for num_person in range(1, num_people+1):
                to_add = []
                to_add.extend(clean_db[num_person-1][num_param])
                noisy_db[num_person-1].append(to_add)
                noise_list.append([])
    return noisy_db, noise_list

def getNoisySamples(num_people, num_samples, noise_level, original, param, method_noise, method_add, param_type, 
                    param_min, param_max, param_clip_min, param_clip_max, num_prob_param=1, bag_of_samples=None, 
                    isReverseLabel=False, isAddConfidenceScore=False, fixedConfScore=None, labels=["Female", "Male"], decimals=3):
#     if param == "F":
#         noise_range_min = 0.0
#         noise_range_max = 1.0
#     elif param == "G":
#         noise_range_min = -1.0
#         noise_range_max = 1.0
#     elif param == "A" or param == "H" or param == "T":
#         noise_range_min = param_min
#         noise_range_max = param_max 


    if param == "A" or param == "H":
        mu = original[0][0]
    else:
        mu = 0.0
        
    noise = getNoise(num_people, num_samples, noise_level, param, method_noise, param_type, mu, param_min, param_max, param_clip_min, param_clip_max, num_prob_param, isAddConfidenceScore, fixedConfScore, bag_of_samples, decimals)
    noisy_samples = addNoise(num_samples, noise_level, param, original, noise, method_add, param_min, param_max, param_clip_min, param_clip_max, isReverseLabel, labels)
    return noisy_samples, noise

 
def getNoise(num_people, num_samples, noise_level, param, method_noise="gaussian-all", param_type="cont",
             mu=0.0, range_min=0.0, range_max=1.0, clip_range_min=0.0, clip_range_max=1.0, num_prob_param=1, 
             isAddConfidenceScore=False, fixedConfScore=None, bag_of_samples=None, decimals=3):
    # num_prob_param is the number of probabilities necessary for each sample for a parameter. For example,
    # num_prob_param = num_people for face, each sample would have [1.0, 0.2, ..., 0.1] as an entry
    # num_prob_param = 1 for gender, age, height, time, each sample would have one entry of probability (for gender 0.8)
    noise = []
    if num_prob_param == 1:
        size_samples = num_samples
    else:
        size_samples = [num_samples, num_prob_param]
        
    if method_noise == "gaussian-all": # for face, gender and time
        sigma = noise_level
        noise = getGaussianRangeVar(size_samples, mu, sigma, range_min = range_min, range_max = range_max, param_type = param_type, decimals = decimals, isClip = False)
    elif method_noise == "gaussian-sample-all": # for age, height
        sigma = noise_level
        noise_samples = getGaussianRangeVar([num_samples, range_max - range_min], mu, sigma, range_min = range_min, range_max = range_max, 
                                            clip_range_min = clip_range_min, clip_range_max = clip_range_max, param_type =  param_type, decimals = decimals)
        mu_noise = np.around(np.mean(noise_samples, axis=0), decimals=decimals)
        sigma_noise = np.std(noise_samples, axis=0)
        if isAddConfidenceScore:    
            if fixedConfScore is not None:
                noise = [[mu_noise[i], fixedConfScore] for i in range(0,num_samples)]
            else:
                for num_sample in range(0, num_samples):
                    confidence = getConfidenceFromDist(mu_noise[num_sample], mu_noise[num_sample], sigma_noise[num_sample], decimals=3)
                    noise.append([mu_noise[num_sample], confidence])
            
    elif method_noise == "gaussian-sample-one": # for age, height
        sigma = noise_level
        noise_samples = getGaussianRangeVar(size_samples, mu, sigma, range_min = range_min, range_max = range_max, 
        clip_range_min = clip_range_min, clip_range_max = clip_range_max, param_type = param_type, decimals = decimals)
        
        if isAddConfidenceScore:    
            if fixedConfScore is not None:
                noise = [[noise_samples[i], fixedConfScore] for i in range(0,num_samples)]
            else:
                for num_sample in range(0, num_samples):
                    confidence = getConfidenceFromDist(noise_samples[num_sample], mu, sigma, decimals=3)
                    noise.append([noise_samples[num_sample], confidence])
         
    elif method_noise == "uniform-sample": # for face only
        noise = getUniformRangeVar([num_samples, getUniformNumNoiseSamples(noise_level, num_people)], range_min, range_max, param_type)
    
    elif method_noise == "uniform-part": # for all
        if num_prob_param == 1:
            noise = getUniformRangeVar(getUniformNumNoiseSamples(noise_level, num_samples), range_min, range_max, param_type)
        else:
            noise = getUniformRangeVar([getUniformNumNoiseSamples(noise_level, num_samples), num_prob_param], range_min, range_max, param_type)
        
        if isAddConfidenceScore:
            if fixedConfScore is not None:
                noise = [[noise[i], fixedConfScore] for i in range(0,num_samples)]
            else:
                noiseProb = getUniformRangeVar(getUniformNumNoiseSamples(noise_level, num_samples), 0.0, 1.0, "cont")
                noise = map(list, zip(noise, noiseProb))
    
    elif method_noise == "uniform-pick": # for time only
        noise = np.random.choice(bag_of_samples, size=getUniformNumNoiseSamples(noise_level, num_samples))
        
    return noise
    
def addNoise(num_samples, noise_level, param, original, noise, method_add="sum-clip", 
             range_min=0.0, range_max=1.0, clip_range_min=0.0, clip_range_max=1.0, isReverseLabel=False, labels=["Female", "Male"]):
    noisy_samples = []
    if method_add == "sum-clip": # for Gaussian face, Gaussian gender and Gaussian time (need to give time_min and time_max for (clip) range_min and range_max)
        if isReverseLabel:
            original_values = [x[1] for x in original]
            noisy_values = np.sum([original_values,noise], axis=0)
            noisy_samples = reverseLabel(noisy_values, original, param, clip_range_min, clip_range_max, labels)
        else:
            noisy_samples = np.sum([original,noise], axis=0)
            noisy_samples = clipRangeSamples(noisy_samples, num_samples, clip_range_min, clip_range_max)
            if param == "T":
                return [int(i) for i in noisy_samples]
    
    elif method_add == "replace-part": # for uniform-part or uniform-pick
        to_change = np.random.choice([x for x in range(0,num_samples)],size=getUniformNumNoiseSamples(noise_level, num_samples), replace=False)
        if isReverseLabel:   
            original_values = [x[1] for x in original]
            noisy_samples.extend(original_values)
        else:
            noisy_samples.extend(original)
            
        noise_counter = 0
        for to_c in to_change:
            noisy_samples[to_c] = noise[noise_counter]
            noise_counter += 1
            
        if isReverseLabel:
            noisy_samples = reverseLabel(noisy_samples, original, param, clip_range_min, clip_range_max, labels)
    
    elif method_add == "replace-sample": # for uniform-sample (face only)
        num_cols = len(original[0])
        size_samples = getUniformNumNoiseSamples(noise_level, num_cols)
        for num_sample in range(0, num_samples):
            noisy_sample =[]
            noisy_sample.extend(original[num_sample])
            to_change = np.random.choice([x for x in range(0,num_cols)],size=size_samples, replace=False)
            noise_counter = 0
            for to_c in to_change:
                noisy_sample[to_c] = noise[num_sample][noise_counter]
                noise_counter += 1
            noisy_samples.append(noisy_sample)
        noisy_samples = np.around(noisy_samples, decimals=3)
    elif method_add == "replace-all": # for Gaussian age and height
        return noise
        
    return noisy_samples

def reverseLabel(noisy_values, original, param, clip_range_min, clip_range_max, labels):
    noisy_samples = []
    if param == "G":
        for num_value in range(0,len(noisy_values)):
            noisy_value = noisy_values[num_value]
            original_label = original[num_value][0]
            if noisy_value < 0:
                if original_label == labels[0]:
                    sample_label = labels[1]
                else:
                    sample_label = labels[0]
                noisy_value *= -1
            else:
                sample_label = original_label
            if noisy_value > clip_range_max:
                noisy_value = clip_range_max
            elif noisy_value < clip_range_min:
                noisy_value = clip_range_min
            noisy_samples.append([sample_label, getGenderProbability(noisy_value)])
    else:
        # randomly choose a label from the remaining labels
        for num_value in range(0,len(noisy_values)):
            original_label = original[num_value][0]
            if noisy_value < 0:
                remaining = [x for x in labels if x!=original_label]
                sample_label = np.random.choice(remaining)
                noisy_value *= -1
            else:
                sample_label = original_label
            if noisy_value > clip_range_max:
                noisy_value = clip_range_max
            elif noisy_value < clip_range_min:
                noisy_value = clip_range_min
            noisy_samples.append([sample_label, noisy_value])
    return noisy_samples

def getUniformNumNoiseSamples(noise_level, num_samples):
    num_noisy_samples = int(noise_level*num_samples)
    if num_noisy_samples == 0:
        return 1
    return num_noisy_samples

def createCrossValidationSet(num_people, num_samples, num_bins, num_folds, params, param_specs, db, db_param_list, base_folder, training_folder, test_folder, init_file, recog_file, db_file, acc=1.0):
    
    sample_info_bin, num_images_per_person_in_bin = cv.divideImagesIntoBinsMod(num_bins, num_people, num_samples, isImage=False)
    fold_bin_order_list = cv.getBinOrderInFold(num_bins)

    recog_order_bin_list = cv.getRecognitionOrderBin(num_bins, num_people, num_images_per_person_in_bin)
    
    recog_order_fold_list, new_id_fold_list = cv.getRecognitionOrderFold(num_people, fold_bin_order_list, recog_order_bin_list)
    
    fold_seq_order_list = cv.getSeqOrderForPersoninFold(num_bins, num_folds, num_people, fold_bin_order_list, recog_order_bin_list, sample_info_bin)
    
    recog_list_training_all = []
    init_recog_list_training_all = []
    recog_list_test_all = []
    init_recog_list_test_all = []
    db_list_all =[]
    clean_times_all = []
    for num_fold in range(1, num_folds+1):
        recog_folder = base_folder + str(num_fold) + "/"
        
        #Training
        recog_training_folder = recog_folder + training_folder
        if not os.path.isdir(recog_training_folder):
            os.makedirs(recog_training_folder)
        
        db_list = saveDBFile(num_samples, params, new_id_fold_list[num_fold-1], db_param_list, recog_training_folder + db_file)
        
        clean_times_all.append(getCleanTimes(num_people, params, new_id_fold_list[num_fold-1], db_param_list))
        
        recog_list_training = saveRecognitionFile(num_people, params, param_specs, db, fold_seq_order_list[num_fold-1][0], new_id_fold_list[num_fold-1], recog_training_folder + recog_file, isInitRecog=False, isTraining=True, acc=acc)
        init_recog_list_training = saveRecognitionFile(num_people, params, param_specs, db, fold_seq_order_list[num_fold-1][0], new_id_fold_list[num_fold-1], recog_training_folder + init_file, isInitRecog=True, isTraining=True, acc=acc)
        
        db_list_all.append(db_list)
        recog_list_training_all.append(recog_list_training)
        init_recog_list_training_all.append(init_recog_list_training)
        
        # Test
        recog_test_folder = recog_folder + test_folder
        if not os.path.isdir(recog_test_folder):
            os.makedirs(recog_test_folder)
        
        shutil.copy2(recog_training_folder + db_file, recog_test_folder + db_file)
        
        recog_list_test = saveRecognitionFile(num_people, params, param_specs, db, fold_seq_order_list[num_fold-1][1], new_id_fold_list[num_fold-1], recog_test_folder + recog_file, isInitRecog=False, acc=acc)
        init_recog_list_test = saveRecognitionFile(num_people, params, param_specs, db, fold_seq_order_list[num_fold-1][1], new_id_fold_list[num_fold-1], recog_test_folder + init_file, isInitRecog=True, acc=acc)
        
        recog_list_test_all.append(recog_list_test)
        init_recog_list_test_all.append(init_recog_list_test)        
        
    return recog_list_training_all, init_recog_list_training_all, recog_list_test_all, init_recog_list_test_all, db_list_all, clean_times_all

def optimParams(args):
    [num_people, num_folds, training_folder, test_folder, cross_val_stats_train_file, cross_val_stats_test_file, 
    db_list, init_list, recogs_list, bn, isTestData,
    normMethod, updateMethod, probThreshold,
    isMultRecognitions, num_mult_recognitions, qualityCoefficient,
    db_file, init_recog_file, final_recog_file, cost_function_alpha, n_iters, optim_params_file, bounds] = args
        
    random_state = np.random.RandomState(1234567890)
    time_loop = time.time()
    
    xp, yp = wop.bayesian_optim(bounds, n_iters=n_iters, n_pre_samples=3, args=args[:-3], random_state=random_state)
    
    # find min error in yp, get values corresponding to that index in xp
    min_error = np.amin(yp)
    min_indices = list(np.where(yp == min_error)[0])
    
    optim_param = []
    i_dir_list = []
    fr_dir_list = []
    i_far_list = []
    fr_far_list = []
    df_stats = pandas.read_csv(cross_val_stats_train_file.replace(".csv","_avg.csv"), usecols = ["I_FAR","F_FAR","I_DIR_1","F_DIR_1","Loss"])
    
    min_error_stats = df_stats.loc[np.isclose(df_stats['Loss'], min_error)].values.tolist()
    
    counter_ind = 0
    for min_ind in min_indices:
        xp_op = xp[min_ind]
        optim_param.append(xp_op)
        i_far_list.append(float("{0:.3f}".format(min_error_stats[counter_ind][0])))
        fr_far_list.append(float("{0:.3f}".format(min_error_stats[counter_ind][1])))   
        i_dir_list.append(float("{0:.3f}".format(min_error_stats[counter_ind][2])))
        fr_dir_list.append(float("{0:.3f}".format(min_error_stats[counter_ind][3])))
        counter_ind += 1
    with open(optim_params_file, 'a') as outcsv:
        writer = csv.writer(outcsv)   
        writer.writerow([updateMethod, normMethod, optim_param, min_error, i_far_list, fr_far_list, i_dir_list, fr_dir_list])
#         writer.writerow([updateMethod, normMethod, optim_param, min_error])

    print "time for loop for " + str(normMethod) + "_" + str(updateMethod) + ":" + str(time.time() - time_loop)
    return min_error, optim_param
    
def getGenderProbability(confidence):
        # difference between the male and female confidences is assumed to be confidence: x - (1.0 - x) = confidence
    prob = 0.5 + (confidence/2.0)
    return np.around(prob,decimals=3)

def getConfidenceFromDist(data_point, mu, sigma, decimals=3):
    confidence = np.around(normpdf(data_point, loc=mu, scale=sigma), decimals=3)
    if confidence < 0:
        confidence = 0.0
    elif confidence > 1:
        confidence = 1.0
    return confidence

def getTimeFromInterval(time_slot, period):
#     time_slot = (int(day)-1)*24*60/self.period + int(hour)*60/self.period + int(minute)/self.period
    p_time = time_slot * period
    day = int(p_time / (24*60)) + 1
    hour = int((p_time % (24*60)) / 60)
    minute = int((p_time % (24*60)) % 60)
    second = np.random.randint(0, high=60)
    if hour < 10:
        s_hour = "0" + str(hour)
    else:
        s_hour = str(hour)
    if minute < 10:
        s_min = "0" + str(minute)
    else:
        s_min = str(minute)
    if second < 10:
        s_sec = "0" + str(second)
    else:
        s_sec = str(second)
    return [s_hour + ":" + s_min + ":" + s_sec, str(day)]

def getCleanTimes(num_people, params, new_id_fold, db_param_list):
    time_index = params.index("T")
    clean_times = []
    for num_person in range(1, num_people+1):
        orig_id = new_id_fold.index(num_person) + 1
        clean_times.append(db_param_list[time_index][orig_id-1][:])
    return clean_times
    
def isRangeVarCorrect(var_est, var_real, bin_size, decimals=0):
    if np.around(var_est, decimals=decimals)/bin_size == np.around(var_real, decimals=decimals):
        return 1
    return 0

def isLabelVarCorrect(var_est, var_real):
    if var_est == var_real:
        return 1
    return 0

def getTimeSlot(p_time, period):
    tp = p_time[0].split(":")
    time_slot = (int(p_time[1])-1)*24*60/period + int(tp[0])*60/period + int(tp[1])/period
    return time_slot

def getParamStats(num_people, num_samples, clean_db, noisy_db, age_bin_size, height_bin_size, time_bin_size):
    
    if not isinstance(num_samples, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_samples_list = [num_samples for _ in range(0, num_people)]
    else:
        num_samples_list = num_samples[:]
        
        
    stats_gender = 0
    stats_age = 0
    stats_height = 0
    stats_time = 0
    total_num_samples = sum(num_samples_list)
    for num_person in range(1, num_people+1):
        for num_sample in range(0, num_samples_list[num_person-1]):
            stats_gender += isLabelVarCorrect(noisy_db[num_person-1][1][num_sample][0], clean_db[num_person-1][1][num_sample][0])
            stats_age += isRangeVarCorrect(noisy_db[num_person-1][2][num_sample][0], clean_db[num_person-1][2][num_sample][0], age_bin_size)
            stats_height += isRangeVarCorrect(noisy_db[num_person-1][3][num_sample][0], clean_db[num_person-1][3][num_sample][0], height_bin_size)
            stats_time += isRangeVarCorrect(int(noisy_db[num_person-1][4][num_sample])/time_bin_size, int(clean_db[num_person-1][4][num_sample])/time_bin_size, 1)
    stats_gender /= total_num_samples*1.0
    stats_age /= total_num_samples*1.0
    stats_height /= total_num_samples*1.0
    stats_time /= total_num_samples*1.0
    
    return stats_gender, stats_age, stats_height, stats_time

def clipRangeSamples(list_samples, num_samples, clip_range_min, clip_range_max):
    for num_sample in range(0, num_samples):
        list_samples[num_sample] = np.clip(list_samples[num_sample], clip_range_min, clip_range_max)    
    return list_samples

def normpdf(x, loc=0, scale=1):
    """x is the value that pdf wants to be read at, loc is the mean, and scale is the stddev
    From: https://stackoverflow.com/questions/8669235/alternative-for-scipy-stats-norm-pdf"""
    u = float(x-loc) / abs(scale)
    y = np.exp(-u*u/2) / (np.sqrt(2*np.pi) * abs(scale))
    return y

def getStddevFromConfidence(confidence):
    """From https://stats.stackexchange.com/questions/269784/calculating-the-probability-of-a-discrete-rv-given-the-mean-and-the-probability"""
    z= normppf(confidence + (1-confidence)/2.0) # z-score
    return 0.5/z

def normppf(y0):     
    """From https://stackoverflow.com/questions/41338539/how-to-calculate-a-normal-distribution-percent-point-function-in-python"""   
    
    s2pi = 2.50662827463100050242E0

    P0 = [
        -5.99633501014107895267E1,
        9.80010754185999661536E1,
        -5.66762857469070293439E1,
        1.39312609387279679503E1,
        -1.23916583867381258016E0,
    ]
    
    Q0 = [
        1,
        1.95448858338141759834E0,
        4.67627912898881538453E0,
        8.63602421390890590575E1,
        -2.25462687854119370527E2,
        2.00260212380060660359E2,
        -8.20372256168333339912E1,
        1.59056225126211695515E1,
        -1.18331621121330003142E0,
    ]
    
    P1 = [
        4.05544892305962419923E0,
        3.15251094599893866154E1,
        5.71628192246421288162E1,
        4.40805073893200834700E1,
        1.46849561928858024014E1,
        2.18663306850790267539E0,
        -1.40256079171354495875E-1,
        -3.50424626827848203418E-2,
        -8.57456785154685413611E-4,
    ]
    
    Q1 = [
        1,
        1.57799883256466749731E1,
        4.53907635128879210584E1,
        4.13172038254672030440E1,
        1.50425385692907503408E1,
        2.50464946208309415979E0,
        -1.42182922854787788574E-1,
        -3.80806407691578277194E-2,
        -9.33259480895457427372E-4,
    ]
    
    P2 = [
        3.23774891776946035970E0,
        6.91522889068984211695E0,
        3.93881025292474443415E0,
        1.33303460815807542389E0,
        2.01485389549179081538E-1,
        1.23716634817820021358E-2,
        3.01581553508235416007E-4,
        2.65806974686737550832E-6,
        6.23974539184983293730E-9,
    ]
    
    Q2 = [
        1,
        6.02427039364742014255E0,
        3.67983563856160859403E0,
        1.37702099489081330271E0,
        2.16236993594496635890E-1,
        1.34204006088543189037E-2,
        3.28014464682127739104E-4,
        2.89247864745380683936E-6,
        6.79019408009981274425E-9,
    ]
    if y0 <= 0 or y0 >= 1:
        raise ValueError("ndtri(x) needs 0 < x < 1")
    negate = True
    y = y0
    if y > 1.0 - 0.13533528323661269189:
        y = 1.0 - y
        negate = False

    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * polevl(y2, P0) / polevl(y2, Q0))
        x = x * s2pi
        return x

    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x

    z = 1.0 / x
    if x < 8.0:
        x1 = z * polevl(z, P1) / polevl(z, Q1)
    else:
        x1 = z * polevl(z, P2) / polevl(z, Q2)
    x = x0 - x1
    if negate:
        x = -x

    return x

def polevl(x, coef):
    accum = 0
    for c in coef:
        accum = x * accum + c
    return accum
    
if __name__ == "__main__":
    
    """
    start_time = time.time()
    num_processors = 1

    num_people = 100
    num_samples = 10
    
    params = ["F", "G", "A", "H", "T"]
        
    age_min = 0
    age_max = 75
    max_threshold = 0.9
    age_bin_size = 1
    z_age = normppf(max_threshold + (1-max_threshold)/2.0)
    age_range_sigma = 30

    height_min = 50
    height_max = 240
    height_bin_size = 1
    height_range_sigma = 20
    
    period = 1 # time is checked every 1 minutes 
    time_range_sigma = 60/period
    time_min = 0
    time_max = (7*24*60/period) -1 # 7(days)*24(hours)*60(minutes)/period ( = num_time_slots)
    time_bin_size = 30
    
    param_noise_ranges = [[0.0, 1.0], [-2.0, 0.0], [age_min, age_max], [height_min, height_max], [time_min, time_max]]
    clip_ranges = [[0.0, 1.0], [0.0, 1.0], [age_min, age_max], [height_min, height_max], [time_min, time_max]]
    
    num_sample_curves = 3
    num_curves = int(num_samples/num_sample_curves)

    if num_curves == 0:
        num_curves = 1
        
    available_noise_methods_list = [["gaussian-all", "uniform-part", "uniform-sample"],                  #F
                                    ["gaussian-all", "uniform-part"],                                    #G    
                                    ["gaussian-sample-all", "gaussian-sample-one", "uniform-part"],      #A
                                    ["gaussian-sample-all", "gaussian-sample-one", "uniform-part"],      #H
                                    ["gaussian-all", "uniform-part", "uniform-pick"]]                    #T                             
  
    chosen_noise_method_list = [2, 1, 2, 2, 2]

    num_folds = 5
    num_bins = 5
    main_folder = "artificial_dataset/"
    #  bn_folder = "BN/"
    bn_folder = "BN_alpha0.9/"


    training_folder = "Training/"
    test_folder = "Test/"
    init_file = "InitialRecognition_data.csv"
    recog_file = "RecogniserBN_data.csv"
    db_file = "db_data.csv"
    optim_params_file_base = "optim_params.csv"
    stats_header = ["Fold", "Evidence_method", "Threshold", "Norm_method", "FER", 
                    "I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1", 
                    "Face_threshold", "Quality_threshold", "Weights", "Loss",
                    "Num_recognitions", "Num_registered"]
    
    bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
    
    norm_method_list = ["softmax", "minmax", "tanh", "norm-sum"]
#     probThreshold_list = [0.0]
    probThreshold = 1.0e-75
    updateMethod_list= ["none", "evidence"]
    noise_level_list = [0.0, 0.02, 0.1, 0.25, 0.5, 0.75]

    noise_indices = [_ for _ in range(0, len(noise_level_list))]
    #  cost_function_alpha = 0.8
    cost_function_alpha = 0.9
    n_iters = 300
    
    variables_list_base = list(itertools.product(noise_indices, norm_method_list, updateMethod_list))
    
#     norm_method_list = [norm_method_list[0]]
#     updateMethod_list = [updateMethod_list[0]]
#     noise_level_list = [noise_level_list[2]]
#     variables_list_base = [variables_list_base[0]]
    
    print "Finished setting params: " + str(time.time() - start_time)
    
    ###
    list_base_folders = []
    list_dbs = []
    list_init_recogs = []
    list_recogs = []
    noise_counter = 0
    variables_list = []
    counter_loop = 0
    for noise_level in noise_level_list:
        print "noise: " + str(noise_level)
        start_time_db = time.time()
        noise_level_folder = str(noise_level) + "/"
        base_folder = main_folder + bn_folder + noise_level_folder
        list_base_folders.append(base_folder)
         
        if not os.path.isdir(str(base_folder)):
            os.makedirs(str(base_folder))
    
        optim_params_file = base_folder + optim_params_file_base
        with open(optim_params_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)   
            writer.writerow(["Evidence_method", "Norm_method","Optim_params", "Loss", "I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1",])   
        cross_val_stats_train_file_base = base_folder + "cvs_train.csv"
        cross_val_stats_test_file_base = base_folder + "cvs_test.csv"
          
        for counter_var in range(0, len(variables_list_base)):
            var_l = variables_list_base[counter_var]
            if var_l[0] == noise_counter:
                train_file_name = cross_val_stats_train_file_base.replace(".csv", "_" + str(var_l[1]) + "_" + str(var_l[2][0]) + ".csv")
                test_file_name = cross_val_stats_test_file_base.replace(".csv", "_" + str(var_l[1]) + "_" + str(var_l[2][0]) + ".csv")
                 
                variables_list.append(variables_list_base[counter_var] + (train_file_name, test_file_name))
                with open(train_file_name, 'wb') as outcsv:
                    writer = csv.writer(outcsv)   
                    writer.writerow(stats_header)
                      
        #         with open(cross_val_stats_test_file, 'wb') as outcsv:
        #             writer = csv.writer(outcsv)   
        #             writer.writerow(stats_header)
                
                with open(train_file_name.replace(".csv", "_avg.csv"), 'wb') as outcsv:
                    writer = csv.writer(outcsv)   
                    writer.writerow(stats_header[1:])
                     
        #         with open(cross_val_stats_test_file.replace(".csv", "_avg.csv"), 'wb') as outcsv:
        #             writer = csv.writer(outcsv)   
        #             writer.writerow(stats_header[1:])
     
        noise_method_list = []
        for num_param in range(0, len(params)):
            chosen_noise_method = chosen_noise_method_list[num_param]
            noise_method_list.append(available_noise_methods_list[num_param][chosen_noise_method])
         
        param_noise_level_list = []
        noise_add_method_list = []
        for num_param in range(0, len(params)):
            noise_method = noise_method_list[num_param]
            param = params[num_param]
            if "gaussian" in noise_method:
                if param == "G":
                    param_noise_ranges[1] = [-2.0, 0.0]
                param_noise_level = (param_noise_ranges[num_param][1]-param_noise_ranges[num_param][0])*noise_level
                if param == "A" or param == "H":
                    noise_add_method = "replace-all"
                else:
                    noise_add_method = "sum-clip"
            elif "uniform" in noise_method:
                if param == "G":
#                     param_noise_ranges[1] = [-1.0, 1.0] # this gives less noise levels than intended (since gender may or may not change)
                    param_noise_ranges[1] = [-1.0, 0.0] # this gives lexact noise level (since gender will change for sure)
 
                param_noise_level = noise_level
                if noise_method == "uniform-sample":
                    noise_add_method = "replace-sample"
                else:
                    noise_add_method = "replace-part"
            param_noise_level_list.append(param_noise_level)
            noise_add_method_list.append(noise_add_method)
        num_samples_people_list = [num_samples for _ in range(1,num_people+1)]
        param_specs = [ 
                      [  # clean dataset param_specs: [param_form_method, param_type, num_samples_per_person, [uniform_range_min, uniform_range_max], [clip_range_min, clip_range_max], mu/num_curves, sigma, extras (labels/ period), decimals
                      ["uniform", "cont", num_people, [0.0, 0.95], [0.0, 1.0], 0.0, 1.0, "", 3],                                             # F
                      ["uniform-label", "discrete", 1, [0.0, 1.0], [0.0, 1.0], "", "", ["Female", "Male"], 3],                                     # G
                      ["uniform-range", "discrete", 1, [age_min, age_max], [age_min, age_max], (age_max+age_min)/2, age_range_sigma, "", 0],                  # A
                      ["uniform-range", "cont", 1, [height_min, height_max], [height_min, height_max], (height_max+height_min)/2, height_range_sigma, "", 1],       # H
                      ["uniform-range", "trunc-discrete", num_samples_people_list, [time_min, time_max], [time_min, time_max], num_curves, time_range_sigma, period, 0]                   # T
                      ],
                      [  # noisy dataset param_specs: 
                         # [[param_noise_level, param_noise_method, param_noise_add_method, isReverseLabel, isAddConfidenceScore], 
                         # param_type, num_samples_per_person, [uniform_range_min, uniform_range_max], [clip_range_min, clip_range_max], mu/num_curves, sigma, extras (labels/ period), decimals]
                      [[param_noise_level_list[0], noise_method_list[0], noise_add_method_list[0], False, False, None], "cont", num_people, param_noise_ranges[0], clip_ranges[0], 0.0, 1.0, "", 3],                                             # F
                      [[param_noise_level_list[1], noise_method_list[1], noise_add_method_list[1], True, False, None], "trunc-cont", 1, param_noise_ranges[1], clip_ranges[1], "", "", ["Female", "Male"], 3],                                     # G
                      [[param_noise_level_list[2], noise_method_list[2], noise_add_method_list[2], False, True, None], "discrete", 1, param_noise_ranges[2], clip_ranges[2], (age_max+age_min)/2, age_range_sigma, "", 1],                  # A
                      [[param_noise_level_list[3], noise_method_list[3], noise_add_method_list[3], False, True, None], "cont", 1, param_noise_ranges[3], clip_ranges[3],  (height_max+height_min)/2, height_range_sigma, "", 1],       # H
                      [[param_noise_level_list[4], noise_method_list[4], noise_add_method_list[4], False, False, None], "cont", 1, param_noise_ranges[4], clip_ranges[4], num_curves, time_range_sigma, period, 0]                   # T                  
                      ]
                      ]
     
      
    #     num_param_to_check = 4
    #     params = [params[num_param_to_check]]    
    #     param_specs = [[param_specs[0][num_param_to_check][:]], [param_specs[1][num_param_to_check][:]]]
    #     print param_noise_level_list[num_param_to_check]
        
        if counter_loop == 0:
            clean_db, db_param_list = getCleanDB(num_people, num_samples, params, param_specs[0])
         
        saveArtificialDB(num_people, num_samples, params, param_specs[0], clean_db, base_folder + "clean_db_file.csv")
        saveParams(num_people, params, db_param_list, base_folder + "db_params.csv")
 
        noisy_db, noise_list = getNoisyDB(num_people, num_samples, params, param_specs[1], clean_db)
     
        saveArtificialDB(num_people, num_samples, params, param_specs[1], noisy_db, base_folder + "noisy_db_file.csv")
          
#         print getParamStats(num_people, num_samples, clean_db, noisy_db, age_bin_size, height_bin_size, time_bin_size)
         
        acc = 1.0 - noise_level
        recog_list_training_all, init_recog_list_training_all, recog_list_test_all, init_recog_list_test_all, db_list_all, clean_times_all = createCrossValidationSet(num_people, num_samples, num_bins, num_folds, params, param_specs[1], noisy_db, db_param_list, base_folder, training_folder, test_folder, init_file, recog_file, db_file, acc=acc)
        list_dbs.append(db_list_all)
        list_init_recogs.append([init_recog_list_training_all, init_recog_list_test_all])
        list_recogs.append([recog_list_training_all, recog_list_test_all])
        print "Database created: " + str(time.time() - start_time_db)
        noise_counter += 1
    #     cv.crossValidationCombine(num_people, 1, base_folder + training_folder, base_folder + test_folder, cross_val_stats_train_file, cross_val_stats_test_file, 
    #                               db_list=db_list_all, init_list=[init_recog_list_training_all, init_recog_list_test_all] , recogs_list=[recog_list_training_all, recog_list_test_all], bn=None, isTestData = False,
    #                                weights = [1.0, 1.0, 1.0, 1.0, 1.0], faceRecogThreshold = 0.3, qualityThreshold = 0.0, normMethod = "softmax", updateMethod = "evidence", probThreshold = 0.000001,
    #                                isMultRecognitions = False, num_mult_recognitions = None, qualityCoefficient = None,
    #                                db_file = None, init_recog_file = None, final_recog_file = None )
    pool = mp.Pool(processes=num_processors)
    results = [pool.apply_async(optimParams, [[num_people, num_folds, list_base_folders[variables_l[0]] + training_folder, list_base_folders[variables_l[0]] + test_folder, variables_l[3],  variables_l[4], 
    list_dbs[variables_l[0]], list_init_recogs[variables_l[0]] , list_recogs[variables_l[0]], None, False,
    variables_l[1], variables_l[2], probThreshold,
    False, 3, None, None, None, None, cost_function_alpha, n_iters, list_base_folders[variables_l[0]] + optim_params_file_base, bounds]]) for variables_l in variables_list]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    ###
    main_folder = "imdb_crop/"
    optim_folder = "optim/"
    updateMethod_list= ["evidence"]
    cross_val_folders = ["N10_gaussianT/", "N10_uniformT/", "Nall_gaussianT/", "Nall_uniformT/"]
    norm_method_list = ["softmax", "minmax", "tanh", "norm-sum", "hybrid"]
    
    norm_method_list = [norm_method_list[4]]
    variables_list_base = list(itertools.product(cross_val_folders, norm_method_list, updateMethod_list))
    
    variables_list = []
    list_base_folders = []

    for noise_level_folder in cross_val_folders:
        base_folder = main_folder + optim_folder + noise_level_folder
        optim_params_file = base_folder + optim_params_file_base
       # with open(optim_params_file, 'wb') as outcsv:
       #     writer = csv.writer(outcsv)   
       #     writer.writerow(["Evidence_method", "Norm_method","Optim_params", "Loss", "I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1"])   
        cross_val_stats_train_file_base = base_folder + "cvs_train.csv"
        cross_val_stats_test_file_base = base_folder + "cvs_test.csv"
          
        for counter_var in range(0, len(variables_list_base)):
            var_l = variables_list_base[counter_var]
            if var_l[0] == noise_level_folder:
                train_file_name = cross_val_stats_train_file_base.replace(".csv", "_" + str(var_l[1]) + "_" + str(var_l[2][0]) + "_partT" + ".csv")
                test_file_name = cross_val_stats_test_file_base.replace(".csv", "_" + str(var_l[1]) + "_" + str(var_l[2][0]) + "_partT" + ".csv")
                 
                variables_list.append(variables_list_base[counter_var] + (train_file_name, test_file_name))
                with open(train_file_name, 'wb') as outcsv:
                    writer = csv.writer(outcsv)   
                    writer.writerow(stats_header)
                      
        #         with open(cross_val_stats_test_file, 'wb') as outcsv:
        #             writer = csv.writer(outcsv)   
        #             writer.writerow(stats_header)
                
                with open(train_file_name.replace(".csv", "_avg.csv"), 'wb') as outcsv:
                    writer = csv.writer(outcsv)   
                    writer.writerow(stats_header[1:])
                     
        #         with open(cross_val_stats_test_file.replace(".csv", "_avg.csv"), 'wb') as outcsv:
        #             writer = csv.writer(outcsv)   
        #             writer.writerow(stats_header[1:])
    
    pool = mp.Pool(processes=num_processors)
    results = [pool.apply_async(optimParams, [[num_people, num_folds,main_folder + optim_folder + variables_l[0] + "folds/" + training_folder, main_folder + optim_folder + variables_l[0] + "folds/" + test_folder, variables_l[3],  variables_l[4], 
    None, None, None, None, False,
    variables_l[1], variables_l[2], probThreshold,
    False, 3, None, db_file, init_file, recog_file, cost_function_alpha, n_iters, 
    main_folder + optim_folder + variables_l[0] + optim_params_file_base, bounds]]) for variables_l in variables_list]
 
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    """
    height_file = "imdb_chosen_heights.csv"
    orig_ids = [[720, 720, 720, 720, 720, 720, 720, 720, 720, 720], [18666, 18666, 18666, 18666, 18666, 18666, 18666, 18666, 18666, 18666], [17572, 17572, 17572, 17572, 17572, 17572, 17572, 17572, 17572, 17572], [10939, 10939, 10939, 10939, 10939, 10939, 10939, 10939, 10939, 10939], [13947, 13947, 13947, 13947, 13947, 13947, 13947, 13947, 13947, 13947], [635, 635, 635, 635, 635, 635, 635, 635, 635, 635], [3206, 3206, 3206, 3206, 3206, 3206, 3206, 3206, 3206, 3206], [1199, 1199, 1199, 1199, 1199, 1199, 1199, 1199, 1199, 1199], [3428, 3428, 3428, 3428, 3428, 3428, 3428, 3428, 3428, 3428], [3448, 3448, 3448, 3448, 3448, 3448, 3448, 3448, 3448, 3448], [9400, 9400, 9400, 9400, 9400, 9400, 9400, 9400, 9400, 9400], [19424, 19424, 19424, 19424, 19424, 19424, 19424, 19424, 19424, 19424], [1105, 1105, 1105, 1105, 1105, 1105, 1105, 1105, 1105, 1105], [10923, 10923, 10923, 10923, 10923, 10923, 10923, 10923, 10923, 10923], [8853, 8853, 8853, 8853, 8853, 8853, 8853, 8853, 8853, 8853], [9397, 9397, 9397, 9397, 9397, 9397, 9397, 9397, 9397, 9397], [16524, 16524, 16524, 16524, 16524, 16524, 16524, 16524, 16524, 16524], [9764, 9764, 9764, 9764, 9764, 9764, 9764, 9764, 9764, 9764], [8907, 8907, 8907, 8907, 8907, 8907, 8907, 8907, 8907, 8907], [9942, 9942, 9942, 9942, 9942, 9942, 9942, 9942, 9942, 9942], [13364, 13364, 13364, 13364, 13364, 13364, 13364, 13364, 13364, 13364], [5773, 5773, 5773, 5773, 5773, 5773, 5773, 5773, 5773, 5773], [10272, 10272, 10272, 10272, 10272, 10272, 10272, 10272, 10272, 10272], [8302, 8302, 8302, 8302, 8302, 8302, 8302, 8302, 8302, 8302], [13977, 13977, 13977, 13977, 13977, 13977, 13977, 13977, 13977, 13977], [3287, 3287, 3287, 3287, 3287, 3287, 3287, 3287, 3287, 3287], [1194, 1194, 1194, 1194, 1194, 1194, 1194, 1194, 1194, 1194], [11490, 11490, 11490, 11490, 11490, 11490, 11490, 11490, 11490, 11490], [18090, 18090, 18090, 18090, 18090, 18090, 18090, 18090, 18090, 18090], [8134, 8134, 8134, 8134, 8134, 8134, 8134, 8134, 8134, 8134], [13146, 13146, 13146, 13146, 13146, 13146, 13146, 13146, 13146, 13146], [1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527, 1527], [17810, 17810, 17810, 17810, 17810, 17810, 17810, 17810, 17810, 17810], [14686, 14686, 14686, 14686, 14686, 14686, 14686, 14686, 14686, 14686], [16790, 16790, 16790, 16790, 16790, 16790, 16790, 16790, 16790, 16790], [13556, 13556, 13556, 13556, 13556, 13556, 13556, 13556, 13556, 13556], [14740, 14740, 14740, 14740, 14740, 14740, 14740, 14740, 14740, 14740], [16892, 16892, 16892, 16892, 16892, 16892, 16892, 16892, 16892, 16892], [1416, 1416, 1416, 1416, 1416, 1416, 1416, 1416, 1416, 1416], [11172, 11172, 11172, 11172, 11172, 11172, 11172, 11172, 11172, 11172], [12748, 12748, 12748, 12748, 12748, 12748, 12748, 12748, 12748, 12748], [11601, 11601, 11601, 11601, 11601, 11601, 11601, 11601, 11601, 11601], [19935, 19935, 19935, 19935, 19935, 19935, 19935, 19935, 19935, 19935], [4582, 4582, 4582, 4582, 4582, 4582, 4582, 4582, 4582, 4582], [9156, 9156, 9156, 9156, 9156, 9156, 9156, 9156, 9156, 9156], [8758, 8758, 8758, 8758, 8758, 8758, 8758, 8758, 8758, 8758], [3308, 3308, 3308, 3308, 3308, 3308, 3308, 3308, 3308, 3308], [18741, 18741, 18741, 18741, 18741, 18741, 18741, 18741, 18741, 18741], [10364, 10364, 10364, 10364, 10364, 10364, 10364, 10364, 10364, 10364], [11042, 11042, 11042, 11042, 11042, 11042, 11042, 11042, 11042, 11042], [16731, 16731, 16731, 16731, 16731, 16731, 16731, 16731, 16731, 16731], [20167, 20167, 20167, 20167, 20167, 20167, 20167, 20167, 20167, 20167], [12815, 12815, 12815, 12815, 12815, 12815, 12815, 12815, 12815, 12815], [17245, 17245, 17245, 17245, 17245, 17245, 17245, 17245, 17245, 17245], [9046, 9046, 9046, 9046, 9046, 9046, 9046, 9046, 9046, 9046], [20245, 20245, 20245, 20245, 20245, 20245, 20245, 20245, 20245, 20245], [11210, 11210, 11210, 11210, 11210, 11210, 11210, 11210, 11210, 11210], [15662, 15662, 15662, 15662, 15662, 15662, 15662, 15662, 15662, 15662], [6276, 6276, 6276, 6276, 6276, 6276, 6276, 6276, 6276, 6276], [10100, 10100, 10100, 10100, 10100, 10100, 10100, 10100, 10100, 10100], [10910, 10910, 10910, 10910, 10910, 10910, 10910, 10910, 10910, 10910], [9031, 9031, 9031, 9031, 9031, 9031, 9031, 9031, 9031, 9031], [10954, 10954, 10954, 10954, 10954, 10954, 10954, 10954, 10954, 10954], [800, 800, 800, 800, 800, 800, 800, 800, 800, 800], [3090, 3090, 3090, 3090, 3090, 3090, 3090, 3090, 3090, 3090], [15169, 15169, 15169, 15169, 15169, 15169, 15169, 15169, 15169, 15169], [4202, 4202, 4202, 4202, 4202, 4202, 4202, 4202, 4202, 4202], [5517, 5517, 5517, 5517, 5517, 5517, 5517, 5517, 5517, 5517], [6952, 6952, 6952, 6952, 6952, 6952, 6952, 6952, 6952, 6952], [9716, 9716, 9716, 9716, 9716, 9716, 9716, 9716, 9716, 9716], [996, 996, 996, 996, 996, 996, 996, 996, 996, 996], [9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666], [2336, 2336, 2336, 2336, 2336, 2336, 2336, 2336, 2336, 2336], [14005, 14005, 14005, 14005, 14005, 14005, 14005, 14005, 14005, 14005], [13095, 13095, 13095, 13095, 13095, 13095, 13095, 13095, 13095, 13095], [11666, 11666, 11666, 11666, 11666, 11666, 11666, 11666, 11666, 11666], [17136, 17136, 17136, 17136, 17136, 17136, 17136, 17136, 17136, 17136], [19395, 19395, 19395, 19395, 19395, 19395, 19395, 19395, 19395, 19395], [13360, 13360, 13360, 13360, 13360, 13360, 13360, 13360, 13360, 13360], [11727, 11727, 11727, 11727, 11727, 11727, 11727, 11727, 11727, 11727], [13150, 13150, 13150, 13150, 13150, 13150, 13150, 13150, 13150, 13150], [18970, 18970, 18970, 18970, 18970, 18970, 18970, 18970, 18970, 18970], [4034, 4034, 4034, 4034, 4034, 4034, 4034, 4034, 4034, 4034], [15037, 15037, 15037, 15037, 15037, 15037, 15037, 15037, 15037, 15037], [1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096, 1096], [13048, 13048, 13048, 13048, 13048, 13048, 13048, 13048, 13048, 13048], [7690, 7690, 7690, 7690, 7690, 7690, 7690, 7690, 7690, 7690], [5846, 5846, 5846, 5846, 5846, 5846, 5846, 5846, 5846, 5846], [17223, 17223, 17223, 17223, 17223, 17223, 17223, 17223, 17223, 17223], [11684, 11684, 11684, 11684, 11684, 11684, 11684, 11684, 11684, 11684], [10078, 10078, 10078, 10078, 10078, 10078, 10078, 10078, 10078, 10078], [9816, 9816, 9816, 9816, 9816, 9816, 9816, 9816, 9816, 9816], [10438, 10438, 10438, 10438, 10438, 10438, 10438, 10438, 10438, 10438], [1225, 1225, 1225, 1225, 1225, 1225, 1225, 1225, 1225, 1225], [16432, 16432, 16432, 16432, 16432, 16432, 16432, 16432, 16432, 16432], [13253, 13253, 13253, 13253, 13253, 13253, 13253, 13253, 13253, 13253], [13141, 13141, 13141, 13141, 13141, 13141, 13141, 13141, 13141, 13141], [41, 41, 41, 41, 41, 41, 41, 41, 41, 41], [4762, 4762, 4762, 4762, 4762, 4762, 4762, 4762, 4762, 4762], [8411, 8411, 8411, 8411, 8411, 8411, 8411, 8411, 8411, 8411]]

    height_min = 50
    height_max = 240
    height_bin_size = 1
    height_range_sigma = 20    
    height_stddev = 6.3
    height_fixed_conf = 0.08

    height_noise_specs = [[[height_stddev, "gaussian-sample-one", "replace-all", False, True, height_fixed_conf], "cont", 1, [height_min, height_max], [height_min, height_max], 
                                            (height_max+height_min)/2, height_stddev, "", 1]]      # H
    time_method = "GMM"
    period = 1 # time is checked every 1 minutes 
    time_range_sigma = 60/period
    time_min = 0
    time_max = (7*24*60/period) -1 # 7(days)*24(hours)*60(minutes)/period ( = num_time_slots)
    time_bin_size = 30

    num_sample_curves = 3
    num_samples = 10
    num_people = 100
    if num_samples is None:
        num_curves = 3
        num_samples_people_list = []
    else:
        num_curves = int(num_samples/num_sample_curves)
        num_samples_people_list = [num_samples for _ in range(1,num_people+1)]
    if num_curves == 0:
        num_curves = 1

    if time_method == "uniform":
        time_specs = [["uniform-range", "trunc-discrete", num_samples_people_list, [time_min, time_max], [time_min, time_max], num_curves, time_range_sigma, period, 0]]
        isSameSetToBeUsed = True # shuffle the set for each num_samples
    else:
        time_specs = [["GMM-range", "trunc-discrete", num_samples_people_list, [time_min, time_max], [time_min, time_max], num_curves, time_range_sigma, period, 0]]
        isSameSetToBeUsed = False # isSameSetToBeUsed = True is used to get the same set as uniform time method for the same number of samples

    orig_heights, noisy_heights, times_list = getRecogInfo(num_people, num_samples, orig_ids, height_file, height_noise_specs, time_specs) 
    print noisy_heights
