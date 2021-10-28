# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  WeightOptimisation script optimises the parameters of the MMIBN, namely, weights of the parameters    #
#  (w_G, w_A, w_H, w_T, and quality of estimation (Q)) using Bayesian optimisation*                      #
#  and tests sets from data, and calls runCrossValidation in RecognitionMemory.                          #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#                                                                                                        #
#    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (2021), "Multi-modal Open World User         #
#    Identification", Transactions on Human-Robot Interaction (THRI), 11 (1), ACM.                       #
#                                                                                                        #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), "Multi-modal Open-Set Person         #
#    Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #
#  * Bayesian optimisation is a variant of the code by Thomas Huijskens:                                 #
#    https://github.com/thuijskens/bayesian-optimization                                                 #
#                                                                                                        #          
#  This script, RecognitionMemory and each script in this project is under the GNU General Public        #
#  License v3.0. You should have received a copy of the license along with MultimodalRecognitionDataset. #
#  If not, see <http://www.gnu.org/licenses>.                                                            # 
#========================================================================================================#

import RecognitionMemory as RM
import time
import numpy as np
import pandas
import ast
import math
import os
import shutil
import csv
import scipy
from scipy import optimize
import itertools # for combination
import matplotlib.pyplot as plt # for plotting the results
import pp
# import collections
from scipy.stats import rankdata as rd
from _ast import Num
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
from operator import add
import gp

import crossValidation as cv

def sample_loss(params, args):

    return cv.crossValidationCombineVar(weights = [1.0, params[0], params[1], params[2], params[3]], qualityThreshold = params[4], args=args)

def bayesian_optim(bounds, n_iters=30, n_pre_samples=3, args = [], dec_precision=3, random_state=None):
    """Variant of https://github.com/thuijskens/bayesian-optimization"""
    # optimize weights, face_recog_rate and quality
    xp, yp = gp.bayesian_optimisation(n_iters=n_iters, 
                                   sample_loss=sample_loss,
                                   bounds=bounds,
                                   n_pre_samples=n_pre_samples,
                                   random_search=100000,args=args,
                                   dec_precision=dec_precision, random_state=random_state)
    return xp, yp

def increment(RB, initial_weights, num_param, recog_folder, start_time, step_size, end_step, param_to_optim = None, data_dir=None):
    # use method L-BFGS-B because the problem is smooth and bounded
    opt_results_file = recog_folder + "optimisation_results.csv"
    weights = initial_weights
    for i in range(0, int(end_step/step_size)):
        if i ==0:
            weights = initial_weights
        else:
            weights = weights + step_size
        # print "weights:" + str(weights)
        res = opt_func(weights, RB, recog_folder, opt_results_file, param_to_optim = param_to_optim,data_dir=data_dir)


def opt_func(weights, RB, recog_folder, opt_results_file, param_to_optim = None, data_dir=None):
    # print "*"*30
    # if os.path.isfile(RB.recog_file):
    RB.resetFiles()
    time.sleep(0.01)
#     print "weights: " + str(weights)

    identity_list, estimated_probabilities_list, false_positive, false_negative, true_positive, true_negative, false_positive_incorrect, false_positive_unknown, num_recognitions, num_unknown, face_stats = getResults(RB, weights, recog_folder, param_to_optim, data_dir=data_dir) # get evidence, fill y with 1.0 for the real person, 0.0 for the rest
#    dist = []
#    for recog_counter in range(0, len(identity_list)):
#        x = estimated_probabilities_list[recog_counter]
#        y = identity_list[recog_counter]
#        param_ranges[0] = len(y)
#        dist.append(euclideanDist(x, y, param_ranges[0]))
         
#    max_dist = max(dist)
    false_rate = false_positive + false_negative    
    df_results = pandas.DataFrame.from_items([ 
                                  ('Weights', [weights]), 
                                  ('True_positive', [true_positive]),
                                  ('True_negative', [true_negative]),
                                  ('False_positive', [false_positive]),
                                  ('False_negative', [false_negative]),
                                  ('Info', [false_rate])])
    with open(opt_results_file, 'a') as fd:
        df_results.to_csv(fd, index=False, header=False)

    return false_rate
#    return max_dist

def getOptimFaceRecogThreshold(num_folds, folds_folder, training_folder, cross_val_stats_train_file, optim_params_file, cost_function_alpha, n_iters, bounds, n_pre_samples, dec_precision):
    random_state = np.random.RandomState(1234567890)
    time_loop = time.time()
    RB = RM.RecogniserBN()
    i_real = []
    f_prob = []
    is_registering = []
    for num_fold in range(1, num_folds+1):
        file_dest = folds_folder + str(num_fold) + "/" + training_folder + RB.comparison_file
        df_stats = pandas.read_csv(file_dest,converters={"F_prob": ast.literal_eval}, dtype={"I_real": object}, usecols = ["I_real", "F_prob", "R"])
        i_real.append(df_stats.I_real.values.tolist())
        f_prob.append(df_stats.F_prob.values.tolist())
        is_registering.append(df_stats.R.values.tolist())
        
    with open(cross_val_stats_train_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Fold", "Threshold","Loss","F_FAR","F_DIR_1"])
        
    with open(cross_val_stats_train_file.replace(".csv","_avg.csv"), 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Threshold","Loss","F_FAR","F_DIR_1"])
        
    # xp, yp = gp.bayesian_optimisation(n_iters=n_iters, 
    #                               sample_loss=fr_loss,
    #                               bounds=bounds,
    #                               n_pre_samples=n_pre_samples,
    #                               random_search=100000,args=[RB, num_folds, i_real, f_prob, is_registering, cost_function_alpha,cross_val_stats_train_file],
    #                               dec_precision=dec_precision, random_state=random_state)
    xp = []
    yp = []
    for fr_threshold in np.arange(bounds[0][0], bounds[0][1], 0.01):
        avg_loss = fr_loss(fr_threshold, [RB, num_folds, i_real, f_prob, is_registering, cost_function_alpha,cross_val_stats_train_file])
        xp.append(fr_threshold)
        yp.append(avg_loss)
    
    min_error = np.amin(yp)
    min_indices = list(np.where(yp == min_error)[0])
    
    optim_param = []
    fr_dir_list = []
    fr_far_list = []
    df_stats = pandas.read_csv(cross_val_stats_train_file.replace(".csv","_avg.csv"), usecols = ["Threshold","Loss","F_FAR","F_DIR_1"])
    
    min_error_stats = df_stats.loc[np.isclose(df_stats['Loss'], min_error)].values.tolist()
    
    counter_ind = 0
    for min_ind in min_indices:
        xp_op = xp[min_ind]
        optim_param.append(xp_op)
        fr_far_list.append(min_error_stats[counter_ind][2])
        fr_dir_list.append(min_error_stats[counter_ind][3])
        
        counter_ind += 1
    with open(optim_params_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Threshold","Loss","F_FAR","F_DIR_1"])
        writer.writerow([optim_param, min_error, fr_far_list, fr_dir_list])

    print "time optim: " + str(time.time() - time_loop)
    return min_error, optim_param
    
def fr_loss(fr_threshold, args):
    [RB, num_folds, i_real, f_prob, is_registering, cost_function_alpha, cross_val_stats_train_file] = args
    avg_loss = 0
    stats_FR_train_avg = [0,0]
    for num_fold in range(1, num_folds+1):
        stats_FR = [0, 0]
        num_unknown = 0
        stats_FR_percent = [0,0]
        for num_counter in range(0, len(i_real[num_fold-1])):
            idPerson = i_real[num_fold-1][num_counter]
            total_face_prob = f_prob[num_fold-1][num_counter]
            isRegistered = not is_registering[num_fold-1][num_counter]
            
            if not isRegistered:
                num_unknown += 1
            face_est = RB.unknown_var

            max_est_value = max(total_face_prob)
            max_est_identity = total_face_prob.index(max_est_value)
            isclose_ar = np.isclose(total_face_prob, max_est_value)
            
            if num_counter >= RB.num_recog_min and max_est_value >= fr_threshold and len(isclose_ar[isclose_ar==True]) ==1:
                face_est = str(max_est_identity)

            stats_FR = RB.getPerformanceMetrics(face_est, idPerson, RB.unknown_var, isRegistered, stats_FR)

        stats_FR_percent = RB.getDIRFAR(stats_FR, len(i_real[num_fold-1]), num_unknown)

        loss = cost_function_alpha*(1.0 - stats_FR_percent[0]) + (1.0-cost_function_alpha)*(stats_FR_percent[1])
        with open(cross_val_stats_train_file, 'a') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow([num_fold, fr_threshold, loss, stats_FR_percent[1], stats_FR_percent[0]])
        avg_loss += loss
        stats_FR_train_avg = map(add, stats_FR_train_avg, stats_FR_percent)
    avg_loss /= num_folds
    stats_FR_train_avg = [x/num_folds for x in stats_FR_train_avg ]
        
    with open(cross_val_stats_train_file.replace(".csv","_avg.csv"), 'a') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow([fr_threshold, avg_loss, stats_FR_train_avg[1], stats_FR_train_avg[0]])
    return avg_loss

def euclideanDist(x, y, num_people):
    # distance_euclidean = sum[(x[i]-y[i])^2], where y[i=real_identity] = 1.0 and y[i!=real_identity] = 0
    sum_dist = 0.0
    for i in range(0, num_people):
        sum_dist += math.pow(x[i]-y[i],2)
    return sum_dist

"""rankData function from: https://stackoverflow.com/questions/13770523/python-ranking-a-list-of-values-using-average-rank-for-non-unique-values"""
def rankData(a):
#     dict_prob_list = collections.defaultdict(list)
#     counter = 0
#     for val in list_values:
#         dict_prob_list[val].append(str(counter))
#         counter += 1
#     print dict_prob_list
#     ranked_key_list = [] 
#     n = v = 1
#     for _, my_list in sorted(dict_prob_list.items()):
#         v = n + (len(my_list)-1)/2 
#         for e in my_list:
#             n += 1
#             ranked_key_list.append((e, v))
#     return collections.OrderedDict(ranked_key_list)
#         ranked_prob = np.searchsorted(np.sort(list_values), list_values)
#         return [i+1 for i in ranked_prob]   
    return (len(a) - rd(a)+1.0).astype(int)
"""End of rankData"""

def getStatsRank(est_prob, real_identity, unknown_var, isRegistered, stats_rank, num_ranks=1, est="0"):
    if num_ranks == 1:
        return getStats(est, real_identity, unknown_var, isRegistered, stats_rank[0][0], stats_rank[0][1])
    else:
        ranked_prob = rankData(est_prob)
        rank_counter = 1
        while rank_counter <= num_ranks:
            if rank_counter == 1:
                stats_rank[rank_counter-1] = getStats(est, real_identity, unknown_var, isRegistered, stats_rank[rank_counter-1][0], stats_rank[rank_counter-1][1])
            elif int(real_identity) < len(ranked_prob):
                if ranked_prob[int(real_identity)] <= rank_counter:
                    stats_rank[rank_counter-1] = getStats(real_identity, real_identity, unknown_var, isRegistered, stats_rank[rank_counter-1][0], stats_rank[rank_counter-1][1])
                else:
                    stats_rank[rank_counter-1] = getStats(est, real_identity, unknown_var, isRegistered, stats_rank[rank_counter-1][0], stats_rank[rank_counter-1][1])

            else:
                stats_rank[rank_counter-1] = getStats(est, real_identity, unknown_var, isRegistered, stats_rank[rank_counter-1][0], stats_rank[rank_counter-1][1])     
            rank_counter += 1
    return stats_rank

def getStats(identity_est, real_identity, unknown_var, isRegistered, stats, stats_openSet, num_recog=None):
#     if num_recog is not None and num_recog < 5: 
#         identity_est = unknown_var
    if isRegistered and identity_est == unknown_var:
        stats[-1] += 1
    elif (isRegistered and identity_est != real_identity) or (not isRegistered and identity_est != unknown_var):
        if isRegistered and identity_est != real_identity:
            stats[2][0] += 1
        elif not isRegistered and identity_est != unknown_var:
            stats[2][1] += 1
            stats_openSet[1] += 1
    elif not isRegistered and identity_est == unknown_var:
        stats[1] += 1
    else:
        stats[0] += 1
        stats_openSet[0] += 1
    return [stats, stats_openSet]

def getStatsPercent(stats, stats_openSet, num_recog, num_unknown):
    for ss_counter in range(0, len(stats)):
        if ss_counter == 0:
            stats[ss_counter] = float("{0:.3f}".format(stats[ss_counter]/((num_recog-num_unknown)*1.0)))
        elif ss_counter == 1:
            if num_unknown > 0:
                stats[ss_counter] = float("{0:.3f}".format(stats[ss_counter]/(num_unknown*1.0)))
            else:
                stats[ss_counter] = 0.0
        elif ss_counter == 2:
            if num_unknown > 0:
                stats[ss_counter] = [float("{0:.3f}".format(stats[ss_counter][0]/((num_recog-num_unknown)*1.0))), float("{0:.3f}".format(stats[ss_counter][1]/(num_unknown*1.0)))]
            else:
                stats[ss_counter] = [float("{0:.3f}".format(stats[ss_counter][0]/(num_recog*1.0))), 0.0]
        elif ss_counter == 3:
            stats[ss_counter] = float("{0:.3f}".format(stats[ss_counter]/((num_recog-num_unknown)*1.0)))
    
    for ss_counter in range(0, len(stats_openSet)):        
        if ss_counter == 0:
            stats_openSet[ss_counter] = float("{0:.3f}".format(stats_openSet[ss_counter]/((num_recog-num_unknown)*1.0)))
        elif ss_counter == 1:
            if num_unknown > 0:
                stats_openSet[ss_counter] = float("{0:.3f}".format(stats_openSet[ss_counter]/(num_unknown*1.0)))
            else:
                stats_openSet[ss_counter] = 0.0
              
    return [stats, stats_openSet]

def getResults(RB, weights, recog_folder, param_to_optim = None, data_dir=None, num_ranks=1):
    isSpanish = False
    isDBinCSV = True
    isMultRecognitions = False
    num_mult_recognitions = 3
    
    if param_to_optim is None:
        RB.setWeights(weights[0], weights[1], weights[2], weights[3], weights[4])
    else:
        RB.setParamWeight(weights, param_to_optim)
    if data_dir is None:
        recog_dir_up = os.path.split(recog_folder[:-1])[0] + "/"
        init_recog_file = recog_dir_up + "InitialRecognition_data.csv"
        final_recog_file = recog_dir_up + "RecogniserBN_data.csv"
       
        db_file = recog_dir_up + "db_data.csv"
    else:
        init_recog_file = data_dir + "InitialRecognition.csv"
        final_recog_file = data_dir + "RecogniserBN.csv"    
        db_file = data_dir + "db.csv"       
    
    df_init = pandas.read_csv(init_recog_file, dtype={"I_est": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
    df_final = pandas.read_csv(final_recog_file, dtype={"I": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
    recogs_list = df_final.values.tolist()
    db_list = []
    if os.path.isfile(db_file):
        df_db = pandas.read_csv(db_file, dtype={"id": object}, converters={"times": ast.literal_eval})
        db_list = df_db.values.tolist()
    
#     evidence_list = []
#     likelihood_list = []
#     bn_list = []
    identity_list = []
    estimated_probabilities_list = []
    false_positive = 0
    false_negative = 0
    true_negative = 0
    true_positive = 0
    count_recogs = 0
    count_rate = 0
    false_positive_incorrect = 0
    false_positive_unknown =0
    num_unknown = 0
#     face_stats = [0,0,[0,0],0]
#     network_stats = [0,0,[0,0],0] # [true_positive, true_negative, [false_positive_another_case, false_positive_unknown_case], false_negative]
#     face_stats_openSet = [0,0] # DIR, FAR
#     network_stats_openSet = [0,0]

    network_stats_rank = [[[0,0,[0,0],0], [0,0]] for _ in range(1, num_ranks+1)]
    face_stats_rank = [[[0,0,[0,0],0], [0,0]] for _ in range(1, num_ranks+1)]
    while count_recogs < len(recogs_list):
        recog_results = []
        # print "p_id: " + str(recogs_list[count_recogs][0])
        isMemoryRobot = True # True if the robot with memory is used (get this from the days maybe?)
        isRegistered =  not recogs_list[count_recogs][6]# False if register button is pressed (i.e. if the person starts the session for the first time)
        isAddPersonToDB = recogs_list[count_recogs][6] # True ONLY IF THE EXPERIMENTS ARE ALREADY STARTED, THE BN IS ALREADY CREATED, ONE NEW PERSON IS BEING ADDED!FOR ADDING MULTIPLE PEOPLE AT THE SAME TIME, DELETE RecogniserBN.bif FILE INSTEAD!!!
        numRecognition = recogs_list[count_recogs][7]
        person = []
        real_identity = recogs_list[count_recogs][0]
        if isAddPersonToDB:
            person = [x for x in db_list if x[0] == recogs_list[count_recogs][0]][0]
            person[0] = str(person[0])
            isRegistered = False
            num_unknown += 1
            
        # Press either register button (isRegistered = False) or start session button (isRegistered = True)
        RB.initSession(isRegistered = isRegistered, isMemoryRobot = isMemoryRobot, isAddPersonToDB = isAddPersonToDB, isDBinCSV = isDBinCSV, personToAdd = person)

        if isRegistered:
            if isMultRecognitions:
                num_mult_recognitions = df_final.loc[df_final['N'] == numRecognition].F.count()
                RB.setDefinedNumMultRecognitions(num_mult_recognitions)
                for num_recog in range(0, num_mult_recognitions):
                    recog_results.append(recogs_list[count_recogs][1:6])
                    if num_recog < num_mult_recognitions - 1:
                        count_recogs += 1
            else:
                recog_results = recogs_list[count_recogs][1:6]
        else:
            if isMultRecognitions:
                init_recog_values = df_init.loc[df_init['N'] == numRecognition].values.tolist()
                num_mult_recognitions = len(init_recog_values)
                RB.setDefinedNumMultRecognitions(num_mult_recognitions)
                for num_recog in range(0, num_mult_recognitions):
                    recog_results.append(init_recog_values[num_recog][1:6])
            else:
                init_recog_values = df_init.loc[df_init['N'] == numRecognition].values.tolist()
                recog_results = init_recog_values[0][1:6]             

        identity_est = RB.startRecognition(recog_results) # get the estimated identity from the recognition network
#         if RB.num_people > 1:
#             print "posterior:" + str(RB.ie.posterior(RB.I))
#         print "identity_est: " + str(identity_est)
        
#         evidence_list.append(RB.getNonweightedProbabilities())
        
#         likelihood_list_cur = []
#         for name in RB.node_names[1:]:
#             likelihood_list_param = []
#             id_v = RB.r_bn.idFromName(name)
#             for p in RB.i_labels:
#                 likelihood_list_param.append(RB.r_bn.cpt(id_v)[{'I':p}][:])
#             likelihood_list_cur.append(likelihood_list_param)
                
#         likelihood_list.append(likelihood_list_cur)

#         bn_list.append(RB.r_bn)

#         [network_stats, network_stats_openSet] = getStats(identity_est, real_identity, RB.unknown_var, isRegistered, network_stats, network_stats_openSet)
#         [face_stats, face_stats_openSet] = getStats(RB.face_est, real_identity, RB.unknown_var, isRegistered, face_stats, face_stats_openSet)
        
        
        identity_list_cur = [0 for i in range(0, len(RB.i_labels))]
        if recogs_list[count_recogs][0] in RB.i_labels:
            identity_list_cur[RB.i_labels.index(real_identity)] = 1
        else:
            identity_list_cur[RB.i_labels.index(RB.unknown_var)] = 1
        identity_list.append(identity_list_cur)
        
        est_prob = RB.getEstimatedProbabilities()
        est_prob = [float("{0:.3f}".format(i)) for i in est_prob]
        estimated_probabilities_list.append(est_prob)
        
        network_stats_rank = getStatsRank(est_prob, real_identity, RB.unknown_var, isRegistered, network_stats_rank, num_ranks=num_ranks,est=identity_est)
        face_stats_rank = getStatsRank(RB.face_prob, real_identity, RB.unknown_var, isRegistered, face_stats_rank, num_ranks=num_ranks,est=RB.face_est)
       
        count_rate += 1
             
        p_id = None
        isRecognitionCorrect = False
        if isMemoryRobot:
            if isRegistered:
                if identity_est != RB.unknown_var:
                    # TODO: ask for confirmation of identity_est on the tablet (isRecognitionCorrect = True if confirmed) 
                    if identity_est == real_identity:
                        isRecognitionCorrect = True # True if the name is confirmed by the patient
        
        if isRecognitionCorrect:
            RB.confirmPersonIdentity(recog_results_from_file = recog_results) # save the network, analysis data, csv for learning and picture of the person in the tablet
        else:
            if isAddPersonToDB:
                p_id = person[0]
                recog_results = []
#                 count_recogs += 1
                if isMultRecognitions:
#                     for num_recog in range(0, num_mult_recognitions):
#                         recog_results.append(recogs_list[count_recogs][1:6])
#                         if num_recog < num_mult_recognitions - 1:
#                             count_recogs += 1
                    num_mult_recognitions = df_final.loc[df_final['N'] == numRecognition].F.count()
                    RB.setDefinedNumMultRecognitions(num_mult_recognitions)
                    for num_recog in range(0, num_mult_recognitions):
                        recog_results.append(recogs_list[count_recogs][1:6])
                        if num_recog < num_mult_recognitions - 1:
                            count_recogs += 1
                else:
                    recog_results = recogs_list[count_recogs][1:6]
                       
                RB.confirmPersonIdentity(p_id = p_id, recog_results_from_file = recog_results)
            
#                 evidence_list.append(RB.getNonweightedProbabilities())
#                 likelihood_list_cur = []
#                 for name in RB.node_names[1:]:
#                     likelihood_list_param = []
#                     id_v = RB.r_bn.idFromName(name)
#                     for p in RB.i_labels:
#                         likelihood_list_param.append(RB.r_bn.cpt(id_v)[{'I':p}][:])
#                     likelihood_list_cur.append(likelihood_list_param)

#                 likelihood_list.append(likelihood_list_cur)

#                 bn_list.append(RB.r_bn)

                identity_list_cur = [0 for i in range(0, len(RB.i_labels))]
                identity_list_cur[RB.i_labels.index(p_id)] = 1
                identity_list.append(identity_list_cur)
                
                es_pr = RB.getEstimatedProbabilities()
                estimated_probabilities_list.append(es_pr)
                identity_est = RB.getEstimatedIdentity(es_pr)
                
#                if identity_est == RB.unknown_var:
#                    false_negative += 1
#                elif identity_est != real_identity:
#                    false_positive += 1
#                else:
#                    true_positive += 1
                
#                count_rate += 1
                
#                 if RB.num_people > 1:
#                     print "posterior:" + str(RB.ie.posterior(RB.I))
                
            else:
                p_id = real_identity # TODO: ask for patient name (p_id) on tablet
                RB.confirmPersonIdentity(p_id = p_id, recog_results_from_file = recog_results)

        # print "-"*10
        count_recogs += 1
    print network_stats_rank[0]
    print count_recogs
    for num_rank in range(1, num_ranks+1):
        face_stats_rank[num_rank-1] = getStatsPercent(face_stats_rank[num_rank-1][0], face_stats_rank[num_rank-1][1], count_rate, num_unknown)
        network_stats_rank[num_rank-1] = getStatsPercent(network_stats_rank[num_rank-1][0], network_stats_rank[num_rank-1][1], count_rate, num_unknown)
    RB.saveConfusionMatrix()   
#     return evidence_list, likelihood_list, identity_list, estimated_probabilities_list
#     return evidence_list, bn_list, identity_list, estimated_probabilities_list
    return identity_list, estimated_probabilities_list, face_stats_rank, network_stats_rank, count_rate, num_unknown
                            
def normEvidence(evidence, weight, index_param):
    sum_weighted = 0.0
    for count in range(0, len(evidence)):
        sum_weighted += math.pow(evidence[count], weight)
    return math.pow(evidence[index_param],weight)/float(sum_weighted)

def resetOptimisationResults(optim_file):
    if os.path.isfile(optim_file):
        os.remove(optim_file)
    with open(optim_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Weights", "True_positive", "True_negative", "False_positive", "False_negative", "Info"])   

def getBestOptimParam(main_folder, results_folder, evidence_folders, threshold_folders, norm_methods, params_optim, optim_stats_file):
    write_header = params_optim[:]
    write_header.insert(0,"norm_method")
    for evidence_folder in evidence_folders:
            
        for threshold_folder in threshold_folders:
            
            folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"
            if not os.listdir(folder_up):
                print "empty directory, skipping"
                continue
            optim_stats_file_set = optim_stats_file.replace(".csv","_"+evidence_folder+"_"+threshold_folder+".csv")
            optim_stats_file_set = main_folder + "/" + optim_stats_file_set
            with open(optim_stats_file_set, 'wb') as outcsv:
                writer = csv.writer(outcsv)
                # writer.writerow([evidence_folder, threshold_folder])          
                writer.writerow(write_header)
            
            for norm_method in norm_methods:
                optim_weights = [norm_method]
                optim_results = [""]
                for param_to_optim in params_optim:
                    recog_folder = (main_folder + "/" + evidence_folder + "/" + threshold_folder + "/" + 
                                   norm_method + "/" +results_folder + "/" + param_to_optim + "/")
                    
                    optim_file = recog_folder + "optimisation_results.csv"
                    info_df = pandas.read_csv(optim_file, usecols = ["Weights","Info"])
                    info_list = info_df.values.tolist()
                    min_error_list = [x[1] for x in info_list]
                    min_error = min(min_error_list)
                    min_weights = info_df.loc[info_df['Info'] == min_error].Weights.tolist()
                    optim_weight = max(min_weights)
                    optim_weights.append(optim_weight)
                    optim_results.append(min_error)
                with open(optim_stats_file_set, 'a') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow(optim_weights)
                    writer.writerow(optim_results)
    
def combineOptimResults(RB, main_folder, evidence_folders, threshold_folders, 
                        results_folder, norm_methods, params_optim, optim_stats_file, optim_stats_comb_file, comb_folders,
                        list_comb_parameters):
    comb_folders_set = comb_folders[:]
    comb_folders_set.insert(0, "norm_method")
    
    for evidence_folder in evidence_folders:
        if evidence_folder == "noUpdate":
            RB.setUpdateMethod("none")
        elif evidence_folder == "updateEvidence":
            RB.setUpdateMethod("evidence")
            
        for threshold_folder in threshold_folders:
            
            folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"
            if not os.listdir(folder_up):
                print "empty directory, skipping"
                continue
            
            optim_stats_file_set = optim_stats_file.replace(".csv","_"+evidence_folder+"_"+threshold_folder+".csv") 
            optim_stats_file_set = main_folder + "/" + optim_stats_file_set
            optim_stats_comb_file_set = optim_stats_comb_file.replace(".csv","_"+evidence_folder+"_"+threshold_folder+".csv")
            optim_stats_comb_file_set = main_folder + "/" + optim_stats_comb_file_set
            with open(optim_stats_comb_file_set, 'wb') as outcsv:
                writer = csv.writer(outcsv)
#                writer.writerow([evidence_folder, threshold_folder])   
                writer.writerow(comb_folders_set)
            
            if threshold_folder == "noProb":
                RB.setProbThreshold(0.0)
            elif threshold_folder == "probThreshold0.000001":
                RB.setProbThreshold(0.000001)
                
            for norm_method in norm_methods:
                
                df_comp = pandas.read_csv(optim_stats_file_set)
                init_weights = df_comp.loc[df_comp['norm_method'] == norm_method].values.tolist()
                init_weights = init_weights[0][1:]
                print "norm_method:" + str(norm_method)
                RB.setNormMethod(norm_method)
                optim_weights = [norm_method]
                optim_results = [norm_method+"-results"]

                for tuple_weights in list_comb_parameters:
                    weights = list(tuple_weights)
                    name_folder = ""
                    count_one = 0
                    for w_counter in range(0,len(params_optim)): 
                        if weights[w_counter] == 1:
                            if name_folder == "":
                                name_folder = params_optim[w_counter]
                            else:
                                name_folder = name_folder + "+" + params_optim[w_counter]
                            count_one += 1
                    if count_one == 0:
                        name_folder = "F"
                    
                    for w_counter in range(0,len(weights)): 
                        if weights[w_counter] == 1:
                            weights[w_counter] = init_weights[w_counter]
                    weights.insert(0,1.0) #for face recognition
                    print weights
                    RB.setWeights(weights[0], weights[1], weights[2], weights[3], weights[4])
                    recog_folder = (main_folder + "/" + evidence_folder + "/" + threshold_folder + "/" + 
                                   norm_method + "/" +results_folder + "/" + name_folder + "/")
                    if not os.path.isdir(recog_folder):
                        os.makedirs(recog_folder)

                    RB.resetFilePaths()
                    RB.setFilePaths(recog_folder)
                    RB.resetFiles()
                    RB.setDebugMode(False)
                    RB.setLogMode(False)
                    identity_list, estimated_probabilities_list, false_positive, false_negative, true_positive, true_negative, false_positive_incorrect, false_positive_unknown, num_recognitions, num_unknown, face_stats = getResults(RB, weights, recog_folder, param_to_optim=None) # get evidence, fill y with 1.0 for the real person, 0.0 for the rest
                    results = [true_positive, true_negative, [false_positive_incorrect, false_positive_unknown], false_negative]
#                     results= RB.getStats(RB.comparison_file, RB.stats_file)[0]

                    optim_weights.append(weights)
                    optim_results.append(results)
                    print norm_method
                    print weights
                    print results
                with open(optim_stats_comb_file_set, 'a') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow(optim_weights)
                    writer.writerow(optim_results)
                    
def getPercentage(values, num_unknown, num_recog):
    values_new = []
    for val_index in range(0,len(values)):
        vn =[]
        for inner_index in range(0, 2):         
            if inner_index == 0:
                divid = (num_recog[val_index]-num_unknown[val_index])*1.0
            elif inner_index == 1:
                divid = num_unknown[val_index]*1.0
            if divid == 0:
                val_per = 0.0
            else:
                val_per = values[val_index][inner_index]/divid
            vn.append(float("{0:.3f}".format(val_per)))
        val_per = (values[val_index][0]+values[val_index][1])/(num_recog[val_index]*1.0)
        vn.append(float("{0:.3f}".format(val_per)))
        values_new.append(vn)
    return values_new

def plotForRangeVar(fig, values, name_x_axis, name_y_axis, linestyle, color, marker, isStyleGiven=True, isYAxisSet=False, minYAxis=0.0, maxYAxis=1.0):
    ax = fig.add_subplot(111)
    ax.set_xlabel(name_x_axis, fontsize=25)
    ax.set_ylabel(name_y_axis, fontsize=25)
    ax.set_xlim(0, len(values))
    dim = np.arange(1,len(values)+1,1)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticks(dim)
    if isYAxisSet:
        ax.set_ylim([minYAxis, maxYAxis])
        
    if isStyleGiven:
        plot, = ax.plot(dim, values, linestyle=linestyle, color=color,marker=marker)
    else:
        plot, = ax.plot(dim, values)

#     # Shrink current axis by 20%
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax, plot

def plotROC(fig, dir_values, far_values, name_y_axis, name_x_axis, face_dir=None, face_far=None, linestyle="-", color= "b"):
    ax = fig.add_subplot(111)
    ax.set_xlabel(name_x_axis, fontsize=38)
    ax.set_ylabel(name_y_axis, fontsize=38)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.set_xticks(dir_values)
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')
    plot, = ax.plot(far_values, dir_values, linestyle= linestyle, color= color)
    if face_dir is not None:
        plt.axhline(face_dir, color ='r', linestyle = ':')
        #     ax.text(min(far_values), face_dir+0.003, "FR_DIR")
    if face_far is not None:
        plt.axvline(face_far, color ='r', linestyle = ':')
        #     ax.text(face_far+0.003, min(dir_values), "FR_FAR")
    ax.tick_params(axis="both", which="major", labelsize=35)

    return ax, plot

def plotCurveForQuality(main_folder, valid_folder, quality_file_name, plot_file_name, face_threshold, linestyle="-", color="b", fig = None):
#     fig = plt.figure()
    df_comp = pandas.read_csv(quality_file_name)
    I_FAR = df_comp.loc[df_comp['Face_threshold'] == face_threshold].I_FAR.tolist()
    F_FAR = df_comp.loc[df_comp['Face_threshold'] == face_threshold].F_FAR.iloc[0]
    I_DIR = df_comp.loc[df_comp['Face_threshold'] == face_threshold].I_DIR_1.tolist()
    F_DIR = df_comp.loc[df_comp['Face_threshold'] == face_threshold].F_DIR_1.iloc[0]    

    ax, plot = plotROC(fig, I_DIR, I_FAR, "DIR", "FAR", face_dir=F_DIR, face_far=F_FAR, linestyle=linestyle, color=color)
 
#     plt.xticks(np.arange(0.2, 0.7, 0.1))
#     plt.yticks(np.arange(0.86, 0.98, 0.02))
   
    plot_file = (main_folder+ "/"+ "plots"+ "/" + plot_file_name + "_" + valid_folder + ".pdf")
          
#     fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0.1, dpi=fig.dpi, format="pdf")

def plotBar(main_folder, plot_file_name):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey= True)
    plt.rc('text', usetex = True)
    plt.rc('font', family='serif')
    
    values = [[0.033, 0.9, 0.033, 0.033], [0.228, 0.189, 0.204, 0.38], [0.039, 0.862, 0.035, 0.065], [0.036, 0.881, 0.034, 0.049]]
    x = [0,1,2,3]
    ax1.bar(x, values[0], align ='center')
    ax1.set_ylim(0, 1)
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel('(a)', fontweight = 'bold', fontsize=32, labelpad=20)
    ax1.set_ylabel('Probability', fontweight = 'bold', fontsize=32, labelpad=20)

    ax1.tick_params(axis="both", which="major", labelsize=30)
    ax1.text(x[0]+0.35, max(values[0])+0.01, str(max(values[0])), fontsize=22)


#     ax1.set_title('Initial P(F$\\vert$I="1")')
    ax2.bar(x, values[1], align ='center')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel('(b)', fontweight = 'bold', fontsize=32, labelpad=20)
    ax2.tick_params(axis="both", which="major", labelsize=30)

#     ax2.set_title('F evidence')

#     plt.xticks(x, ids)
    ax3.bar(x, values[2], align ='center')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.set_xlabel('(c)', fontweight = 'bold', fontsize=32, labelpad=20)
    ax3.tick_params(axis="both", which="major", labelsize=30)
    ax3.text(x[0]+0.35, max(values[2])+0.01, str(max(values[2])), fontsize=22)


#     ax3.set_title('P(F$\\vert$F=f,G=g,A=a,$\\newline$H=h,T=t,I="1")')

#     plt.xticks(x, ids)
    ax4.bar(x, values[3], align ='center')
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.set_xlabel('(d)', fontweight = 'bold', fontsize=32, labelpad=20)
    ax4.tick_params(axis="both", which="major", labelsize=30)
    ax4.text(x[0]+0.35, max(values[3])+0.01, str(max(values[3])), fontsize=22)

    plt.rcParams["figure.figsize"] = [20, 20]
#     ax4.set_title('Updated P(F$\\vert$I="1")')

#     plt.xticks(x, ids)
    plt.subplots_adjust(wspace=0.4)
    plot_file = (main_folder+ "/"+ "plots"+ "/" + plot_file_name + ".pdf")
    fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0.1, dpi=fig.dpi, format="pdf")
    
def plotCurveForFaceThreshold(main_folder, valid_folder, face_threshold_file_name, plot_file_name):
    fig = plt.figure()
    df_comp = pandas.read_csv(face_threshold_file_name)
    I_FAR = df_comp.I_FAR.tolist()
    F_FAR = df_comp.F_FAR.tolist()
    I_DIR = df_comp.I_DIR_1.tolist()
    F_DIR = df_comp.F_DIR_1.tolist() 
    face_thresholds = df_comp.Face_threshold.tolist()
    df_minmax = pandas.read_csv("cross_validation/results/avg_faceThreshold_results_minmaxTraining.csv")
    minmax_FAR = df_minmax.I_FAR.tolist()
    minmax_DIR = df_minmax.I_DIR_1.tolist()
    
    ax, plot = plotROC(fig, I_DIR, face_thresholds, "Performance", "$\\theta_{FR}$", linestyle="-", color= "b")
    ax, plot = plotROC(fig, F_DIR, face_thresholds, "Performance", "$\\theta_{FR}$", linestyle=":", color= "b")
    ax, plot = plotROC(fig, minmax_DIR, face_thresholds, "Performance", "$\\theta_{FR}$", linestyle="--", color= "b")

    ax.text(min(face_thresholds)+0.05, max(I_DIR)+0.003, "DIR", fontsize=30)
    
    ax, plot = plotROC(fig, I_FAR, face_thresholds, "Performance", "$\\theta_{FR}$", linestyle="-", color= "r")
    ax, plot = plotROC(fig, F_FAR, face_thresholds, "Performance", "$\\theta_{FR}$", linestyle=":", color= "r")
    ax, plot = plotROC(fig, minmax_FAR, face_thresholds, "Performance", "$\\theta_{FR}$", linestyle="--", color= "r")

    ax.text(min(face_thresholds)+0.05, max(I_FAR)+0.003, "FAR", fontsize=30)
    
    plot_file = (main_folder+ "/"+ "plots"+ "/" + plot_file_name + "_" + valid_folder + ".pdf")
    
    fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0.1, dpi=fig.dpi, format="pdf")
    
def plotForLabelVar(name_file, name_axis, valuess, labelss, isToMax = True, isToMin = False, containsBase = False):
    values = valuess[:]
    labels = labelss[:]
    
    size_var = len(labels)
    ra = np.arange(size_var)
    ra_reverse = np.arange(size_var - 1, -1, -1)  # reverse order
    vx = ["{0}:{1:1.3f}".format(labels[i], values[i]) for i in ra_reverse]

    fig = plt.figure()
    fig.set_figheight(size_var / 4.0)
    fig.set_figwidth(2)

    ax = fig.add_subplot(111)
    vals = values
    vals.reverse()
    if isToMax:
        if containsBase:
            max_val = max(vals[:-1]) # last element is base condition
        else:
            max_val = max(vals)
        indices = [i for i, x in enumerate(vals) if x == max_val]
    if isToMin:
        if containsBase:
            min_val = min(vals[:-1]) # last element is base condition
        else:
            min_val = min(vals)
        indices = [i for i, x in enumerate(vals) if x == min_val]
    
    bar_list = ax.barh(ra, vals, align='center')
    
    if containsBase:
        bar_list[-1].set_color('g')
    for ii in indices:
        bar_list[ii].set_color('r')
    
    ax.set_xlim(0, 1)
    ax.set_yticks(np.arange(size_var))
    ax.set_yticklabels(vx)
    ax.set_xticklabels([])
    # ax.set_xlabel('Probability')
    ax.set_title(name_axis)
    ax.get_xaxis().grid(True)
    fig.savefig(name_file+".pdf", bbox_inches='tight', transparent=True, pad_inches=0, dpi=fig.dpi, format="pdf")

def plotOptimStats(main_folder, evidence_folders, threshold_folders, norm_methods, 
                   comb_folders, labels_folders, optim_stats_comb_file, num_recog, num_unknown, num_days=None):
    labels = labels_folders[:]
    labels.insert(0, "FR")
    containsBase = True
    axis_labels = ["Recognition rate (%)", "True positive (%)", "True negative (%)"]
    name_files = ["RR","TP","TN"]
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'k',
              'b', 'g', 'r', 'c', 'm', 'k', 
              'b', 'g', 'r', 'c']
    markers = [".", ",", "o", "v", "^", "<", ">", "+", "x", "*", "p", "s", "s", "D", "|", "_"]
    converters = dict([ (comb_folder, ast.literal_eval) for comb_folder in comb_folders ])
    for evidence_folder in evidence_folders:
            
        for threshold_folder in threshold_folders:
            folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"

            if not os.listdir(folder_up):
                print "empty directory, skipping"
                continue
            
            optim_stats_comb_file_set = optim_stats_comb_file.replace(".csv","_"+evidence_folder+"_"+threshold_folder+".csv")
            optim_stats_comb_file_set = main_folder + "/" + optim_stats_comb_file_set
            for norm_method in norm_methods:
                
                df_comp = pandas.read_csv(optim_stats_comb_file_set, dtype={"norm_method": object}, converters=converters)
                
                results = df_comp.loc[df_comp['norm_method'] == norm_method+"-results"].values.tolist()
                results = results[0][1:]
                print results
                for index_axis_label in range(0,len(axis_labels)):
                    axis_label = axis_labels[index_axis_label]              
                    plots = []
                    
                    fig = plt.figure()
                    values = []
                    for index_comb_folder in range(0, len(comb_folders)):
                        if index_axis_label == 0:
                            values.append((results[index_comb_folder][0] + results[index_comb_folder][1]))
                        elif index_axis_label == 1:
                            values.append(results[index_comb_folder][0]*num_recog/((num_recog-num_unknown)*1.0))
                        elif index_axis_label == 2:
                            values.append(results[index_comb_folder][1]*num_recog/(num_unknown*1.0))
#                         for num_day in num_days:
#                             if index_axis_label == 0:
#                                 values.append((results[index_comb_folder][0] + results[index_comb_folder][1])/num_recog[num_day])
#                             elif index_axis_label == 1:
#                                 values.append(results[index_comb_folder][0]/((num_recog[num_day]-num_unknown[num_day])*1.0))
#                             elif index_axis_label == 2:
#                                 values.append(results[index_comb_folder][1]/(num_unknown[num_day]*1.0))
                                                   
#                         comb_folder = comb_folders[index_comb_folder]
#                         linestyle = linestyles[index_comb_folder]
#                         color = colors[index_comb_folder]
#                         marker = markers[index_comb_folder]
#                         ax, plot = plotForRangeVar(fig, values, 'Day', axis_label, linestyle, color, marker)
#                         plots.append(plot)
                        
                    # Put a legend next to the plot
#                     ax.legend(plots, comb_folder[1:],loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
            
#                     fig.savefig("stats"+"_"+evidence_folder+"_"+threshold_folder+"_"+norm_method+"_"+name_files[index_axis_label]+".pdf", bbox_inches='tight', transparent=True, pad_inches=0, dpi=fig.dpi, format="pdf")
                    
                    if name_files[index_axis_label] == "RR":
                        values.insert(0, 0.8885)
                    elif name_files[index_axis_label] == "TN":
                        values.insert(0, 0.6)
                    elif name_files[index_axis_label] == "TP":
                        values.insert(0, 0.9022)
                    
                    if threshold_folder == "probThreshold0.000001":
                        threshold_folder_name = "prob"
                    else:
                        threshold_folder_name = "noprob"
                    
                    if evidence_folder == "updateEvidence":
                        evidence_folder_name = "upev"
                    elif evidence_folder == "noUpdate":
                        evidence_folder_name = "noup"
                        
                    plot_file = (main_folder+ "/"+ "plots"+ "/" + name_files[index_axis_label] + "/" 
                                 + evidence_folder_name + "_" + threshold_folder_name + "_" + norm_method + 
                                 "_" + name_files[index_axis_label])
                    plotForLabelVar(plot_file, axis_label, values, labels, containsBase=containsBase)

def plotForRank(best_methods_file_name, plot_file_name, num_ranks, main_folder, evidence_folders, threshold_folders, norm_methods,
                optimised_weights, params_list, linestyles, colors, markers, 
                isShowMethodName = True, isFaceRecogThreshold=False, isQualityThreshold=False, thresholdList = None):
    cols = ['Fold', 'Evidence_method', 'Threshold', 'Norm_method', 'I_FAR', 'F_FAR']
    for num_rank in range(1, num_ranks+1):
        cols.append("I_DIR_" + str(num_rank))
        cols.append("F_DIR_" + str(num_rank))
    if isFaceRecogThreshold:
        cols.append("Face_threshold")
    elif isQualityThreshold:
        cols.append("Quality_threshold")

    df_comp = pandas.read_csv(best_methods_file_name, usecols=cols)
#     df_minmax = pandas.read_csv("cross_validation/results/avg_faceThreshold_results_minmaxTraining.csv", usecols=cols)

    plots = []
    file_legend = []
    fig = plt.figure()
    style_counter = 0
    w_counter = 0
    i_rank_cols = []
    f_rank_cols = []
    for num_rank in range(1, num_ranks+1):
        i_rank_cols.append("I_DIR_" + str(num_rank))
        f_rank_cols.append("F_DIR_" + str(num_rank))
    
    isFaceAdded = False
    condition = False
    
    threshold_counter = 0
    if thresholdList is not None:
        threshold_values = thresholdList
    elif isFaceRecogThreshold:
        threshold_values = df_comp["Face_threshold"].values.tolist()
        print threshold_values
    elif isQualityThreshold:
        threshold_values = df_comp["Quality_threshold"].values.tolist()
    len_threshold = len(threshold_values)
    while condition is False:
        # NETWORK
        for evidence_folder in evidence_folders:              
            for threshold_folder in threshold_folders:
                linestyle = linestyles[style_counter]
                color = colors[style_counter]
                marker = markers[style_counter]
                if isFaceRecogThreshold:
                    param_df_values = df_comp[(df_comp['Fold'] == "AVG") &
                                       (df_comp['Evidence_method'] == evidence_folder) &
                                       (df_comp['Threshold'] == threshold_folder) &
                                       (df_comp['Face_threshold'] == threshold_values[threshold_counter])]
#                     param_df_minmax = df_minmax[(df_minmax['Fold'] == "AVG") &
#                                        (df_minmax['Evidence_method'] == "noup") &
#                                        (df_minmax['Threshold'] == "prob") &
#                                        (df_minmax['Face_threshold'] == threshold_values[threshold_counter])]
                elif isQualityThreshold:
                    param_df_values = df_comp[(df_comp['Fold'] == "AVG") &
                                        (df_comp['Evidence_method'] == evidence_folder) &
                                        (df_comp['Threshold'] == threshold_folder) &
                                        (df_comp['Quality_threshold'] == threshold_values[threshold_counter])]
                else:
                    param_df_values = df_comp[(df_comp['Fold'] == "AVG") &
                                       (df_comp['Evidence_method'] == evidence_folder) &
                                       (df_comp['Threshold'] == threshold_folder)]
                if not param_df_values.empty:
                    norm_method =  param_df_values['Norm_method'].values.tolist()[0]
                    
#                     if not isFaceAdded:
#                         # FACE RECOGNITION
#                         values = param_df_values[f_rank_cols].values.tolist()[0]
#                         far = float("{0:.3f}".format(param_df_values["F_FAR"].values.tolist()[0]))
#                         ax, plot = plotForRangeVar(fig, values, 'Rank', 'DIR', linestyle, color, marker)
#                         if isFaceRecogThreshold or isQualityThreshold:
#                             ax.text(0, values[0], str(far))
#                         plots.append(plot)
#                         file_legend.append("FR" + "," + str(far))
#                         style_counter += 1
#                         linestyle = linestyles[style_counter]
#                         color = colors[style_counter]
#                         marker = markers[style_counter]
#                         isFaceAdded = True
                                             
                    values = param_df_values[i_rank_cols].values.tolist()[0]
                    print values
                    far = float("{0:.3f}".format(param_df_values["I_FAR"].values.tolist()[0]))
                    ax, plot = plotForRangeVar(fig, values, 'Rank', 'DIR', "-", color, marker)
                    if isFaceRecogThreshold or isQualityThreshold:
                        ax.text(0+0.55, values[0]-0.001, str(far))
                    plots.append(plot)
 
                    face_values = param_df_values[f_rank_cols].values.tolist()[0]
                     
                    f_far = float("{0:.3f}".format(param_df_values["F_FAR"].values.tolist()[0]))
                    ax, plot = plotForRangeVar(fig, face_values, 'Rank', 'DIR', ":", color, marker)
                    if isFaceRecogThreshold or isQualityThreshold:
                        ax.text(0+0.55, face_values[0]-0.001, str(f_far))
                    plots.append(plot)

#                     minmax_values = param_df_minmax[i_rank_cols].values.tolist()[0]
#                     print minmax_values
#                     i_far = float("{0:.3f}".format(param_df_minmax["I_FAR"].values.tolist()[0]))
#                     ax, plot = plotForRangeVar(fig, minmax_values, 'Rank', 'DIR', "--", color, marker)
#                     if isFaceRecogThreshold or isQualityThreshold:
#                         ax.text(0+0.55, minmax_values[0]-0.001, str(i_far))
#                     plots.append(plot)
                                       
                    norm_counter = norm_methods.index(norm_method)
                    if isShowMethodName:
                        weights = optimised_weights[w_counter][norm_counter]
                        name_method = evidence_folder + "-" + threshold_folder + "-" + norm_method + "-"
                        isFirstParam = False
                        for weight_counter in range(0, len(weights)):
                            weight = weights[weight_counter]
                            if weight > 0.0:
                                if not isFirstParam:
                                    name_method += params_list[weight_counter]
                                    isFirstParam = True
                                else:
                                    name_method += "+" + params_list[weight_counter]
                        name_method += "," + str(far)
                    else:
                        name_method = str(far)
                        if isFaceRecogThreshold or isQualityThreshold:
                            name_method += "," + str(threshold_values[threshold_counter])
                    file_legend.append(name_method)
                    style_counter +=1
                    w_counter += 1
        if isFaceRecogThreshold or isQualityThreshold:
            if threshold_counter < len_threshold -1:
                threshold_counter += 1
            else:
                condition = True
        else:
            condition = True

    # SAVE FIGURE
    plot_file = (main_folder+ "/"+ "plots"+ "/" + plot_file_name + "_" + valid_folder + ".pdf")
          
    legend_fig = plt.figure(figsize=(4,2))
    legend = legend_fig.legend(plots, file_legend, 'right')
         
    fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=fig.dpi, format="pdf")
    legend_fig.savefig(plot_file.replace(".pdf","leg.pdf"))       

def plotConfusionMatrix(main_folder, fold_folder, num_fold, valid_folder, threshold_folder, threshold, results_folder, conf_matrix_file_set, plot_file_name, num_people):
    conf_matrix_file = main_folder + "/" + fold_folder + "/" + str(num_fold) + "/" + valid_folder + "/" + threshold_folder + "/" + str(threshold) + "/" + results_folder + "/" + conf_matrix_file_set + ".csv"
    conf_matrix_file_percent = main_folder + "/" + fold_folder + "/" + str(num_fold) + "/" + valid_folder + "/" + threshold_folder + "/" + str(threshold) + "/" + results_folder + "/" + conf_matrix_file_set + "Percent" + ".csv"

    df = pandas.read_csv(conf_matrix_file)
    df_percent = pandas.read_csv(conf_matrix_file_percent)

    color_array = []
    values_array = []
    for num_person in range(0, num_people+1):
        color_array.append(df_percent.loc[df_percent["Real/Estimate"] == num_person].values.tolist()[0][1:-1])
        values_array.append(df.loc[df["Real/Estimate"] == num_person].values.tolist()[0][1:-1])
    np_color= np.array(color_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.matshow(np_color, cmap=plt.cm.jet, interpolation="nearest")
    
    # Annotation code from: https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
    width, height = np.array(values_array).shape
    
    for x in xrange(width):
        for y in xrange(height):
            if values_array[x][y] > 0:
                ax.annotate(str(values_array[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    cb=fig.colorbar(res)
    plot_file = (main_folder+ "/"+ "plots"+ "/" + plot_file_name + "_" + valid_folder + ".pdf")
    fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=fig.dpi, format="pdf")
        
def getAverageFoldsThreshold(avg_file_set, optim_file_set, num_folds, num_ranks, valid_folders, isFaceRecogThreshold=False, isQualityThreshold=False):
    cols = {}

    for num_rank in range(1, num_ranks+1):
        i_col_name = "I_Stats_" + str(num_rank)
        f_col_name = "F_Stats_" + str(num_rank)
        cols[i_col_name] = ast.literal_eval
        cols[f_col_name] = ast.literal_eval
        
    for cur_folder_name in valid_folders:
        optim_file = optim_file_set.replace(".csv", cur_folder_name + ".csv")
        avg_file = avg_file_set.replace(".csv", cur_folder_name + ".csv")
        if not os.path.isfile(avg_file):
            with open(avg_file, 'wb') as outcsv:
                writer = csv.writer(outcsv)
                row = ["Fold", "Evidence_method", "Threshold", "Norm_method", "FER"]
                row.append("I_FAR")
                row.append("F_FAR")
                for num_rank in range(1,num_ranks+1):
                    row.append("I_DIR" + "_" + str(num_rank))    
                    row.append("F_DIR" + "_" + str(num_rank))
                for num_rank in range(1,num_ranks+1):  
                    row.append("I_Stats" + "_" + str(num_rank))
                    row.append("F_Stats" + "_" + str(num_rank))
                row.append("Num_recognitions")
                row.append("Num_registered")
                if isFaceRecogThreshold:
                    row.append("Face_threshold")
                if isQualityThreshold:
                    row.append("Quality_threshold")                
                writer.writerow(row)
                
        df_comp = pandas.read_csv(optim_file, converters = cols)
        if isFaceRecogThreshold:
            df_comp.round({'Face_threshold': 2})
            group_v = df_comp.loc[:].groupby('Face_threshold')
        elif isQualityThreshold:
            df_comp.round({'Quality_threshold': 3})
            group_v = df_comp.loc[:].groupby('Quality_threshold')
    
        recog_threshold = 0.0
        num_rows, num_cols = df_comp.shape
        for counter in range(0,len(group_v)):
            gr = group_v.get_group(recog_threshold)
            row = []
            for num_col in range(0, num_cols):
                value = 0
                
                for g_counter in range(0, len(gr)):
                    fold_item = gr.iloc[g_counter,num_col]
                    if num_col == 0:
                        row.append("AVG")
                        break
                    elif type(fold_item) is str: # string
                        if g_counter == 0:
                            row.append(fold_item)
                    elif type(fold_item) is list: # Stats list
                        if g_counter == 0:
                            avg_list = [0.0, 0.0, [0.0, 0.0], 0.0]
                        for item_counter in range(0, len(fold_item)):
                            if type(fold_item[item_counter]) is list:
                                for i_i_counter in range(0, len(fold_item[item_counter])):
                                    avg_list[item_counter][i_i_counter] += fold_item[item_counter][i_i_counter]
                            else:
                                avg_list[item_counter] += fold_item[item_counter]
                                
                        if g_counter == len(gr) - 1:
                            for item_counter in range(0, len(fold_item)):
                                if type(fold_item[item_counter]) is list:
                                    for i_i_counter in range(0, len(fold_item[item_counter])):
                                        avg_list[item_counter][i_i_counter] = float("{0:.3f}".format(avg_list[item_counter][i_i_counter]/len(gr)))
                                else:
                                    avg_list[item_counter] = float("{0:.3f}".format(avg_list[item_counter]/len(gr)))
                            row.append(avg_list)
                    else:
                        value += fold_item
                        if g_counter == len(gr) - 1:
                            value /= len(gr)
                            row.append(float("{0:.3f}".format(value)))
            with open(avg_file, 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(row)
            if isFaceRecogThreshold:
                recog_threshold = float("{0:.3f}".format(recog_threshold + 0.05))
            
def optimThreshold(RB, num_people, num_images_per_person, num_folds, num_ranks, optimised_weights,
                              optim_file_set, main_folder_set, valid_folders, evidence_folders, threshold_folders, norm_methods, 
                              isFaceRecogThreshold=False, isQualityThreshold=False):
    recog_threshold = 0.0
    I_FAR = 1.0
    while I_FAR > 0 and recog_threshold < 1.0:
        print "Recognition threshold:" + str(recog_threshold)
        if isFaceRecogThreshold:
            I_FAR_list = crossValidationParamOptim(RB, num_people, num_images_per_person, num_folds, num_ranks, optimised_weights,
                                  optim_file_set, main_folder_set, valid_folders, evidence_folders, threshold_folders, norm_methods, 
                                  faceRecogThreshold=recog_threshold)
        elif isQualityThreshold:
            I_FAR_list = crossValidationParamOptim(RB, num_people, num_images_per_person, num_folds, num_ranks, optimised_weights,
                                  optim_file_set, main_folder_set, valid_folders, evidence_folders, threshold_folders, norm_methods, 
                                  qualityThreshold=recog_threshold)            
        I_FAR = I_FAR_list[0][0]
        if isFaceRecogThreshold:
            recog_threshold = float("{0:.3f}".format(recog_threshold + 0.05))
        elif isQualityThreshold:
            recog_threshold = float("{0:.3f}".format(recog_threshold + 0.001))

def setQualityForFaceThreshold(optim_file, RB, num_folds, main_folder, face_threshold_folder, threshold_values, 
                               quality_threshold_start, quality_threshold_end, num_quality_increment, evidence_folders, threshold_folders, norm_methods):
    
    quality_values = np.linspace(quality_threshold_start, quality_threshold_end, num_quality_increment)
#     quality_values = [0.0]
    threshold_values = [0.45]
    optim_file = optim_file.replace(".csv", "_training.csv")
    if not os.path.isfile(optim_file):
        with open(optim_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            row = ["I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1", "Face_threshold", "Quality_threshold"]
#             row = ["Fold", "Evidence_method", "Threshold", "Norm_method", "I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1"] 
            writer.writerow(row)
    for face_threshold in threshold_values:
        for quality_threshold in quality_values:
#             for evidence_folder in evidence_folders:
#                 for threshold_folder in threshold_folders:
#                     for norm_method in norm_methods:
            network_stats_avg = [0,0]
            face_stats_avg = [0,0]
            for num_fold in range(1, num_folds+1):
                network_stats = [[0,0,[0,0],0], [0,0]]
                face_stats = [[0,0,[0,0],0], [0,0]]
#                             face_threshold_analysis_file = (main_folder + "/" + "folds" + "/" + str(num_fold) + "/" + "Training" + "/" 
#                                                                         + "optim_norm_methods" + "/" +
#                                                              evidence_folder + "/" + threshold_folder + "/" + norm_method + "/" + "Results/" + "AnalysisFolder/Comparison.csv")
                face_threshold_analysis_file = (main_folder + "/" + "folds" + "/" + str(num_fold) + "/" + "Training" + "/" + face_threshold_folder + "/" +
                                                str(face_threshold) + "/" + "Results/" + "AnalysisFolder/Comparison.csv")
                df_list = pandas.read_csv(face_threshold_analysis_file, usecols =["I_real" , "I_est", "F_est", "R", "Quality"], 
                                          dtype={"I_real": object, "I_est": object, "F_est":object}).values.tolist()
                num_unknown = 0
                num_recognitions = len(df_list)
                num_recog = 0
                for df in df_list:
                    
                    if df[3] == 1:
                        isRegistered = False
                        num_unknown += 1
                    else:
                        isRegistered = True

                    if df[1] != RB.unknown_var and df[4] < quality_threshold:
                        df[1] = RB.unknown_var
#                     
#                     if num_recog < 5:
#                         network_stats = getStats(RB.unknown_var, df[0], RB.unknown_var, isRegistered, network_stats[0], network_stats[1],num_recog=num_recog)
#                         face_stats = getStats(RB.unknown_var, df[0], RB.unknown_var, isRegistered, face_stats[0], face_stats[1],num_recog=num_recog)
#                     else:   
                    network_stats = getStats(df[1], df[0], RB.unknown_var, isRegistered, network_stats[0], network_stats[1],num_recog=num_recog)
                    face_stats = getStats(df[2], df[0], RB.unknown_var, isRegistered, face_stats[0], face_stats[1],num_recog=num_recog)
                    num_recog += 1
                network_stats_percent = getStatsPercent(network_stats[0], network_stats[1], num_recognitions, num_unknown)
                network_stats_avg = [x+y for x,y in zip(network_stats_avg, network_stats_percent[1])] 
                
                face_stats_percent = getStatsPercent(face_stats[0], face_stats[1], num_recognitions, num_unknown)
                face_stats_avg = [x+y for x,y in zip(face_stats_avg, face_stats_percent[1])]
#                 with open(optim_file, 'a') as outcsv:
#                     writer = csv.writer(outcsv)
#                     row = [network_stats_avg[1],face_stats_avg[1],network_stats_avg[0],face_stats_avg[0],float("{0:.2f}".format(face_threshold)), float("{0:.3f}".format(quality_threshold))]
# #                                 row = [num_fold,evidence_folder, threshold_folder, norm_method, network_stats_percent[1][1],face_stats_percent[1][1],network_stats_percent[1][0],face_stats_percent[1][0]]
# 
#                     writer.writerow(row)
     
            network_stats_avg = [float("{0:.3f}".format(i/num_folds)) for i in network_stats_avg]
            face_stats_avg = [float("{0:.3f}".format(i/num_folds)) for i in face_stats_avg]
            with open(optim_file, 'a') as outcsv:
                writer = csv.writer(outcsv)
                row = [network_stats_avg[1],face_stats_avg[1],network_stats_avg[0],face_stats_avg[0],float("{0:.2f}".format(face_threshold)), float("{0:.3f}".format(quality_threshold))]
#                             row = ["AVG",evidence_folder, threshold_folder, norm_method, network_stats_avg[1],face_stats_avg[1],network_stats_avg[0],face_stats_avg[0]]

                writer.writerow(row)
          
        
def crossValidationParamOptim(RB, num_people, num_images_per_person, num_folds, num_ranks, optimised_weights,
                              optim_file_set, main_folder_set, valid_folders, evidence_folders, threshold_folders, norm_methods, 
                              faceRecogThreshold=None, qualityThreshold=None):
    num_total = num_people*num_images_per_person
    I_FAR_list = []
    if faceRecogThreshold is not None:
        RB.setFaceRecognitionThreshold(faceRecogThreshold)
    
    if qualityThreshold is not None:
        RB.setQualityThreshold(qualityThreshold)
        
    for cur_folder_name in valid_folders:
        if cur_folder_name == "Training":
            num_total = num_people*num_images_per_person*(num_folds-1)/num_folds
        elif cur_folder_name == "Test":
            num_total = num_people*num_images_per_person/num_folds

        optim_file = optim_file_set.replace(".csv", cur_folder_name + ".csv")
        I_FAR_folder_list = []
        if not os.path.isfile(optim_file):
            with open(optim_file, 'wb') as outcsv:
                writer = csv.writer(outcsv)
                row = ["Fold", "Evidence_method", "Threshold", "Norm_method", "FER"]
                row.append("I_FAR")
                row.append("F_FAR")
                for num_rank in range(1,num_ranks+1):
                    row.append("I_DIR" + "_" + str(num_rank))    
                    row.append("F_DIR" + "_" + str(num_rank))
                for num_rank in range(1,num_ranks+1):  
                    row.append("I_Stats" + "_" + str(num_rank))
                    row.append("F_Stats" + "_" + str(num_rank))
                row.append("Num_recognitions")
                row.append("Num_registered")
                if faceRecogThreshold is not None:
                    row.append("Face_threshold")
                if qualityThreshold is not None:
                    row.append("Quality_threshold")                
                writer.writerow(row)
                
        for num_fold in range(1,num_folds+1):
            print "num_fold:" + str(num_fold)
            fold_folder = "cross_validation/folds/" + str(num_fold) + "/" + cur_folder_name + "/" # test
            
            main_folder = fold_folder + main_folder_set
            w_counter = 0
#             makeDirectory(main_folder)
            for evidence_folder in evidence_folders:
#                 print "evidence_folder:" + str(evidence_folder)
                if evidence_folder == "noup":
                    RB.setUpdateMethod("none")
                elif evidence_folder == "upev":
                    if cur_folder_name == "Test":
                        RB.setUpdateMethod("none")
                    else:
                        RB.setUpdateMethod("evidence")
                makeDirectory(main_folder + "/" + evidence_folder)
                for threshold_folder in threshold_folders:
#                     print "threshold_folder:" + str(threshold_folder)
                    folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"
                    makeDirectory(folder_up)
                    if threshold_folder == "noprob":
                        RB.setProbThreshold(0.0)
                    elif threshold_folder == "prob":
                        RB.setProbThreshold(0.000001)
                    norm_counter = 0
                    for norm_method in norm_methods:
                        weights = optimised_weights[w_counter][norm_counter]
#                         print "norm_method:" + str(norm_method)
#                         print "weights" + str(weights)
                        RB.setNormMethod(norm_method)
                        RB.setWeights(weights[0], weights[1], weights[2], weights[3], weights[4])
                        
                        if faceRecogThreshold is not None:
                            recog_folder = (main_folder + "/" + str(faceRecogThreshold) + "/" + "Results" + "/")
                        elif qualityThreshold is not None:
                            recog_folder = (main_folder + "/" + str(qualityThreshold) + "/" + "Results" + "/")
                        else:
                            recog_folder = (main_folder + "/" + evidence_folder + "/" + threshold_folder + "/" + 
                                       norm_method + "/" + "Results" + "/")
                        makeDirectory(recog_folder)
                        RB.resetFilePaths()
                        RB.setFilePaths(recog_folder)
                        RB.resetFiles()
                        if cur_folder_name == "Test":
                            RB.resetFilePaths()
                            if faceRecogThreshold is not None:
                                val_folder = ("cross_validation/folds/" + str(num_fold) + "/" + "Training" + "/" + main_folder_set + "/" + str(faceRecogThreshold) + "/" + "Results" + "/")
                            elif qualityThreshold is not None:
                                val_folder = ("cross_validation/folds/" + str(num_fold) + "/" + "Training" + "/" + main_folder_set + "/" + str(qualityThreshold) + "/" + "Results" + "/")
                            else:
                                val_folder = ("cross_validation/folds/" + str(num_fold) + "/" + "Training" + "/" + main_folder_set + "/" + evidence_folder + "/" + threshold_folder + "/" + 
                                       norm_method + "/" + "Results" + "/")
                            RB.copyNetworkDBFromValidation(val_folder, recog_folder)
                            RB.setFilePaths(recog_folder)
                        
                        RB.setDebugMode(False)
                        RB.setLogMode(False)
                        (identity_list, estimated_probabilities_list, 
                         face_stats_rank, network_stats_rank,
                         num_recognitions, num_unknown) = getResults(RB, weights, recog_folder, None, data_dir=fold_folder,num_ranks=num_ranks) # get evidence, fill y with 1.0 for the real person, 0.0 for the rest                   
                        
                        FER = float("{0:.3f}".format((num_total - num_recognitions)*1.0/num_total)) # failure to enroll rate
                        with open(optim_file, 'a') as outcsv:
                            writer = csv.writer(outcsv)
                            row = [num_fold, evidence_folder, threshold_folder, norm_method, FER]
                            row.append(network_stats_rank[0][1][1])
                            I_FAR_folder_list.append(network_stats_rank[0][1][1])
                            row.append(face_stats_rank[0][1][1])
                            for num_rank in range(1, num_ranks+1):
                                row.append(network_stats_rank[num_rank-1][1][0])
                                row.append(face_stats_rank[num_rank-1][1][0])
                            for num_rank in range(1, num_ranks+1):
                                row.append(network_stats_rank[num_rank-1][0]) 
                                row.append(face_stats_rank[num_rank-1][0])
                            row.append(num_recognitions)
                            row.append(num_unknown)
                            if faceRecogThreshold is not None:
                                row.append(faceRecogThreshold)
                            if qualityThreshold is not None:
                                row.append(qualityThreshold)
                            writer.writerow(row)
                        norm_counter += 1                   
                    w_counter += 1
        I_FAR_list.append(I_FAR_folder_list)
    return I_FAR_list

def makeDirectory(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    
if __name__ == "__main__":
    
    start_time = time.time()
    RB = RM.RecogniserBN()
    params_list = ["F", "G", "A", "H", "T"]
    num_params = len(params_list)
    params_optim = ["G","A","H","T"]
#     initial_weights = [1.0, 0.35, 0.15, 0.16, 0.23]
#     initial_weights = [1.0, 0.5, 0.5, 0.5, 0.5]
#    initial_weights = [1., 0.6, 0.8, 0.6, 0.7]
#     RB.setWeights(1.0, 0.5, 0.08, 0.03, 0.0)
#     RB.setWeights(1.0,0.0,0.0,0.0,0.0)
    initial_weights = 0.1
    step_size = 0.1
    end_step = 1.0
#     param_to_optim = "T"
    param_ranges = [1, len(RB.g_labels), RB.age_max-RB.age_min+1, RB.height_max-RB.height_min+1, RB.time_max-RB.time_min+1]
    dist_function = "euclidean"

#     evidence_folders = ["noUpdate", "updateEvidence"]
#     threshold_folders = ["noProb", "probThreshold0.000001"]
#     norm_methods = ["minmax", "norm-sum", "softmax", "tanh"]

    job_server = pp.Server()
    evidence_folders = ["noup", "upev"]

    threshold_folders = ["noprob", "prob"]
    norm_methods = ["norm-sum", "minmax", "softmax", "tanh"]
    cross_validation_folder = "cross_validation" 
    main_folder_set = "optim_norm_methods"
    results_folder = "optim"
    optim_stats_file = "optim_stats.csv"
    optim_stats_comb_file = "optim_stats_combinations.csv"
    stats_folder = "results"
    num_folds = 5

    optimised_weights = [[[1.0, 0, 0, 0.1, 0],  [1.0, 0.2, 0, 0.1, 0],[1.0, 0.1, 0, 0.6, 0],    [1.0, 0, 0, 0.1, 0]], #noup+ noprob
                         [[1.0, 0, 0, 0.1, 0.1],[1.0, 0.2, 0, 0.1, 0],[1.0, 0.1, 0.1, 0.1, 0.1],[1.0, 0, 0, 0.3, 0.1]], #noup+ prob
                         [[1.0, 0.1, 0, 0.1, 0],[1.0, 0, 0, 0.1, 0],  [1.0, 0.1, 0, 0.6, 0],    [1.0, 0, 0, 0.1, 0]], #upev+ noprob
                         [[1.0, 0.1, 0, 0.1, 0],[1.0, 0, 0, 0.1, 0],  [1.0, 0.1, 0, 0.6, 0],    [1.0, 0, 0, 0.3, 0]]] #upev+ prob
    
    optim_file_set = cross_validation_folder + "/" + "optim_results.csv"
    
    num_people = 14
    num_images_per_person = 65
    num_total = num_people*num_images_per_person
    num_ranks = num_people+1
    valid_folders = ["Training", "Test"]
#     crossValidationParamOptim(RB, num_people, num_images_per_person, num_folds, num_ranks, optimised_weights,
#                               optim_file_set.replace(".csv", "_threshold0_4.csv"), "norm_methods_threshold0.4", valid_folders, evidence_folders, threshold_folders, norm_methods)
    linestyles = ['-', '-', '-', '-', '-', '-',
                  '--', '--', '--', '--', '--','--',
                  '-', '-', '-', '-', '-', '-',
                  '--', '--', '--', '--', '--','--',
                  ':', ':', ':', ':', ':', ':'
                  ]
    colors = ['b', 'g', 'r', 'c', 'm', 'k',
              'b', 'g', 'r', 'c', 'm', 'k', 
              'b', 'g', 'r', 'c', 'm', 'k',
              'b', 'g', 'r', 'c', 'm', 'k',
              'b', 'g', 'r', 'c', 'm', 'k']

    markers = ["o", "x", "s", "*", ".", "+",
               "o", "x", "s", "*", ".", "+",
               "o", "x", "s", "*", ".", "+",
               "o", "x", "s", "*", ".", "+",
               "o", "x", "s", "*", ".", "+"]
    
    face_threshold_optim_file = cross_validation_folder + "/" + stats_folder + "/" +"faceThreshold_results.csv" #tanh
    face_threshold_optim_file_softmax = cross_validation_folder + "/" + stats_folder + "/" +"faceThreshold_results_softmax.csv"
    face_threshold_optim_file_minmax = cross_validation_folder + "/" + stats_folder + "/" +"faceThreshold_results_minmax.csv"

    avg_face_threshold_optim_file = cross_validation_folder + "/" + stats_folder + "/" +"avg_faceThreshold_results.csv" #tanh
    avg_face_threshold_optim_file_softmax = cross_validation_folder + "/" + stats_folder + "/" +"avg_faceThreshold_results_softmax.csv"
    avg_face_threshold_optim_file_minmax = cross_validation_folder + "/" + stats_folder + "/" +"avg_faceThreshold_results_minmax.csv"
    
    
    avg_quality_threshold_optim_file = cross_validation_folder + "/" + stats_folder + "/" +"avg_qualityThreshold_results.csv" #tanh
    avg_quality_threshold_optim_file_softmax = cross_validation_folder + "/" + stats_folder + "/" +"avg_qualityThreshold_results_softmax.csv"
    avg_quality_threshold_optim_file_minmax = cross_validation_folder + "/" + stats_folder + "/" +"avg_qualityThreshold_results_minmax.csv"
    avg_quality_threshold_optim_file_minmax_point4 = cross_validation_folder + "/" + stats_folder + "/" +"avg_qualityThreshold_results_minmax_0.4.csv"
    avg_quality_threshold_optim_file_minmax_point4_extended = cross_validation_folder + "/" + stats_folder + "/" +"avg_qualityThreshold_results_minmax_0.4_extended2.csv"
    avg_quality_threshold_optim_file_softmax_point4_extended = cross_validation_folder + "/" + stats_folder + "/" +"avg_qualityThreshold_results_softmax_0.4_extended.csv"

 
#     plotBar(cross_validation_folder, "bar_plots")
#     
#     fig = plt.figure()
#          
#     plotCurveForQuality(cross_validation_folder, "Training", avg_quality_threshold_optim_file_softmax_point4_extended, "ROC_quality_compare", 0.4, linestyle="-", color="b", fig=fig)
#     plotCurveForQuality(cross_validation_folder, "Training", avg_quality_threshold_optim_file_minmax_point4_extended, "ROC_quality_compare", 0.4, linestyle="--", color="b", fig=fig)
#     
# #     plotCurveForQuality(cross_validation_folder, "Training", avg_quality_threshold_optim_file_minmax_point4_extended, "ROC_quality_minmax4_extended", 0.4)
#     
#     
#     fig.savefig(cross_validation_folder+ "/" + "plots/" + "ROC_quality_compare_0.4.pdf", bbox_inches='tight', transparent=True, pad_inches=0.1, dpi=fig.dpi, format="pdf")
    
#     legend_fig = plt.figure(figsize=(4,2))
#     red_dot = mlines.Line2D([], [], color='r', linestyle = ':', label='FR')
#     blue_dashed_line = mlines.Line2D([], [], color='b', linestyle = '--', label='N minmax')
#     blue_line = mlines.Line2D([], [], color='b', linestyle = '-', label='NL softmax')
#     handles = [red_dot, blue_dashed_line, blue_line]
#     labels = [h.get_label() for h in handles]
#     legend = legend_fig.legend(handles=handles, labels=labels, loc='right')
#          
#     legend_fig.savefig(cross_validation_folder+ "/" + "plots/" + "ROC_quality_compare_0.4_leg_2.pdf")       
# 
#     
#     plotCurveForFaceThreshold(cross_validation_folder, "Training", avg_face_threshold_optim_file_softmax.replace(".csv", "Training.csv"), "ROC_face")
#     for valid_folder in valid_folders:
#         best_methods_file_name = cross_validation_folder + "/" + "best_methods_" + valid_folder + ".csv"
#         plotForRank(best_methods_file_name, "ROC_methods", num_ranks, cross_validation_folder, evidence_folders, threshold_folders, norm_methods,
#                 optimised_weights, params_list, linestyles, colors, markers, isShowMethodName = True)
# 
#     optimThreshold(RB, num_people, num_images_per_person, num_folds, num_ranks, [[[1.0, 0.2, 0, 0.1, 0]]],
#                               face_threshold_optim_file_minmax, "minmax_faceThreshold", valid_folders, 
#                               ["noup"], ["prob"], ["minmax"],isFaceRecogThreshold=True)
#     getAverageFoldsThreshold(avg_face_threshold_optim_file_minmax, face_threshold_optim_file_minmax, num_folds, num_ranks, valid_folders, isFaceRecogThreshold=True, isQualityThreshold=False)

#     for valid_folder in valid_folders:
#         file_to_get_data = avg_face_threshold_optim_file_minmax.replace(".csv", valid_folder + ".csv")
#         plotForRank(file_to_get_data, "ROC_face_minmax", 4, cross_validation_folder, ["noup"], ["prob"], ["minmax"],
#                 [[[1.0, 0.2, 0, 0.1, 0]]], params_list, linestyles, colors, markers, 
#                 isShowMethodName = False, isFaceRecogThreshold=True, isQualityThreshold=False, thresholdList=[0.3, 0.5, 0.7, 0.8])


#     setQualityForFaceThreshold(cross_validation_folder + "/" +"OPTIM_RESULTS_CORRECTED.csv", RB, num_folds, cross_validation_folder, "optim_norm_methods", np.linspace(0, 0.95, 20) , 
#                                0, 0.2, 21, evidence_folders, threshold_folders, norm_methods)

#     setQualityForFaceThreshold(cross_validation_folder + "/results/" +"avg_qualityThreshold_results_minmax_0.45_extended.csv", RB, num_folds, cross_validation_folder, "minmax_faceThreshold", np.linspace(0, 0.95, 20) , 
#                                0, 0.8, 81, evidence_folders, threshold_folders, norm_methods)
#        
#     correctConfMatrix(cross_validation_folder+"/" +"folds" + "/" + str(5) +"/" + "Test" +"/" + "softmax_faceThreshold" + "/" + str(0.3) + "/" + "Results" + "/" + "AnalysisFolder/Comparison.csv" , cross_validation_folder+"/" +"folds" + "/" + str(5) +"/" + "Test" +"/" + "softmax_faceThreshold" + "/" + str(0.3) + "/" + "Results" + "/" + "confusionMatrix.csv",num_people+1)
#     plotConfusionMatrix(cross_validation_folder, "folds", 5, "Training", "softmax_faceThreshold", 0.3, "Results", "confusionMatrixNetwork", "conf_matrix_plot_network", num_people)
#     plotConfusionMatrix(cross_validation_folder, "folds", 5, "Training", "minmax_faceThreshold", 0.3, "Results", "confusionMatrixNetwork", "conf_matrix_plot_network_minmax", num_people)
# 
#     plotConfusionMatrix(cross_validation_folder, "folds", 5, "Test", "softmax_faceThreshold", 0.3, "Results", "confusionMatrixNetwork", "conf_matrix_plot_network", num_people)
# 
#     plotConfusionMatrix(cross_validation_folder, "folds", 5, "Test", "softmax_faceThreshold", 0.3, "Results", "confusionMatrixFaceRecognition", "conf_matrix_plot_fr", num_people)

#     for num_fold in range(1,num_folds+1):
#         print "num_fold:" + str(num_fold)
#         fold_folder = "cross_validation/folds/" + str(num_fold) + "/Training/"
#         main_folder = fold_folder + main_folder_set
#         makeDirectory(main_folder)
#         for evidence_folder in evidence_folders:
#             print "evidence_folder:" + str(evidence_folder)
#             if evidence_folder == "noUpdate":
#                 RB.setUpdateMethod("none")
#             elif evidence_folder == "updateEvidence":
#                 RB.setUpdateMethod("evidence")
#             makeDirectory(main_folder + "/" + evidence_folder)
#             for threshold_folder in threshold_folders:
#                 print "threshold_folder:" + str(threshold_folder)
#                 folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"
#     #             if not os.listdir(folder_up):
#     #                 print "empty directory, skipping"
#     #                 continue
#                 makeDirectory(folder_up)
#                 if threshold_folder == "noProb":
#                     RB.setProbThreshold(0.0)
#                 elif threshold_folder == "probThreshold0.000001":
#                     RB.setProbThreshold(0.000001)
#                        
#                 for norm_method in norm_methods:
#                        
#                     print "norm_method:" + str(norm_method)
#                     RB.setNormMethod(norm_method)
#                     makeDirectory(folder_up + "/" + norm_method)
#                     for param_to_optim in params_optim:
#                         makeDirectory(folder_up + "/" + norm_method + "/" + param_to_optim)
#                         print "param_to_optim:" + str(param_to_optim)
#                         RB.setWeights(1.0,0.0,0.0,0.0,0.0)
# #                         recog_folder = (main_folder + "/" + evidence_folder + "/" + threshold_folder + "/" + 
# #                                        norm_method + "/" +results_folder + "/" + param_to_optim + "/")
#                         recog_folder = (main_folder + "/" + evidence_folder + "/" + threshold_folder + "/" + 
#                                        norm_method + "/" + param_to_optim + "/")
#                         optim_file = recog_folder + "optimisation_results.csv"
#                        
#                         resetOptimisationResults(optim_file)
#                         RB.resetFilePaths()
#                         RB.setFilePaths(recog_folder)
#                         RB.setDebugMode(False)
#                         RB.setLogMode(False)
#                     #     gradientDescent(RB, initial_weights, num_param, param_ranges, dist_function, recog_folder)
#                     #     basinhopping(RB, initial_weights, num_param, recog_folder, start_time)
#                     #     basinhopping(RB, initial_weights, num_param, recog_folder, start_time, param_to_optim)
#  
#                         increment(RB, initial_weights, num_params, recog_folder, start_time, step_size, end_step, param_to_optim, data_dir=fold_folder)
#                     #    job_server.submit(increment, (RB, initial_weights, num_params, recog_folder, start_time, step_size, end_step, param_to_optim, fold_folder))

#     getBestOptimParam(main_folder, results_folder, evidence_folders, threshold_folders, norm_methods, params_optim, optim_stats_file)
    
#     list_comb_parameters = set(itertools.product([1,0], repeat=len(params_optim)))

    list_comb_parameters =[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                           [1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],
                           [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],
                           [1,1,1,1]]
    labels_folders = ["F", "F+G", "F+A", "F+H", "F+T", 
                    "F+G+A", "F+G+H","F+G+T","F+A+H","F+A+T", "F+H+T",
                    "F+G+A+H", "F+G+A+T", "F+G+H+T", "F+A+H+T",
                    "F+G+A+H+T"]
    comb_folders = ["F", "G", "A", "H", "T", 
                    "G+A", "G+H","G+T","A+H","A+T", "H+T",
                    "G+A+H", "G+A+T", "G+H+T", "A+H+T",
                    "G+A+H+T"]
#     for tuple_weights in list_comb_parameters:
#         weights = list(tuple_weights)
#         name_folder = ""
#         label_f = "F"
#         count_one = 0
#         for w_counter in range(0,len(params_optim)): 
#             if weights[w_counter] == 1:
#                 if name_folder == "":
#                     name_folder = params_optim[w_counter]
#                 else:
#                     name_folder = name_folder + "+" + params_optim[w_counter]
#                 label_f = label_f + "+" + params_optim[w_counter]
#                 count_one += 1
#         
#         if count_one == 0:
#             name_folder = "F"
#         comb_folders.append(name_folder)    
#         labels_folders.append(label_f)
#     comb_folders.insert(0,"norm_method")
    
    

#  
#     combineOptimResults(RB, main_folder, evidence_folders, threshold_folders, 
#                         results_folder, norm_methods, params_optim, optim_stats_file, optim_stats_comb_file, comb_folders,
#                         list_comb_parameters)
#     plotOptimStats(main_folder, evidence_folders, threshold_folders, norm_methods, 
#                    comb_folders, labels_folders, optim_stats_comb_file, 332, 15)
#     RB.resetFilePaths()
#     num_recog = [26,49,71,103,136,160,192,226,253,288,332]
#     num_unknown = [11,11,13,15,15,15,15,15,15,15,15] 
#     stats_file = main_folder + "/" + "stats_optim_graph.csv"

#     with open(stats_file.replace(".csv","_day.csv"), 'wb') as outcsv:
#         writer = csv.writer(outcsv)
#         writer.writerow(["Day", "Update_method", "Threshold_method", "Norm_method", "Optim_param","Stats_network","Stats_face","Num_unknown","Num_total_recognitions"])
#     with open(stats_file, 'wb') as outcsv:
#         writer = csv.writer(outcsv)
#         writer.writerow(["Day", "Update_method", "Threshold_method", "Norm_method", "Optim_param","Stats_network","Stats_face","Num_unknown","Num_total_recognitions"])    
#      
#     for evidence_folder in evidence_folders:
#             
#         for threshold_folder in threshold_folders:
#             folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"
# 
#             if not os.listdir(folder_up):
#                 print "empty directory, skipping"
#                 continue
#             
#             for norm_method in norm_methods:
#                 
#                 for comb_folder in comb_folders:
#                     
#                     recog_folder = (main_folder + "/" + evidence_folder + "/" + threshold_folder + "/" + 
#                                    norm_method + "/" +results_folder + "/" + comb_folder + "/")
#                     comp_file = recog_folder + RB.comparison_file
# 
#                     stats, stats_percent, stats_graph = RB.getStats(comp_file, stats_file,
#                                 num_total_recogs_list = num_recog, num_total_unknown_list = num_unknown, 
#                                 isWriteToFile = False, optim_param=comb_folder, norm_method=norm_method,
#                                 update_method=evidence_folder, threshold_method=threshold_folder)
#                     print stats_graph
#                     
    params_to_get_names = [[["F+H"],["F+G"],["F+H"],["F+H"]],
                     [["F+H"],["F+G", "F+H"],["F+H"],["F+H"]],
                     [["F+G"],["F+G"],["F+H"],["F+H"]],
                     [["F+H"],["F+G"],["F+A","F+H"],["F+H"]]
                     ]
    
    params_to_get = [[["H"],["G"],["H"],["H"]],
                     [["H"],["G", "H"],["H"],["H"]],
                     [["G"],["G"],["H"],["H"]],
                     [["H"],["G"],["A","H"],["H"]]
                     ]
    stats_full = []
    axis_labels = ["Recognition rate (%)", "True positive (%)", "True negative (%)"]
    name_files = ["RR","TP","TN"]

"""
    stats_files = [main_folder + "/" + "stats_optim_graph.csv", main_folder + "/" + "stats_optim_graph_day.csv"]
    for stats_file_counter in range(0, len(stats_files)):
        df_comp = pandas.read_csv(stats_files[stats_file_counter], dtype={"Optim_param": object}, converters={"Stats_network": ast.literal_eval, "Stats_face": ast.literal_eval})
        line_counter = 0

        for evidence_folder in evidence_folders:
            if evidence_folder == "updateEvidence":
                evidence_folder_name = "upev"
            elif evidence_folder == "noUpdate":
                evidence_folder_name = "noup"
                
            for threshold_folder in threshold_folders:
                if threshold_folder == "probThreshold0.000001":
                    threshold_folder_name = "prob"
                else:
                    threshold_folder_name = "noprob"
                folder_up = main_folder + "/" + evidence_folder + "/" + threshold_folder + "/"
    
                if not os.listdir(folder_up):
                    print "empty directory, skipping"
                    continue
                
                for index_axis_label in range(0,len(axis_labels)):
                    axis_label = axis_labels[index_axis_label]              
                    plots = []
                    file_legend = []
                    fig = plt.figure()
                    style_counter = 0

                    isFaceAdded = False
                    index_norm_method = 0
                    for norm_method in norm_methods:
                        print line_counter
                        print index_norm_method
                        ptg = params_to_get[line_counter][index_norm_method]
                        for index_comb_folder in range(0, len(ptg)):
                            optim_param = ptg[index_comb_folder]
                            linestyle = linestyles[style_counter]
                            color = colors[style_counter]
                            marker = markers[style_counter]
                            param_df_values = df_comp[
                                                   (df_comp['Update_method'] == evidence_folder) &
                                                   (df_comp['Threshold_method'] == threshold_folder) &
                                                   (df_comp['Norm_method'] == norm_method) &
                                                   (df_comp['Optim_param'] == optim_param)]
                            
                            if not isFaceAdded:
                            
                                values_full = param_df_values.Stats_face.tolist()[:]
                                values = [val[index_axis_label] for val in values_full]
                                print values
                                isFaceAdded = True
                                ax, plot = plotForRangeVar(fig, values, axis_label, linestyle, color, marker)
                                plots.append(plot)
                                style_counter+= 1
                                
                                linestyle = linestyles[style_counter]
                                color = colors[style_counter]
                                marker = markers[style_counter]
                                param_df_values = df_comp[
                                                       (df_comp['Update_method'] == evidence_folder) &
                                                       (df_comp['Threshold_method'] == threshold_folder) &
                                                       (df_comp['Norm_method'] == norm_method) &
                                                       (df_comp['Optim_param'] == optim_param)]
                                file_legend.append("FR")
                            values_full = param_df_values.Stats_network.tolist()[:]
                            values = [val[index_axis_label] for val in values_full]
                            print values
                            if index_axis_label == 2: #todo: this is hack for true negative, change this
                                ax, plot = plotForRangeVar(fig, values, axis_label, linestyle, color, marker, isYAxisSet=True)
                            else:
                                ax, plot = plotForRangeVar(fig, values, axis_label, linestyle, color, marker)
                            plots.append(plot)
#                             leg = evidence_folder_name + "-" + threshold_folder_name + "-" + norm_method + "-" + optim_param
                            leg = norm_method + "-" + "F"+ "+" + optim_param

                            file_legend.append(leg)
                            style_counter +=1
                        index_norm_method += 1
                                            

                    if stats_file_counter == 0:
                        plot_file = (main_folder+ "/"+ "plots"+ "/" + "overall" + "/" + name_files[index_axis_label] + "/"+ evidence_folder_name + "-" + threshold_folder_name + "-" + name_files[index_axis_label] + ".pdf")
                    else:
                        plot_file = (main_folder+ "/"+ "plots"+ "/"+ "day" + "/" + name_files[index_axis_label] + "/"+ evidence_folder_name + "-" + threshold_folder_name + "-" + name_files[index_axis_label] + "_day.pdf")
                    # create a second figure for the legend
#                     if index_axis_label == 0:
#                         ax.legend(plots, file_legend ,loc='bottom right', bbox_to_anchor=(1, 0.5), fontsize=20)

                    
                    legend_fig = plt.figure(figsize=(3,2))
                    legend = legend_fig.legend(plots, file_legend, 'right')
#                     legend_fig.canvas.draw()
                    legend_fig.savefig(plot_file.replace(".pdf","leg.pdf"),
                                        bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))
        
                    fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=fig.dpi, format="pdf")
                    legend_fig.savefig(plot_file.replace(".pdf","leg.pdf"))
                line_counter += 1 
"""
