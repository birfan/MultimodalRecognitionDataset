# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  run_CrossValidationOnRobot script runs the cross validation on robot (Pepper Naoqi 2.4 or 2.5)        #
#  for a specified directory. It also has the functionality to clean images in a path which do not have  #
#  a face detected by ALFaceDetection.                                                                   #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #   
#    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (under review), 'Multi-modal Incremental     #
#    Bayesian Network with Online Learning for Open World User Identification', ACM Transactions on      #
#    Human-Robot Interaction (THRI).                                                                     #
#                                                                                                        #         
#  This script, RecognitionMemory and each script in this project is under the GNU General Public        #
#  License.                                                                                              #
#========================================================================================================#

import RecognitionMemory
import crossValidation as cv
import time
import csv
import ast
import pandas
import numpy as np
import matplotlib.pyplot as plt # for plotting the results

def plotConfusionMatrix(fold_folder_set, num_fold, set_folder, conf_matrix_file_set):
    conf_matrix_file = fold_folder_set + str(num_fold) + "/" + set_folder + conf_matrix_file_set
    conf_matrix_file_percent = conf_matrix_file.replace(".csv", "Percent" + ".csv")

    df = pandas.read_csv(conf_matrix_file)
    df_percent = pandas.read_csv(conf_matrix_file_percent)

    color_array = []
    for i in range(0, len(df_percent)):
        num_person = df_percent.iloc[i,0]
        color_array.append(df_percent.iloc[i,1:-1])
    np_color= np.array(color_array)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.matshow(np_color, cmap=plt.cm.jet, interpolation="nearest")
    
    cb=fig.colorbar(res)
    plot_file = conf_matrix_file.replace(".csv", ".pdf")
    fig.savefig(plot_file, bbox_inches='tight', transparent=True, pad_inches=0, dpi=1200, format="pdf")

if __name__ == "__main__":
    
    isAnalysisIteration = True
    num_repeat_start = 6
    num_repeats = 6

    num_people = 100
    num_folds = 5
    num_bins = num_folds
    update_methods = ["none", "evidence"]
    norm_methods = ["softmax", "minmax", "tanh", "norm-sum", "hybrid"]
    cross_val_folders = ["N10_gaussianT/", "Nall_gaussianT/", "N10_uniformT/", "Nall_uniformT/"]
    train_folder = "Training/"
    test_folder = "Test/"
    validation_info_file = "validation_info_fold.csv"
    main_folder = "recog_values_robot/Repeated5fold/"
    db_file = "db_data.csv"
    init_recog_file = "InitialRecognition_data.csv"
    final_recog_file = "RecogniserBN_data.csv"
    conf_matrix_file = "confusionMatrix.csv"
    analysis_file = None
    setT = "train"
    face_weights = [1.0]
    isSaveRecogFiles = False
    isSaveImageAn = False
    models_folder = "Models/"

    if isAnalysisIteration:
        face_weights = [0.0, 1.0]
        analysis_file = "analysis.csv"
        norm_methods = ["hybrid"]
        num_repeat_start = 6
        num_repeats = 6
        isSaveRecogFiles = True
        isSaveImageAn = False

    for num_repeat in range(num_repeat_start, num_repeats+1):

        for eval_folder in ["training", "closed-test", "open", "open-closed"]:

            if isAnalysisIteration:
                results_file = analysis_file.replace(".csv", "-" + eval_folder + str(num_repeat) + ".csv")
            else:
                results_file = eval_folder + str(num_repeat) + ".csv"
#            with open(results_file, 'wb') as outcsv:
#                writer = csv.writer(outcsv)   
#                writer.writerow(["Num_repeat", "Fold", "Time_type", "Sample_size", "Learning_method", "Norm_method", "I_FAR", "F_FAR", "I_DIR", "F_DIR", "I_Loss", "F_Loss",  "Num_recognitions", "Num_registered"])

        print "num_repeat:" + str(num_repeat)

        for faceWeight in face_weights:
            dest_main = main_folder + "cross_validation_" + setT + "/"
            for cross_val in cross_val_folders:
                if "N10" in cross_val:
                    sample_size = "ten"
                else:
                    sample_size = "all"
                if "gaussian" in cross_val:
                    time_type = "gaussian"
                else:
                    time_type = "uniform"
                optim_file = cross_val.replace("/","_") + "optim_params.csv"
                optim_values = pandas.read_csv(optim_file, dtype={"Evidence_method": object, "Norm_method": object, "Optim_params":object}, usecols = {"Evidence_method", "Norm_method", "Optim_params"})
                start_time = time.time()
                for updateMethod in update_methods:

                    if isAnalysisIteration:
                        if faceWeight == 0.0:
                            model_name = "SB"
                            isSaveImageAn = False
                        else:
                            model_name = "BN"
                            isSaveImageAn = True
                        if updateMethod == "evidence":
                            model_name += "-OL"
                        cross = dest_main + str(num_repeat) + "/" + models_folder + model_name + "/"
                    else:
                        model_name = None
                        cross = dest_main + str(num_repeat) + "/"

                    cross_validation_folder = cross + "train/" + cross_val
                    print "Folder:" + str(cross_validation_folder)
                    fold_folder_set = cross_validation_folder + "folds/"

                    cross_validation_folder_open = cross + "open/" + cross_val
                    fold_folder_set_open = cross_validation_folder_open + "folds/"

                    for normMethod in norm_methods:
                        optim_specs_val = optim_values[(optim_values['Evidence_method'] == str(updateMethod)) & (optim_values['Norm_method'] == str(normMethod))].Optim_params.tolist()[0]
                        optim_specs = ast.literal_eval(optim_specs_val.replace('array(','').replace(')',''))
                        for optimSpec in optim_specs:
                            qualityThreshold = optimSpec[-1]
    
                            weights = optimSpec[:-1]
                            weights.insert(0, faceWeight)
                        print cross_val + ":" + updateMethod + ":" + normMethod

#                         cv.repeatedKFoldCrossValidation(num_people, num_repeat, num_folds, 
#                                  fold_folder_set, fold_folder_set_open, train_folder, test_folder, 
#                                  db_file, init_recog_file, final_recog_file, validation_info_file, analysis_file = analysis_file,
#                                  time_type = time_type, sample_size =sample_size, 
#                                  weights = weights, normMethod = normMethod, updateMethod = updateMethod, model_name = model_name, 
#                                  isSaveRecogFiles = isSaveRecogFiles, isSaveImageAn = isSaveImageAn, 
#                                  cost_function_alpha = 0.9, faceRecogThreshold = None, qualityThreshold = qualityThreshold, probThreshold = None,
#                                  evidence_norm_methods = None, update_partial_params = None, isUpdateFaceLikelihoodsEqually = False)       
                        if isAnalysisIteration and faceWeight == 1.0:
                            for num_fold in range(1, num_folds+1):
                                for set_folder in [train_folder, test_folder]:
                                    for matrix_f in ["Network", "FaceRecognition"]:
                                        conf_file = conf_matrix_file.replace(".csv", matrix_f + ".csv")
                                        plotConfusionMatrix(fold_folder_set, num_fold, set_folder, conf_file)

                print "time for " + cross_val + ":" + str(time.time() -start_time)
