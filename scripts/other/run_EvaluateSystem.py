# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  run_EvaluateSystem script runs the cross validation for a specified directory, and plots confusion    #
#  matrix.                                                                                               #
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
#  This script, RecognitionMemory and each script in this project is under the GNU General Public        #
#  License v3.0. You should have received a copy of the license along with MultimodalRecognitionDataset. #
#  If not, see <http://www.gnu.org/licenses>.                                                            #  
#========================================================================================================#

import RecognitionMemory
import time
import csv
import ast
import pandas
import numpy as np
import matplotlib.pyplot as plt # for plotting the results

def plotConfusionMatrix(results_folder, conf_matrix_file_set):
    conf_matrix_file = results_folder + conf_matrix_file_set
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
    
    num_people = 100

    update_methods = ["none", "evidence"]
    dest_main = "HRIExperiment/"
    normMethod = "hybrid"
    db_file = "db_data.csv"
    init_recog_file = "InitialRecognition_data.csv"
    final_recog_file = "RecogniserBN_data.csv"
    conf_matrix_file = "confusionMatrix.csv"
    stats_file = "analysis.csv"
    faceWeight = 1.0
    cost_function_alpha = 0.9
    isSaveRecogFiles = True
    isSaveImageAn = False
    models_folder = "Models/"
    optim_file = "optim_params.csv"

    optim_values = pandas.read_csv(dest_main + optim_file, dtype={"Evidence_method": object, "Norm_method": object, "Optim_params":object}, usecols = {"Evidence_method", "Norm_method", "Optim_params"})

    with open(dest_main + stats_file, 'w') as outcsv:
        writer = csv.writer(outcsv)   
        writer.writerow(["Model","FAR","FR_FAR","DIR","FR_DIR","Loss","FR_Loss","Num_recog","Num_enrolled","Total_Time"])

    for updateMethod in update_methods:
        start_time = time.time()
        if updateMethod == "evidence":
	    model_name = "BN-OL"
        else:
	    model_name = "BN"
        results_folder = dest_main + models_folder + model_name + "/"
        db_file_model = results_folder + db_file
        init_recog_file_model = results_folder + init_recog_file
        final_recog_file_model = results_folder + final_recog_file
        optim_specs_val = optim_values[(optim_values['Evidence_method'] == str(updateMethod)) & (optim_values['Norm_method'] == str(normMethod))].Optim_params.tolist()[0]
        optim_specs = ast.literal_eval(optim_specs_val.replace('array(','').replace(')',''))
        for optimSpec in optim_specs:
	    qualityThreshold = optimSpec[-1]
            weights = optimSpec[:-1]
            weights.insert(0, faceWeight)

        RB = RecognitionMemory.RecogniserBN()
        start_time_run = time.time()
        num_recog, FER, stats_openSet, stats_FR, num_unknown = RB.runCrossValidation(num_people, results_folder, None, 
	   None, None, None, None, isTestData = False, isOpenSet = False,
	   weights = weights, faceRecogThreshold = None, qualityThreshold = qualityThreshold, normMethod = normMethod, updateMethod = updateMethod, probThreshold = None,
	   isMultRecognitions = False, num_mult_recognitions = 3, qualityCoefficient = None,
	   db_file = db_file_model, init_recog_file = init_recog_file_model, final_recog_file = final_recog_file_model, valid_info_file = None,
	   isSaveRecogFiles = isSaveRecogFiles, isSaveImageAn = isSaveImageAn)
        time_run = time.time()-start_time_run
        loss = cost_function_alpha*(1.0 - stats_openSet[0]) + (1.0-cost_function_alpha)*(stats_openSet[1])
        F_Loss = cost_function_alpha*(1.0 - stats_FR[0]) + (1.0-cost_function_alpha)*(stats_FR[1])

        with open(dest_main + stats_file, 'a') as outcsv:
            writer = csv.writer(outcsv)   
            writer.writerow([model_name, stats_openSet[1], stats_FR[1], stats_openSet[0], stats_FR[0], loss, F_Loss, num_recog, num_unknown, time_run])      

        for matrix_f in ["Network", "FaceRecognition"]:
            conf_file = conf_matrix_file.replace(".csv", matrix_f + ".csv")
            plotConfusionMatrix(results_folder, conf_file)
