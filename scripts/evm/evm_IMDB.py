# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2018-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  evm_IMDB script runs Extreme Value Machine (Rudd et al., 2018) on the Multi-modal Long-Term User      #
#  Recognition Dataset.                                                                                  #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (under review), 'Multi-modal Incremental     #
#    Bayesian Network with Online Learning for Open World User Identification', ACM Transactions on      #
#    Human-Robot Interaction (THRI).                                                                     #
#                                                                                                        #
#    Ethan M. Rudd, Lalit P. Jain, Walter J. Scheirer and Terrance E. Boult (2018), "The Extreme Value   #
#    Machine" in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 3.         #
#                                                                                                        #
#  evm_IMDB and each script in this project is under the GNU General Public License.                     #
#========================================================================================================#

import RecognitionMemory as RM
import evm
import time
import ast
import pandas
import os
import shutil
import numpy as np
import csv
import matplotlib.pyplot as plt # for plotting the results
import crossValidation as cv

global tailsize

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
    
def getNonweightedProbabilities(RB, recog_results, max_num_people, incl):
    """The evidence (recognition probabilities of F,G,A,H,T) without weights applied"""

    # P(e|F)
    face_result = RB.setFaceProbabilities(recog_results[0], 1.0, isNormalisationOn = False)
    face_result = RB.normalise(face_result, norm_method = "norm-sum")
    # NOTE THAT I NEEDED TO PAD THIS FOR THE METHOD TO WORK!
    face_result.extend([0]*(max_num_people-len(face_result)))

    # P(e|G)
    gender_result = RB.setGenderProbabilities(recog_results[1], 1.0)

    # P(e|A)
    age_result = RB.getCurve(conf = recog_results[2][1], mean = recog_results[2][0], 
                             min_value = RB.age_min, max_value = RB.age_max, 
                             weight = 1.0, norm_method = "tanh")

    # P(e|H)
    height_result = RB.getCurve(conf = recog_results[3][1], mean = recog_results[3][0], 
                                stddev = RB.stddev_height, min_value = RB.height_min, max_value = RB.height_max, 
                                weight = 1.0, norm_method = "norm-sum")
    
    # P(e|T)   
    time_result = RB.getCurve(mean = RB.getTimeSlot(recog_results[4]), stddev = RB.stddev_time, 
                              min_value = RB.time_min, max_value = RB.time_max, 
                              weight = 1.0, norm_method = "softmax")

    #result = face_result + gender_result + age_result + height_result
    # result = face_result + gender_result + age_result + height_result + time_result
    if incl == "face": # using only face information
        result = face_result
    elif incl == "soft": # using only soft biometrics
        result = gender_result + age_result + height_result + time_result
    else: # using all info
        result = face_result + gender_result + age_result + height_result + time_result
    return result


    
def recogniseEVM(RB, Xtrain, weibulls, ytrain, recog_results, open_set_threshold, num_people, max_num_people, incl):
    """
    Recognise the user using the network:
    (2) Get recognition results from modalities
    (3) Set evidence
    (4) Estimate identity using EVM
    (5) Estimate identity using face recognition (for comparison)
    isRegistered = False if register button is pressed"""
    r_time_t = time.time()
    RB.nonweighted_evidence = recog_results
    if not recog_results:
        print "No face detected in the image"
        return ""
    Xtest = getNonweightedProbabilities(RB, recog_results, max_num_people, incl) # (3)
    Xtest_np = np.array(Xtest)
    Xtest_np = Xtest_np.reshape(1, -1) 
    if num_people > tailsize + 2:
        prediction,probs = evm.predict(Xtest_np,Xtrain,weibulls,ytrain) # (4)
        if max(probs) < open_set_threshold:
            prediction = ['0']
            
        face_est, face_prob = RB.getFaceRecogEstimate() # (5) 
    else:
        face_est, face_prob = RB.getFaceRecogEstimate() # (5)
        prediction = face_est
        probs = face_prob
    if RB.isDebugMode:
        print "time for recognise: " + str(time.time() - r_time_t)
    return prediction[0], probs, face_est, face_prob, Xtest, Xtest_np

def learnFromFileEVM(RB, max_num_people, num_people = 0, open_set_threshold = 0.3, incl = "all", 
                        isTestData = False, isOpenSet = False,
                        Xtrain_np = None, weibulls = None, ytrain_np = None,
                        db_list=None, init_list=None, recogs_list=None,
                        db_file = None, init_recog_file = None, final_recog_file = None, 
                        valid_info_file = None, isSaveImageAn = False, orig_image_dir = None):
    """Creates the network from files/lists. The evidence is fed from existing file or information one by one."""
    learn_start_time = time.time() 

    if db_list is None and os.path.isfile(db_file):
        df_db = pandas.read_csv(db_file, dtype={"id": object}, converters={"times": ast.literal_eval}, usecols = ["id","name","gender","age","height","times"])
        db_list = df_db.values.tolist()

    if init_list is None and os.path.isfile(init_recog_file):
        df_init = pandas.read_csv(init_recog_file, dtype={"I_est": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
        init_list = df_init.values.tolist()
    
    if recogs_list is None and os.path.isfile(final_recog_file):
        df_final = pandas.read_csv(final_recog_file, dtype={"I": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
        recogs_list = df_final.values.tolist()
        
    if isSaveImageAn:
        df_info = pandas.read_csv(valid_info_file, usecols ={"Original_image", "Validation_image"}).values.tolist()          

    num_unknown = 0
    stats_openSet = [0,0] #DIR, FAR
    stats_FR = [0,0]
    numNoFaceImages = 0
    count_recogs = 0
    num_recog = 0
    RB.isMemoryOnRobot = True
    Xtrain = []
    ytrain = []
    Xtest = []
    batch_init_data = []
    batch_recog_data = []
    init_recog_timer = 0.0
    if weibulls is None:
        weibulls = np.empty(shape=[0, 2])
    if Xtrain_np is None:
        Xtrain_np = np.empty(shape=[0, 2])
    if ytrain_np is None:
        ytrain_np = np.empty(shape=[0, 2])
    if len(RB.i_labels) == 0:
        RB.i_labels.append(RB.unknown_var)
    while count_recogs < len(recogs_list):
        idPerson = str(recogs_list[count_recogs][0])
        isRegistered = not recogs_list[count_recogs][6]
        isAddPersonToDB = recogs_list[count_recogs][6]
        numRecognition = recogs_list[count_recogs][7]
        person = []
        recog_results = []
        if isAddPersonToDB:    
            person = [x for x in db_list if str(x[0]) == idPerson][0]
            person[0] = str(person[0])
            person[1] = str(person[1])
            isRegistered = False
            
        RB.setSessionVar(isRegistered = isRegistered, isAddPersonToDB = isAddPersonToDB, personToAdd = person)    
            
        if isRegistered:
            if RB.isMultipleRecognitions:
                recog_values = [x for x in recogs_list if x[7] == numRecognition]
                num_mult_recognitions = len(recog_values)
                RB.setDefinedNumMultRecognitions(num_mult_recognitions)
                for num_rec in range(0, num_mult_recognitions):
                    recog_results.append(recog_values[num_rec][1:6])
                    if num_rec < num_mult_recognitions - 1:
                        count_recogs += 1
            else:
                recog_results = recogs_list[count_recogs][1:6]
        else:
            if RB.isMultipleRecognitions:
                init_recog_values = [x for x in init_list if x[6] == numRecognition]
                num_mult_recognitions = len(init_recog_values)
                RB.setDefinedNumMultRecognitions(num_mult_recognitions)
                for num_rec in range(0, num_mult_recognitions):
                    recog_results.append(init_recog_values[num_rec][1:6])
            else:
                init_recog_values = [x for x in init_list if x[6] == numRecognition][0]
                recog_results = init_recog_values[1:6]
        
        start_recog_timer = time.time()                                                
        identity_est, probs, face_est, face_prob, Xtest, Xtest_np = recogniseEVM(RB, Xtrain_np, weibulls, ytrain_np, recog_results, open_set_threshold, num_people, max_num_people, incl) # get the estimated identity from the recognition network
        init_recog_timer += time.time() - start_recog_timer
        if isRegistered:
            batch_init_data.append([idPerson] + Xtest)
        else:
            batch_init_data.append([RB.unknown_var] + Xtest)
        RB.saveInitialRecognitionCSV(RB.initial_recognition_file, recog_results, identity_est)
        
        if identity_est == "":
            numNoFaceImages += 1
            continue
        p_id = None
        
        stats_openSet = RB.getPerformanceMetrics(identity_est, idPerson, RB.unknown_var, isRegistered, stats_openSet)
        
        stats_FR = RB.getPerformanceMetrics(face_est, idPerson, RB.unknown_var, isRegistered, stats_FR)
        
        isRecognitionCorrect = False

        if isRegistered and identity_est != RB.unknown_var and identity_est == idPerson:
            isRecognitionCorrect = True # True if the name is confirmed by the user
                    
        if isSaveImageAn:
            copy_dir = ""
            if isRegistered:
                if isRecognitionCorrect:
                    copy_dir = "Known_True/"
                elif identity_est == RB.unknown_var:
                    copy_dir = "Known_Unknown/"
                else:
                    copy_dir = "Known_False/"
            else:
                if identity_est == RB.unknown_var:
                    copy_dir = "Unknown_True/"
                else:
                    copy_dir = "Unknown_False/"
            
            orig_image = str(df_info[num_recog][0])

            valid_image = df_info[num_recog][-1]
            shutil.copy2(orig_image,RB.image_save_dir + copy_dir + str(num_recog+1) + "_" + valid_image + ".jpg")
                                    
        if isRecognitionCorrect:
            Xtrain_np,weibulls,ytrain_np = evm.update_iter_single(Xtest_np,np.array([idPerson]),Xtrain_np,weibulls,ytrain_np)
        else:
            if isAddPersonToDB:
                p_id = person[0]
                num_unknown += 1
                recog_results = []
                if RB.isMultipleRecognitions:
                    recog_values = [x for x in recogs_list if x[7] == numRecognition]
                    num_mult_recognitions = len(recog_values)
                    RB.setDefinedNumMultRecognitions(num_mult_recognitions)
                    for num_recog in range(0, num_mult_recognitions):
                        recog_results.append(recog_values[num_recog][1:6])
                        if num_recog < num_mult_recognitions - 1:
                            count_recogs += 1
                else:
                    recog_results = recogs_list[count_recogs][1:6]
                num_people += 1
                RB.i_labels.append(p_id)
                identity_est_2, probs_2, face_est_2, face_prob_2, Xtest, Xtest_np = recogniseEVM(RB, Xtrain_np, weibulls, ytrain_np, recog_results, open_set_threshold, num_people, max_num_people, incl)

            else:
                p_id = idPerson 
            
            if not isTestData:
                if isOpenSet:
                    Xtrain_np,weibulls,ytrain_np = evm.update_iter_single(Xtest_np,np.array([idPerson]),Xtrain_np,weibulls,ytrain_np)
                else:
                    if num_recog < tailsize + 1:
                        Xtrain.append(Xtest)
                        ytrain.append(idPerson)
                    elif num_recog == tailsize + 1:
                        Xtrain.append(Xtest)
                        ytrain.append(idPerson)
                        Xtrain_np = np.array(Xtrain)
                        ytrain_np = np.array(ytrain)
                        weibulls = evm.fit(Xtrain_np, ytrain_np)
                        Xtrain_np,weibulls,ytrain_np = evm.reduce_model(Xtrain_np,weibulls,ytrain_np)
                    else:
                        Xtrain_np,weibulls,ytrain_np = evm.update_iter_single(Xtest_np,np.array([idPerson]),Xtrain_np,weibulls,ytrain_np)
        
        batch_recog_data.append([idPerson] + Xtest)
        RB.saveRecogniserCSV(RB.recogniser_csv_file, idPerson, num_recog=None)
        end_recog_timer = time.time()
        RB.saveComparisonCSV(RB.comparison_file, idPerson, identity_est, face_est, probs, face_prob, end_recog_timer - start_recog_timer, -1)

        num_recog += 1
        count_recogs += 1
        RB.num_recognitions += 1
    RB.num_people = num_people +1
    init_recog_timer /= num_recog
    if RB.isDebugMode:
        print "time to learn:" + str(time.time() - learn_start_time) 
    return stats_openSet, stats_FR, num_recog, numNoFaceImages/(num_recog+numNoFaceImages), num_unknown, num_people, Xtrain_np, ytrain_np, weibulls, batch_init_data, batch_recog_data, init_recog_timer

def runCrossValidationEVM(RB, training_folder, test_folder, max_num_people, num_people = 0, open_set_threshold = 0.3, incl = "all",
                          db_list=None, init_list=None, recogs_list=None, 
                           isTestData = False, isOpenSet = False, Xtrain_np = None, weibulls = None, ytrain_np = None,
                           faceRecogThreshold = None, 
                           isMultRecognitions = False, num_mult_recognitions = None,
                           db_file = None, init_recog_file = None, final_recog_file = None, valid_info_file = None, isSaveRecogFiles = True, isSaveImageAn = True):
    
    """Run cross validation offline from recognition files or recognition lists"""
    
    start_time_run = time.time()
    
    """BEGIN: set params"""
    
    
    RB.isSaveRecogFiles = isSaveRecogFiles
    
    recog_folder = ""
    
    if isTestData:
        recog_folder = test_folder
        if RB.isSaveRecogFiles:
            RB.resetFilePaths()
            RB.setFilePaths(recog_folder)
            RB.resetFiles()
            RB.resetFilePaths()
            RB.setFilePaths(recog_folder)
    else:
        recog_folder = training_folder
        if RB.isSaveRecogFiles:
            RB.resetFilePaths()
            RB.setFilePaths(recog_folder)
            RB.resetFiles()
      
    if faceRecogThreshold is not None:
        RB.setFaceRecognitionThreshold(faceRecogThreshold)

    """END: set params"""

    stats_openSet, stats_FR, num_recog, FER, num_unknown, num_people, Xtrain_np, ytrain_np, weibulls, batch_init_data, batch_recog_data, init_recog_timer = learnFromFileEVM(RB, max_num_people, num_people, open_set_threshold, incl,
isTestData = isTestData, isOpenSet = isOpenSet, Xtrain_np = Xtrain_np, weibulls = weibulls, ytrain_np = ytrain_np, db_list=db_list, init_list=init_list, recogs_list=recogs_list,
                                                                            db_file = db_file, init_recog_file = init_recog_file, final_recog_file = final_recog_file, 
                                                                            valid_info_file = valid_info_file, 
                                                                            isSaveImageAn = isSaveImageAn, orig_image_dir = os.path.abspath(os.path.join(recog_folder,"../../../bins")) + "/")

    stats_openSet_percent = RB.getDIRFAR(stats_openSet, num_recog, num_unknown)
    stats_FR_percent = RB.getDIRFAR(stats_FR, num_recog, num_unknown)
    RB.saveConfusionMatrix()    
    print "time to run: " + str(time.time()-start_time_run)
    return num_recog, FER, stats_openSet_percent, stats_FR_percent, num_unknown, num_people, Xtrain_np, ytrain_np, weibulls, batch_init_data, batch_recog_data, init_recog_timer

if __name__ == "__main__":

    db_file = "db_data.csv"
    init_recog_file = "InitialRecognition_data.csv"
    final_recog_file = "RecogniserBN_data.csv"
    conf_matrix_file = "confusionMatrixNetwork.csv"
    stats_file = "analysis.csv"
    cost_function_alpha = 0.9
    isSaveRecogFiles = True
    isSaveImageAn = False
    training_folder = "Training/"
    test_folder = "Test/"
    dest_main = "EVM_tests/"
    valid_info_file = "validation_info_fold.csv"
    batch_init_file = "batch_init.txt"
    batch_recog_file = "batch_recog.txt"
    max_num_people = 201
    tailsize = 3 # Tao in the article (first 5 recognitions should be recognised as unknown (fair comparison with MMIBN, and EVM at least needs 4))
    cover_threshold = 1.0 # sigma in the article
    num_to_fuse = 1 # k in the article
    distance = "cosine" # or "euclidean" in the article, cosine is said to be better
    evm.update_params_small(tailsize, cover_threshold, num_to_fuse, distance)
    open_set_threshold = 0.3 # if the highest probability below this value, the prediction should be unknown (set this for decreasing FAR)
    increment_threshold = np.arange(0.05, 0.35, 0.05)
    dataset_folder = "Nall_gaussianT/"
    print "dataset_folder:" + dataset_folder
    num_folds = 5
    num_start_fold = 1
    num_end_fold = num_start_fold
    cov_threshold = 1.0
    stats_file = stats_file.replace(".csv", "_" + str(cov_threshold)+ ".csv")
    for opt_result in ["EVM_all_hybrid/"]:
        incl = opt_result.split("_")[1]
        incl = incl.replace("/", "")
        print incl
        cross_val_folder = dest_main + opt_result + "train/" + dataset_folder + "folds/"
        cross_val_folder_open = dest_main + opt_result + "open/" + dataset_folder + "folds/"
        # with open(dest_main + opt_result + stats_file, 'wb') as outcsv:
        #    writer = csv.writer(outcsv)   
        #    writer.writerow(["Cover_threshold", "Open_set_threshold", "Dataset_folder", "Evaluation set", "Num_fold", "EVM_FAR", "FR_FAR", "EVM_DIR", "FR_DIR", "EVM_Loss", "FR_Loss", "Num_recog", "Num_enrolled", "Avg_recognition_time"])   
        
        for cover_threshold in [cov_threshold]:
            evm.update_params_small(tailsize, cover_threshold, num_to_fuse, distance)
            print "cover_threshold:" + str(cover_threshold)
            # for open_set_threshold in increment_threshold:
            for open_set_threshold in [0.05]:      
                print "open_set_threshold:" + str(open_set_threshold)
                for num_fold in range(num_start_fold, num_folds+1):
                    print "num_fold:" + str(num_fold)
                    RB = RM.RecogniserBN()
                    for eval_folder in ["training", "closed-test", "open", "open-closed"]:
                        print "eval_folder:" + str(eval_folder)
                        training_folder_fold = cross_val_folder + str(num_fold) + "/" + training_folder
                        test_folder_fold = cross_val_folder + str(num_fold) + "/" + test_folder
                        if eval_folder == "training":
                            isTestData = False
                            eval_folder_fold = training_folder_fold
                        elif eval_folder == "closed-test":
                            isTestData = True
                            eval_folder_fold = test_folder_fold
                        elif eval_folder == "open":
                            isTestData = False
                            training_folder_fold = cross_val_folder_open + str(num_fold) + "/" + training_folder
                            test_folder_fold = cross_val_folder_open + str(num_fold) + "/" + test_folder
                            eval_folder_fold = training_folder_fold
                        else:
                            isTestData = True
                            training_folder_fold = cross_val_folder_open + str(num_fold) + "/" + training_folder
                            test_folder_fold = cross_val_folder_open + str(num_fold) + "/" + test_folder
                            eval_folder_fold = test_folder_fold
                
                        db_file_model = eval_folder_fold + db_file
                        init_recog_file_model = eval_folder_fold + init_recog_file
                        final_recog_file_model = eval_folder_fold + final_recog_file
                        valid_info_file_model = eval_folder_fold + valid_info_file
                        batch_init_file_model = eval_folder_fold + batch_init_file
                        batch_recog_file_model = eval_folder_fold + batch_recog_file

                        isOpenSet = False
                        isTestData = False
                        if eval_folder == "training":                   
                            Xtrain_np = None
                            ytrain_np = None
                            weibulls = None
                            num_people = 0 
                        elif eval_folder == "open":
                            isOpenSet = True
                        elif eval_folder == "open-closed":
                            isOpenSet = True
                            isTestData = True
                        else:
                            isTestData = True
                
                        num_recog, FER, stats_openSet, stats_FR, num_unknown, num_people, Xtrain_np, ytrain_np, weibulls, batch_init_data, batch_recog_data, init_recog_timer = runCrossValidationEVM(RB, training_folder_fold, test_folder_fold, max_num_people, num_people,
                                           open_set_threshold = open_set_threshold, incl = incl,
                                           db_list=None, init_list=None, recogs_list=None, isTestData = isTestData, isOpenSet = isOpenSet,
                                           Xtrain_np = Xtrain_np, weibulls = weibulls, ytrain_np = ytrain_np,
                                           faceRecogThreshold = None, 
                                           isMultRecognitions = False, num_mult_recognitions = None,
                                           db_file = db_file_model, init_recog_file = init_recog_file_model, final_recog_file = final_recog_file_model, valid_info_file = valid_info_file_model, isSaveRecogFiles = True, isSaveImageAn = False)
                      
                        loss = cost_function_alpha*(1.0 - stats_openSet[0]) + (1.0-cost_function_alpha)*(stats_openSet[1])
                        F_Loss = cost_function_alpha*(1.0 - stats_FR[0]) + (1.0-cost_function_alpha)*(stats_FR[1])
                
                        with open(dest_main + opt_result + stats_file, 'a') as outcsv:
                            writer = csv.writer(outcsv)   
                            writer.writerow([cover_threshold, open_set_threshold, dataset_folder, eval_folder, num_fold, stats_openSet[1], stats_FR[1], stats_openSet[0], stats_FR[0], loss, F_Loss, num_recog, num_unknown, init_recog_timer])

 
                        with open(batch_init_file_model, 'wb') as binit:
                            for bid in batch_init_data:
                                str_to_write = ','.join(str(e) for e in bid)
                                str_to_write += "\n"
                                binit.write(str_to_write)

                        with open(batch_recog_file_model, 'wb') as brecog:
                            for brd in batch_recog_data:
                                str_to_write = ','.join(str(e) for e in brd)
                                str_to_write += "\n"
                                brecog.write(str_to_write)
                        
                        plotConfusionMatrix(eval_folder_fold, conf_matrix_file)
                                        
    # TODO: copy Trainall for saving pictures
    # TODO: need to Bayesian optimize parameters using long-term recognition loss: tailsize, cover_threshold, open_set_threshold, margin_threshold for the combination of all parameters vs. only face
    # in the paper, it states that num_to_fuse (k > 1) does not increase the accuracy a lot (at most 1-2%), and I need 
    # TODO: check performance of batch recognition by saving all the 'evidence' values into a file
    # TODO: check other EVM code
    

