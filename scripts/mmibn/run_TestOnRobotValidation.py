# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  run_TestOnRobotValidation script gets the recognition values from the robot (Pepper Naoqi 2.4 or 2.5) #
#  for images in a specified directory. This script works 6 times faster than the                        #
#  run_CrossValidationOnRobot (run times: 6.8 min for Nall dataset for one fold, 9.45 min for N10),      #
#  hence it should be used if only recognition values would like to be obtained for images.              #
#  It also has the functionality to clean images in a path which do not have a face detected by          #
#  ALFaceDetection.                                                                                      #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, N. Lyubova, M. Garcia Ortiz, and T. Belpaeme (2018), 'Multi-modal Open-Set Person         #
#    Identification in HRI', 2018 ACM/IEEE International Conference on Human-Robot Interaction Social    #
#    Robots in the Wild workshop.                                                                        #
#                                                                                                        #
#    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (under review), 'Multi-modal Open World User #
#    Identification', ACM Transactions on Human-Robot Interaction (THRI).                                #
#                                                                                                        #            
#  This script, RecognitionMemory and each script in this project is under the GNU General Public        #
#  License.                                                                                              #
#========================================================================================================#

import RecognitionMemory
import time
import os

if __name__ == "__main__":

    num_people = 100
    num_folds = 5
    num_bins = num_folds
    num_repeats = 11
    num_start_repeat = 1
    updateMethod = "evidence"

    cross_val_folders = ["N10_gaussianT/", "Nall_gaussianT/", "N10_uniformT/", "Nall_uniformT/"]
    
    train_folder = "Training/"
    test_folder = "Test/"
    openvalid_folder = "OpenValidation/"
    validation_info_file = "validation_info_fold.csv"
    #  main_folder = "Repeated5fold/"
    main_folder_list = ["Repeated5fold/", "RepeatedOptim5fold/"]

    RB = RecognitionMemory.RecogniserBN()
    
    # image_path = "/home/nao/dev/src/imdb_recognition/"
    image_path = "/home/nao/dev/src/cross_validation_tests/imdb_new_folds/"
    # image_path = "/home/nao/dev/src/imdb_recognition/cleaned_dataset/" # for testing images for face detection
    RB.connectToRobot("127.0.0.1", useSpanish=False, isImageFromTablet = True, isMemoryOnRobot=True, imagePath = image_path)
    # RB.resetFaceDetectionDB() # for testing images for face detection
    # RB.testImagesForFace(image_path) # for testing images for face detection

    # weights = [1.0, 0.35, 0.15, 0.16, 0.23]
    weights = [1.0, 0.044, 0.538, 0.136, 0.906]
    for main_folder in main_folder_list:
        if main_folder == "Repeated5fold/":
            continue
        for setT in ["train"]:
            dest_main = main_folder + "cross_validation_" + setT + "/"
            for num_repeat in range(num_start_repeat, num_repeats+1):
                cross = dest_main + str(num_repeat) + "/"
 
                for cross_val in cross_val_folders:
                    cross_validation_folder = cross + cross_val
                    print "Folder:" + str(cross_validation_folder)
                    bin_folder = cross_validation_folder + "bins/"
                    fold_folder_set = cross_validation_folder + "folds/"
                    db_file = cross_validation_folder + "db_data.csv"
    
                    cross_validation_folder_open = cross.replace("train", "open") + cross_val
                    bin_folder_open = cross_validation_folder_open + "bins/"
                    fold_folder_set_open = cross_validation_folder_open + "folds/"
                    db_file_open = cross_validation_folder_open + "db_data.csv"
                    
                    for num_fold in range(1, num_folds+1):
                        print "Fold:" + str(num_fold)
    
                        fold_folder = fold_folder_set + str(num_fold) + "/"
                        training_folder = fold_folder + train_folder
                        
                        testing_folder = fold_folder + test_folder

                        start_fold_time = time.time()
                        print "Training set:" + str(training_folder)        
                        # Training
                        db_new, num_recog_in_fold = RB.getRecogValuesForImagesOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                     validation_info_file, db_file,
                                     isTestData = False, db_order_list = None, isOpenSet = False, start_db_size = 0,
                                     isMultRecognitions = False, num_mult_recognitions = None, 
                                     isRobotLearning = True, qualityCoefficient = None, weights = weights, updateMethod = updateMethod,
                                     comparison_file = None, stats_file = None)
                        train_time_fin = time.time()
                        print "Training time:" + str(train_time_fin - start_fold_time)
    
                        print "Closed set:" + str(testing_folder)

                        if os.path.exists(fold_folder + test_folder):
                            testing_folder = fold_folder + test_folder
                            # Closed Set Test
                            db_new_2, num_recog_in_fold_test = RB.getRecogValuesForImagesOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                     validation_info_file, db_file, 
                                     isTestData = True, db_order_list = db_new, isOpenSet = False, start_db_size = 0,
                                     isMultRecognitions = False, num_mult_recognitions = None, 
                                     isRobotLearning = False, qualityCoefficient = None, weights = weights, updateMethod = updateMethod,
                                     comparison_file = None, stats_file = None)
                            closed_time_fin = time.time()
                            print "Closed set time:" + str(closed_time_fin - train_time_fin)

                        else:
                            testing_folder = fold_folder + openvalid_folder
                            # Open Set Validation
                            db_new_2, num_recog_in_fold_test = RB.getRecogValuesForImagesOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                     validation_info_file, db_file, 
                                     isTestData = True, db_order_list = db_new, isOpenSet = True, start_db_size = 0,
                                     isMultRecognitions = False, num_mult_recognitions = None, 
                                     isRobotLearning = True, qualityCoefficient = None, weights = weights, updateMethod = updateMethod,
                                     comparison_file = None, stats_file = None)
                            closed_time_fin = time.time()
                            print "Closed set time:" + str(closed_time_fin - train_time_fin)    

    
                        fold_folder_open = fold_folder_set_open + str(num_fold) + "/"
                        training_folder_open = fold_folder_open + train_folder
                        
                        testing_folder_open = fold_folder_open + test_folder
                        
    
                        print "Open set train:" + str(training_folder_open)
    
                        # Open Set Train
                        db_new_3, num_recog_in_fold_test = RB.getRecogValuesForImagesOnRobot(num_people, training_folder_open, testing_folder_open, bin_folder_open,
                                     validation_info_file, db_file_open, 
                                     isTestData = False, db_order_list = db_new, isOpenSet = True, start_db_size = len(db_new),
                                     isMultRecognitions = False, num_mult_recognitions = None, 
                                     isRobotLearning = True, qualityCoefficient = None, weights = weights, updateMethod = updateMethod,
                                     comparison_file = None, stats_file = None)
                        open_train_time_fin = time.time()
                        print "Open set training time:" + str(open_train_time_fin - closed_time_fin)
    
                        print "Open set test:" + str(testing_folder_open)
                        
                        if os.path.exists(fold_folder_open + test_folder):
                            testing_folder_open = fold_folder_open + test_folder
                            # Open Set - Closed Test
                            db_new_4, num_recog_in_fold_test = RB.getRecogValuesForImagesOnRobot(num_people, training_folder_open, testing_folder_open, bin_folder_open,
                                     validation_info_file, db_file_open,
                                     isTestData = True, db_order_list = db_new_3, isOpenSet = True, start_db_size = len(db_new),
                                     isMultRecognitions = False, num_mult_recognitions = None, 
                                     isRobotLearning = False, qualityCoefficient = None, weights = weights, updateMethod = updateMethod,
                                     comparison_file = None, stats_file = None)
    
                            open_test_time_fin = time.time()
                            print "Open set test time:" + str(open_test_time_fin - open_train_time_fin)
                        else:
                            testing_folder_open = fold_folder_open + openvalid_folder
                            # Open Set Validation
                            db_new_4, num_recog_in_fold_test = RB.getRecogValuesForImagesOnRobot(num_people, training_folder_open, testing_folder_open, bin_folder_open,
                                     validation_info_file, db_file_open,
                                     isTestData = True, db_order_list = db_new_3, isOpenSet = True, start_db_size = len(db_new),
                                     isMultRecognitions = False, num_mult_recognitions = None, 
                                     isRobotLearning = True, qualityCoefficient = None, weights = weights, updateMethod = updateMethod,
                                     comparison_file = None, stats_file = None)
    
                            open_test_time_fin = time.time()
                            print "Open set test time:" + str(open_test_time_fin - open_train_time_fin)                            
                        print "fold time:" + str(time.time() - start_fold_time)
                        print "-"*40



