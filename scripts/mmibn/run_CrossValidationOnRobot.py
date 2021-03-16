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
import time

if __name__ == "__main__":

    num_people = 100
    num_folds = 5
    num_bins = num_folds
    num_total_folds = 11
    updateMethod = "evidence"

    cross_val_folders = ["N10_gaussianT/", "Nall_gaussianT/", "N10_uniformT/", "Nall_uniformT/"]
    
    train_folder = "Training/"
    test_folder = "Test/"
    validation_info_file = "validation_info_fold.csv"
    main_folder = "Repeated5fold/"

    RB = RecognitionMemory.RecogniserBN()
    
    # image_path = "/home/nao/dev/src/imdb_recognition/"
    image_path = "/home/nao/dev/src/cross_validation_tests/imdb_new_folds/"
    # image_path = "/home/nao/dev/src/imdb_recognition/cleaned_dataset/" # for testing images for face detection
    RB.connectToRobot("127.0.0.1", useSpanish=False, isImageFromTablet = True, isMemoryOnRobot=True, imagePath = image_path)
    # RB.resetFaceDetectionDB() # for testing images for face detection
    # RB.testImagesForFace(image_path) # for testing images for face detection

    # weights = [1.0, 0.35, 0.15, 0.16, 0.23]
    weights = [1.0, 0.044, 0.538, 0.136, 0.906]
    for setT in ["train"]:
        dest_main = "cross_validation_" + setT + "/"
        for num_f in range(1, num_total_folds):
            cross = dest_main + str(num_f) + "/"
            
            for cross_val in cross_val_folders:
                cross_validation_folder = cross + cross_val
                db_file = cross_validation_folder + "db_data.csv"
                print "Folder:" + str(cross_validation_folder)
                bin_folder = cross_validation_folder + "bins/"
                fold_folder_set = cross_validation_folder + "folds/"

                cross_validation_folder_open = cross.replace("train", "open") + cross_val
                bin_folder_open = cross_validation_folder_open + "bins/"
                fold_folder_set_open = cross_validation_folder_open + "folds/"
                
                for num_fold in range(1, num_folds+1):
         
                    print "Fold:" + str(num_fold)
                    fold_folder = fold_folder_set + str(num_fold) + "/"
                    training_folder = fold_folder + train_folder
                    
                    testing_folder = fold_folder + test_folder
                    start_fold_time = time.time()
                    print "Training set:" + str(training_folder)          
                    # Training
                    db_train, stats_openSet_train, stats_FR_train, num_recog_train, FER_train, num_unknown_train = RB.runCrossValidationOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                         validation_info_file, db_file, isTestData = False, db_order_list = None, isOpenSet = False,
                                         isMultRecognitions = False, num_mult_recognitions = None, isRobotLearning = True, qualityCoefficient = None,
                                         weights = weights, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = updateMethod, probThreshold = None,
                                         isSaveRecogFiles = True, isSaveImageAn = True)
                    train_time_fin = time.time()
                    print "Training time:" + str(train_time_fin - start_fold_time)

                    print "Closed set:" + str(testing_folder)
                    # Closed Set Test
                    db_test, stats_openSet_test, stats_FR_test, num_recog_test, FER_test, num_unknown_test = RB.runCrossValidationOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                         validation_info_file, db_file, isTestData = True, db_order_list = db_train, isOpenSet = False,
                                         isMultRecognitions = False, num_mult_recognitions = None, isRobotLearning = False, qualityCoefficient = None,
                                         weights = weights, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = updateMethod, probThreshold = None,
                                         isSaveRecogFiles = True, isSaveImageAn = True)
                    closed_time_fin = time.time()
                    print "Closed set time:" + str(closed_time_fin - train_time_fin)

                    fold_folder_open = fold_folder_set_open + str(num_fold) + "/"
                    training_folder_open = fold_folder_open + train_folder
                    
                    testing_folder_open = fold_folder_open + test_folder

                    print "Open set train:" + str(training_folder_open)
                    # Open Set Train
                    db_new_3, num_recog_in_fold_test = RB.runCrossValidationOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                         validation_info_file, db_file, isTestData = False, db_order_list = None, isOpenSet = True,
                                         isMultRecognitions = False, num_mult_recognitions = None, isRobotLearning = True, qualityCoefficient = None,
                                         weights = weights, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = updateMethod, probThreshold = None,
                                         isSaveRecogFiles = True, isSaveImageAn = True)
                    open_train_time_fin = time.time()
                    print "Open set training time:" + str(open_train_time_fin - closed_time_fin)

                    print "Open set test:" + str(testing_folder_open)

                    # Open Set Test
                    db_new_4, num_recog_in_fold_test = RB.runCrossValidationOnRobot(num_people, training_folder, testing_folder, bin_folder, 
                                         validation_info_file, db_file, isTestData = True, db_order_list = db_train, isOpenSet = True,
                                         isMultRecognitions = False, num_mult_recognitions = None, isRobotLearning = False, qualityCoefficient = None,
                                         weights = weights, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = updateMethod, probThreshold = None,
                                         isSaveRecogFiles = True, isSaveImageAn = True)
                    open_test_time_fin = time.time()
                    print "Open set test time:" + str(open_test_time_fin - open_train_time_fin)
         
                    print "fold time:" + str(time.time() - start_fold_time)
                    print "-"*40
