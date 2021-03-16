# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2018-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  imdb_prepareImages script creates the cross validation set for evaluation of MMIBN in                 #
#  RecogniserMemory using crossValidation functions and artificialDataset for creating artificial        #
#  estimates of height and time of interaction, which are missing from the IMDB dataset*. The images are #
#  previously cleaned by NAOqi face detection and manually for removing images without a face detecting. #
#  See the MATLAB code imdb_face_crossval_extraction.m, for details of choosing the images in the        #
#  dataset.                                                                                              # 
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
#  * Face cropped images of IMDB dataset in IMDB-Wiki dataset are used for this purpose:                 #
#    R. Rothe and R. Timofte and L. Van Gool (2016), 'Deep expectation of real and apparent age from a   #
#    single image without facial landmarks', International Journal of Computer Vision (IJCV).            #
#                                                                                                        #
#    R. Rothe, R. Timofte and L. Van Gool (2018), 'Deep expectation of real and apparent age from a      #
#    single image without facial landmarks', International Journal of Computer Vision, vol. 126, no. 2-4.#
#                                                                                                        #
#  imdb_prepareImages, RecognitionMemory and each script in this project is under the GNU General Public #
#  License.                                                                                              #
#========================================================================================================#

from PIL import Image
import shutil
import pandas
import os
import crossValidation as cv
import artificialDataset as ad
import numpy as np
import csv
import WeightOptimisation as wop               
import ast
import copy
import collections

random_state = np.random.RandomState(1234567890)

def getMinResImagesAll(src_folder, target_folder, min_width, min_height, num_get_last=None):
    subdirs = [x[0] for x in os.walk(src_folder)]
    for subdir in subdirs[1:]:
        files = os.walk(subdir).next()[2]
        if len(files) > 0:
            if not os.path.exists(target_folder + subdir.split('/')[-1]):
                os.makedirs(target_folder + subdir.split('/')[-1])
            for file_name in files:
                im = Image.open(subdir+"/"+file_name)
                width, height = im.size
                if width >= min_width and height >= min_height:
                    # write image to folder
                    new_image = target_folder + subdir.split('/')[-1] +"/"+ file_name
                    shutil.copy2(subdir+"/"+file_name, new_image)
                                
def getMinResImages(src_folder, min_width, min_height, info_file, num_get_last=None):
    if not os.path.exists(src_folder):
        os.makedirs(src_folder)
    df_images = pandas.read_csv(info_file)
    df_images_list = df_images.values.tolist()
    if num_get_last is not None:
        images_list = df_images_list[-1*num_get_last:]
    else:
        images_list = df_images_list
    for row in images_list:
        folder_name = str(row[0]) + '/'
        target_dir = main_folder + folder_name
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for col in row[4:]:
            im = Image.open(col)
            width, height = im.size
            if width >= min_width and height >= min_height:
            # write image to folder
                new_image = target_dir + col.split('/')[-1]
                shutil.copy2(col, new_image)

def renameImagesToSequential(src_folder, dest_folder, num_samples=None, previous_sampled=None, previous_sub_dirs_order=None, isSameSetToBeUsed=False):
    image_info_sequential = [] # [[[Original_image_1, Sequential_image_1],[Original_image_2, Sequential_image_2]...] for num_person in range(1, num_people+1)]
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)
    subdirs = [x[0] for x in os.walk(src_folder)]
    id_counter = 1
    orig_ids = []
    num_samples_list = []
    subdir_counter = 0
    sub_dirs = subdirs[1:]
    if isSameSetToBeUsed:
        sub_dirs = copy.deepcopy(previous_sub_dirs_order)
    else:
        random_state.shuffle(sub_dirs) # ADDED TO INCREASE RANDOMNESS
        
    current_sampled = []
    if previous_sampled:
        current_sampled = copy.deepcopy(previous_sampled)
    for subdir in sub_dirs:
        orig_id_repeated = [] #to have the same format as crossValidation function
        files = os.walk(subdir).next()[2]
        num_seq = 1
        orig_id = int(subdir.split('/')[-1])
        if len(files) > 0:
            image_person_info = []
            if previous_sampled:
                # this is used for creating a cross validation set with the same images in each fold if the number of samples is determined
                for counter_x in range(0, len(current_sampled)):
                    x = current_sampled[counter_x]
                    if x[0] == orig_id:
                        sampled = x[1]
                        if not isSameSetToBeUsed:
                            # if a previous set has been chosen, but the set needs to be shuffled, update the current order in current_sampled
                            random_state.shuffle(sampled) # ADDED TO INCREASE RANDOMNESS
                            current_sampled[counter_x][:] = [orig_id, sampled]
                        break
            elif num_samples is None:
                # all images are chosen
                sampled = copy.deepcopy(files)
                random_state.shuffle(sampled) # ADDED TO INCREASE RANDOMNESS
                current_sampled.append([orig_id, sampled])
            else:
                # if it is the first the samples are chosen
                # sampled = random_state.choice(files, size=num_samples) #NOTE: THE CODE USED TO GENERATE IMDB DATASET USED IN THE EXPERIMENTS HAD REPLACE=TRUE
                sampled = random_state.choice(files, size=num_samples, replace=False)
                random_state.shuffle(sampled) # ADDED TO INCREASE RANDOMNESS
                current_sampled.append([orig_id, sampled])
    
            for file_name in sampled:
                orig_id_repeated.append(orig_id)
                new_image = dest_folder + str(id_counter) + "_" + str(num_seq) + ".jpg"
                shutil.copy(subdir + "/" + file_name, new_image) # DON'T USE COPY2 BECAUSE IT TRANSFERS THE METADATA OF THE FILE (TRANSFERRING THE THUMBNAIL OF IT WHICH LOOKS WRONG!)
                image_person_info.append([id_counter, subdir + "/" + file_name, new_image])
                num_seq += 1
            num_samples_list.append(num_seq-1)
            image_info_sequential.append(image_person_info)
            id_counter += 1
            orig_ids.append(orig_id_repeated)
        subdir_counter += 1
    
    return image_info_sequential, orig_ids, num_samples_list, current_sampled, sub_dirs


def shuffleImagesInDirectory(src_folder, dest_folder):
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)
    subdirs = [x[0] for x in os.walk(src_folder)]
    for subdir in subdirs[1:]:
        files = os.walk(subdir).next()[2]
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder + subdir)
        if len(files) > 0:
            shuffled_files = random_state.shuffle(files)
            counter_file = 0
            for file_name in files:
                shuffled_image = dest_folder + sub_dir + "/" +  shuffled_files[counter_file]
                shutil.copy2(subdir + "/" + file_name, shuffled_image)
                counter_file += 1

def separateImagesIntoFolders(src_folder, dest_folder, num_people, validation_info_file):
    df_info = pandas.read_csv(validation_info_file, usecols = ["Identity","Original_image"]).values.tolist()
    id_folder = []
    for num_person in range(0, num_people):
        orig_id = df_info[num_person][1].split("/")[2]
        saved_id = df_info[num_person][0]
        id_folder.append([orig_id, saved_id])
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    files = os.walk(src_folder).next()[2]
    for id_fold in id_folder:
        sub_dir = dest_folder + id_fold[0] + "/"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)    
        id_files = [i for i in files if i.startswith(str(id_fold[1]) + "_")]
        for fil in id_files:
            shutil.copy(src_folder + fil, sub_dir)
    
def getTimeInfo(num_people, num_samples, time_specs):
    
    clean_db, db_param_list = ad.getCleanDB(num_people, num_samples, ["T"], time_specs) #artificiaDataset function
    
    return db_param_list[0]

def getHeightInfo(num_people, num_samples, orig_ids, height_file, height_noise_specs):
    df_heights = pandas.read_csv(height_file).values.tolist()
    ids_h = [x[0] for x in df_heights]
    height_list = []
    orig_heights = []
    for orig_id_repeated in orig_ids:
        height_id = float(df_heights[ids_h.index(orig_id_repeated[0])][2])
        orig_heights.append(height_id)
        #to have the same format as crossValidation function
        height_list.append([[[float("{0:.1f}".format(height_id)), 0.08] for _ in range(0, len(orig_id_repeated))]])
    noisy_db, noise_list = ad.getNoisyDB(num_people, num_samples, ["H"], height_noise_specs, height_list)
    noisy_heights = [x[0] for x in noisy_db]
    return orig_heights, noisy_heights

def getRecogInfo(num_people, num_samples, orig_ids, height_file, height_noise_specs, time_specs):
    orig_heights, noisy_heights = getHeightInfo(num_people, num_samples, orig_ids, height_file, height_noise_specs)
    times_list = getTimeInfo(num_people, num_samples, time_specs)
    return orig_heights, noisy_heights, times_list

def saveDBFile(num_people, orig_ids, info_file, orig_heights, time_list, db_file, db_headers):
    df_info = pandas.read_csv(info_file, usecols = ["Id","Name","Gender","Age"]).values.tolist()
    id_list = [x[0] for x in df_info]
    counter = 1
    if os.path.isfile(db_file):
        os.remove(db_file)
    with open(db_file, 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(db_headers)
        for orig_id_repeated in orig_ids:
            ind = id_list.index(orig_id_repeated[0])
            height = orig_heights[counter-1]
            time_init = time_list[counter-1][0]
            row = [counter, df_info[ind][1], df_info[ind][2], df_info[ind][3], height , time_init, [0,0,0]]
            writer.writerow(row)
            counter += 1

def shuffleDataset(num_people, num_folds, sorted_validation_info, sorted_test_info, folds_folder, training_folder, test_folder, validation_info_file, valid_headers, 
                  prev_shuffle_order = [], prev_fold_bin_order_list = []):
    # combine dataset
    full_info = copy.deepcopy(sorted_validation_info) # validation information file of one fold only
    full_info.extend(sorted_test_info) # test information file of one fold only
    
    # shuffle the order
    if prev_shuffle_order: 
        # use previous shuffle order for creating same order in Gaussian and Uniform times
        shuffle_order = copy.deepcopy(prev_shuffle_order)
    else:
        shuffle_order = range(1, len(full_info)+1)
        random_state.shuffle(shuffle_order)

    full_info_new = []
    for num_valid in shuffle_order:
        full_info_new.append(full_info[num_valid-1][:])
    
    # divide to "bins"
    validation_bin_list = [ full_info_new[i::num_folds] for i in xrange(num_folds) ]
    
    if prev_fold_bin_order_list:
        # use same split for creating same order in Gaussian and Uniform times
        fold_bin_order_list = copy.deepcopy(prev_fold_bin_order_list)
    else:
        fold_bin_order_list = cv.getBinOrderInFold(num_folds, num_folds)
    fold_counter = 1
    for fold_bin_order in fold_bin_order_list:
        validation_info_new = []
        # get training set from first num_folds-1 bins
        for fold_bin in fold_bin_order[:-1]:
            validation_info_new.extend(validation_bin_list[fold_bin-1])
        
        # get test set from the last bin
        test_info_new = validation_bin_list[fold_bin_order[-1]-1]
        # correct the N_validation numbers to sequential
        for count_v in range(1, len(validation_info_new)+1):
            validation_info_new[count_v-1][0] = count_v
        for count_v in range(1, len(test_info_new)+1):
            test_info_new[count_v-1][0] = count_v

        # correct the validation images
        new_id_fold = [0 for _ in range(0, num_people)]
        valid_counter = 0
        id_counter = 1

        for valid_info in [validation_info_new, test_info_new]:
            valid_ids_list = [x[1] for x in valid_info]
            for valid_id in valid_ids_list:
                if new_id_fold[valid_id-1] == 0:
                    new_id_fold[valid_id-1] = id_counter
                    id_counter += 1
        for valid_info in [validation_info_new, test_info_new]:
            valid_ids_list = [x[1] for x in valid_info]
            for num_person in range(1, num_people+1):
                appearance_im_list = [i+1 for i, x in enumerate(valid_ids_list) if x == num_person]
                
                if appearance_im_list:
                    im_counter = 1
                    for appearance_im in appearance_im_list:
                        valid_info[appearance_im-1][6] = str(new_id_fold[num_person-1]) + "_" + str(im_counter) # Validation_image
                        im_counter +=1 
        val_train_file_dir = folds_folder + str(fold_counter) + "/" + training_folder
        if os.path.exists(val_train_file_dir):
            shutil.rmtree(val_train_file_dir)
        os.makedirs(val_train_file_dir)
        with open(val_train_file_dir + validation_info_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(valid_headers)
            for ite in validation_info_new:
                writer = csv.writer(outcsv)
                writer.writerow(ite)
        val_test_file_dir = folds_folder + str(fold_counter) + "/" + test_folder
        if os.path.exists(val_test_file_dir):
            shutil.rmtree(val_test_file_dir)
        os.makedirs(val_test_file_dir)
        with open(val_test_file_dir + validation_info_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(valid_headers)
            for ite in test_info_new:
                writer = csv.writer(outcsv)
                writer.writerow(ite)
        fold_counter += 1
    return fold_bin_order_list, validation_info_new, test_info_new, shuffle_order

def divideIntoTrainingAndOpenSetForOptimisation(num_people, num_folds, sorted_validation_info, sorted_test_info, folds_folder, training_folder, test_folder, validation_info_file, valid_headers, prev_shuffle_people_order = [], prev_shuffle_order_train_list = [], prev_shuffle_order_test_list = [], prev_fold_bin_order_list = []):
    test_folder = "OpenValidation/"
    shuffle_order_train_list = []
    shuffle_order_test_list = []
    # combine dataset
    full_info = copy.deepcopy(sorted_validation_info) # validation information file of one fold only
    full_info.extend(sorted_test_info) # test information file of one fold only
    
    # shuffle the order
    if prev_shuffle_people_order: 
        # use previous shuffle order for creating same order in Gaussian and Uniform times
        shuffle_people_order = copy.deepcopy(prev_shuffle_people_order)
    else:
        shuffle_people_order = range(1, num_people +1)
        random_state.shuffle(shuffle_people_order)
    people_order_bins = [ shuffle_people_order[i::num_folds] for i in xrange(num_folds) ]

    # put all images of people in same "bins"
    full_info_new = []
    validation_bin_list = []
    valid_ids_list = [x[1] for x in full_info]
    for people_order in people_order_bins:
        validation_bin = []
        for num_person in people_order:
            appearance_im_list = [i+1 for i, x in enumerate(valid_ids_list) if x == num_person]
            for appearance in appearance_im_list:
                validation_bin.append(full_info[appearance-1][:] )
            
        validation_bin_list.append(validation_bin)

    if prev_fold_bin_order_list:
        # use same split for creating same order in Gaussian and Uniform times
        fold_bin_order_list = copy.deepcopy(prev_fold_bin_order_list)
    else:
        fold_bin_order_list = cv.getBinOrderInFold(num_folds, num_folds)

    fold_counter = 1
    for fold_bin_order in fold_bin_order_list:
        validation_info_med = []
        # get training set from first num_folds-1 bins
        for fold_bin in fold_bin_order[:-1]:
            validation_info_med.extend(validation_bin_list[fold_bin-1])
        # get test set from the last bin
        test_info_med = validation_bin_list[fold_bin_order[-1]-1]
        if prev_shuffle_order_train_list:
            # use same shuffle for creating same order in Gaussian and Uniform times
            shuffle_order_train = copy.deepcopy(prev_shuffle_order_train_list[fold_counter-1])
        else:
            shuffle_order_train = range(1, len(validation_info_med)+1)
            random_state.shuffle(shuffle_order_train)
        shuffle_order_train_list.append(shuffle_order_train)

        validation_info_new = []
        for num_valid in shuffle_order_train:
            validation_info_new.append(validation_info_med[num_valid-1][:])

        if prev_shuffle_order_test_list:
            # use same shuffle for creating same order in Gaussian and Uniform times
            shuffle_order_test = copy.deepcopy(prev_shuffle_order_test_list[fold_counter-1])
        else:
            shuffle_order_test = range(1, len(test_info_med)+1)
            random_state.shuffle(shuffle_order_test)
        shuffle_order_test_list.append(shuffle_order_test)

        test_info_new = []
        for num_valid in shuffle_order_test:
            test_info_new.append(test_info_med[num_valid-1][:])

        # correct the N_validation numbers to sequential
        for count_v in range(1, len(validation_info_new)+1):
            validation_info_new[count_v-1][0] = count_v
        for count_v in range(1, len(test_info_new)+1):
            test_info_new[count_v-1][0] = count_v


        # correct the validation images
        new_id_fold = [0 for _ in range(0, num_people)]
        valid_counter = 0
        id_counter = 1

        for valid_info in [validation_info_new, test_info_new]:
            valid_ids_list = [x[1] for x in valid_info]
            for valid_id in valid_ids_list:
                if new_id_fold[valid_id-1] == 0:
                    new_id_fold[valid_id-1] = id_counter
                    id_counter += 1
        for valid_info in [validation_info_new, test_info_new]:
            valid_ids_list = [x[1] for x in valid_info]
            for num_person in range(1, num_people+1):
                appearance_im_list = [i+1 for i, x in enumerate(valid_ids_list) if x == num_person]
                
                if appearance_im_list:
                    im_counter = 1
                    for appearance_im in appearance_im_list:
                        valid_info[appearance_im-1][6] = str(new_id_fold[num_person-1]) + "_" + str(im_counter) # Validation_image
                        im_counter +=1 
        val_train_file_dir = folds_folder + str(fold_counter) + "/" + training_folder
        if os.path.exists(val_train_file_dir):
            shutil.rmtree(val_train_file_dir)
        os.makedirs(val_train_file_dir)
        with open(val_train_file_dir + validation_info_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(valid_headers)
            for ite in validation_info_new:
                writer = csv.writer(outcsv)
                writer.writerow(ite)
        val_test_file_dir = folds_folder + str(fold_counter) + "/" + test_folder
        if os.path.exists(val_test_file_dir):
            shutil.rmtree(val_test_file_dir)
        os.makedirs(val_test_file_dir)
        with open(val_test_file_dir + validation_info_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(valid_headers)
            for ite in test_info_new:
                writer = csv.writer(outcsv)
                writer.writerow(ite)
        fold_counter += 1
    return fold_bin_order_list, validation_info_new, test_info_new, shuffle_people_order, shuffle_order_train_list, shuffle_order_test_list


def createCrossValidationSet(num_bins, num_folds, num_people, num_samples, cross_valid_folder_set, training_folder, test_folder, orig_images_folder, info_file, validation_info_file, valid_headers, height_file, height_noise_specs, time_method, time_spec_constants, db_headers, 
                             previous_sampled = None, previous_sub_dirs_order = None, isCreateDifferentTimeSets = False,
                             isRepeatedKFold = False, previous_db_file_list = [], prev_sorted_validation_info_list = [], prev_sorted_test_info_list = []):

    recog_order_bin_list = []
    recog_order_fold_list = []

    if isCreateDifferentTimeSets:
        time_method_list = ["GMM", "uniform"]
    else:
        time_method_list = [time_method]

    prev_shuffle_order = []
    prev_fold_bin_order_list = []
    sorted_validation_info_list = []
    sorted_test_info_list = []
    time_method_counter = 0
    db_file_list = []

    prev_shuffle_people_order = []
    prev_shuffle_order_train_list = []
    prev_shuffle_order_test_list = []
    
    for time_method in time_method_list:
        
        if time_method == "uniform":
            cross_valid_folder = cross_valid_folder_set + "_uniformT/"
            time_specs = [["uniform-range", "trunc-discrete", time_spec_constants[0], time_spec_constants[1], time_spec_constants[2], time_spec_constants[3], time_spec_constants[4], time_spec_constants[5], time_spec_constants[6]]]
            if len(time_method_list) == 1:
                # if we are only creating uniform dataset
                isSameSetToBeUsed = False # shuffle the set for each num_samples
            else:
                isSameSetToBeUsed = True # isSameSetToBeUsed = True is used to get the same set as uniform time method for the same number of samples
        else:
            cross_valid_folder = cross_valid_folder_set + "_gaussianT/"
            time_specs = [["GMM-range", "trunc-discrete", time_spec_constants[0], time_spec_constants[1], time_spec_constants[2], time_spec_constants[3], time_spec_constants[4], time_spec_constants[5], time_spec_constants[6]]]
            isSameSetToBeUsed = False # shuffle the set for each num_samples

        if os.path.exists(cross_valid_folder):
            shutil.rmtree(cross_valid_folder)
        os.makedirs(cross_valid_folder)
        seq_images_folder = cross_valid_folder + "images_sequential/"
        bin_folder = cross_valid_folder + "bins/"
        fold_folder_set = cross_valid_folder + "folds/"
        db_file = cross_valid_folder + "db_data.csv"
        db_file_list.append(db_file)
        if isRepeatedKFold:
            current_sampled = previous_sampled
            cur_sub_dirs = previous_sub_dirs_order
            if os.path.exists(fold_folder_set):
                shutil.rmtree(fold_folder_set)
            os.makedirs(fold_folder_set)
            if time_method_counter == 1:
                for ite_counter in range(0, len(prev_sorted_validation_info_list[0])):
                    ite_gaussian = prev_sorted_validation_info_list[0][ite_counter]
                    ite_uniform = prev_sorted_validation_info_list[1][ite_counter]
                    if ite_gaussian[:-1] != ite_uniform[:-1]:
                        print "UNIFORM AND GAUSSIAN DATASETS ARE NOT THE SAME"
                        print prev_sorted_validation_info_list[0][ite_counter][:-1]
                        print prev_sorted_validation_info_list[1][ite_counter][:-1]
                        break

            # fold_bin_order_list, sorted_validation_info, sorted_test_info, shuffle_order = shuffleDataset(num_people, num_folds, prev_sorted_validation_info_list[time_method_counter], prev_sorted_test_info_list[time_method_counter], fold_folder_set, training_folder, test_folder, validation_info_file, valid_headers, prev_shuffle_order, prev_fold_bin_order_list)
            # prev_fold_bin_order_list = copy.deepcopy(fold_bin_order_list)
            # prev_shuffle_order = copy.deepcopy(shuffle_order)
            # shutil.copy(previous_db_file_list[time_method_counter], db_file)
            fold_bin_order_list, sorted_validation_info, sorted_test_info, shuffle_people_order, shuffle_order_train_list, shuffle_order_test_list = divideIntoTrainingAndOpenSetForOptimisation(num_people, num_folds, prev_sorted_validation_info_list[time_method_counter], prev_sorted_test_info_list[time_method_counter], fold_folder_set, training_folder, test_folder, validation_info_file, valid_headers, prev_shuffle_people_order, prev_shuffle_order_train_list, prev_shuffle_order_test_list, prev_fold_bin_order_list)
            prev_fold_bin_order_list = copy.deepcopy(fold_bin_order_list)
            prev_shuffle_people_order = copy.deepcopy(shuffle_people_order)
            prev_shuffle_order_train_list = copy.deepcopy(shuffle_order_train_list)
            prev_shuffle_order_test_list = copy.deepcopy(shuffle_order_test_list)
            shutil.copy(previous_db_file_list[time_method_counter], db_file)
        else:
    
            if time_method_counter == 0:
                for num_bin in range(1, num_bins+1):
                    bin_name = bin_folder + str(num_bin)
                    if not os.path.isdir(bin_name):
                        os.makedirs(bin_name)

                image_info_sequential, orig_ids, num_images_per_person, current_sampled, cur_sub_dirs = renameImagesToSequential(orig_images_folder, seq_images_folder, num_samples, previous_sampled, previous_sub_dirs_order, isSameSetToBeUsed) #FUNCTION SPECIFIC FOR IMDB
                previous_sub_dirs_order = copy.deepcopy(cur_sub_dirs)
                previous_sampled = copy.deepcopy(current_sampled)
                image_info_bin, num_images_per_person_in_bin = cv.divideImagesIntoBins(num_bins, num_people, num_images_per_person, seq_images_folder, bin_folder, image_info_sequential)
                # image_info_bin, num_images_per_person_in_bin = cv.dividePeopleIntoBins(num_bins, num_people, num_images_per_person, seq_images_folder, bin_folder, image_info_sequential)
                # use same order/heights for uniform and GMM if isCreateDifferentTimeSets = True
                orig_heights, noisy_heights = getHeightInfo(num_people, num_images_per_person, orig_ids, height_file, height_noise_specs) #FUNCTION SPECIFIC FOR IMDB
                fold_bin_order_list = cv.getBinOrderInFold(num_bins)
    
                recog_order_bin_list = cv.getRecognitionOrderBin(num_bins, num_people, num_images_per_person_in_bin)
        
                recog_order_fold_list, new_id_fold_list = cv.getRecognitionOrderFold(num_people, fold_bin_order_list, recog_order_bin_list)
        
                FRO, num_im_in_bin_list = cv.getRecognitionOrderFoldPerImage(num_bins, num_folds, num_people, recog_order_bin_list, fold_bin_order_list, num_images_per_person_in_bin)

            time_list = getTimeInfo(num_people, num_images_per_person, time_specs) #FUNCTION SPECIFIC FOR IMDB
            saveDBFile(num_people, orig_ids, info_file, orig_heights, time_list, db_file, db_headers)
    
            sorted_validation_info, sorted_test_info = cv.saveValidationInformationFile(num_bins, num_folds, num_people, image_info_bin, fold_bin_order_list, FRO, orig_ids, new_id_fold_list, num_images_per_person_in_bin, num_im_in_bin_list, noisy_heights, time_list, validation_info_file, fold_folder_set)

        sorted_validation_info_list.append(sorted_validation_info)
        sorted_test_info_list.append(sorted_test_info)
        time_method_counter += 1
        
    return fold_bin_order_list, recog_order_bin_list, recog_order_fold_list, current_sampled, cur_sub_dirs, db_file_list, sorted_validation_info_list, sorted_test_info_list

def createCrossValidationSetForAllConditions(main_folder, num_repeats, num_folds, num_bins, num_people, time_method, isCreateDifferentTimeSets = True):
    period = 1 # time is checked every 1 minutes 
    time_range_sigma = 60/period
    time_min = 0
    time_max = (7*24*60/period) -1 # 7(days)*24(hours)*60(minutes)/period ( = num_time_slots)
    time_bin_size = 30
                
    height_min = 50
    height_max = 240
    height_bin_size = 1
    height_range_sigma = 20    
    height_stddev = 6.3
    height_fixed_conf = 0.08

    height_noise_specs = [[[height_stddev, "gaussian-sample-one", "replace-all", False, True, height_fixed_conf], "cont", 1, [height_min, height_max], [height_min, height_max], 
                                            (height_max+height_min)/2, height_stddev, "", 1]]      # H
    db_headers = ["id", "name", "gender", "age", "height", "times", "occurrence"]

    validation_info_file = "validation_info_fold.csv"
    valid_headers = ["N_validation", "Identity", "Original_image", "Sequential_image", "Bin", "Bin_image", "Validation_image", "N_original", "Height", "Time"]
    training_folder = "Training/"
    test_folder = "Test/"
    for setT in ["train", "open"]:
        dest_main = main_folder + "cross_validation_" + setT + "/"
        if setT == "train":
            orig_images_folder_set = "Train"
            info_file = "imdb_chosen_train.csv"
        else:
            orig_images_folder_set = "Test"
            info_file = "imdb_chosen_open.csv"
        height_file = info_file.replace(".csv", "_heights.csv")
        if not os.path.exists(dest_main):
            os.makedirs(dest_main)

        for num_samples in [10, None]:
            current_sampled = None 
            cur_sub_dirs_order = None
            cur_db_file_list = []
            cur_sorted_validation_info_list = []
            cur_sorted_test_info_list = []

            for num_f in range(1, num_repeats+1):
                if num_repeats > 1:
                    cross_valid_folder_set = dest_main + str(num_f) + "/" 
                else:
                    cross_valid_folder_set = dest_main
                if num_samples == 10:
                    cross_valid_folder_ss = cross_valid_folder_set + "N10"
                    orig_images_folder = orig_images_folder_set + "All/" # "Ten"
                else:
                    cross_valid_folder_ss = cross_valid_folder_set +  "Nall"
                    orig_images_folder = orig_images_folder_set + "All/"
                
                num_sample_curves = 3
                if num_samples is None:
                    num_curves = 3
                    num_samples_people_list = []
                else:
                    num_curves = int(num_samples/num_sample_curves)
                    num_samples_people_list = [num_samples for _ in range(1,num_people+1)]
                if num_curves == 0:
                    num_curves = 1
                time_spec_constants = [num_samples_people_list, [time_min, time_max], [time_min, time_max], num_curves, time_range_sigma, period, 0]

                     #   [[param_noise_level, param_noise_method, param_noise_add_method, isReverseLabel, isAddConfidenceScore], 
                                         # param_type, num_samples_per_person, [uniform_range_min, uniform_range_max], [clip_range_min, clip_range_max], mu/num_curves, sigma, extras (labels/ period), decimals]# T
                if num_f  > 1:
                    isRepeatedKFold = True
                else:
                    isRepeatedKFold = False

                fold_bin_order_list, recog_order_bin_list, recog_order_fold_list, current_sampled, cur_sub_dirs_order, cur_db_file_list, cur_sorted_validation_info_list, cur_sorted_test_info_list = createCrossValidationSet(num_bins, num_folds, num_people, num_samples, cross_valid_folder_ss, training_folder, test_folder, orig_images_folder, info_file, validation_info_file, valid_headers, height_file, height_noise_specs, time_method, time_spec_constants, db_headers, 
                             previous_sampled = current_sampled, previous_sub_dirs_order = cur_sub_dirs_order, isCreateDifferentTimeSets = isCreateDifferentTimeSets,
                             isRepeatedKFold = isRepeatedKFold, previous_db_file_list = cur_db_file_list, prev_sorted_validation_info_list = cur_sorted_validation_info_list, prev_sorted_test_info_list = cur_sorted_test_info_list)
                
         
if __name__ == "__main__":

    """
    min_width = 150
    min_height = 150
    main_folder = "imdb_crop/"
    info_file = main_folder + "imdb_chosen_open.csv"
    height_file = info_file.replace(".csv", "_heights.csv")
    num_get_last = 9
    cross_valid_folder = "cross_validation/test/Nall_gaussianT/"
    orig_images_folder = main_folder + "chosen_clean_open/"
    seq_images_folder = main_folder + cross_valid_folder + "images_sequential/"
    bin_folder = main_folder + cross_valid_folder + "bins/"
    fold_folder_set = main_folder + cross_valid_folder + "folds/"
    validation_info_file = "validation_info_fold.csv"
    db_file = main_folder + cross_valid_folder + "db_data.csv"
    training_folder = "Training/"
    test_folder = "Test/"

#    GET IMAGES WITH MINIMUM RESOLUTION CORRESPONDING TO CHOSEN CELEBRITY IMAGES
#     getMinResImages(src_folder, min_width, min_height, info_file, num_get_last=num_get_last)

#    GET IMAGES WITH MINIMUM RESOLUTION OF ALL IMAGES IN A FOLDER
#     getMinResImagesAll(main_folder + "orig/", main_folder + "orig_res/", min_width, min_height, num_get_last=None)

#     FIND OPTIMAL FACE RECOGNITION THRESHOLD:
#     cost_function_alpha = 0.9
#     n_iters = 200
#     bounds = np.array([[0.0, 1.0]])
#     n_pre_samples = 3
#     dec_precision = 2
#     cross_val_stats_train_file = main_folder + cross_valid_folder + "face_rec_threshold_folds.csv"
#     optim_params_file = main_folder + cross_valid_folder + "opt_face_rec_threshold.csv"
#     wop.getOptimFaceRecogThreshold(num_folds, fold_folder_set, training_folder, cross_val_stats_train_file, optim_params_file, cost_function_alpha, n_iters, bounds, n_pre_samples, dec_precision) 

#     CREATE CROSS VALIDATION SET FOR ALL CONDITIONS
    num_people = 100
    num_bins = 5
    num_repeats = 1 # 1 for old cross validation
    num_folds = 10 # 5 for old cross validation
    main_folder = "TenFolds"
    createCrossValidationSetForAllConditions(main_folder, num_repeats, num_folds, num_bins, num_people)
    
#    COPY THE RECOGNISERBN.CSV AND INITIALRECOGNITION.CSV FROM CROSS_VALIDATION FOLDERS TO OPTIM FOLDER
    main_folder = ""
    
    num_people = 100
    num_bins = 5
    num_repeats = 15 # 1 for old cross validation
    num_folds = 1 # 5 for old cross validation

    cross_valid_const_folder =  "sim_scores_robot/"
    cross_valid_list = ["N10_gaussianT/", "Nall_gaussianT/", "N10_uniformT/", "Nall_uniformT/"]

#    optim_folder = "cross_validation_test/MMIBN-OL/"
    optim_folder_list = ["optim/Nall_gaussianT/MMIBN/", "cross_validation_optim_weights/Nall_uniformT/MMIBN/", 
                         "cross_validation_optim_weights/Nall_gaussianT/MMIBN-OL/", "cross_validation_optim_weights/Nall_uniformT/MMIBN-OL/"]
    optim_params_folder = "optim/"
    files_to_copy = ["validation_info_fold.csv", "InitialRecognition.csv", "RecogniserBN.csv", "db.csv"]
    for optim_folder in optim_folder_list:
        for cross_valid in cross_valid_list:
            optim_cross_dir = main_folder + optim_folder + cross_valid
            if not os.path.isdir(optim_cross_dir):
                os.makedirs(optim_cross_dir)
            fold_dir = main_folder + cross_valid_const_folder + cross_valid + "folds/"
            optim_cross_fold_dir = optim_cross_dir + "folds/"
            if not os.path.isdir(optim_cross_fold_dir):
                os.makedirs(optim_cross_fold_dir)
            if "Nall_gaussianT" in optim_cross_dir:
                shutil.copy2("imdb_crop/" + optim_params_folder + "Nall_gaussianT/" + "optim_params.csv", optim_cross_dir + "optim_params.csv")
            elif "Nall_uniformT" in optim_cross_dir:
                shutil.copy2("imdb_crop/" + optim_params_folder + "Nall_uniformT/" + "optim_params.csv", optim_cross_dir + "optim_params.csv")
            for num_fold in range(1, num_folds+1):
                
                optim_cross_num_fold_dir = optim_cross_fold_dir + str(num_fold) + "/"
                if not os.path.isdir(optim_cross_num_fold_dir):
                    os.makedirs(optim_cross_num_fold_dir)
                
                for t_fold in [training_folder, test_folder]:
                    src_cross_num_fold_t_dir = fold_dir + str(num_fold) + "/" + t_fold
                    optim_cross_num_fold_t_dir = optim_cross_num_fold_dir + t_fold
    
                    if not os.path.isdir(optim_cross_num_fold_t_dir):
                        os.makedirs(optim_cross_num_fold_t_dir)
                    if os.path.isdir(optim_cross_num_fold_t_dir + "images"):
                        shutil.rmtree(optim_cross_num_fold_t_dir + "images")
                    os.makedirs(optim_cross_num_fold_t_dir + "images")
                    os.makedirs(optim_cross_num_fold_t_dir + "images/Known_True")
                    os.makedirs(optim_cross_num_fold_t_dir+ "images/Known_False")
                    os.makedirs(optim_cross_num_fold_t_dir + "images/Known_Unknown")
                    os.makedirs(optim_cross_num_fold_t_dir + "images/Unknown_True")
                    os.makedirs(optim_cross_num_fold_t_dir + "images/Unknown_False")
                    for file_to_cp in files_to_copy:
#                         shutil.copy2(src_cross_num_fold_t_dir + file_to_cp, optim_cross_num_fold_t_dir + file_to_cp.replace(".csv", "_data.csv"))
                        if "validation_info_fold" in file_to_cp:
#                             values_train = pandas.read_csv(src_cross_num_fold_t_dir + file_to_cp.replace(".csv", "_data.csv"), converters={"H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
                            values_train = []
                            values_test = pandas.read_csv(src_cross_num_fold_t_dir + file_to_cp, converters={"H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
                            for count_v in range(0, len(values_test)):
                                values_test[count_v][0] = len(values_train) + count_v + 1 #N
#                                 values_test[count_v][1] = num_people + int(values_test[count_v][1]) #I
#                                 val_im = values_test[count_v][6].split("_")
#                                 values_test[count_v][6] = str(num_people + int(val_im[0])) + "_" + val_im[1]
                                 
                        elif "InitialRecognition" in file_to_cp or "RecogniserBN" in file_to_cp:
                            values_train = []
#                             values_train = pandas.read_csv(src_cross_num_fold_t_dir + file_to_cp.replace(".csv", "_data.csv"), converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
                            values_test = pandas.read_csv(src_cross_num_fold_t_dir + file_to_cp, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
                            values_test_val = pandas.read_csv(src_cross_num_fold_t_dir + "validation_info_fold.csv",  converters={"H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
 
                            if "InitialRecognition" in file_to_cp:
                                for count_v in range(0, len(values_test)):
                                    values_test[count_v][6] = len(values_train) + count_v + 1 #N
                                    values_test[count_v][5] = values_test_val[count_v][-1] #T
                            else:
 
                                for count_v in range(0, len(values_test)):
                                    values_test[count_v][7] = len(values_train) + count_v + 1 #N
                                    values_test[count_v][5] =  values_test_val[count_v][-1]#T
                                    if t_fold in training_folder:
                                        if count_v < num_people:
                                            values_test[count_v][6] = 1 # R
                                        else:
                                            values_test[count_v][6] = 0
                                    else:
                                        values_test[count_v][6] = 0 
                                    
                                 
                        elif "db" in file_to_cp:
                            if t_fold in training_folder:
                                values_test_val = pandas.read_csv(src_cross_num_fold_t_dir + "validation_info_fold.csv",  converters={"H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
                                db_test_data = pandas.read_csv( main_folder + cross_valid_const_folder + cross_valid + "db_data.csv",  converters={"H": ast.literal_eval, "T": ast.literal_eval}).values.tolist()
                                values_train = []
    #                             values_train = pandas.read_csv(src_cross_num_fold_t_dir + file_to_cp.replace(".csv", "_data.csv"), converters={"times": ast.literal_eval, "occurrence": ast.literal_eval}).values.tolist()
                                values_test = []
                                for count_v in range(0, num_people):
                                    id_orig = values_test_val[count_v][1]
                                    person_row = db_test_data[id_orig-1][1:]
                                    person_row[-2] = [ast.literal_eval(values_test_val[count_v][-1])]#T
                                    person_row[-1] = [1,1,1]
    #                                 id_new = count_v + 1 + num_people
                                    id_new = count_v + 1
                                    person_row.insert(0, id_new)
                                    values_test.append(person_row)
                            else:
                                shutil.copy2(src_cross_num_fold_t_dir.replace(test_folder, training_folder) + file_to_cp, optim_cross_num_fold_t_dir + file_to_cp.replace(".csv", "_data.csv"))
                                continue
                        values_to_write = values_train + values_test
                        
                        headers =  pandas.read_csv(src_cross_num_fold_t_dir + file_to_cp, nrows=1).columns
                        with open(optim_cross_num_fold_t_dir + file_to_cp.replace(".csv", "_data.csv"), 'wb') as outcsv:
                            writer = csv.writer(outcsv)   
                            writer.writerow(headers)
                            for value_row in values_to_write:
                                writer.writerow(value_row)
     

    
    num_repeats = 15
    num_bins = 5
    num_people = 100

    probThreshold = 1.0e-75

#     COPY FILES TO OPTIM FOLDERS

#     OPTIMISE PARAMETERS ON THE FOLDS
    main_folder = ""
    optim_folder = "optim/"
    updateMethod_list= ["none", "evidence"]
    cross_val_folders = ["N10_gaussianT/", "N10_uniformT/", "Nall_gaussianT/", "Nall_uniformT/"]
    norm_method_list = ["softmax", "minmax", "tanh", "norm-sum", "hybrid"]
    optim_params_file_base = "optim_params.csv"
    folds_folder = "folds/"
    training_folder = "Training/"
    test_folder = "Test/"
    open_training_folder = "OpenTraining/"
    open_test_folder = "OpenTest/"
    stats_header = ["Fold", "Evidence_method", "Threshold", "Norm_method", "FER", 
                    "I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1", 
                    "Face_threshold", "Quality_threshold", "Weights", "Loss",
                    "Num_recognitions", "Num_registered"]
    stats_header_avg = ["Evidence_method", "Threshold", "Norm_method", "FER", "Std",
                    "I_FAR", "Std", "F_FAR", "Std", "I_DIR_1", "Std", "F_DIR_1", "Std", 
                    "Face_threshold", "Quality_threshold", "Weights", "Loss", "Std",
                    "Num_recognitions", "Num_registered"]
    variables_list_base = list(itertools.product(cross_val_folders, norm_method_list, updateMethod_list))
    
    variables_list = []
    list_base_folders = []

    for cross_val_folder in cross_val_folders:
        base_folder = main_folder + optim_folder + cross_val_folder
        optim_params_file = base_folder + optim_params_file_base
        if not os.path.isfile(optim_params_file):
            with open(optim_params_file, 'wb') as outcsv:
                writer = csv.writer(outcsv)   
                writer.writerow(["Evidence_method", "Norm_method", "Optim_params", "Loss", "Std", "I_FAR", "Std", "F_FAR", "Std", "I_DIR_1", "Std", "F_DIR_1", "Std"])   
        cross_val_stats_train_file_base = base_folder + "cvs_train/"
        cross_val_stats_open_train_file_base = base_folder + "cvs_open_train/"
        if not os.path.isdir(cross_val_stats_train_file_base):
            os.makedirs(cross_val_stats_train_file_base)
        if not os.path.isdir(cross_val_stats_open_train_file_base):
            os.makedirs(cross_val_stats_open_train_file_base)          
        for counter_var in range(0, len(variables_list_base)):
            var_l = variables_list_base[counter_var]
            if var_l[0] == cross_val_folder:
                train_file_name = cross_val_stats_train_file_base + str(var_l[1]) + "_" + str(var_l[2][0]) + ".csv"
                open_train_file_name = cross_val_stats_open_train_file_base + str(var_l[1]) + "_" + str(var_l[2][0]) + ".csv"
                 
                variables_list.append(variables_list_base[counter_var] + (train_file_name, open_train_file_name))
                if not os.path.isfile(train_file_name):
                    with open(train_file_name, 'wb') as outcsv:
                        writer = csv.writer(outcsv)   
                        writer.writerow(stats_header)
                train_file_name_avg = train_file_name.replace(".csv", "_avg.csv")
                if not os.path.isfile(train_file_name_avg):
                    with open(train_file_name_avg, 'wb') as outcsv:
                        writer = csv.writer(outcsv)   
                        writer.writerow(stats_header_avg)
          
                if not os.path.isfile(open_train_file_name):
                    with open(open_train_file_name, 'wb') as outcsv:
                        writer = csv.writer(outcsv)   
                        writer.writerow(stats_header)
                open_train_file_name_avg = open_train_file_name.replace(".csv", "_avg.csv")
                if not os.path.isfile(open_train_file_name_avg):
                    with open(open_train_file_name_avg, 'wb') as outcsv:
                        writer = csv.writer(outcsv)   
                        writer.writerow(stats_header_avg)
                   

    pool = mp.Pool(processes=num_processors)

    # [num_people, num_folds, training_folder, test_folder, cross_val_stats_train_file, cross_val_stats_test_file, 
    # db_list, init_list, recogs_list, bn, isTestData,
    # normMethod, updateMethod, probThreshold,
    # isMultRecognitions, num_mult_recognitions, qualityCoefficient,
    # db_file, init_recog_file, final_recog_file, cost_function_alpha, n_iters, optim_params_file, bounds]

    results = [pool.apply_async(ad.optimParams, [[num_people, num_repeats, main_folder + optim_folder + variables_l[0] + folds_folder + training_folder, main_folder + optim_folder + variables_l[0] + folds_folder + open_training_folder, variables_l[3],  variables_l[4], 
    None, None, None, None, False,
    variables_l[1], variables_l[2], probThreshold,
    False, 3, None, db_file, init_file, recog_file, cost_function_alpha, n_iters, 
    main_folder + optim_folder + variables_l[0] + optim_params_file_base, bounds]]) for variables_l in variables_list]
 
    output = [p.get() for p in results]
    pool.close()
    pool.join()

"""
#     CREATE CROSS VALIDATION SET FOR ALL CONDITIONS
    num_people = 100
    num_folds = 5 # 5 for old cross validation
    num_bins = num_folds
    num_repeats = 11 # 1 for old cross validation 10+1
    main_folder = "RepeatedOptim5fold/"
    time_method = "GMM" # "uniform"
    createCrossValidationSetForAllConditions(main_folder, num_repeats, num_folds, num_bins, num_people, time_method, isCreateDifferentTimeSets = True)

#     GET STATISTICAL SIGNIFICANCE RESULTS, STDDEV AND AVERAGE FOR NORMALISATION METHOD WITH OPTIMISED PARAMETERS AND FOR COMPARING IT TO FACE RECOGNITION AND SOFT BIOMETRICS

#     USE OPTIMISED WEIGHTS AND QUALITY WITH OPTIMUM NORMALISATION METHOD FOR RERUNNING CROSS VALIDATION ON ROBOT

