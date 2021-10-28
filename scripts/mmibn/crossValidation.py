# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  CrossValidation script evaluates the RecogniserMemory in a cross validation setting. It creates       #
#  training  and tests sets from data, and calls runCrossValidation in RecognitionMemory.                #
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
#  crossValidation, RecognitionMemory and each script in this project is under the GNU General Public    #
#  License v3.0. You should have received a copy of the license along with MultimodalRecognitionDataset. #
#  If not, see <http://www.gnu.org/licenses>.                                                            #   
#========================================================================================================#

import random
import os
import os.path
import shutil
import pandas
import ast
import csv
import numpy as np
import time
import RecognitionMemory
from operator import add
# Initial seeding
random.seed(1234)

def renameImagesToSequential(num_people, num_mult_images, orig_images_folder, seq_images_folder, num_samples_per_person=None):
    """
        Brief: 
        Example: 
        Returns: 
    """
    image_info_sequential = [] # [[[Original_image_1, Sequential_image_1],[Original_image_2, Sequential_image_2]...] for num_person in range(1, num_people+1)]
    mult_image_info = [] # [[#mult_image_recognition1_person1, #mult_image_recognition2_person1], [#mult_image_recognition1_person2, #mult_image_recognition2_person2],..]
    
    if not os.path.exists(seq_images_folder):
        os.makedirs(seq_images_folder)

    num_images_per_person = []
        
    for num_person in range(1, num_people+1):
        image_person_info = []
        mult_image_info_person = []
        num_seq = 1
        if num_person >= 12:
            # I am missing enrolment images for the first 12 people, so i am making everything equal
            start_num = 2         
        else:
            start_num = 1
        num_recog = start_num
        
        files = sorted([filename for filename in os.listdir(orig_images_folder) if filename.startswith(str(num_person) + "_") and not filename.startswith(str(num_person) + "_000"+str(start_num-1)) and not filename.startswith(str(num_person) + "_000"+str(start_num-2))])
        sampled = files[:]

        for num_f in range(0, len(sampled)):
            num_mult_counter = 0
            for num_im in range(0, num_mult_images):
                if num_recog <10:
                    zeros = "_000"
                elif num_recog <100:
                    zeros = "_00"
                elif num_recog <1000:
                    zeros = "_0"
                else:
                    zeros = "_"
                    
                filename = str(num_person) + zeros + str(num_recog) + "-" + str(num_im) + ".jpg"
                if filename in sampled:
                    new_image = seq_images_folder + str(num_person) + "_" + str(num_seq) + ".jpg"
                    shutil.copy2( orig_images_folder + filename, new_image)
                    image_person_info.append([num_person, str(num_person) + zeros + str(num_recog) + "-" + str(num_im), str(num_person) + "_" + str(num_seq)])
                    num_mult_counter += 1
                    num_seq += 1
                    if num_samples_per_person is not None and num_seq > num_samples_per_person:
                        break
            if num_mult_counter > 0:
                mult_image_info_person.append(num_mult_counter)
            num_recog += 1
            if num_samples_per_person is not None and num_seq > num_samples_per_person:
                break
        num_images_per_person.append(num_seq-1)
        mult_image_info.append(mult_image_info_person)
        image_info_sequential.append(image_person_info)
        
    return image_info_sequential, mult_image_info, num_images_per_person
    
def divideImagesIntoBins(num_bins, num_people, num_images_per_person, seq_images_folder=None, bin_folder=None, image_info_sequential=None, isImage=True):
    """ 
        Brief: create bins by dividing the sequential images one by one to the bins and then saving them in random order
        Example: Images for person #1: 
        sequential order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,.. 
        If you have 5 bins, 1-> bin#1 ->saved as 1_7 (randomised order within the bin instead of 1_1 within bin), 2-> bin#2 -> 1_3, .. 6-> bin#1 -> 1_2
        Returns: information on which sequential image was saved as which number within each bin (combined with information on original_image and sequential_image)
    """    
    for num_bin in range(1, num_bins+1):
        bin_name = bin_folder + str(num_bin)
        if os.path.isdir(bin_name):
            shutil.rmtree(bin_name)
        os.makedirs(bin_name)

    if not isinstance(num_images_per_person, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_images_per_person_list = [num_images_per_person for _ in range(0, num_people)]
    else:
        num_images_per_person_list = num_images_per_person[:]
    
    if image_info_sequential:
        image_info_bin = image_info_sequential[:]
    else:
        image_info_bin = [[[] for x in range(0, num_images_per_person_list[i])] for i in range(0, num_people)]
        
    num_images_per_person_in_bin = [[0 for _ in range(1, num_bins+1)] for _ in range(1, num_people+1)]
    # divide the sequential images one by one to each bin and randomise the order of images within the bin
    for num_person in range(1, num_people+1):
        counter_num_images = [0 for _ in range(1, num_bins+1)]
        num_samples = num_images_per_person_list[num_person-1]
        
        num_remaining = num_samples % num_bins
        
        rand_order_list = [range(1, (num_samples/num_bins)+1) for _ in range(1, num_bins+1)]
        if num_remaining != 0:
#             remaining_bins = random.sample(range(1,num_bins+1), num_remaining)
            remaining_bins = range(1, num_remaining+1)
            for remain_bin in remaining_bins:
                rand_order_list[remain_bin-1].append((num_samples/num_bins)+1)
                
        for num_bin in range(1, num_bins+1):
            random.shuffle(rand_order_list[num_bin-1])

        for num_im in range(1, num_samples+1):
            num_bin = num_im % num_bins
            if num_bin == 0:
                num_bin = num_bins
            
            index_im = rand_order_list[num_bin-1][counter_num_images[num_bin-1]]

            if isImage:
                filename = seq_images_folder + str(num_person) + "_" + str(num_im) + ".jpg"            
                image_dir = bin_folder + str(num_bin) + "/" + str(num_person) + "_" + str(index_im) + ".jpg"
                shutil.copy2(filename, image_dir)
                
            image_info_bin[num_person-1][num_im-1].append(num_bin)
            num_images_per_person_in_bin[num_person-1][num_bin-1] += 1
            image_info_bin[num_person-1][num_im-1].append(str(num_person) + "_" + str(index_im))
            counter_num_images[num_bin-1] += 1
    return image_info_bin, num_images_per_person_in_bin

def dividePeopleIntoBins(num_bins, num_people, num_images_per_person, seq_images_folder=None, bin_folder=None, image_info_sequential=None, isImage=True):
    """ 
        Brief: create bins by dividing the sequential images one by one to the bins and then saving them in random order
        Example: Images for person #1: 
        sequential order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,.. 
        If you have 5 bins, 1-> bin#1 ->saved as 1_7 (randomised order within the bin instead of 1_1 within bin), 2-> bin#2 -> 1_3, .. 6-> bin#1 -> 1_2
        Returns: information on which sequential image was saved as which number within each bin (combined with information on original_image and sequential_image)
    """    
    for num_bin in range(1, num_bins+1):
        bin_name = bin_folder + str(num_bin)
        if os.path.isdir(bin_name):
            shutil.rmtree(bin_name)
        os.makedirs(bin_name)

    if not isinstance(num_images_per_person, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_images_per_person_list = [num_images_per_person for _ in range(0, num_people)]
    else:
        num_images_per_person_list = num_images_per_person[:]
    
    if image_info_sequential:
        image_info_bin = image_info_sequential[:]
    else:
        image_info_bin = [[[] for x in range(0, num_images_per_person_list[i])] for i in range(0, num_people)]
    
    people_list = list(range(1, num_people+1))
    random.shuffle(people_list)

    people_list_in_bins = partitionList(people_list, num_bins)
        
    num_images_per_person_in_bin = [[0 for _ in range(1, num_bins+1)] for _ in range(1, num_people+1)]
    # 
    for num_bin in range(1, num_bins+1):
        for num_person_counter in range(1, len(people_list_in_bins[num_bin - 1]) +1):
            num_person = people_list_in_bins[num_bin - 1][num_person_counter - 1]
            num_samples = num_images_per_person_list[num_person-1]   
            rand_order_list = list(range(1,num_samples+1))
            random.shuffle(rand_order_list)
            for index_im in range(1, len(rand_order_list)+1):
                num_im = rand_order_list[index_im-1]

                if isImage:
                    filename = seq_images_folder + str(num_person) + "_" + str(num_im) + ".jpg"            
                    image_dir = bin_folder + str(num_bin) + "/" + str(num_person) + "_" + str(index_im) + ".jpg"
                    shutil.copy2(filename, image_dir)
                
                image_info_bin[num_person-1][num_im-1].append(num_bin)
                num_images_per_person_in_bin[num_person-1][num_bin-1] += 1
                image_info_bin[num_person-1][num_im-1].append(str(num_person) + "_" + str(index_im)) 
    return image_info_bin, num_images_per_person_in_bin

def divideImagesIntoBinsMod(num_bins, num_people, num_images_per_person, seq_images_folder=None, bin_folder=None, image_info_sequential=None, isImage=True):
    """ 
        Brief: create bins by dividing the sequential images one by one to the bins and then saving them in random order
        Example: Images for person #1: 
        sequential order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,.. 
        If you have 5 bins, 1-> bin#1 ->saved as 1_7 (randomised order within the bin instead of 1_1 within bin), 2-> bin#2 -> 1_3, .. 6-> bin#1 -> 1_2
        Returns: information on which sequential image was saved as which number within each bin (combined with information on original_image and sequential_image)
    """    

    if not isinstance(num_images_per_person, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_images_per_person_list = [num_images_per_person for _ in range(0, num_people)]
    else:
        num_images_per_person_list = num_images_per_person[:]
        
    image_info_bin_mod = [[[0 for _ in range(0, num_images_per_person_list[num_person]/num_bins)] for _ in range(0, num_bins)] for num_person in range(0, num_people)]
    
    num_images_per_person_in_bin = [[0 for _ in range(0, num_bins)] for _ in range(0, num_people)]
    
    # divide the sequential images one by one to each bin and randomise the order of images within the bin
    for num_person in range(1, num_people+1):
        counter_num_images = [0 for _ in range(1, num_bins+1)]
        num_samples = num_images_per_person_list[num_person-1]
        num_remaining = num_samples % num_bins
        
        rand_order_list = [range(1, (num_samples/num_bins)+1) for _ in range(1, num_bins+1)]
        if num_remaining != 0:
#             remaining_bins = random.sample(range(1,num_bins+1), num_remaining)
            remaining_bins = range(1, num_remaining+1)
            for remain_bin in remaining_bins:
                rand_order_list[remain_bin-1].append((num_samples/num_bins)+1)
                
        for num_bin in range(1, num_bins+1):
            random.shuffle(rand_order_list[num_bin-1])
            
        for num_im in range(1, num_samples+1):
            num_bin = num_im % num_bins
            if num_bin == 0:
                num_bin = num_bins
            index_im = rand_order_list[num_bin-1][counter_num_images[num_bin-1]]

            if isImage:
                filename = seq_images_folder + str(num_person) + "_" + str(num_im) + ".jpg"            
                image_dir = bin_folder + str(num_bin) + "/" + str(num_person) + "_" + str(index_im) + ".jpg"
                shutil.copy2(filename, image_dir)
                
            image_info_bin_mod[num_person-1][num_bin-1][index_im-1] = num_im
            num_images_per_person_in_bin[num_person-1][num_bin-1] += 1
            counter_num_images[num_bin-1] += 1
    return image_info_bin_mod, num_images_per_person_in_bin

def getSeqOrderForPersoninFold(num_bins, num_folds, num_people, fold_bin_order_list, recog_order_bin_list, image_info_bin_mod):
    """
    """
    fold_seq_order_list = []
    for num_fold in range(1, num_folds+1):
        train_seq_order = []
        test_seq_order = []
        bin_counter = 0
        for num_bin in fold_bin_order_list[num_fold-1]:
            bin_person_im_counter = [0 for _ in range(1, num_people+1)]
            for num_person in recog_order_bin_list[num_bin-1]:
                orig_num_sample = image_info_bin_mod[num_person-1][num_bin-1][bin_person_im_counter[num_person-1]]
                if bin_counter == num_bins-1:
                    test_seq_order.append([num_person, orig_num_sample])
                else:
                    train_seq_order.append([num_person, orig_num_sample])
                bin_person_im_counter[num_person-1] += 1
            bin_counter += 1
        fold_seq_order_list.append([train_seq_order, test_seq_order])
    return fold_seq_order_list

def getBinOrderInFold(num_bins, num_folds = None):
    """ 
        Brief: randomise validation set order, the first item and the last item in each order is unique in each bin 
        The reason: first item to ensure a different order of enrolment in each fold, and last item to ensure a different test set in each fold
        Returns: list of bin order for each fold 
    """
    if num_folds is None:
        num_folds = num_bins
    fold_bin_order_list = randomiseOrder(num_folds, range(1, num_bins+1), isEndUnique = True)
    return fold_bin_order_list
    
def getRecognitionOrderBin(num_bins, num_people, num_images_per_person_in_bin):
    """ 
        Brief: randomise people order for recognition (first everyone is introduced to the system in a random order, and then recognitions start in random order)
        Returns: list of recognition order for each bin 
    """
    if not isinstance(num_images_per_person_in_bin, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_images_per_person_in_bin_list = [[num_images_per_person_in_bin for _ in range(0, num_bins)] for _ in range(0, num_people)]
    else:
        num_images_per_person_in_bin_list = num_images_per_person_in_bin[:]
         
    enrolment_order_list = randomiseOrder(num_bins, range(1, num_people+1))
    recog_order_bin_list = []
    for num_bin in range(1, num_bins+1):
        recog_order_per_bin = []
        enrolment_order = enrolment_order_list[num_bin-1]
        for num_person in range(1,num_people+1):
            num_images_per_person_in_bin_p = num_images_per_person_in_bin_list[num_person-1][num_bin-1]
            for num_im in range(1,num_images_per_person_in_bin_p):
                recog_order_per_bin.append(num_person)
        random.shuffle(recog_order_per_bin) # remaining order of people are shuffled
        recog_order_per_bin = enrolment_order + recog_order_per_bin # order of the recognition
        recog_order_bin_list.append(recog_order_per_bin)
    return recog_order_bin_list
   
def getRecognitionOrderFold(num_people, fold_bin_order_list, recog_order_bin_list):
    """ 
        Brief: get recognition order per fold combining fold_bin_order_list and recognition order within each bin
        Returns: list of recognition order for each fold 
    """
    recog_order_fold_list = []
    for fold in fold_bin_order_list:
        recog_order_per_fold = []
        for num_bin in fold:
            recog_order_per_fold = recog_order_per_fold + recog_order_bin_list[num_bin-1]
        recog_order_fold_list.append(recog_order_per_fold)
    
    new_id_fold_list = []
    for num_fold in range(1, len(fold_bin_order_list)+1):
        new_id_fold = []
        recog_order_fold = recog_order_fold_list[num_fold-1]
        for num_person in range(1,num_people+1):
            id = [i+1 for i, x in enumerate(recog_order_fold) if x == num_person][0]
            new_id_fold.append(id)
        new_id_fold_list.append(new_id_fold)
    return recog_order_fold_list, new_id_fold_list

def getRecognitionOrderFoldPerImage(num_bins, num_folds, num_people, recog_order_bin_list, fold_bin_order_list, num_images_per_person_in_bin):
    """
    """
    FRO = []
    if not isinstance(num_images_per_person_in_bin, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_im_in_bin_list = [num_images_per_person_in_bin*num_people for _ in range(0, num_bins)]
    else:
        num_im_in_bin_list = map(sum, zip(*num_images_per_person_in_bin))
    
    for num_person in range(1, num_people+1):
        recog_numbers_bin_person_list = []
        recog_numbers_fold_person_list = []
        for num_bin in range(1, num_bins+1):
            recog_numbers_bin_person = [i+1 for i, x in enumerate(recog_order_bin_list[num_bin-1]) if x == num_person]
            recog_numbers_bin_person_list.append(recog_numbers_bin_person)
            
        for num_fold in range(1, num_folds+1):
            fold_bin_order = fold_bin_order_list[num_fold-1][:]
            recog_numbers_fold_person = [[] for _ in range(1, num_bins+1)]
            bin_counter = 0
            tot_img = 0
            for num_bin in fold_bin_order:
                recog_numbers_bin_person = recog_numbers_bin_person_list[num_bin-1][:]
                if bin_counter == num_bins-1: # Test bin
                    recog_numbers_fold_person[num_bin-1] = recog_numbers_bin_person
                else: # Training bin
                    recog_numbers_fold_person[num_bin-1] = [i + tot_img for i in recog_numbers_bin_person]
                tot_img += num_im_in_bin_list[num_bin-1]
                bin_counter += 1
            recog_numbers_fold_person_list.append(recog_numbers_fold_person)
                
        FRO.append(recog_numbers_fold_person_list)
    return FRO, num_im_in_bin_list
                
def getOriginalRecognitionOrder(num_people, recogniser_csv_file):
    """
    """
    df_orig = pandas.read_csv(recogniser_csv_file, dtype={"I": object}, converters={"H": ast.literal_eval, "T": ast.literal_eval}, usecols = {"I", "H", "T", "N"})
#     identity_list = df_orig.I.tolist()
#     orig_recog_list = []
#     iden = ""
#     for ite in identity_list:
#         if ite is not np.nan:
#             iden = ite
#         orig_recog_list.append(iden)
#     orig_recog_list = [int(i) for i in orig_recog_list]
#     height_list = []
#     time_list = []
#     recog_numbers_person_list = []
#     for num_person in range(1, num_people+1):
#         list_n = df_orig.loc[df_orig["I"] == str(num_person), 'N'].values.tolist()
#         recog_numbers_person_list.append(list_n)
#     
#     for recog_numbers_person in recog_numbers_person_list:
#         heights_per_person = []
#         times_per_person = []
#         for recog_number in recog_numbers_person:
#             heights_per_person_recog = df_orig.loc[df_orig["N"] == recog_number, 'H'].values.tolist()
#             heights_per_person = heights_per_person + heights_per_person_recog
#         
#             times_per_person_recog = df_orig.loc[df_orig["N"] == recog_number, 'T'].values.tolist()
#             times_per_person = times_per_person + times_per_person_recog
#         height_list.append(heights_per_person)
#         time_list.append(times_per_person)
# 
#     return orig_recog_list, height_list, time_list
#     
    orig_recog_list = df_orig.I.dropna().tolist()
    orig_recog_list = [int(i) for i in orig_recog_list]
    return orig_recog_list, df_orig

def getRecogInfo(num_people, num_images_per_person, mult_image_info, recogniser_csv_file):
    """
    """
    orig_recog_list, df_orig = getOriginalRecognitionOrder(num_people, recogniser_csv_file)
    
    if not isinstance(num_images_per_person, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_images_per_person_list = [num_images_per_person for _ in range(0, num_people)]
    else:
        num_images_per_person_list = num_images_per_person[:]
        
    height_list = []
    time_list = []
    orig_recog_numbers_list = []
    for num_person in range(1, num_people+1):
        orig_recog_numbers_person = [i+1 for i, x in enumerate(orig_recog_list) if x == num_person]
        heights_per_person = []
        times_per_person = []
        recog_numbers_person = []
        
        list_recog = orig_recog_numbers_person[1:] # skip enrolment (I was missing enrolment images)
        list_mult = mult_image_info[num_person-1][:]

        person_recog_counter = 0
        
        num_seq = 0
        
        while num_seq < num_images_per_person_list[num_person-1]:
            recog_number = list_recog[person_recog_counter]
            
            heights_per_person_recog = df_orig.loc[df_orig["N"] == recog_number, 'H'].iloc[0]
            times_per_person_recog = df_orig.loc[df_orig["N"] == recog_number, 'T'].iloc[0]
            for num_mult in range(0, list_mult[person_recog_counter]):
                heights_per_person.append(heights_per_person_recog)
                times_per_person.append(times_per_person_recog)
                recog_numbers_person.append(recog_number)
                num_seq += 1
            person_recog_counter += 1
        height_list.append(heights_per_person)
        time_list.append(times_per_person)
        orig_recog_numbers_list.append(recog_numbers_person)
    
    return orig_recog_numbers_list, height_list, time_list

def makeDirectory(dir_name):
    """
    """    
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)   
 
def saveValidationInformationFile(num_bins, num_folds, num_people, image_info_bin, fold_bin_order_list, FRO, orig_recog_numbers_list, new_id_fold_list,
                                  num_images_per_person, num_im_in_bin_list, height_list, time_list, validation_info_file, fold_folder):
    """
    """    
    if not isinstance(num_images_per_person, (list,)):
        # if it is not list, make it list (to avoid compatibility errors)
        num_images_per_person_list = [num_images_per_person for _ in range(0, num_people)]
    else:
        num_images_per_person_list = num_images_per_person[:]
    for num_fold in range(1, num_folds+1):
        fold_name = fold_folder + str(num_fold)
        fold_bin_order = fold_bin_order_list[num_fold-1]
        makeDirectory(fold_name)
        validation_folder = fold_name + "/" + "Training"
        makeDirectory(validation_folder)
        val_info_file = validation_folder + "/" + validation_info_file
        test_folder = fold_name + "/" + "Test"
        makeDirectory(test_folder)
        test_info_file = test_folder + "/" + validation_info_file
        if os.path.isfile(val_info_file):
            os.remove(val_info_file)
        with open(val_info_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["N_validation", "Identity", "Original_image", "Sequential_image", "Bin", "Bin_image", "Validation_image", "N_original", "Height", "Time"])
        if os.path.isfile(test_info_file):
            os.remove(test_info_file)
        with open(test_info_file, 'wb') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["N_validation", "Identity", "Original_image", "Sequential_image", "Bin", "Bin_image", "Validation_image", "N_original", "Height", "Time"])
        validation_folder_images = validation_folder + "/" + "images"
        makeDirectory(validation_folder_images)
        validation_folder_discarded_images = validation_folder_images + "/" + "discarded"
        makeDirectory(validation_folder_discarded_images)
        test_folder_images = test_folder + "/" + "images"
        makeDirectory(test_folder_images)
        test_folder_discarded_images = test_folder_images + "/" + "discarded"
        makeDirectory(test_folder_discarded_images)
        image_info_full = []
        for num_person in range(1, num_people+1):
            num_person_im_bin_tot_order = []
            num_tot = 0
            for num_bin in fold_bin_order:
                num_person_im_bin_tot_order.append(num_tot)
                num_tot += num_images_per_person_list[num_person-1][num_bin-1]

            orig_recog_numbers_person = orig_recog_numbers_list[num_person-1][:]
            for num_im in range(1, num_tot+1):
                num_bin = image_info_bin[num_person-1][num_im-1][3]
                num_bin_order = fold_bin_order.index(num_bin)
                bin_image_filename = image_info_bin[num_person-1][num_im-1][4]
                bin_image_num = int(bin_image_filename.split("_",1)[1])
                recog_numbers_person = FRO[num_person-1][num_fold-1][num_bin-1][bin_image_num-1]
                row = image_info_bin[num_person-1][num_im-1][:]
                if num_bin_order == num_bins - 1: # Test
                    validation_image = str(new_id_fold_list[num_fold-1][num_person-1]) + "_" + str(bin_image_num)
                else: # Training
                    validation_image = str(new_id_fold_list[num_fold-1][num_person-1]) + "_" + str(bin_image_num + num_person_im_bin_tot_order[num_bin_order]) 
                row.append(validation_image)
                row.append(orig_recog_numbers_person[num_im-1])
                row.insert(0, recog_numbers_person) # insert validation number in the first column
                row.append(height_list[num_person-1][num_im-1])
                row.append(time_list[num_person-1][num_im-1])
                row.append(num_bin_order) #  for sorting according to bin order
                image_info_full.append(row)

        sorted_info = sorted(image_info_full, key=lambda x: x[-1]) # sort according to bin order
        
        test_bin = fold_bin_order[num_bins-1]
        test_info = sorted_info[-1*num_im_in_bin_list[test_bin-1]:]
        test_info = [i[:-1] for i in test_info]
        sorted_test_info = sorted(test_info, key=lambda x: x[0]) # sort according to validation number
        validation_info = sorted_info[:-1*num_im_in_bin_list[test_bin-1]]
        validation_info = [i[:-1] for i in validation_info]
        sorted_validation_info = sorted(validation_info, key=lambda x: x[0]) # sort according to validation number

        for ite in sorted_test_info:
            with open(test_info_file, 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(ite)
        for ite in sorted_validation_info:
            with open(val_info_file, 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(ite)
    return sorted_validation_info, sorted_test_info

def createCrossValidationSet(num_bins, num_folds, num_people, num_mult_images,
                             orig_images_folder, seq_images_folder, bin_folder, fold_folder_set, recogniser_csv_file, validation_info_file, num_samples_per_person=None):
    """
    """
    
    image_info_sequential, mult_image_info, num_images_per_person = renameImagesToSequential(num_people, num_mult_images, orig_images_folder, seq_images_folder, num_samples_per_person=num_samples_per_person)
    image_info_bin, num_images_per_person_in_bin = divideImagesIntoBins(num_bins, num_people, num_images_per_person, seq_images_folder, bin_folder, image_info_sequential)
    fold_bin_order_list = getBinOrderInFold(num_bins) 
    recog_order_bin_list = getRecognitionOrderBin(num_bins, num_people, num_images_per_person_in_bin)
    recog_order_fold_list, new_id_fold_list = getRecognitionOrderFold(num_people, fold_bin_order_list, recog_order_bin_list)
    orig_recog_numbers_list, height_list, time_list = getRecogInfo(num_people, num_images_per_person, mult_image_info, recogniser_csv_file)
    FRO, num_im_in_bin_list = getRecognitionOrderFoldPerImage(num_bins, num_folds, num_people, recog_order_bin_list, fold_bin_order_list, num_images_per_person_in_bin)
    saveValidationInformationFile(num_bins, num_folds, num_people, image_info_bin, fold_bin_order_list, FRO, orig_recog_numbers_list, new_id_fold_list, 
                                  num_images_per_person_in_bin, num_im_in_bin_list, height_list, time_list, validation_info_file, fold_folder_set)
    
    return fold_bin_order_list, recog_order_bin_list, recog_order_fold_list

def partitionList ( lst, n ):
    # From https://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
    return [ lst[i::n] for i in xrange(n) ]

def randomiseOrder(num_bins, list_randomise, isEndUnique = False):
        
    """
        Brief: randomise order of items (unique order for each bin), with unique first item for each bin (and unique last item for each bin if isEndUnique True)
        Returns: randomised list
    """
    rand_order_list = []
    # randomly pick the first item in the list (this is to ensure that the first person/image will always be different in each bin)
    if len(list_randomise) >= num_bins:
        first_item_list = random.sample(list_randomise, num_bins)
    else:
        first_item_list = [random.choice(list_randomise) for _ in range(1,num_bins+1)]
    
    # if end has to be unique across bins (for example, for creating the order of validation set)
    if isEndUnique:
        uniqueSetFound = False
        while not uniqueSetFound:
            exclude_list_for_last = []
            last_item_list = []
            # pick last items such that it is not the same as the first item in a bin and it is not the same with previously chosen last items
            for num_bin in range(1, num_bins+1):
                first_item = first_item_list[num_bin-1]
                exclude_list_for_last.append(first_item)
                available_list = list(set([x for x in list_randomise if x not in exclude_list_for_last]))
                if not available_list:
                    break
                last_item = random.choice(available_list)
                exclude_list_for_last.pop()
                exclude_list_for_last.append(last_item)
                last_item_list.append(last_item)
                if num_bin == num_bins:
                    uniqueSetFound = True
    
    exclude_list = []                      
    for num_bin in range(1, num_bins+1):
        exclude_first = first_item_list[num_bin-1]
        exclude_list = [exclude_first]
        
        # add last item to the list to exclude such that the filling items are unique (not the same as first or last)
        if isEndUnique:
            exclude_last = last_item_list[num_bin-1]
            exclude_list.append(exclude_last)
        include_list = [x for x in list_randomise if x not in exclude_list]
        
        # randomise the order of the rest of the items
        if isEndUnique: 
            rand_order = random.sample(include_list, len(list_randomise) - 2)
        else:
            rand_order = random.sample(include_list, len(list_randomise) - 1)
        
        # insert the first item
        rand_order.insert(0, exclude_first)
        
        # insert the last item
        if isEndUnique:
            rand_order.append(exclude_last)
            
        # append the list into the list of lists
        rand_order_list.append(rand_order)
        
    return rand_order_list

    
def crossValidationCombineVar(weights = None, faceRecogThreshold = 0.4, qualityThreshold = None, args = []):
    """
    """
    # TODO: make global variable file and include it in here and artificialDataset files!!

#     [num_people, num_folds, training_folder, test_folder, 
#     cross_val_stats_train_file, cross_val_stats_test_file, 
#     db_list, init_list, recogs_list, bn, isTestData,
#     normMethod, updateMethod, probThreshold] = args

    [num_people, num_folds, training_folder, test_folder, cross_val_stats_train_file, cross_val_stats_test_file, 
    db_list, init_list, recogs_list, bn, isTestData,
    normMethod, updateMethod, probThreshold,
    isMultRecognitions, num_mult_recognitions, qualityCoefficient,
    db_file, init_recog_file, final_recog_file, cost_function_alpha] = args
                             
    avg_loss = crossValidationCombine(num_people, num_folds, training_folder, test_folder, cross_val_stats_train_file, cross_val_stats_test_file, db_list, init_list, recogs_list, bn, isTestData,
                               weights, faceRecogThreshold, qualityThreshold, normMethod, updateMethod, probThreshold,
                               isMultRecognitions, num_mult_recognitions, qualityCoefficient,
                               db_file, init_recog_file, final_recog_file, cost_function_alpha)
    return avg_loss
    
    
def crossValidationCombine(num_people, num_folds, training_folder, test_folder, cross_val_stats_train_file, cross_val_stats_test_file, db_list=None, init_list=None, recogs_list=None, bn=None, isTestData = False,
                               weights = None, faceRecogThreshold = None, qualityThreshold = None, normMethod = None, updateMethod = "evidence", probThreshold = None,
                               isMultRecognitions = False, num_mult_recognitions = None, qualityCoefficient = None,
                               db_file = None, init_recog_file = None, final_recog_file = None, valid_info_file = None, 
                               cost_function_alpha = 0.9, evidence_norm_methods = None, update_partial_params = None, isUpdateFaceLikelihoodsEqually = False,
                               isSaveRecogFiles = False, isSaveImageAn = False, isOptim = True):
    """
    """
    num_recog_train_avg = 0
    num_recog_test_avg = 0
    FER_train_avg = 0 
    FER_test_avg = 0
    num_unknown_train_avg = 0
    num_unknown_test_avg = 0
    stats_openSet_train_avg = [0, 0]
    stats_openSet_test_avg = [0, 0]
    stats_FR_train_avg = [0, 0]
    stats_FR_test_avg = [0, 0]
    avg_loss = 0
    avg_loss_test = 0
    
    for num_fold in range(1, num_folds+1):
        RB = RecognitionMemory.RecogniserBN()
        RB.setSaveRecogFiles(isSaveRecogFiles)
        RB.setUpdatePartialParams(update_partial_params)
        RB.setUpdateFaceLikelihoodsEqually(isUpdateFaceLikelihoodsEqually)
        if evidence_norm_methods is not None:
            RB.setEvidenceNormMethods(evidence_norm_methods)
            
        training_folder_fold = training_folder.replace("Training", str(num_fold) + "/Training")
        db_file_train_fold = training_folder_fold + db_file
        init_recog_train_fold = training_folder_fold + init_recog_file
        final_recog_train_fold = training_folder_fold + final_recog_file
        valid_info_train_fold = training_folder_fold + valid_info_file
        db_fold = None
        if db_list is not None:
            db_fold = db_list[num_fold-1]
        init_fold = None
        if init_list is not None:
            init_fold = init_list[0][num_fold-1]
        recogs_fold = None
        if recogs_list is not None:
            recogs_fold = recogs_list[0][num_fold-1]
            
        test_folder_fold = test_folder.replace("Test", str(num_fold) + "/Test")
        num_recog_train, FER_train, stats_openSet_train, stats_FR_train, num_unknown_train = RB.runCrossValidation(num_people, training_folder_fold, test_folder_fold, 
                               db_fold, init_fold, recogs_fold, bn, isTestData =False,
                               weights = weights, faceRecogThreshold = faceRecogThreshold, qualityThreshold =qualityThreshold, normMethod = normMethod, updateMethod = updateMethod, probThreshold = probThreshold,
                               isMultRecognitions = isMultRecognitions, num_mult_recognitions = num_mult_recognitions, qualityCoefficient = qualityCoefficient,
                               db_file = db_file_train_fold, init_recog_file = init_recog_train_fold, final_recog_file = final_recog_train_fold, valid_info_file = valid_info_train_fold,
                               isSaveRecogFiles = isSaveRecogFiles, isSaveImageAn = isSaveImageAn)
        
        # loss = cost_function_alpha*(FR_DIR_mean - DIR_mean) + (1-cost_function_alpha)*(FAR_mean - FR_FAR_mean)
#         loss = cost_function_alpha*(stats_FR_train[0] - stats_openSet_train[0]) + (1.0-cost_function_alpha)*(stats_openSet_train[1] - stats_FR_train[1])
        loss = cost_function_alpha*(1.0 - stats_openSet_train[0]) + (1.0-cost_function_alpha)*(stats_openSet_train[1])

        num_recog_train_avg += num_recog_train
        FER_train_avg += FER_train
        num_unknown_train_avg += num_unknown_train
        stats_openSet_train_avg = map(add, stats_openSet_train_avg, stats_openSet_train)
        stats_FR_train_avg = map(add, stats_FR_train_avg, stats_FR_train)
        avg_loss += loss
        
        with open(cross_val_stats_train_file, 'a') as outcsv:
            writer = csv.writer(outcsv)   
            writer.writerow([num_fold, updateMethod, probThreshold, normMethod, FER_train, stats_openSet_train[1], stats_FR_train[1], stats_openSet_train[0], stats_FR_train[0], 
                            faceRecogThreshold, qualityThreshold, weights, loss, num_recog_train, num_unknown_train])


        if not isOptim:
            # Test
            
            db_file_test_fold = test_folder_fold + db_file
            init_recog_test_fold = test_folder_fold + init_recog_file
            final_recog_test_fold = test_folder_fold + final_recog_file
            valid_info_test_fold = test_folder_fold + valid_info_file
            db_fold = None
            if db_list is not None:
                db_fold = db_list[num_fold-1]
            init_fold = None
            if init_list is not None:
                init_fold = init_list[1][num_fold-1]
            recogs_fold = None
            if recogs_list is not None:
                recogs_fold = recogs_list[1][num_fold-1]
                  
            num_recog_test, FER_test, stats_openSet_test, stats_FR_test, num_unknown_test = RB.runCrossValidation(num_people, training_folder_fold, test_folder_fold, 
                                   db_fold, init_fold, recogs_fold, bn, isTestData =True,
                                   weights = weights, faceRecogThreshold = faceRecogThreshold, qualityThreshold =qualityThreshold, normMethod = normMethod, updateMethod = updateMethod, probThreshold = probThreshold,
                                   isMultRecognitions = isMultRecognitions, num_mult_recognitions = num_mult_recognitions, qualityCoefficient = qualityCoefficient,
                                   db_file = db_file_test_fold, init_recog_file = init_recog_test_fold, final_recog_file = final_recog_test_fold, valid_info_file = valid_info_test_fold,
                                   isSaveRecogFiles = isSaveRecogFiles, isSaveImageAn = isSaveImageAn)
            
            loss_test = cost_function_alpha*(1.0 - stats_openSet_test[0]) + (1.0-cost_function_alpha)*(stats_openSet_test[1])
              
            num_recog_test_avg += num_recog_test
            FER_test_avg += FER_test
            num_unknown_test_avg += num_unknown_test
            stats_openSet_test_avg = map(add, stats_openSet_test_avg, stats_openSet_test)
            stats_FR_test_avg = map(add, stats_FR_test_avg, stats_FR_test)
            avg_loss_test += loss_test
    
            with open(cross_val_stats_test_file, 'a') as outcsv:
                writer = csv.writer(outcsv)   
                writer.writerow([num_fold, updateMethod, probThreshold, normMethod, FER_test, stats_openSet_test[1], stats_FR_test[1], stats_openSet_test[0], stats_FR_test[0], 
                                faceRecogThreshold, qualityThreshold, weights, loss_test, num_recog_test, num_unknown_test])    

    num_recog_train_avg /= num_folds
    FER_train_avg /= num_folds
    num_unknown_train_avg /= num_folds
    stats_openSet_train_avg = [x/num_folds for x in stats_openSet_train_avg ]
    stats_FR_train_avg = [x/num_folds for x in stats_FR_train_avg ]
    avg_loss /= num_folds
    
    with open(cross_val_stats_train_file.replace(".csv", "_avg.csv"), 'a') as outcsv:
        writer = csv.writer(outcsv)   
        writer.writerow([updateMethod, probThreshold, normMethod, FER_train_avg, stats_openSet_train_avg[1], stats_FR_train_avg[1], stats_openSet_train_avg[0], stats_FR_train_avg[0], 
                         faceRecogThreshold, qualityThreshold, weights, avg_loss, num_recog_train_avg, num_unknown_train_avg])
    
    if not isOptim:
        num_recog_test_avg /= num_folds
        FER_test_avg /= num_folds
        num_unknown_test_avg /= num_folds    
        stats_openSet_test_avg = [x/num_folds for x in stats_openSet_test_avg ]
        stats_FR_test_avg = [x/num_folds for x in stats_FR_test_avg ]
        avg_loss_test /= num_folds
          
        with open(cross_val_stats_test_file.replace(".csv", "_avg.csv"), 'a') as outcsv:
            writer = csv.writer(outcsv)   
            writer.writerow([updateMethod, probThreshold, normMethod, FER_test_avg, stats_openSet_test_avg[1], stats_FR_test_avg[1], stats_openSet_test_avg[0], stats_FR_test_avg[0], 
                             faceRecogThreshold, qualityThreshold, weights, avg_loss_test, num_recog_test_avg, num_unknown_test_avg])
    
    return avg_loss

def repeatedKFoldCrossValidation(num_people, num_repeat, num_folds, 
                                 cross_val_folder, cross_val_folder_open, training_folder, test_folder,
                                 db_file, init_recog_file, final_recog_file, valid_info_file, analysis_file = None,
                                 time_type = None, sample_size = None, 
                                 weights = None, normMethod = None, updateMethod = "evidence", model_name = None, 
                                 isSaveRecogFiles = False, isSaveImageAn = False, 
                                 cost_function_alpha = 0.9, faceRecogThreshold = None, qualityThreshold = None, probThreshold = None,
                                 evidence_norm_methods = None, update_partial_params = None, isUpdateFaceLikelihoodsEqually = False):
    """
    """


    if model_name is not None:
        learning_method = model_name
    elif updateMethod == "none":
        learning_method = "none"
    else:
        learning_method = "online"

    for num_fold in range(1, num_folds+1):
        db_fold = None
        init_fold = None
        recogs_fold = None
        bn = None
        RB = RecognitionMemory.RecogniserBN()
        for eval_folder in ["training", "closed-test", "open", "open-closed"]:
            training_folder_fold = cross_val_folder + str(num_fold) + "/" + training_folder
            test_folder_fold = cross_val_folder + str(num_fold) + "/" + test_folder
            if analysis_file is None:
                cross_val_stats_file = eval_folder + str(num_repeat) + ".csv"
            else:
                cross_val_stats_file = analysis_file.replace(".csv", "-" + eval_folder + str(num_repeat) + ".csv")
            if eval_folder == "training":
                isTestData = False
                isOpenSet = False
                eval_folder_fold = training_folder_fold
            elif eval_folder == "closed-test":
                isTestData = True
                isOpenSet = False
                eval_folder_fold = test_folder_fold
            elif eval_folder == "open":
                isTestData = False
                isOpenSet = True
                training_folder_fold = cross_val_folder_open + str(num_fold) + "/" + training_folder
                test_folder_fold = cross_val_folder_open + str(num_fold) + "/" + test_folder
                eval_folder_fold = training_folder_fold
            else:
                isOpenSet = True
                isTestData = True
                training_folder_fold = cross_val_folder_open + str(num_fold) + "/" + training_folder
                test_folder_fold = cross_val_folder_open + str(num_fold) + "/" + test_folder
                eval_folder_fold = test_folder_fold

            RB.setSaveRecogFiles(isSaveRecogFiles)
            RB.setUpdatePartialParams(update_partial_params)
            RB.setUpdateFaceLikelihoodsEqually(isUpdateFaceLikelihoodsEqually)
            if evidence_norm_methods is not None:
                RB.setEvidenceNormMethods(evidence_norm_methods)
                
            db_file_train_fold = eval_folder_fold + db_file
            init_recog_train_fold = eval_folder_fold + init_recog_file
            final_recog_train_fold = eval_folder_fold + final_recog_file
            valid_info_train_fold = eval_folder_fold + valid_info_file

            num_recog, FER, stats_openSet, stats_FR, num_unknown = RB.runCrossValidation(num_people, training_folder_fold, test_folder_fold, 
                                   db_fold, init_fold, recogs_fold, bn, isTestData = isTestData,isOpenSet =isOpenSet,
                                   weights = weights, faceRecogThreshold = faceRecogThreshold, qualityThreshold =qualityThreshold, normMethod = normMethod, updateMethod = updateMethod, probThreshold = probThreshold,
                                   isMultRecognitions = False, num_mult_recognitions = 3, qualityCoefficient = None,
                                   db_file = db_file_train_fold, init_recog_file = init_recog_train_fold, final_recog_file = final_recog_train_fold, valid_info_file = valid_info_train_fold,
                                   isSaveRecogFiles = isSaveRecogFiles, isSaveImageAn = isSaveImageAn)
            
            loss = cost_function_alpha*(1.0 - stats_openSet[0]) + (1.0-cost_function_alpha)*(stats_openSet[1])
            F_Loss = cost_function_alpha*(1.0 - stats_FR[0]) + (1.0-cost_function_alpha)*(stats_FR[1])
            
            with open(cross_val_stats_file, 'a') as outcsv:
                writer = csv.writer(outcsv)   
                writer.writerow([num_repeat, num_fold, time_type, sample_size, learning_method, normMethod, stats_openSet[1], stats_FR[1], stats_openSet[0], stats_FR[0], loss, F_Loss, num_recog, num_unknown, RB.init_recognising_time])
                    
                    
def combineOptimResultsIntoFile(main_folder, optim_weight_folders, update_method_folders, dataset_folders, training_avg_file, test_avg_file, combined_file, cost_function_alpha):
    """
    """
    for datasetFolder in dataset_folders:
        combined_dataset_file = main_folder + combined_file.replace(".csv", "_" + datasetFolder.replace("/","") + ".csv")
        with open(combined_dataset_file, 'wb') as outcsv:  
            writer = csv.writer(outcsv)   
            writer.writerow(["", "I_L", "F_L", "S_L", "I_FAR", "F_FAR", "S_FAR", "I_DIR", "F_DIR", "S_DIR"])
        for f_file in [training_avg_file, test_avg_file]:
            with open(combined_dataset_file, 'a') as outcsv:  
                writer = csv.writer(outcsv)
                if f_file in training_avg_file:
                    writer.writerow(["OPEN-SET TEST"])
                if f_file in test_avg_file:
                    writer.writerow(["CLOSED-SET TEST"])
            for optimWeightFolder in optim_weight_folders:
                optim_weight_ab = ""
                if "gaussian" in optimWeightFolder:
                    optim_weight_ab = "gt:"
                elif "uniform" in optimWeightFolder:
                    optim_weight_ab = "ut:"
                    
                for updateMethodFolder in update_method_folders:
                    
                    dir_file = main_folder + optimWeightFolder + updateMethodFolder + datasetFolder
                    if os.path.isdir(dir_file):
                        info_file = pandas.read_csv(dir_file + f_file, usecols = {"I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1", "Weights", "Loss"}, converters={"Weights": ast.literal_eval}).values.tolist()
                        f_loss = info_file[0][1]*(1-cost_function_alpha) + (1-info_file[0][3])*cost_function_alpha
                        method_name = optim_weight_ab
                        if "MMIBN-OL" in updateMethodFolder:
                            method_name += "H-OL"
                        elif "EvidenceT-OL" in updateMethodFolder:
                            method_name += "H-OL-T"
                        elif "MMIBN-eq" in updateMethodFolder:
                            method_name += "H-Eq"
                        elif "MMIBN" in updateMethodFolder:
                            method_name += "H"
                        elif "minmax" in updateMethodFolder:
                            method_name += "MM"
                        elif "softmax-OL" in updateMethodFolder:
                            method_name += "SM-OL"
                        soft_row = 0
                        all_info_row = 1
                        if info_file[1][4][0] == 0.0:
                            soft_row = 1
                            all_info_row = 0
                        row = [method_name, info_file[all_info_row][-1], f_loss, info_file[soft_row][-1], info_file[all_info_row][0], info_file[soft_row][1], info_file[soft_row][0], info_file[all_info_row][2], info_file[soft_row][3], info_file[soft_row][2]]
                        with open(combined_dataset_file, 'a') as outcsv:  
                            writer = csv.writer(outcsv)
                            writer.writerow(row)
                    
def combineOptimResultsIntoFileWithStd(num_folds, main_folder, optim_weight_folders, update_method_folders, dataset_folders, training_folds_file, test_folds_file, combined_mean_std_file, cost_function_alpha):
    """
    """
    soft_values = []
    all_values = []
    fr_values = []
    num_recog_values = []
    num_unknown_values = []
    for datasetFolder in dataset_folders:
        soft_values_ds = []
        all_values_ds = []
        fr_values_ds = []
        num_recog_ds = []
        num_unknown_ds = []
        combined_dataset_file = main_folder + combined_mean_std_file.replace(".csv", "_" + datasetFolder.replace("/","") + ".csv")
        with open(combined_dataset_file, 'wb') as outcsv:  
            writer = csv.writer(outcsv)   
            writer.writerow(["", "I_L", "I_L_STD", "F_L", "F_L_STD", "S_L", "S_L_STD", "I_FAR", "I_FAR_STD", "F_FAR", "F_FAR_STD", "S_FAR", "S_FAR_STD", "I_DIR", "I_DIR_STD", "F_DIR", "F_DIR_STD", "S_DIR", "S_DIR_STD"])
        for f_file in [training_folds_file, test_folds_file]:
            with open(combined_dataset_file, 'a') as outcsv:  
                writer = csv.writer(outcsv)
                if f_file in training_folds_file:
                    writer.writerow(["TRAINING"])
                if f_file in test_folds_file:
                    writer.writerow(["CLOSED-SET TEST"])
            for optimWeightFolder in optim_weight_folders:
                optim_weight_ab = ""
                if "gaussian" in optimWeightFolder:
                    optim_weight_ab = "gt:"
                elif "uniform" in optimWeightFolder:
                    optim_weight_ab = "ut:"
                    
                for updateMethodFolder in update_method_folders:
                    
                    dir_file = main_folder + optimWeightFolder + updateMethodFolder + datasetFolder
                    
                    if os.path.isdir(dir_file):
                        
                        
                        df_info_file = pandas.read_csv(dir_file + f_file, usecols = {"I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1", "Weights", "Loss", "Num_recognitions", "Num_registered"}, converters={"Weights": ast.literal_eval})
                        soft_row_start = 0
                        all_row_start = num_folds
                        
                        if df_info_file.iloc[0, 4][0] == 1.0:
                            soft_row_start = num_folds
                            all_row_start = 0
                        
                        if f_file in training_folds_file:
                            soft_values_ds.append(df_info_file[[0,2,5]].iloc[list(range(soft_row_start, soft_row_start+num_folds))].values.tolist())
                        soft_mean_values = df_info_file[[0,2,5]].iloc[list(range(soft_row_start, soft_row_start+num_folds))].mean(axis=0).values.tolist()
                        soft_std_values = df_info_file[[0,2,5]].iloc[list(range(soft_row_start, soft_row_start+num_folds))].std(axis=0).values.tolist()
                        
                        if f_file in training_folds_file:
                            all_values_ds.append(df_info_file[[0,2,5]].iloc[list(range(all_row_start, all_row_start+num_folds))].values.tolist())
                        all_mean_values = df_info_file[[0,2,5]].iloc[list(range(all_row_start, all_row_start+num_folds))].mean(axis=0).values.tolist()
                        all_std_values = df_info_file[[0,2,5]].iloc[list(range(all_row_start, all_row_start+num_folds))].std(axis=0).values.tolist()
                        
                        fr_mean_values = df_info_file[[1,3]].iloc[list(range(soft_row_start, soft_row_start+num_folds))].mean(axis=0).values.tolist()
                        fr_std_values = df_info_file[[1,3]].iloc[list(range(soft_row_start, soft_row_start+num_folds))].std(axis=0).values.tolist()
                        
                        fr_far_values = df_info_file.iloc[list(range(soft_row_start, soft_row_start+num_folds)),1].values.tolist()
                        fr_dir_values = df_info_file.iloc[list(range(soft_row_start, soft_row_start+num_folds)),3].values.tolist()
                        fr_loss_values = np.zeros(num_folds)
                        
                        if f_file in training_folds_file:
                            num_recog_ds = df_info_file.iloc[list(range(soft_row_start, soft_row_start+num_folds)), 6].values.tolist()
                            num_unknown_ds = df_info_file.iloc[list(range(soft_row_start, soft_row_start+num_folds)), 7].values.tolist()
                        
                        for num_fold in range(0, num_folds):
                            fr_loss_values[num_fold] = fr_far_values[num_fold]*(1-cost_function_alpha) + (1- fr_dir_values[num_fold])*cost_function_alpha
                        
                        fr_mean_values.append(np.mean(fr_loss_values, axis=0))
                        fr_std_values.append(np.std(fr_loss_values, ddof=1, axis=0))
                        
                        if f_file in training_folds_file:
                            fr_values_ds.append([list(a) for a in zip(fr_far_values, fr_dir_values, fr_loss_values)])
                        
                        method_name = optim_weight_ab
                        if "MMIBN-OL" in updateMethodFolder:
                            method_name += "H-OL"
                        elif "EvidenceT-OL" in updateMethodFolder:
                            method_name += "H-OL-T"
                        elif "MMIBN-eq" in updateMethodFolder:
                            method_name += "H-Eq"
                        elif "MMIBN" in updateMethodFolder:
                            method_name += "H"
                        elif "minmax" in updateMethodFolder:
                            method_name += "MM"
                        elif "softmax-OL" in updateMethodFolder:
                            method_name += "SM-OL"
                        row = [method_name, 
                               all_mean_values[-1], all_std_values[-1], fr_mean_values[-1], fr_std_values[-1], soft_mean_values[-1], soft_std_values[-1], #loss
                               all_mean_values[0], all_std_values[0], fr_mean_values[0], fr_std_values[0], soft_mean_values[0], soft_std_values[0], #FAR
                               all_mean_values[1], all_std_values[1], fr_mean_values[1], fr_std_values[1], soft_mean_values[1], soft_std_values[1]] #DIR
                        with open(combined_dataset_file, 'a') as outcsv:  
                            writer = csv.writer(outcsv)
                            writer.writerow(row)
        soft_values.append(soft_values_ds)
        all_values.append(all_values_ds)
        fr_values.append(fr_values_ds)
        num_recog_values.append(num_recog_ds)
        num_unknown_values.append(num_unknown_ds)
    
    return all_values, fr_values, soft_values, num_recog_values, num_unknown_values
                        
def getOpenSetTestResultsFromCombined(main_folder, dataset_folders, update_method_folders, training_folds_file, combined_mean_std_file, num_folds,
                                      num_recog_values_train, num_unknown_values_train, num_recog_values_train_test, num_unknown_values_train_test,
                                      train_values, train_test_values, cost_function_alpha):
    
    ds_counter = 0
    
    for datasetFolder in dataset_folders:
        
        combined_dataset_file = main_folder + combined_mean_std_file.replace(".csv", "_" + datasetFolder.replace("/","") + ".csv")
        
        with open(combined_dataset_file, 'a') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["OPEN-SET TEST"])
        update_method_counter = 0
        for updateMethodFolder in update_method_folders:
            open_values = []
            
            for param in range(len(train_values)):
                train = train_values[param][ds_counter][update_method_counter][:]
                train_test = train_test_values[param][ds_counter][update_method_counter][:]
                
                dir_open = np.zeros(num_folds)
                far_open = np.zeros(num_folds)
                loss_open = np.zeros(num_folds)

                for num_fold in range(0, num_folds):
                    num_recog_train = num_recog_values_train[ds_counter][num_fold]
                    num_unknown_train = num_unknown_values_train[ds_counter][num_fold]
                    
                    num_recog_train_test = num_recog_values_train_test[ds_counter][num_fold]
                    num_unknown_train_test = num_unknown_values_train_test[ds_counter][num_fold]
                    
                    num_recog_open = num_recog_train_test - num_recog_train
                    num_unknown_open = num_unknown_train_test- num_unknown_train
                    
                    dir_open[num_fold] = (train_test[num_fold][1]* (num_recog_train_test - num_unknown_train_test) - (train[num_fold][1] * (num_recog_train - num_unknown_train)))/(num_recog_open - num_unknown_open) 
                    far_open[num_fold] = (train_test[num_fold][0]* num_unknown_train_test - (train[num_fold][0] * num_unknown_train))/num_unknown_open
                    loss_open[num_fold] = far_open[num_fold]*(1-cost_function_alpha) + (1- dir_open[num_fold])*cost_function_alpha
            
                dir_mean = np.mean(dir_open, axis=0)
                dir_std = np.std(dir_open, ddof=1, axis=0)
                
                far_mean = np.mean(far_open, axis=0)
                far_std = np.std(far_open, ddof=1, axis=0)
                
                loss_mean = np.mean(loss_open, axis=0)
                loss_std = np.std(loss_open, ddof=1, axis=0)
                
                open_values.append([far_mean, far_std, dir_mean, dir_std, loss_mean, loss_std])
            method_name = ""
            if "MMIBN-OL" in updateMethodFolder:
                method_name += "H-OL"
            elif "EvidenceT-OL" in updateMethodFolder:
                method_name += "H-OL-T"
            elif "MMIBN-eq" in updateMethodFolder:
                method_name += "H-Eq"
            elif "MMIBN" in updateMethodFolder:
                method_name += "H"
            elif "minmax" in updateMethodFolder:
                method_name += "MM"
            elif "softmax-OL" in updateMethodFolder:
                method_name += "SM-OL"
            
            row = [method_name, 
                   open_values[0][4], open_values[1][5], open_values[1][4], open_values[1][5], open_values[2][4], open_values[2][5], #loss
                   open_values[0][0], open_values[1][1], open_values[1][0], open_values[1][1], open_values[2][0], open_values[2][1], #FAR
                   open_values[0][2], open_values[1][3], open_values[1][2], open_values[1][3], open_values[2][2], open_values[2][3] #DIR
                   ]
            with open(combined_dataset_file, 'a') as outcsv:  
                writer = csv.writer(outcsv)
                writer.writerow(row)
            update_method_counter += 1
        ds_counter += 1                    
if __name__ == "__main__":
    
    """
    num_people = 14
    num_bins = 5
    num_samples_per_person = 65
#     num_samples_per_person = None

    num_mult_images = 3
    num_folds = 5
    cross_validation_folder = "plymouth_dataset/"
    train_folder = "Training/"
    test_folder = "Test/"
    
#     optim_folder = ""
    optim_folder = "optimised_weights/N65/"
#     optim_folder = "optimised_weights/Nall/"

    orig_images_folder = cross_validation_folder + "images_wrong_people_removed/"
#     orig_images_folder = cross_validation_folder + "images_cleaned/"

    seq_images_folder = cross_validation_folder + "images_sequential/"
#     seq_images_folder = cross_validation_folder + "images_sequential_cleaned/"

    bin_folder = cross_validation_folder + optim_folder + "bins/"
    fold_folder_set = cross_validation_folder + optim_folder + "folds/"
    recogniser_csv_file = cross_validation_folder + "RecogniserBN_orig.csv"
    validation_info_file = "validation_info_fold.csv"
    fold_bin_order_list, recog_order_bin_list, recog_order_fold_list = createCrossValidationSet(num_bins, num_folds, num_people, num_mult_images, 
                            orig_images_folder, seq_images_folder, bin_folder, fold_folder_set, recogniser_csv_file, validation_info_file, num_samples_per_person)
    
    
    RB = RecognitionMemory.RecogniserBN()
     
    image_path = "/home/nao/dev/src/recognition/"
    RB.connectToRobot("127.0.0.1", useSpanish=False, isImageFromTablet = True, isMemoryOnRobot=True, imagePath = image_path)
      
    for num_fold in range(1, num_folds+1):
 
        print "Fold:" + str(num_fold)
        fold_folder = fold_folder_set + str(num_fold) + "/"
        training_folder = fold_folder + train_folder
        tes_folder = fold_folder + test_folder
        start_fold_time = time.time()
        # Training
        db_new, num_recog_in_fold = RB.runTestOnRobotValidation(num_people, cross_validation_folder, training_folder, tes_folder, bin_folder, validation_info_file, 
                                 isTestData = False, db_order_list = None, 
                                 isMultRecognitions = False, num_mult_recognitions = None, isRobotLearning = True, qualityCoefficient = None, weights = None,
                                comparison_file = None, stats_file = None)
        # Test
        db_new_2, num_recog_in_fold_test = RB.runTestOnRobotValidation(num_people, cross_validation_folder, training_folder, tes_folder, bin_folder, validation_info_file, 
                                 isTestData = True, db_order_list = db_new, 
                                 isMultRecognitions = False, num_mult_recognitions = None, isRobotLearning = False, qualityCoefficient = None, weights = None,
                                comparison_file = None, stats_file = None)
         
        print "fold time:" + str(time.time() - start_fold_time)
        print "-"*40
    
    
    """
    main_folder = "imdb_crop/"
#     main_folder = "plymouth_dataset/"
#     num_people = 14
    num_people = 100

    num_folds = 5
#     optim_folder = "cross_validation_test/MMIBN-OL/"
    optim_folder_list = ["criss_cross_weights/uniform/cross_validation/MMIBN/", "criss_cross_weights/uniform/cross_validation_test_only/MMIBN/", "criss_cross_weights/uniform/cross_validation_test+training/MMIBN/"]
#     optim_folder_list = ["cross_validation_optim_weights/Nall_uniformT/None-eq/"]
    # optim_folder = "optim/"
    # cross_val_folder = "hybrid_tries/Nall_gaussianT/"
    
    folds_folder = "folds/"
    train_folder = "Training/"
    test_folder = "Test/"
    faceRecogThreshold = 0.4
    probThreshold = 1.0e-75

    init_file = "InitialRecognition_data.csv"
    recog_file = "RecogniserBN_data.csv"
    db_file = "db_data.csv"
    valid_info_file = "validation_info_fold_data.csv"
    cost_function_alpha = 0.9
    stats_header = ["Fold", "Evidence_method", "Threshold", "Norm_method", "FER", 
                    "I_FAR", "F_FAR", "I_DIR_1", "F_DIR_1", 
                    "Face_threshold", "Quality_threshold", "Weights", "Loss",
                    "Num_recognitions", "Num_registered"]

    norm_methods = ["softmax", "minmax", "tanh", "norm-sum", "hybrid"]
#     update_methods = ["none", "evidence"]
    update_methods = ["none"]
#     cross_val_folders = ["N65/", "Nall/"]
    cross_val_folders = ["N10_gaussianT/", "Nall_gaussianT/", "N10_uniformT/", "Nall_uniformT/"]
#     cross_val_folders = ["N10_gaussianT/"]

    norm_methods = [norm_methods[4]]
    face_weights = [0.0, 1.0]
#     face_weights = [face_weights[1]]
    count_f = 0
    for faceWeight in face_weights:
        isSaveRecogFiles = True
        isSaveImageAn = False
#         if faceWeight == 0.0:
#             isSaveImageAn = False
#         else:
#             isSaveImageAn = True
        for optim_folder in optim_folder_list:
            if "OL" in optim_folder:
                update_methods = ["evidence"]
            else:
                update_methods = ["none"]
                
            if "EvidenceT" in optim_folder:
                update_partial_params = ["T"]
            else:
                update_partial_params = None
            
            if "-eq" in optim_folder:
                isUpdateFaceLikelihoodsEqually = True
            else:
                isUpdateFaceLikelihoodsEqually = False
                
            for cross_val_folder in cross_val_folders:
                base_folder = main_folder + optim_folder + cross_val_folder
                print base_folder
                cross_val_stats_train_file = base_folder + "cvs_train.csv"
                cross_val_stats_test_file = base_folder + "cvs_test.csv"
                optim_file = base_folder + "optim_params.csv"
                 
                if count_f == 0:
                     with open(cross_val_stats_train_file, 'wb') as outcsv:  
                         writer = csv.writer(outcsv)   
                         writer.writerow(stats_header)
                     
                     with open(cross_val_stats_train_file.replace(".csv","_avg.csv"), 'wb') as outcsv:
                         writer = csv.writer(outcsv)   
                         writer.writerow(stats_header[1:])
                           
                     with open(cross_val_stats_test_file, 'wb') as outcsv:
                         writer = csv.writer(outcsv)   
                         writer.writerow(stats_header)
                     
                     with open(cross_val_stats_test_file.replace(".csv","_avg.csv"), 'wb') as outcsv:
                         writer = csv.writer(outcsv)   
                         writer.writerow(stats_header[1:])
             
        #         weights = [0.0, 0.35 , 0.58 , 0.312, 0.275]
        #         qualityThreshold = 0.059
                optim_values = pandas.read_csv(optim_file, dtype={"Evidence_method": object, "Norm_method": object, "Optim_params":object}, usecols = {"Evidence_method", "Norm_method", "Optim_params"})
                
                for updateMethod in update_methods:
                    for normMethod in norm_methods:
                        if "MMIBN-eq" in optim_folder:
                            updateMethodName = "none-eq"
                        elif "EvidenceT" in optim_folder:
                            updateMethodName = "evidence-partT"
                        else:
                            updateMethodName = updateMethod
                        optim_specs_val = optim_values[(optim_values['Evidence_method'] == str(updateMethodName)) & (optim_values['Norm_method'] == str(normMethod))].Optim_params.tolist()[0]
                        optim_specs = ast.literal_eval(optim_specs_val.replace('array(','').replace(')',''))
#                         optim_specs = [[0.2, 0.0, 0.1 , 0.0, 0.11]]
                        for optimSpec in optim_specs:
                            qualityThreshold = optimSpec[-1]
    
                            weights = optimSpec[:-1]
                            weights.insert(0, faceWeight)
                            crossValidationCombine(num_people, num_folds, base_folder+folds_folder+train_folder, base_folder+folds_folder+test_folder, cross_val_stats_train_file, cross_val_stats_test_file, 
                                           db_list=None, init_list=None, recogs_list=None, bn=None, isTestData = False,
                                           weights = weights, faceRecogThreshold = faceRecogThreshold, qualityThreshold = qualityThreshold, normMethod = normMethod, updateMethod = updateMethod, probThreshold = probThreshold,
                                           isMultRecognitions = False, num_mult_recognitions = None, qualityCoefficient = None,
                                           db_file = db_file, init_recog_file = init_file, final_recog_file = recog_file, valid_info_file = valid_info_file, 
                                           cost_function_alpha = cost_function_alpha, evidence_norm_methods = None, update_partial_params = update_partial_params, isUpdateFaceLikelihoodsEqually = isUpdateFaceLikelihoodsEqually,
                                           isSaveRecogFiles = isSaveRecogFiles, isSaveImageAn = isSaveImageAn, isOptim= False)
        count_f += 1
   
    
#     combineOptimResultsIntoFile("plymouth_dataset/cross_validation_optim_weights/", ["Nall_gaussianT/", "Nall_uniformT/", "SRIW/"], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "None-eq/", "minmax/", "softmax-OL/"], ["N65/", "Nall/"], "cvs_train_avg.csv", "cvs_test_avg.csv", "combined_optim_results.csv", 0.9)
#     combineOptimResultsIntoFile("imdb_crop/criss_cross_weights/uniform/cross_validation/", [""], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], "cvs_train_avg.csv", "cvs_test_avg.csv", "combined_optim_results.csv", 0.9)
#     combineOptimResultsIntoFile("imdb_crop/criss_cross_weights/uniform/cross_validation_test_only/", [""], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], "cvs_train_avg.csv", "cvs_test_avg.csv", "combined_optim_results.csv", 0.9)
#     combineOptimResultsIntoFile("imdb_crop/criss_cross_weights/uniform/cross_validation_test+training/", [""], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], "cvs_train_avg.csv", "cvs_test_avg.csv", "combined_optim_results.csv", 0.9)
#    num_folds = 5
#    all_values_train, fr_values_train, soft_values_train, num_recog_values_train, num_unknown_values_train = combineOptimResultsIntoFileWithStd(num_folds, "imdb_crop/cross_validation/", [""], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], "cvs_train.csv", "cvs_test.csv", "combined_optim_results_with_std.csv", 0.9)
#    combineOptimResultsIntoFileWithStd(num_folds, "imdb_crop/cross_validation_test_only/", [""], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], "cvs_train.csv", "cvs_test.csv", "combined_optim_results_with_std.csv", 0.9)
#    all_values_train_test, fr_values_train_test, soft_values_train_test, num_recog_values_train_test, num_unknown_values_train_test = combineOptimResultsIntoFileWithStd(num_folds, "imdb_crop/cross_validation_test+training/", [""], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], "cvs_train.csv", "cvs_test.csv", "combined_optim_results_with_std.csv", 0.9)

#    getOpenSetTestResultsFromCombined("imdb_crop/cross_validation_test+training/", ["N10_uniformT/", "N10_gaussianT/", "Nall_uniformT/", "Nall_gaussianT/"], ["MMIBN/", "MMIBN-OL/", "EvidenceT-OL/", "MMIBN-eq/"], "cvs_train.csv", "combined_optim_results_with_std.csv", num_folds, 
#                                      num_recog_values_train, num_unknown_values_train, num_recog_values_train_test, num_unknown_values_train_test, 
#                                       [all_values_train, fr_values_train, soft_values_train], [all_values_train_test, fr_values_train_test, soft_values_train_test], 0.9)
