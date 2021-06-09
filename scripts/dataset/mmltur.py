# coding: utf-8

#! /usr/bin/env python

#========================================================================================================#
#  Copyright (c) 2017-present, Bahar Irfan                                                               #
#                                                                                                        #                      
#  mmltur script creates a smaller version of the dataset, using the 6th repeat second fold (used for the#
# evaluations).                                                                                          #
#                                                                                                        #
#  Please cite the following work if using this code:                                                    #
#    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (under review), 'Multi-modal Open World User #
#    Identification', ACM Transactions on Human-Robot Interaction (THRI).                                #
#                                                                                                        #            
#  Each script in this project is under the GNU General Public License.                                  #
#========================================================================================================#

import pandas
import csv
import os
import ast
import shutil

main_folder = "../../dataset/"
orig_folder = main_folder + "BN"
init_recog_file_name = "InitialRecognition.csv"
final_recog_file_name = "RecogniserBN.csv"
valid_info_file_name = "validation_info_fold.csv"
db_file_name = "db.csv"
orig_image_folder = main_folder + "IMDB_chosen_images"
mmltur_folder = main_folder + "MMLTUR-dataset"
mmltur_images_folder = "images"
mmltur_biometric_file = "multiModalBiometricData.csv"
mmltur_info_file = "info.csv"
datasets = ["N10_gaussianT", "N10_uniformT", "Nall_gaussianT", "Nall_uniformT"]
open_sets = ["train", "open"]
train_test_sets = ["Training", "Test"]
folds_folder = "folds"
selected_repeat = "6"
selected_fold = "2"
num_users = 200
for dataset in datasets:
    mmltur_path = os.path.join(mmltur_folder, dataset)
    multimodal_data = []
    info_data = []
    num_recog = 1
    user_appearance = [1 for _ in range(0, num_users)]
    im_path = os.path.join(mmltur_path, mmltur_images_folder)
    if os.path.isdir(im_path):
        shutil.rmtree(im_path)
    os.makedirs(im_path)
    for oset in open_sets:
        for tset in train_test_sets:
            fold_path = os.path.join(orig_folder, oset, dataset, folds_folder, selected_fold, tset)
            init_recog_file = os.path.join(fold_path, init_recog_file_name)
            final_recog_file = os.path.join(fold_path,final_recog_file_name)
            valid_info_file = os.path.join(fold_path,valid_info_file_name)
            db_file = os.path.join(fold_path,db_file_name)

            df_init = pandas.read_csv(init_recog_file, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval}, usecols ={"F", "G",  "A", "H", "T", "N"})
            init_list = df_init.values.tolist()
            
            df_final = pandas.read_csv(final_recog_file, dtype={"I": object}, converters={"F": ast.literal_eval, "G": ast.literal_eval, "A": ast.literal_eval, "H": ast.literal_eval, "T": ast.literal_eval})
            recogs_list = df_final.values.tolist()

            df_valid = pandas.read_csv(valid_info_file, converters={"Height": ast.literal_eval, "Time": ast.literal_eval}, usecols ={"Original_image", "Height", "Time"})
            valid_list = df_valid.values.tolist()

            df_db = pandas.read_csv(db_file, converters={"height": ast.literal_eval, "times": ast.literal_eval, "occurrence": ast.literal_eval})
            db_list = df_db.values.tolist()

            count_recogs = 0
            
            while count_recogs < len(recogs_list): 
                recog_results = recogs_list[count_recogs][:]
                isRegistering = recog_results[6]
                recog_results[3] = [int(recog_results[3][0]), recog_results[3][1]]
                user_id = recog_results[0]
                init_recog_results = init_list[count_recogs][:]
                init_recog_results[2] = [int(init_recog_results[2][0]), init_recog_results[2][1]]
                init_recog_results.insert(0, 0)
                init_recog_results.insert(-1, isRegistering)
                if isRegistering:
                    multimodal_data.append(init_recog_results)
                multimodal_data.append(recog_results)

                orig_image_file = os.path.join(orig_image_folder, valid_list[count_recogs][0])
                image_id = user_id + "_" + str(user_appearance[int(user_id)-1]) + ".jpg"
                info_data.append([num_recog, user_id, image_id, valid_list[count_recogs][1], valid_list[count_recogs][2], isRegistering])
                shutil.copy2(orig_image_file, os.path.join(mmltur_path, mmltur_images_folder, image_id))

                user_appearance[int(user_id)-1] += 1
                count_recogs+= 1
                num_recog+= 1

    with open(os.path.join(mmltur_path, mmltur_biometric_file), 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["I", "F", "G", "A", "H", "T", "R", "N"])
        for mm_data in multimodal_data:
            writer.writerow(mm_data)
    with open(os.path.join(mmltur_path, db_file_name), 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["id", "name", "gender", "age", "height", "times", "occurrence"])
        for db_data in db_list:
            writer.writerow(db_data)
    with open(os.path.join(mmltur_path, mmltur_info_file), 'wb') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["N", "id", "image", "height", "time", "R"])
        for i_data in info_data:
            writer.writerow(i_data)


