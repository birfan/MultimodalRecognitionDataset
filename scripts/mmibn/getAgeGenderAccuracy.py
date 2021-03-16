
import RecognitionMemory as RM
import math

def getAverageAgeStddev(RB, main_folder, dataset_folder, num_folds, num_people):
    prev_estimates_mean = None
    total_estimates_mean = [[] for _ in range(0, num_people)]
    stddev_true_mean = [0.0 for i in range(0, num_people)]
    stddev_est_list = [0.0 for i in range(0, num_people)]  
    stddev_true_mean = [0.0 for i in range(0, num_people)]
    avg_val = [0.0 for i in range(0, num_people)]
    avg_stddev_true = 0
    avg_stddev_est = 0
    overall_stddev_true = 0
    for num_fold in range(1,num_folds+1):
        for fold_set in ["train/", "open/"]:
            for fold_t in ["Training/", "Test/"]:
                recog_folder = main_folder + fold_set + dataset_folder + "folds/" + str(num_fold) + "/" + fold_t
                RB.resetFilePaths()
                RB.setFilePaths(recog_folder)                
                _, estimates_mean = RB.getAgeStddev(isReturnWithoutAveraging=True)
                if fold_t == "Test/":
                    estimates_mean = [a + b for (a, b) in zip(prev_estimates_mean, estimates_mean)]
                    if fold_set == "train/":
                        total_estimates_mean[0:100] = estimates_mean
                    else:
                        total_estimates_mean[100:200] = estimates_mean[100:200]
                        
                        for p_id_index in range(1,len(total_estimates_mean)+1):
                            for est_mean in total_estimates_mean[p_id_index-1]:
                                stddev_true_mean[p_id_index-1] += math.pow(est_mean - RB.ages[p_id_index], 2)
                            avg_val = [sum(a) / float(len(a)) for a in total_estimates_mean]

                            
                            stddev_true_mean[p_id_index-1] = math.sqrt(stddev_true_mean[p_id_index-1]/(len(total_estimates_mean[p_id_index-1])-1))
                            for est_mean in total_estimates_mean[p_id_index-1]:
                                stddev_est_list[p_id_index-1] += math.pow(est_mean - avg_val[p_id_index-1], 2)

                            stddev_est_list[p_id_index-1] = math.sqrt(stddev_est_list[p_id_index-1]/(len(total_estimates_mean[p_id_index-1])-1))
                prev_estimates_mean = estimates_mean
        avg_stddev_true += sum(stddev_true_mean) / float(len(stddev_true_mean))
        avg_stddev_est += sum(stddev_est_list) / float(len(stddev_est_list))
    avg_stddev_true = avg_stddev_true/num_folds
    avg_stddev_est = avg_stddev_est/num_folds
    return avg_stddev_true, avg_stddev_est
    
def getAverageGenderRecogRate(RB, main_folder, dataset_folder, num_folds, num_people):
    overall_gender_recog_rate = 0.0
    overall_female_recog_rate = 0.0
    overall_male_recog_rate = 0.0
    for num_fold in range(1,num_folds+1):
        fold_gender_rate = 0.0
        fold_female_recog_rate = 0.0
        fold_male_recog_rate = 0.0
        fold_counter = 0
        fold_female_counter = 0
        fold_male_counter = 0
        for fold_set in ["train/", "open/"]:
            for fold_t in ["Training/", "Test/"]:
                recog_folder = main_folder + fold_set + dataset_folder + "folds/" + str(num_fold) + "/" + fold_t
                RB.resetFilePaths()
                RB.setFilePaths(recog_folder)                
                overall_gender_rate, female_recog_rate, male_recog_rate, count_recogs, female_counter, male_counter = RB.getGenderDetectionRate()
                fold_gender_rate += overall_gender_rate*float(count_recogs)
                fold_counter += count_recogs

                fold_female_recog_rate += female_recog_rate*float(female_counter)
                fold_female_counter += female_counter

                fold_male_recog_rate += male_recog_rate*float(male_counter)
                fold_male_counter += male_counter
        overall_gender_recog_rate += fold_gender_rate/float(fold_counter)
        overall_female_recog_rate += fold_female_recog_rate/float(fold_female_counter)
        overall_male_recog_rate += fold_male_recog_rate/float(fold_male_counter)
    overall_gender_recog_rate /= num_folds
    overall_female_recog_rate /= num_folds
    overall_male_recog_rate /= num_folds
    return overall_gender_recog_rate, overall_female_recog_rate, overall_male_recog_rate
                        
if __name__ == "__main__":

    RB = RM.RecogniserBN()
    main_folder = "ChosenRepeat/BN/"
    dataset_folder = "Nall_gaussianT/"
    num_folds = 5
    num_people = 200
    avg_stddev_true, avg_stddev_est = getAverageAgeStddev(RB, main_folder, dataset_folder, num_folds, num_people)
    print(avg_stddev_true, avg_stddev_est)
    gender_recog_rate, female_recog_rate, male_recog_rate = getAverageGenderRecogRate(RB, main_folder, dataset_folder, num_folds, num_people)
    print(gender_recog_rate, female_recog_rate, male_recog_rate)
