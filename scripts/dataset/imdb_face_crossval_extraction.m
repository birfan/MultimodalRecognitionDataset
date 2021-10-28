%========================================================================================================%
%  Copyright (c) 2018-present, Bahar Irfan                                                               %
%                                                                                                        %                      
%  imdb_face_crossval_extraction script chooses the users from the IMDB dataset using crossValidation    %
%  functions and artificialDataset for creating artificial estimates of height and time of interaction,  %
%  which are missing from the IMDB dataset*. The images are previously cleaned by NAOqi face detection   %
%  and manually for removing images without a face detected.                                             %
%                                                                                                        %
%  Please cite the following work if using this code:                                                    %
%                                                                                                        %
%    B. Irfan, M. Garcia Ortiz, N. Lyubova, and T. Belpaeme (2021), "Multi-modal Open World User         %
%    Identification", Transactions on Human-Robot Interaction (THRI), 11 (1), ACM.                       %
%                                                                                                        %
%  * Face cropped images of IMDB dataset in IMDB-Wiki dataset are used for this purpose:                 %
%                                                                                                        %
%    R. Rothe, R. Timofte and L. Van Gool (2018), "Deep expectation of real and apparent age from a      %
%    single image without facial landmarks", International Journal of Computer Vision, vol. 126, no. 2-4.%
%                                                                                                        %
%    R. Rothe and R. Timofte and L. Van Gool (2016), "Deep expectation of real and apparent age from a   %
%    single image without facial landmarks", International Journal of Computer Vision (IJCV).            %  
%                                                                                                        %         
%  imdb_face_crossval_extraction and each script in this project is under the GNU General Public         %
%  License v3.0. You should have received a copy of the license along with MultimodalRecognitionDataset. %
%  If not, see <http://www.gnu.org/licenses/>.                                                           %
%========================================================================================================%

% in matlab:
% in "celeb_id" get ids of celebs that have number of images per person 
% that correspond to the same age > num_samples
% from those ids randomly select ids
% sample age based on frequency
% get name of user from "name"
% get gender for each celeb
% get images corresponding to id and age
% write to csv

% in python:
% if image pixel less than 150x150, remove image
% for each id, randomly select image
% save them to folder
% manually remove images that are distorted (face cropped, but has multiple
% faces)
% sequentially assign ids starting from 1, and rename images to 1_1, 1_2 ..
% save them to folder images_sequential
% run cross validation functions for binning images into different folders

rng(123456789,'twister');
stream = RandStream.getGlobalStream();

isCleaned = 1;
img_src_folder = 'cleaned_dataset';
isRemoveChosen = 0;
isTestSet = 1;
num_people = 100;
num_samples = 50;
min_age = 10;
max_age = 75;
% to_erase = [2632, 3360, 5015, 8408, 12417, 13373, 14907, 19069, 20187];
%to_erase = [1981, 9767, 13150,1096,9031,11210,16892,1527,720,18970,9397,3308,19395,8302,13360,11666,800,1105,18666,12748,7690,996,1225,19935,13141,13364,14740,13095,18090,9046,41,11490,13048,20245,9400,17245,13947,10438,10939,2336,11172,10364,16432,14005,11601,13146,9156,635,5846,10910,4034,3090,15662,9716,11727,17136,10078,16790,9816,11684,1194,10923,13556,15169,17223,10272,3448,8853,17810,5773,4582,16731,16524,3428,11042,9942,6952,1199,4202,8134,19424,3206,1416,13253,13977,9666,18741,8411,8907,3287,10954,5517,12815,9764,8758,14686,10100,17572,20167,6276,4762,15037];
%to_erase = [41,635,720,800,996,1096,1105,1194,1199,1225,1416,1527,2336,3090,3206,3287,3308,3428,3448,4034,4202,4582,4762,5517,5773,5846,6276,6952,7690,8134,8302,8411,8758,8853,8907,9031,9046,9156,9397,9400,9666,9716,9764,9816,9942,10078,10100,10272,10364,10438,10910,10923,10939,10954,11042,11172,11210,11490,11601,11666,11684,11727,12748,12815,13048,13095,13141,13146,13150,13253,13360,13364,13556,13947,13977,14005,14686,14740,15037,15169,15662,16432,16524,16731,16790,16892,17136,17223,17245,17572,17810,18090,18666,18741,18970,19395,19424,19935,20167,20245,476,858,1188,2177,2686,2711,3474,3606,3670,3798,3877,3925,3964,4036,4125,4220,4620,4644,5506,5559,5685,5827,5843,5871,5891,6003,6163,6654,6928,7487,7505,7762,8132,8221,8451,8462,8700,8803,8811,8817,8837,8860,9048,9074,9186,9409,9700,9806,10017,10055,10145,10259,10264,10474,10538,10588,10994,11267,11365,11520,11807,11966,12205,12207,12344,12509,12797,13000,13149,13207,13285,13495,13746,13980,14017,14420,14655,15212,15238,15283,15571,17178,17255,17538,17673,17826,17903,18029,18150,18624,18968,19105,19122,19154,19589,19705,19807,20152,20188,20250];
prev_file = 'imdb_chosen_train.csv';
% isTestSet && if exist(prev_file, 'file') == 2
%    prev_values = xlsread(prev_file, '');
%    to_erase = prev_values(:, 1);
% end

fullFileName = 'imdb_chosen_open.csv';
headers = 'Id,Name,Gender,Age';
type_list = '%d,%s,%s,%d,';

for i=1:num_samples
    im_num = strcat('Im_', num2str(i,'%d'));
    headers= strcat(headers, ',');
    headers= strcat(headers, im_num);
    if i<num_samples
        type_list = strcat(type_list,'%s,');
    else
        type_list = strcat(type_list,'%s\n');
        headers = strcat(headers,'\n');
    end
end

load('imdb.mat');

[age,~] = datevec(datenum(imdb.photo_taken,7,1)-imdb.dob);
cat_id_age = cat(2,imdb.celeb_id',age');
i_count = 1;
info_list = [];
imgs_ind = [];
while i_count <= length(imdb.celeb_id)
    id_celeb = imdb.celeb_id(i_count);
    same_ids = find(imdb.celeb_id==id_celeb);
    if isCleaned
        % if the image dataset is cleaned (low res images are removed and 
        % passed through NaoQi face detection)
        imgs_id_celeb = [];
        for ind = same_ids
            file_name = strcat(img_src_folder,'/');
            file_name = strcat(file_name, imdb.full_path{ind});
            
            if exist(file_name, 'file') == 2
                imgs_id_celeb = [imgs_id_celeb;ind];
                imgs_ind = [imgs_ind;ind];
            end
        end
        celeb_ages = age(imgs_id_celeb);
    else
        celeb_ages = age(same_ids);
    end
    
    celeb_reasonable_ages = celeb_ages(celeb_ages >= min_age & celeb_ages <= max_age);
    
    [num_im_with_age, ages_celeb_list] = hist(celeb_reasonable_ages,unique(celeb_reasonable_ages));
        
    list_to_concat = [];
    for num_age_count=1:length(ages_celeb_list)
        if num_im_with_age(num_age_count) >= num_samples
            gender = imdb.gender(i_count);
            conc = [id_celeb, ages_celeb_list(num_age_count), gender];
            list_to_concat = [list_to_concat; conc];
        end
    end
    info_list = [info_list;list_to_concat];
    i_count = i_count + length(same_ids);
end
[num_age_dist, ages_dist] = hist(info_list(:,2),unique(info_list(:,2)));

[num_age_sorted, num_age_order] = sort(num_age_dist, 'descend');
max_age_indices = num_age_order(1:5);

chosen_users = [];
chosen_ids = [];
num_chosen = 0;

if isTestSet && ~isempty(to_erase)
    for te = to_erase
        ind = find(info_list(:, 1)==te);
        while ~isempty(ind)
        	info_list(ind(1),:) = [];
            ind = find(info_list(:, 1)==te);
        end
    end
    if length(unique(info_list(:, 1))) < num_people
        error('The remaining number of people satisfying the criteria is less than the necessary number. Criteria need to be changed!');
        exit;
    end
end

if isRemoveChosen && exist(fullFileName, 'file') == 2
    filehere = which(fullFileName);
    if ~isempty(filehere)  
        fid = fopen(fullFileName,'r');
        chosen_users_cell = textscan(fid,strrep(strrep(type_list,',',' '), '\n',''), 'headerlines', 1, 'delimiter', ',');
        fclose(fid);
        sz = length(chosen_users_cell{:,1});  % Line count
        chosen_users=cell(sz,size(chosen_users_cell,2));             
        for k = 1:size(chosen_users_cell,2)
            t1 = chosen_users_cell(k);
            t2 = [t1{:}];
            if isnumeric(t2)              % Takes care of floats
                chosen_users(:,k) = num2cell(t2);
            else
                chosen_users(:,k) = t2;
            end
        end
        
        chosen_ids = [chosen_users{:,1}]';
        if ~isempty(to_erase)
            for te = to_erase
                ind = find([chosen_users{:,1}]' == te);
                chosen_users(ind,:) = [];
            end
        end
        num_chosen = length([chosen_users{:,1}]);
        
        % remove chosen_ids from info_list
        for id = chosen_ids'
            ind = find(info_list(:, 1)==id);
            while ~isempty(ind)
                info_list(ind(1),:) = [];
                ind = find(info_list(:, 1)==id);
            end
        end
        if length(unique(info_list(:, 1))) < num_people - num_chosen
            error('The remaining number of people satisfying the criteria is less than the necessary number. Criteria need to be changed!');
            exit;
        end

    end
end


while num_chosen < num_people
    [user_data, user_ind] = datasample(stream, info_list, 1, 'Replace', false);
    chosen_id = info_list(user_ind,1);
    chosen_age = info_list(user_ind,2);
    if (ismember(chosen_age, ages_dist(max_age_indices)) == 1 && randi([0 1], 1, 1) == 0)
        continue;
    end
    % user_id, user_name, user_gender, user_age, image_nums*100
   
    chosen_g = info_list(user_ind,3);
    if chosen_g == 0
        chosen_gender = 'Female';
    elseif chosen_g == 1
        chosen_gender = 'Male';
    else
        continue;
    end
    
    chosen_name = imdb.celeb_names(chosen_id);
    if isCleaned
        chosen_ind_init = find(cat_id_age(:,1)== chosen_id & ...
            cat_id_age(:,2)==chosen_age);
        chosen_ind = intersect(imgs_ind, chosen_ind_init);
    else
        chosen_ind = find(cat_id_age(:,1)== chosen_id & cat_id_age(:,2)==chosen_age);
    end
    all_ims = imdb.full_path(chosen_ind);
    chosen_im_ids = datasample(stream, 1:length(all_ims), num_samples,'Replace', false);
    chosen_ims = all_ims(chosen_im_ids);
    conc_list = [chosen_id, chosen_name, chosen_gender, chosen_age, chosen_ims];
    chosen_users = [chosen_users; conc_list];
    
    ind = find(info_list(:, 1)==chosen_id);
    while ~isempty(ind)
        info_list(ind(1),:) = [];
        ind = find(info_list(:, 1)==chosen_id);
    end
            
    num_chosen = num_chosen + 1;
end


% write to csv
fid = fopen(fullFileName,'wt');
if fid>0
    fprintf(fid,headers);
    for k=1:size(chosen_users,1)
        fprintf(fid,type_list,chosen_users{k,:});
    end
    fclose(fid);
end

img_dest_folder = 'chosen_clean_open/';

for k=1:size(chosen_users,1)
    folder_name = strcat(img_dest_folder, num2str(chosen_users{k,1}));
    mkdir(folder_name);
    for l=5:5+num_samples-1
        file_name = chosen_users{k,l};
        split_f = strsplit(file_name, '/');
        new_filename = strcat(folder_name, '/');
        new_filename = strcat(new_filename, split_f{2});
        orig_filename = strcat(img_src_folder, '/');
        orig_filename = strcat(orig_filename, file_name);
        copyfile(orig_filename, new_filename);
    end
end
