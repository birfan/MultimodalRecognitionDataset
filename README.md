# Multi-modal Long-Term User Recognition Dataset and Multi-modal Incremental Bayesian Network

Multi-modal Incremental Bayesian Network (MMIBN) is described in the papers below:

 * Bahar Irfan, Michael Garcia Ortiz, Natalia Lyubova, and Tony Belpaeme (under review), "Multi-modal Open World User Identification", ACM Transactions on Human-Robot Interaction (THRI).

 * Bahar Irfan, Natalia Lyubova, Michael Garcia Ortiz, and Tony Belpaeme (2018), "Multi-modal Open-Set Person Identification in HRI", 2018 ACM/IEEE International Conference on Human-Robot Interaction Social Robots in the Wild workshop, http://socialrobotsinthewild.org/wp-content/uploads/2018/02/HRI-SRW_2018_paper_6.pdf

The most up-to-date version of Multi-modal Incremental Bayesian Network can be found in: https://github.com/birfan/MultimodalRecognition

First paper also presents the Multi-modal Long-Term User Recognition Dataset, with evaluations of the MMIBN, [http://doc.aldebaran.com/2-5/naoqi/](NAOqi) face recognition and [https://github.com/EMRResearch/ExtremeValueMachine](Extreme Value Machine) (Rudd et al., 2017) on this dataset and three real-world human-robot interaction (HRI) experiments. [https://agrum.gitlab.io/](pyAGrum) library (Gonzales et al., 2017) is used for implementing the Bayesian network structure.

Please cite both papers if you are using the Multi-modal Incremental Bayesian Network; cite the first paper for the Multi-modal Long-Term User Recognition Dataset; cite the first paper for the evaluations on the dataset; cite the first and second paper if you are referring to the real-world long-term HRI experiments for user recognition.

## Multi-modal Long-Term User Recognition Dataset

The dataset is provided in the *dataset* folder. The cross-validation datasets are provided in *cross-validation-datasets* release. The trained models and optimisation datasets are available in *trained-models* release. The code to create the dataset and reproduce the experiments in Irfan et al. (under review) is available through:

    $ git clone https://github.com/birfan/MultimodalRecognitionDataset.git

Multi-modal Long-Term User Recognition Dataset contains 5735 images and corresponding face recognition similarity scores, gender and age estimations, along with simulated height and (random and patterned) time of interactions for 200 users. The images are taken from [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](IMDB-WIKI dataset) (Rothe et al., 2015; Rothe et al., 2018) which is the largest publicly available dataset of face images of celebrities with gender and age labels, taken at events or still frames from movies.The proprietary algorithms of the Pepper robot (SoftBank Robotics Europe) were used to obtain multi-modal biometric information from these images (face, gender and age), while the height and time of interaction are artificially generated to simulate a long-term HRI scenario. 

The 200 users in the dataset are randomly sampled out of 20k celebrities, choosing only celebrities which have more than 10 images each corresponding to the same age, using *imdb_face_crossval_extraction.m* script. The resulting dataset contains 101 females, 98 males and one transgender person. In the dataset, each image of the user was chosen from the same year in order to simulate an open world HRI scenario, where the users will be met in consecutive days or weeks. The images that correspond to an age that is within the five most common ages in the set were randomly rejected during the selection. The resulting age range is 10-63, with the mean age of 33.04 (standard deviation (SD) is 9.28). To keep the data realistic and model the differences between the estimated heights, Gaussian noise with SD=6.3 cm found in (Irfan et al., 2018) added to the heights obtained from the web. Each image has a resolution greater than (or equal to) 150x150, has a single face in the image that corresponds to the correct celebrity (i.e., the IMDB-Wiki dataset was cleaned to ensure these criteria are met, using *imdb_prepareImages.py* script).

Based on the frequency of user encounters, we created the following datasets: (1) *D-Ten*, where each user is observed precisely ten times, e.g., ten return visits to a robot therapist, and (2) *D-All*, in which each user is encountered a different amount of times (10 to 41 times). Two types of distribution are considered for the time of interaction: (1) patterned interaction times in a week modelled through a Gaussian mixture model (*Gaussian*), where the user will be encountered certain times on specific days, which applies to HRI in rehabilitation and education areas, and (2) random interaction times represented by a uniform distribution (*Uniform*), such as in domestic applications with companion robots, where the user can be seen at any time of the day in the week, resulting in a total of four datasets (in *N10_gaussianT*, *N10_uniformT*, *Nall_gaussianT*, *Nall_uniformT* folders). The only difference between Gaussian and uniform datasets is the time of the interaction for each sample. Each dataset contains the following folders/files:

* *images*: Contains (10 to 41) images for each user in the dataset. E.g., 1_5 is the fifth image of user with ID 1.
* *db.csv*: This file contains the ID of the user (corresponding to the order of enrolment), the user's name, gender (as taken from IMDB-Wiki dataset), height (as taken from the web), the time of interaction when the user enrolled (*times*), and *occurrence* which corresponds to \[*number_occurrences_of_user*, *number_of_images_taken_while_enrolling*, *number_total_images_of_user*\]. The *occurrence* is \[0,0,0\] as default.
* *info.csv*: This file contains the order of recognitions (*N*), the user recognised (*id*), the image for recognition (*image*), (artificially generated) estimated height *height*, (artificially generated) time of interaction *time*, and whether the user is enrolling (*R* 1 if the user is new, 0 if enrolled). 
* *multiModalBiometricData.csv*: Contains the multi-modal biometric data (NAOqi estimations and artificially generated height and time of interactions for each recognition). This file corresponds to the *info.csv* where NAOqi proprietary algorithms are applied on each image to obtain face similarity scores (*F*), gender estimations (*G*), age estimations (*A*), in addition to the artificially generated height estimations (*H*) and time of interactions (*T*) for each user (*I*) in order of recognition (*N*).
* *simplifiedData.csv*: The results in the *multiModalBiometricData.csv* are simplied by taking the best match evidence for modalities  (i.e., confidence scores are not used). For instance, the most similar user (or unknown) is taken as the face recognition estimate by taking into account the face recognition threshold (0.4), and the evidence for gender, age and height are taken as the estimated values. 

## Multi-modal Incremental Bayesian Network

Multi-modal Incremental Bayesian Network (MMIBN) is the first user recognition method that can continuously and incrementally learn users, without the need for any preliminary training, for fully autonomous user recognition in long-term human-robot interaction. It is also the first method that combines a primary biometric (face recognition) with weighted soft biometrics (gender, age, height and time of interaction) for improving open world user identification in real-time human-robot interaction. In order to learn the changes in the user appearance, we extend MMIBN with online learning (MMIBN:OL) that adapts the likelihoods in the Bayesian network with incoming data.

Clone the Multi-modal Incremental Bayesian Network and follow the instructions to build the libraries:

    $ git clone https://github.com/birfan/MultimodalRecognition.git

For reproducing the results in Irfan et al. (under review), replace *RecognitionMemory.py* with the one provided under *scripts*, download the cross-validation datasets in the *cross-validation-datasets* release, and read the description of files in the readme file provided. Note that the MultimodalRecognition repository contains the latest code, and it may have slight changes with the code provided here. See the ReadMe file provided in that repository for setting up and using the code, and for information on the resulting files. For more information about the MMIBN models, see Irfan et al. (2018; under review).

## Extreme Value Machine

Extreme Value Machine (EVM, Rudd et al. (2017)) is a state-of-the-art open world recognition (i.e., incremental learning of new classes, in addition to recognising previously learned classes) method. To accept sequential and incremental data for online learning, we adapted EVM by adjusting its hyperparameters to use it as a baseline (as provided in *evm.py* file under *scripts*). In the original work, batch learning of 50 classes was used with an average of 63806 data points at each update, instead of a single data point that we used in this work. We compared MMIBN models with the performance of two EVM models: (a) EVM:FR, using NAOqi face recognition similarity scores as data (trained models in *EVM_face* in *trained-models* release), (b) EVM:MM using the same multi-modal data as in MMIBN (trained models in *EVM_all* in *trained-models* release). *EVM_soft* also provides results using only soft biometrics data (gender, age, height and time of interaction). The parameters of EVM (cover threshold and open-set threshold) are optimised for EVM:FR and EVM:MM and the optimum parameters and the corresponding results are provided in *EVM_scripts_params_times.tar.gz* in *trained-models* release. Note that *EVM-0.1.2.zip* contains the original code and the README to install necessary libraries. *evm_IMDB.py* script is used to evaluate EVM on the Multi-modal Long-Term User Recognition Dataset. Functions from *RecognitionMemory.py* and *crossValidation.py* are necessary to reproduce the results for EVM.

## User Recognition Evaluations on the Dataset

Multi-modal Incremental Bayesian Network (MMIBN) by Irfan et al. (2018; under review) and Extreme Value Machine (EVM) by Rudd et al. (2017) are evaluated on the Multi-modal Long-Term User Recognition Dataset. The results show that MMIBN models achieve significantly lower long-term recognition performance loss compared to the baseline face recognition (NAOqi) and EVM. For more details about the evaluations and the models, refer to Irfan et al. (under review). The *trained-models* contains the trained models and the results for reproducibility of the reported results in Irfan et al. (under review). *cross-validation-datasets* release contains the cross-validation datasets to evaluate the algorithms.

## License

The Multi-modal Long-Term User Recognition Dataset is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Multi-modal Incremental Bayesian Network is released under a GNU General Public License v3.0. 
In other words, the dataset is made available for academic research purpose only, and the scripts for user recognition allow modification and reuse only with attribution and releasing under the same license. The licenses are included with the data and the scripts.

## Contact

Irfan et al. (under review) contains more information about the dataset and the evaluations. For any other information on the dataset or the MMIBN, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.

## Acknowledgments

We would like to thank Pierre-Henri Wuillemin for his substantial help with pyAgrum, and Ethan M. Rudd for his suggestions in adapting the Extreme Value Machine for online recognition.

## References

 * Ethan M. Rudd, Lalit P. Jain, Walter J. Scheirer and Terrance E. Boult (2018), "The Extreme Value Machine" in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 3, pp. 762-768, [https://doi.org/10.1109/TPAMI.2017.2707495](DOI:10.1109/TPAMI.2017.2707495)
 * Christophe Gonzales, Lionel Torti and Pierre-Henri Wuillemin (2017), "aGrUM: a Graphical Universal Model framework", International Conference on Industrial Engineering, Other Applications of Applied Intelligent Systems, Springer, [https://doi.org/10.1007/978-3-319-60045-1_20](DOI:10.1007/978-3-319-60045-1_20)
 * Rasmus Rothe, Radu Timofte and Luc Van Gool (2015), "DEX: Deep EXpectation of apparent age from a single image", IEEE International Conference on Computer Vision Workshops (ICCVW)
 * Rasmus Rothe, Radu Timofte and Luc Van Gool (2018), "Deep expectation of real and apparent age from a single image without facial landmarks", International Journal of Computer Vision, vol. 126, no. 2-4, pp. 144-157, Springer, [https://doi.org/10.1007/s11263-016-0940-3](DOI:10.1007/s11263-016-0940-3)
