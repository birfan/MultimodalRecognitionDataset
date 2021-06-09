import pyAgrum as gum
import os
import pandas as pd
import csv
import itertools
from scipy.stats import pearsonr
from scipy.stats import spearmanr

data_file = "../../dataset/MMLTUR-dataset/Nall_gaussianT/simplifiedData.csv"
bn_file = "../../dataset/MMLTUR-dataset/Nall_gaussianT/RecogniserBN.bif"
out_file = "../../dataset/MMLTUR-dataset/Nall_gaussianT/generatedData.csv"

age_min = 0 # min age that can be detected (NAOqi) 
age_max = 75 # max age that can be detected (NAOqi)
height_min = 50 # min height that can be detected (NAOqi)
height_max = 240 # max height that can be detected (NAOqi)
height_list = [str(x) for x in range(height_min,height_max+1)]
        
period = 30 # time is checked every 30 minutes 
time_min = 0
time_max = int(7*24*60/period) -1 # 7(days)*24(hours)*6

"""
# THIS DOESN'T WORK BECAUSE THERE IS CONDITIONAL INDEPENDENCE BETWEEN VARIABLES:
df = pd.read_csv(data_file)
df_list = df.values.tolist()

perm_two = list(itertools.permutations([0,1,2,3,4,5], r=2))
var_names = ['I','F','G','A','H','T']
mod_df_list = []
for df_item in df_list:
    if df_item[2] == "Female":
        df_item[2] = 0
    else:
        df_item[2] = 1
    mod_df_list.append(df_item)
for x, y in perm_two:
    corr, _ = pearsonr(mod_df_list[x], mod_df_list[y])
    corr_spearman, _ = spearmanr(mod_df_list[x], mod_df_list[y])
    print("Pearson correlation between {} and {} is {}".format(var_names[x], var_names[y], corr))
    print("Spearman correlation between {} and {} is {}".format(var_names[x], var_names[y], corr_spearman))

# GENERATE DATA:
gum.generateCSV(bn,out_file,5935,True)
df = pd.read_csv(out_file)
df_col_list = df.columns.tolist()
df_list = df.values.tolist()
modified_df_list = []
for df in df_list:
    if df[df_col_list.index('A')] > age_max:
        df[df_col_list.index('A')] = age_max
    if df[df_col_list.index('H')] > height_max:
        df[df_col_list.index('H')] = height_max
    elif df[df_col_list.index('H')] < height_min:
        df[df_col_list.index('H')] = height_min
    if df[df_col_list.index('G')] == 0:
        df[df_col_list.index('G')] = "Female"
    else:
        df[df_col_list.index('G')] = "Male"
    if df[df_col_list.index('T')] > time_max:
        df[df_col_list.index('T')] = time_max
    modified_df_list.append(df)

with open(out_file, 'w') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(df_col_list)
    for row in modified_df_list:
        writer.writerow(row)
"""

bn=gum.loadBN(bn_file)
print(bn.toDot())
print("*"*10)

learner=gum.BNLearner(data_file,bn) #using bn as template for variables
#learner=gum.BNLearner(out_file,bn) 
learner.useGreedyHillClimbing()
#learner.useK2([0,1,2,3,4,5])
#learner.useLocalSearchWithTabuList()
learner.addMandatoryArc("I","F")
learner.addMandatoryArc("I","G")
learner.addMandatoryArc("I","A")
learner.addMandatoryArc("I","H")
learner.addMandatoryArc("I","T")
#learner.useScoreLog2Likelihood() #->doesn't work well, creates illogical connections (e.g., T -> A)!
#learner.setMaxIndegree(2) # no more than 2 parents by node
bn2=learner.learnBN()
#kl=gum.BruteForceKL(bn,bn2) -> in old version
print(bn2.toDot())
print("="*10)
#kl=gum.ExactBNdistance(bn,bn2)
#kl.compute()
#learner=gum.BNLearner(data_file,bn) #using bn as template for variables
#learner.useLocalSearchWithTabuList()
#learner.useGreedyHillClimbing()
#learner.useK2([0,1,2,3,4,5])

#learner.setInitialDAG(bn2.dag())

#bn3=learner.learnBN()
#print(bn3.toDot())

#parameter learning -> does not give a good learned BN because face recognition mostly recognises users as unknown.
learner.setInitialDAG(bn.dag())
bn3=learner.learnParameters()
gum.saveBN(bn3, "../../dataset/MMLTUR-dataset/Nall_gaussianT/learnedBN.bif")
print(bn3.toDot())
