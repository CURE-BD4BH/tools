
#  python auroc.py <survival file> <predictions>
from __future__ import division
import sys
import math
import time
import operator
import numpy as np
from sklearn.metrics import roc_auc_score
import os.path
from os import path

outputLine = ""

def ca151SurvivalDataReader(fileName):
    rowIndex = -1
    patientSurvival = {}
    patientDic = {}
    unknownS5 = 0
    unknownS10 = 0
    unknownDf5 = 0
    tempCountS5 = 0
    tempCountS10 = 0
    tempCountDf5 = 0

    with open(fileName, "r") as myfile:
       for line in myfile:
            if rowIndex == -1:
                rowIndex = rowIndex + 1
                continue
            rowIndex = rowIndex + 1
            line = line.strip()
            spline = line.split(',')
            patientID = spline[0]
            s5 = spline[1]
            s10 = spline[2]
            df5 = spline[3]
            patientSurvival[patientID] = {}
            patientDic[patientID] = 1

            if s5.startswith('GTE'):
                patientSurvival[patientID]['s5'] = 1
            elif s5.startswith('LT'):
                patientSurvival[patientID]['s5'] = 0
            elif s5.startswith('Un'):
                patientSurvival[patientID]['s5'] = -1
                unknownS5 = unknownS5 + 1
            else:
                tempCountS5 = tempCountS5 + 1
            try:
                patientSurvival[patientID]['s5out'] = s5
            except Exception as e:
                pass

            if s10.startswith('GTE'):
                patientSurvival[patientID]['s10'] = 1
            elif s10.startswith('LT'):
                patientSurvival[patientID]['s10'] = 0
            elif s10.startswith('Un'):
                patientSurvival[patientID]['s10'] = -1
                unknownS10 = unknownS10 + 1
            else:
                tempCountS10 = tempCountS10 + 1
            try:
                patientSurvival[patientID]['s10out'] = s10
            except Exception as e:
                pass

            if df5.startswith('GTE'):
                patientSurvival[patientID]['df5'] = 1
            elif df5.startswith('LT'):
                patientSurvival[patientID]['df5'] = 0
            elif df5.startswith('Un'):
                patientSurvival[patientID]['df5'] = -1
                unknownDf5 = unknownDf5 + 1
            else:
                tempCountDf5 = tempCountDf5 + 1
            try:
                patientSurvival[patientID]['df5out'] = df5
            except Exception as e:
                pass
 

    #print(tempCountS5, tempCountS10, tempCountDf5) # Sanity check
    print("The number of patients: " + str(len(patientDic)))
    print("The number of patients with Survive5 unknown: " + str(unknownS5))
    print("The number of patients with Survive10 unknown: " + str(unknownS10))
    print("The number of patients with DiseaseFree5 unknown: " + str(unknownDf5))
    return patientDic, patientSurvival


def resultFileReader(fileName, patientDic, rowIndex):
    patientPrediction = {}

    with open(fileName, "r") as myfile:
       for line in myfile:
            if rowIndex == -1:
                rowIndex = rowIndex + 1
                continue
            rowIndex = rowIndex + 1
            line = line.strip()
            spline = line.split('\t')
            patientID = spline[0]
            if patientID not in patientDic:
                print("***Warning: patient ID " + patientID + " does not exist in the CA151 dataset. Predictions for this patient are ignored.")
                continue
            try:
                s5 = float(spline[1])
            except:
                s5 = 0.74
            try:
                s10 = float(spline[2])
            except:
                s10 = 0.146
            try:
                df5 = float(spline[3])
            except:
                df5 = 0.753
            patientPrediction[patientID] = {}

            if s5 > 1 or s5 < 0:
                patientPrediction[patientID]['s5'] = 0.74
                print("***Warning: patient " + patientID + "'s Survive5 prediction is out of range [0, 1]. Prediciton will be replaced with training data proportion.")
            else:
                patientPrediction[patientID]['s5'] = s5

            if s10 > 1 or s10 < 0:
                patientPrediction[patientID]['s10'] = 0.146
                print("***Warning: patient " + patientID + "'s Survive10 prediction is out of range [0, 1]. Prediciton will be replaced with training data proportion.")
            else:
                patientPrediction[patientID]['s10'] = s10

            if df5 > 1 or df5 < 0:
                patientPrediction[patientID]['df5'] = 0.753
                print("***Warning: patient " + patientID + "'s DiseaseFree5 prediction is out of range [0, 1]. Prediciton will be replaced with training data proportion.")
            else:
                patientPrediction[patientID]['df5'] = df5

    print("The number of patients with predictions: " + str(len(patientPrediction)))
    return patientPrediction


def bootstrapCI(y_true, y_pred):
    # Reference: https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
    n_bootstraps = 1000
    rng_seed = 4 # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower, confidence_upper


def aurocScore(patientSurvival, patientPrediction):
    global outputLine
    # Prepare the input data for sklearn
    s5ListTrue = []
    s5ListPred = []
    s10ListTrue = []
    s10ListPred = []
    df5ListTrue = []
    df5ListPred = []
    outPutFile = open(analysisTextFile, 'w')
    outPutFile.write("File: " + analysisTextFile)
    outPutFile.write("\n")
    outPutFile.write("The number of patients with predictions" + '\t' + str(len(patientPrediction)) + '\n')

    for p in patientSurvival:
        if p not in patientPrediction:
            print("***Warning: patient " + p + "'s prediction is missing. Prediciton will be replaced with training data proportion.")
            patientPrediction[p] = {}
            patientPrediction[p]['s5'] = 0.74
            patientPrediction[p]['s10'] = 0.146
            patientPrediction[p]['df5'] = 0.753
        if patientSurvival[p]['s5'] != -1 and patientPrediction[p]['s5'] != -1:
            s5ListTrue.append(patientSurvival[p]['s5'])
            s5ListPred.append(patientPrediction[p]['s5'])
        if patientSurvival[p]['s10'] != -1 and patientPrediction[p]['s10'] != -1:
            s10ListTrue.append(patientSurvival[p]['s10'])
            s10ListPred.append(patientPrediction[p]['s10'])
        if patientSurvival[p]['df5'] != -1 and patientPrediction[p]['df5'] != -1:
            df5ListTrue.append(patientSurvival[p]['df5'])
            df5ListPred.append(patientPrediction[p]['df5'])

    s5ArrayTrue = np.asarray(s5ListTrue)
    s5ArrayPred = np.asarray(s5ListPred)
    s10ArrayTrue = np.asarray(s10ListTrue)
    s10ArrayPred = np.asarray(s10ListPred)
    df5ArrayTrue = np.asarray(df5ListTrue)
    df5ArrayPred = np.asarray(df5ListPred)

    outputLine = outputLine + str(len(patientPrediction))
    print('')
    outPutFile.write("\n")

    if s5ArrayPred.size > 0:
        # Compute the scores
        s5Score = roc_auc_score(s5ArrayTrue, s5ArrayPred)
        # Estimate the confidence interval with bootstrapping
        confidence_lower_s5, confidence_upper_s5 = bootstrapCI(s5ArrayTrue, s5ArrayPred)

        outputLine = outputLine + ',' + str(round(s5Score, 3))
        outputLine = outputLine + ',"' + str(round(confidence_lower_s5, 3)) + ' - ' + str(round(confidence_upper_s5, 3)) + '"'

        print("AUROC score for Survive5: " + str(round(s5Score, 3)))
        outPutFile.write("AUROC score for Survive5" + '\t' + str(round(s5Score, 3)) + '\n')
        print("Confidence interval (90%, bootstrap) for the score: [{:0.3f} - {:0.3}]".format(confidence_lower_s5, confidence_upper_s5))
        outPutFile.write("Confidence interval (90%, bootstrap) for the score" + '\t' + str(round(confidence_lower_s5, 3)) + '\t' + str(round(confidence_upper_s5, 3)) + '\n')
        outPutFile.write("\n")
        print('')

    else:
        noPredMessage(outPutFile, "Survive5")

    if s10ArrayPred.size > 0:
        s10Score = roc_auc_score(s10ArrayTrue, s10ArrayPred)
        confidence_lower_s10, confidence_upper_s10 = bootstrapCI(s10ArrayTrue, s10ArrayPred)

        outputLine = outputLine + ',' + str(round(s10Score, 3))
        outputLine = outputLine + ',"' + str(round(confidence_lower_s10, 3)) + ' - ' + str(round(confidence_upper_s10, 3)) + '"'

        print("AUROC score for Survive10: " + str(round(s10Score, 3)))
        outPutFile.write("AUROC score for Survive10" + '\t' + str(round(s10Score, 3)) + '\n')
        print("Confidence interval (90%, bootstrap) for the score: [{:0.3f} - {:0.3}]".format(confidence_lower_s10, confidence_upper_s10))
        outPutFile.write("Confidence interval (90%, bootstrap) for the score" + '\t' + str(round(confidence_lower_s10, 3)) + '\t' + str(round(confidence_upper_s10, 3)) + '\n')
        outPutFile.write("\n")
        print('')

    else:
        noPredMessage(outPutFile, "Survive10")

    if df5ArrayPred.size > 0:
        df5Score = roc_auc_score(df5ArrayTrue, df5ArrayPred)
        confidence_lower_df5, confidence_upper_df5 = bootstrapCI(df5ArrayTrue, df5ArrayPred)

        outputLine = outputLine + ',' +str(round(df5Score, 3))
        outputLine = outputLine + ',"' + str(round(confidence_lower_df5, 3)) + ' - ' + str(round(confidence_upper_df5, 3)) + '"'

        print("AUROC score for DiseaseFree5: " + str(round(df5Score, 3)))
        outPutFile.write("AUROC score for DiseaseFree5" + '\t' + str(round(df5Score, 3)) + '\n')
        print("Confidence interval (90%, bootstrap) for the score: [{:0.3f} - {:0.3}]".format(confidence_lower_df5, confidence_upper_df5))
        outPutFile.write("Confidence interval (90%, bootstrap) for the score" + '\t' + str(round(confidence_lower_df5, 3)) + '\t' + str(round(confidence_upper_df5, 3)) + '\n')

    else:
        noPredMessage(outPutFile, "DiseaseFree5")

    #outPutFile.write("The number of patients with predictions" + '\t' + str(len(patientPrediction)) + '\n')

    outPutFile.close()
    return


def noPredMessage(outPutFile, which):
    global outputLine
    outputLine = outputLine + ',No predictions'
    outputLine = outputLine + ',No predictions'
    print("AUROC score for " + which + ": No predictions")
    outPutFile.write("AUROC score for " + which + ": No predictions")
    outPutFile.write("\n")
    print('')


def ece(y_true, y_pred):
    binSize = 10
    binInterval = 1 / binSize
    binBoundary = []
    for i in range(binSize):
        binBoundary.append(i * binInterval)
    binBoundary.append(1)
    N_pos = len(y_true[y_true == 1])
    N = len(y_true)
    binProb = {}
    binPos = {}
    for j in range(binSize):
        binProb[j] = []
        binPos[j] = 0

    for i in range(N):
        for j in range(binSize):
            if y_pred[i] >= binBoundary[j] and y_pred[i] < binBoundary[j + 1]:
                binProb[j].append(y_pred[i])
                if y_true[i] == 1:
                    binPos[j] = binPos[j] + 1
                break
        if y_pred[i] == binBoundary[binSize]:
            binProb[binSize - 1].append(y_pred[i])
            if y_true[i] == 1:
                binPos[binSize - 1] = binPos[binSize - 1] + 1

    score = 0
    for i in range(binSize):
        if len(binProb[i]) > 0:
            score = score + (len(binProb[i]) / N) * abs(binPos[i] / N_pos - np.mean(binProb[i]))
    return score


def bootstrapCI_ECE(y_true, y_pred):
    # Reference: https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
    n_bootstraps = 1000
    rng_seed = 4 # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = ece(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower, confidence_upper


def eceScore(patientSurvival, patientPrediction):
    global outputLine
    # Prepare the data
    s5ListTrue = []
    s5ListPred = []
    s10ListTrue = []
    s10ListPred = []
    df5ListTrue = []
    df5ListPred = []

    for p in patientSurvival:
        if p not in patientPrediction:
            patientPrediction[p] = {}
            patientPrediction[p]['s5'] = 0.74
            patientPrediction[p]['s10'] = 0.146
            patientPrediction[p]['df5'] = 0.753
        if patientSurvival[p]['s5'] != -1 and patientPrediction[p]['s5'] != -1:
            s5ListTrue.append(patientSurvival[p]['s5'])
            s5ListPred.append(patientPrediction[p]['s5'])
        if patientSurvival[p]['s10'] != -1 and patientPrediction[p]['s10'] != -1:
            s10ListTrue.append(patientSurvival[p]['s10'])
            s10ListPred.append(patientPrediction[p]['s10'])
        if patientSurvival[p]['df5'] != -1 and patientPrediction[p]['df5'] != -1:
            df5ListTrue.append(patientSurvival[p]['df5'])
            df5ListPred.append(patientPrediction[p]['df5'])

    s5ArrayTrue = np.asarray(s5ListTrue)
    s5ArrayPred = np.asarray(s5ListPred)
    s10ArrayTrue = np.asarray(s10ListTrue)
    s10ArrayPred = np.asarray(s10ListPred)
    df5ArrayTrue = np.asarray(df5ListTrue)
    df5ArrayPred = np.asarray(df5ListPred)

    outPutFile = open(analysisTextFile, 'a')
    outPutFile.write("\n")

    if s5ArrayPred.size > 0:
        s5Score = ece(s5ArrayTrue, s5ArrayPred)
        confidence_lower_s5, confidence_upper_s5 = bootstrapCI_ECE(s5ArrayTrue, s5ArrayPred)

        outputLine = outputLine + ',' + str(round(s5Score, 3))
        outputLine = outputLine + ',"' + str(round(confidence_lower_s5, 3)) + ' - ' + str(round(confidence_upper_s5, 3)) + '"'

        print("ECE score for Survive5: " + str(round(s5Score, 3)))
        outPutFile.write("ECE score for Survive5" + '\t' + str(round(s5Score, 3)) + '\n')
        print("Confidence interval (90%, bootstrap) for the score: [{:0.3f} - {:0.3}]".format(confidence_lower_s5, confidence_upper_s5))
        outPutFile.write("Confidence interval (90%, bootstrap) for the score" + '\t' + str(round(confidence_lower_s5, 3)) + '\t' + str(round(confidence_upper_s5, 3)) + '\n')
        print('')
    else:
        outputLine = outputLine + ',No predictions'
        outputLine = outputLine + ',No predictions'
        print("ECE score for Survive5: No predictions")
        outPutFile.write("ECE score for Survive5\tNo predictions\n")

    if s10ArrayPred.size > 0:
        s10Score = ece(s10ArrayTrue, s10ArrayPred)
        confidence_lower_s10, confidence_upper_s10 = bootstrapCI_ECE(s10ArrayTrue, s10ArrayPred)

        outputLine = outputLine + ',' + str(round(s10Score, 3))
        outputLine = outputLine + ',"' + str(round(confidence_lower_s10, 3)) + ' - ' + str(round(confidence_upper_s10, 3)) + '"'

        print("ECE score for Survive10: " + str(round(s10Score, 3)))
        outPutFile.write("ECE score for Survive10" + '\t' + str(round(s10Score, 3)) + '\n')
        print("Confidence interval (90%, bootstrap) for the score: [{:0.3f} - {:0.3}]".format(confidence_lower_s10, confidence_upper_s10))
        outPutFile.write("Confidence interval (90%, bootstrap) for the score" + '\t' + str(round(confidence_lower_s10, 3)) + '\t' + str(round(confidence_upper_s10, 3)) + '\n')
        print('')
    else:
        outputLine = outputLine + ',No predictions'
        outputLine = outputLine + ',No predictions'
        print("ECE score for Survive10: No predictions")
        outPutFile.write("ECE score for Survive10\tNo predictions\n")

    if df5ArrayPred.size > 0:
        df5Score = ece(df5ArrayTrue, df5ArrayPred)
        confidence_lower_df5, confidence_upper_df5 = bootstrapCI_ECE(df5ArrayTrue, df5ArrayPred)

        outputLine = outputLine + ',' + str(round(df5Score, 3))
        outputLine = outputLine + ',"' + str(round(confidence_lower_df5, 3)) + ' - ' + str(round(confidence_upper_df5, 3)) + '"' + '\n'

        print("ECE score for DiseaseFree5: " + str(round(df5Score, 3)))
        outPutFile.write("ECE score for DiseaseFree5" + '\t' + str(round(df5Score, 3)) + '\n')
        print("Confidence interval (90%, bootstrap) for the score: [{:0.3f} - {:0.3}]".format(confidence_lower_df5, confidence_upper_df5))
        outPutFile.write("Confidence interval (90%, bootstrap) for the score" + '\t' + str(round(confidence_lower_df5, 3)) + '\t' + str(round(confidence_upper_df5, 3)) + '\n')
    else:
        outputLine = outputLine + ',No predictions'
        outputLine = outputLine + ',No predictions'
        print("ECE score for DiseaseFree5: No predictions")
        outPutFile.write("ECE score for DiseaseFree5\tNo predictions\n")


    outPutFile.close()
    return

def writeStatsCSV(predictionFile, analysisFile):
    global outputLine
    addHeader = True

    if path.exists(analysisFile):
        addHeader =  False

    csvFile = open(analysisFile, 'a+')

    if addHeader:
        header = "Prediction File,# of Patients,AUROC (Srv5),Confidence interval-S5 (90% bootstrap)," + \
            "AUROC (Srv10),Confidence interval-S10 (90% bootstrap)," + \
            "AUROC (DF5),Confidence interval-DF5 (90% bootstrap)," + \
            "ECE (Srv5),Confidence interval-S5 (90% bootstrap),ECE (Srv10),Confidence interval-S10 (90% bootstrap)," + \
            "ECE (DF5),Confidence interval-DF5(90% bootstrap)\n"
        csvFile.write(header)

    outputLine = predictionFile +  ',' + outputLine
    csvFile.write(outputLine)
    csvFile.close()
    return

# writes out all the details of the individual predictions and survival outcomes
def writePatientLevelPredictions(ppatientPrediction, patientSurv, predictionFile, plpFile):
    pFile = open(plpFile, 'a+')
    pFile.write(predictionFile+"\n")
    pFile.write("TCGA_ID, Surv5, Srv10, DiseaseFree5, Surv5, Srv10, DiseaseFree5, Surv5, Srv10, DiseaseFree5\n")
    keys = ppatientPrediction.keys()
    sortedKeys = list(keys)
    sortedKeys.sort()

    for p in sortedKeys:
        pre = ppatientPrediction[p]
        s5 = pre['s5']
        s10 = pre['s10']
        df5 = pre['df5']
        s5out = patientSurv[p]['s5out']
        s10out = patientSurv[p]['s10out']
        df5out = patientSurv[p]['df5out']
        if s5out.startswith('GTE'):
            s5outctr = 1
        elif s5out.startswith('LT'):
            s5outctr = 0
        else:
            s5outctr = -1

        if s10out.startswith('GTE'):
            s10outctr = 1
        elif s10out.startswith('LT'):
            s10outctr = 0
        else:
            s10outctr = -1

        if df5out.startswith('GTE'):
            df5outctr = 1
        elif df5out.startswith('LT'):
            df5outctr = 0
        else:
            df5outctr = -1

        out = p + ',' + str(s5) +  ',' + str(s10) + ',' + str(df5) + ',' + str(s5out) + ',' + str(s10out) + ',' + str(df5out) + \
            ',' + str(s5outctr) + ',' + str(s10outctr) + ',' + str(df5outctr) +'\n'
        pFile.write(out)
    pFile.close()


#===================================#
################Main#################
#===================================#

if __name__=="__main__":
    startTime = time.time()
    # Read the files and process
    survivalFile = sys.argv[1]
    predictionFile = sys.argv[2]
    tokens = predictionFile.split(".")
    analysisFile = "BreastCA151_analysis-" + time.strftime("%Y%m%d") + ".csv";  # single analysis file for all results
    ppatientPredictionFile = "BreastCA151_prediction_details-" + time.strftime("%Y%m%d") + ".csv";
    analysisTextFile = tokens[0] + "_analysis.txt";

    print("################")
    print("Reading CA151 survival file (ground truth)...")
    patientDic, patientSurvival = ca151SurvivalDataReader(survivalFile)
    print('')

    print("################")
    print("Reading results file...")
    rowIndex = 0
    if len(sys.argv) > 3 and sys.argv[3] == '1':
        rowIndex = -1
    patientPrediction = resultFileReader(predictionFile, patientDic, rowIndex)
    print('')

    print("################")
    print("Computing AUROC score and confidence interval...")
    aurocScore(patientSurvival, patientPrediction)
    print('')

    print("################")
    print("Computing ECE score...")
    eceScore(patientSurvival, patientPrediction)
    print('')

    writeStatsCSV(predictionFile, analysisFile)

    writePatientLevelPredictions(patientPrediction, patientSurvival, predictionFile, ppatientPredictionFile)

    timeSpent = round(time.time() - startTime, 2)
    print('Done.\nTime spent: ' + str(timeSpent) + 's.')
