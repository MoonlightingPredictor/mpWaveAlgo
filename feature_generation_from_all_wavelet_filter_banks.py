import numpy as np
import sys
import scipy.integrate

np.set_printoptions(threshold=sys.maxsize)
# Standardize the data attributes for the Iris dataset.
from sklearn.datasets import load_iris
from sklearn import preprocessing

import matplotlib.pyplot as plt
import pywt
import pywt.data
import csv
import os
from pylab import *
import threading
import pandas

import scipy.special as special
from scipy.stats import entropy
from scipy.stats import linregress
from collections import Counter
from scipy.stats import kurtosis
from scipy.stats import moment

# definition of variables
Concatinate_Wavelet_vectore_Tobe_right_in_CSV_file = []  # This vectore will concatinate (LL,LH, HL and HH) to write on the one row of the csv file


class ObtainSignalresult:
    # This class would take one argument which is the file of Amino Acids surface charges and hydrophobicity
    # The file will be opened by SW and change the 2D feauture and create the wavelet features
    def __init__(self, twoD_features_extracted_txt_file):  # initialize the entrance txt file
        self.twoD_features_extracted_txt_file = twoD_features_extracted_txt_file

    def creating_signal_vector(self):  # The function which will give us the Wavelet feature vectors.

        # local variaable definition
        twoD_feature_of_Amino_acides_before_wavelet_transform = []  # Vectore for append all the wavelet produced matrixes
        coeffs = []

        counter = 0

        #using different filterbanks name
        for filter in ['db2','db3' ,'db4','db5','db6','db7','db8','db9','bior1.3','bior1.5','bior2.2','bior2.4','bior2.6','bior3.1','bior3.3','bior3.5','bior3.7'
                       ,'bior4.4','bior5.5','coif1','coif2','coif3','coif4','coif5','sym2','sym3','sym4','sym5','sym6','sym7','sym8']:
            print("Generating Features For ",filter, " Filter bank started")
            f3 = open("Filter_banks_feature_vectors\\"+ filter+'Coeffs_Avrage_features_for_Training_Data.txt' ,'w+')
            for line in open(self.twoD_features_extracted_txt_file):
                list = line.split()  # split the file by each tab in a dragon of the list

                if list == [] or list[0][
                    0] == ">":  # Chk the file in order to make sure it is not the Protein Name and Chk the end of the file with second term

                    if len(twoD_feature_of_Amino_acides_before_wavelet_transform) != 0:

                        counter = counter + 1
                        coeffs = pywt.dwt2(twoD_feature_of_Amino_acides_before_wavelet_transform, filter)
                        # LL, (LH, HL, HH),(LH1,HL1,HH1),(LH2, HL2, HH2),(LH3, HL3, HH3) = coeffs
                        LL, (LH, HL, HH) = coeffs # separate the Approximate coefficient and detail coefficient to 4 signals

                       #generating the criteria including max, min, median, average, Standard deviation,energy, entopy,
                       #slope and SED error for details coeeficient towards approximate coefficient
                       #overal energy and wavelength for overal signal and write in the file for each filter bank
                        LLenergy = np.sqrt(np.sum(np.array(LL[-1]) ** 2)) / len(LL[-1])
                        LHenergy = np.sqrt(np.sum(np.array(LH[-1]) ** 2)) / len(LH[-1])
                        HLenergy = np.sqrt(np.sum(np.array(HL[-1]) ** 2)) / len(HL[-1])
                        HHenergy = np.sqrt(np.sum(np.array(HH[-1]) ** 2)) / len(HH[-1])
                        overalEnergy = np.sqrt(np.sum(np.array(coeffs[-1]) ** 2)) / len(coeffs[-1])
                        LLwavelength = len(LL)

                        LL = np.concatenate(LL)
                        LH = np.concatenate(LH)
                        HL = np.concatenate(HL)
                        HH = np.concatenate(HH)

                        LL1energy = np.sqrt(np.sum(np.array(LL[-1]) ** 2)) / len(LL)
                        LH1energy = np.sqrt(np.sum(np.array(LH[-1]) ** 2)) / len(LH)
                        HL1energy = np.sqrt(np.sum(np.array(HL[-1]) ** 2)) / len(HL)
                        HH1energy = np.sqrt(np.sum(np.array(HH[-1]) ** 2)) / len(HH)

                        LHslope, LHintercept, LHr_value, LHp_value, LHstd_err = linregress(LL, LH)
                        HLslope, HLintercept, HLr_value, HLp_value, HLstd_err = linregress(LL, HL)
                        HHslope, HHintercept, HHr_value, HHp_value, HHstd_err = linregress(LL, HH)

                        maxLL = np.max(LL)
                        minLL = np.min(LL)
                        AvLL = np.average(LL)
                        MedianLL = (minLL + maxLL) / 2

                        pd_series = pandas.Series(LL)
                        counts = pd_series.value_counts()
                        LLentropy = entropy(counts)

                        maxLH = np.max(LH)
                        minLH = np.min(LH)
                        AvLH = np.average(LH)
                        MedianLH = (minLH + maxLH) / 2

                        pd_series = pandas.Series(LH)
                        counts = pd_series.value_counts()
                        LHentropy = entropy(counts)

                        maxHL = np.max(HL)
                        minHL = np.min(HL)
                        AvHL = np.average(HL)
                        MedianHL = (minHL + maxHL) / 2

                        pd_series = pandas.Series(HL)
                        counts = pd_series.value_counts()
                        HLentropy = entropy(counts)

                        maxHH = np.max(HH)
                        minHH = np.min(HH)
                        AvHH = np.average(HH)
                        MedianHH = (minHH + maxHH) / 2

                        pd_series = pandas.Series(HH)
                        counts = pd_series.value_counts()
                        HHentropy = entropy(counts)

                        f3.write(str(maxLL) + '\t' + str(maxLH) + '\t' + str(maxHL) + '\t' + str(maxHH) + '\t' + str(
                            minLL) + '\t' + str(minLH) + '\t' + str(minHL) + '\t' + str(minHH) + '\t' + str(
                            AvLL) + '\t' + str(AvLH) + '\t' + str(AvHL) + '\t' + str(AvHH) + '\t' + str(
                            MedianLL) + '\t' + str(
                            MedianLH) + '\t' + str(MedianHL) + '\t' + str(MedianHH) + '\t' + str(
                            np.std(LL)) + '\t' + str(
                            np.std(LH)) + '\t' + str(np.std(HL)) + '\t' + str(np.std(HH)) + '\t' + str(
                            LLentropy) + '\t' + str(LHentropy) + '\t' + str(HLentropy) + '\t' +
                                 str(HHentropy) + '\t' + str(LLenergy) + '\t' + str(LHenergy) + '\t' + str(
                            HLenergy) + '\t' + str(HHenergy)

                                 + '\t' + str(LHslope) + '\t' + str(HLslope) + '\t' + str(HHslope) +
                                 '\t' + str(LHstd_err) + '\t' + str(HLstd_err) + '\t' + str(HHstd_err) + '\t' + str(
                            LLwavelength) + '\t' + str(overalEnergy) + '\n')


                        twoD_feature_of_Amino_acides_before_wavelet_transform = []
                        if list != []:

                            f3.write("\t".join(list) + '\n')

                    else:  # len(twoD_feature_of_Amino_acides_before_wavelet_transform) == 0:

                        # csvwriter = csv.writer(f2)
                        f3.write("\t".join(list) + '\n')



                else:
                    twoD_feature_of_Amino_acides_before_wavelet_transform.append(list)
        print("Feature Extraction Finished")


p1 = ObtainSignalresult("Training_data\\Feature_Results_for_Train.txt")
t1 = threading.Thread(target=p1.creating_signal_vector)
t1.start()
#p2 = ObtainSignalresult("Feature_Results_for_Test.txt")
#t2 = threading.Thread(target=p2.creating_signal_vector)
#t2.start()
