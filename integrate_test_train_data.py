
import threading
import os
def integrate_Train_data_to_a_file () :
    path = os.path.join(os.getcwd(), "Training_data")
    path1 = os.path.join(os.getcwd(), "Result")
    path2 = os.path.join(os.getcwd(), "Filter_banks_feature_vectors")
#make 3 essential directories for data saving
    try:

        os.mkdir(path)

    except OSError:

        print("Creation of the directory %s failed" % path)
    try:

        os.mkdir(path1)

    except OSError:

        print("Creation of the directory %s failed" % path1)

    try:
        os.mkdir(path2)

    except OSError:

            print("Creation of the directory %s failed" % path2)
#****************************************************************************

#in F parameter operator should specufied the exact address of Moonlighting proteins
#In f2 parameters operator should specified the address of non MPs protein
    f = open("Training_data\\Train_Data.txt", 'w+')
    f1 = open("Moonlighting proteins_Dataset1.fasta" , 'r')
    f2 = open("non_Moonlighting Proteins_dataset_1_2.fasta" , 'r')

    for line in f1 :
        f.write((line[0:5].rstrip() if line[0]==">" else line.rstrip())  + ('\t1\n'  if line[0] == '>' else '\n') )
    for line in f2 :
        f.write((line[0:5].rstrip() if line[0]==">" else line.rstrip())  + ('\t2\n'  if line[0] == '>' else '\n') )

    f.close()

t1 = threading.Thread(integrate_Train_data_to_a_file())

t1.start()
t1.join()




