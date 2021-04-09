import numpy as np
import pywt
import pywt.data
import threading




class ReadToFile:

    def __init__(self, proteindata):
        self.proteindata = proteindata

    def aminoacid_to_wave(self, argument):
# biochemical index of each amino acides

        switcher2 = {
            'A': "-0.591\t1.80\t-1.302\t-0.733\t1.57\t-0.146\n",
            'C': "-1.343\t2.50\t0.465\t-0.862\t-1.02\t-0.255\n",
            'D': "1.05\t-3.50\t0.302\t-3.656\t-0.259\t-3.242\n",
            'E': "1.357\t-3.50\t-1.453\t1.477\t0.113\t-0.837\n",
            'F': "-1.006\t2.80\t-0.59\t1.891\t-0.397\t0.412\n",
            'G': "-0.384\t-0.40\t1.652\t1.33\t1.045\t2.064\n",
            'H': "0.336\t-3.20\t-0.417\t-1.673\t-1.474\t-0.078\n",
            'I': "-1.239\t4.50\t-0.547\t2.131\t0.393\t0.816\n",
            'K': "1.831\t-3.90\t-0.561\t0.533\t-0.277\t1.648\n",
            'L': "-1.019\t3.80\t-0.987\t-1.505\t1.266\t-0.912\n",
            'M': "-0.663\t1.90\t-1.524\t2.219\t-1.005\t1.212\n",
            'N': "0.945\t-3.50\t0.828\t1.299\t-0.169\t0.933\n",
            'P': "0.189\t-1.60\t2.081\t-1.628\t0.421\t-1.392\n",
            'Q': "0.931\t-3.50\t-0.179\t-3.005\t-0.503\t-1.853\n",
            'R': "1.538\t-4.50\t-0.055\t1.502\t0.44\t2.897\n",
            'S': "-0.228\t-0.80\t1.399\t-4.76\t0.67\t-2.647\n",
            'T': "-0.032\t-0.70\t0.326\t2.213\t0.908\t1.313\n",
            'V': "-1.337\t4.20\t-0.279\t-0.544\t1.242\t-1.262\n",
            'W': "-0.595\t-0.90\t0.009\t0.672\t-2.128\t-0.184\n",
            'Y': "0.26\t-1.30\t0.83\t3.097\t-0.838\t1.512\n",
            'X': "0\t-0.24\t0.06\t0.40\t0\t0\n",
            '\n' : ""
        }

        # get() method of dictionary data type returns
        # value of passed argument if it is present
        # in dictionary otherwise second argument will
        # be assigned as default value of passed argument
        return switcher2.get(argument, "\n")

    def change(self):
        f = open(self.proteindata)
        w = open("Training_data\\Feature_Results_for_Train.txt",'w+')

        while True :
            char = f.read(1)
            if char == ">":
                w.write(">"+ f.readline())

            else:

                w.write(self.aminoacid_to_wave(char))

            if not char: break

p1 = ReadToFile("Training_data\\Train_Data.txt")
t1 = threading.Thread(target= p1.change)

t1.start()


























































#A, cD = pywt.dwt([1,2,3,4,5,8,1,5,1,12,18,2,19,20,22,11,18,142,15], 'bior2.4')
#a= [(1,8,1),(5,4,5)]
#ch1,(ch2,ch3,ch4) = ((pywt.dwt2(a,'bior2.4')))
#print(pywt.idwt2 ((ch1,(ch2,ch3,ch4)),'bior2.4'))

