import numpy as np
import pandas as pd
from math import log

global XDATA
#[Samples, Classes, Clusters]
XDATA = pd.read_excel('Gldn_Std.xls')

def PrepareData(XSamples):
    """
    Prepar Input Data inorder to be readable to Entropy
    """
    Samples  = XSamples['Sample']._values.tolist()
    Classes  = XSamples['Class']._values.tolist()
    Clusters = XSamples['Cluster']._values.tolist()

    # Some Samples may have more than one class or cluster
    # and because each line of Excel is read as a string
    # This turns it into an array.
    for i in range(len(Samples)):
        if not isinstance(Classes[i], int): # type(Classes[i]) != int
            Classes[i] = list(map(int, Classes[i].split(',')))
        else:
            Classes[i] = [Classes[i]]

        if not isinstance(Clusters[i], int):
            Clusters[i] = list(map(int, Clusters[i].split(',')))
        else:
            Clusters[i] = [Clusters[i]]
    return Samples, Classes, Clusters



def PrepareDataForSpecialWindow(XSamples,Clusters,WinNum):
    """
    Prepar Input Data inorder to be readable to Entropy
    """

    Samples  = XSamples['Sample']._values.tolist()
    Classes  = XSamples['Class']._values.tolist()

    if WinNum == 14:
        Samples  = Samples[0:94]
        Classes  = Classes[0:94]
    elif WinNum == 15:
        Samples  = Samples[94:150]
        Classes  = Classes[94:150]
    elif WinNum == 16:
        Samples  = Samples[150:508]
        Classes  = Classes[150:508]
    elif WinNum == 17:
        Samples  = Samples[508:]
        Classes  = Classes[508:]
    else:
        print('ERROR - For This Window Number Dose Not have Golden Standard')


    # Some Samples may have more than one class or cluster
    # and because each line of Excel is read as a string
    # This turns it into an array.
    for i in range(len(Samples)):
        if not isinstance(Classes[i], int): # type(Classes[i]) != int
            Classes[i] = list(map(int, Classes[i].split(',')))
        else:
            Classes[i] = [Classes[i]]

        if not isinstance(Clusters[i], int):
            Clusters[i] = list(map(int, Clusters[i].split(',')))
        else:
            Clusters[i] = [Clusters[i]]
    return Samples, Classes, Clusters




def Entropy(Samples,Evals, Desires):#(Data, Data_W, Data_C):
    """ Calculate the Entropy
        It calculates Entropy based on the order of second and third parameters.
        if second parameter is Clusters and third parameter is classes  it caculate Cluster Entropy.
        if second parameter is classes  and third parameter is Clusters it caculate Class Entropy.
        \n
        (Evals = Clusters & Desires = Classes) => Cluster Entropy
        \n
        (Evals = Classes  & Desires = Clusters) => Class Entropy)
        \n
        The first parameters is the samples\n
        Returns HO as total (Cluster or Class) Entropy
        (This Code is written based on Cluster Entropy. There is no difference between Class and Cluster Entropy but to swap thier palces)
    """

    global Results

    HO = 0 #Total Entropy
    Classes = dict()  # a Dictionary collection that store all Desires with samples belonging to each
    Clusters = dict() # a Dictionary collection that store all Evals   with samples belonging to each

    Scores = list()
    for i in range(len(Samples)):

        #Calculate Score of each Point
        Scores.append(1 / len(Evals[i]))

        # Initiate Classes & Clusters
        # region Classes
        for D in Desires[i]:
            if D in Classes:
                 Classes[D].append(i) # The Index of Samples
                # Classes[D].append(Samples[i]) # age moshkeli nadare Idx ezafe beshe
            else:
                Classes.update({D : [i]})
                # Classes.update({D : [Samples[i]]})
         # endregion

        # region Clusters
        for E in Evals[i]:
            if E in Clusters:
                Clusters[E].append(i)
                # Clusters[E].append(Samples[i])
            else:
                Clusters.update({E : [i]})
        # endregion

    # Calculates total Score of each Cluster  & Score of each Class in Cluster & computes the Entropy of each Cluster
    for Cluster in Clusters:

        Sum = 0 # The total Score of each Cluster
        H = 0   #Entropy of each Cluster

        # Initiate The score of each class in this Cluster
        S = dict()
        for Class in Classes:
            S.update({Class:0})

        SC = dict() # all of samples that exists in this Cluster
        FC = dict() #Frequency of each Class in this Cluster

        # The number of accurance of this Clusters' Samples in each Class
        for Idx in Clusters[Cluster]:
            # Idx = Samples.index(S)
            for c in Desires[Idx]:
                if c in FC:
                    FC[c] += 1
                else:
                    FC.update({c : 1})
            SC.update({Idx : Desires[Idx].copy()}) # A copy of Samples of this Cluster and thiers Classes

        # Find Class of each Sample in this Cluster based on the freq of Classes in this Cluster
        while len(FC) != 0:
            MaxD = max(list(FC.items()), key = lambda x:x[1]) #ret = (a,b): a->key, b -> value
            if MaxD[1] == 0: # There is no sample left that it's class is not specified
                break
            m = MaxD[0] # The Class that has maximum number of samples of this Cluster
            for s in SC:
                if m in SC[s]:
                    for i in SC[s]:
                        FC[i] -= 1
                    SC[s] = [m] # Class of every samples that has m in it's classes has to be m
            FC.pop(m)

        #Cal Score of Cluster and it's Entropy
        for Idx in Clusters[Cluster]:

            Sum += Scores[Idx] # Score of Cluster

            C = SC[Idx][0] # Select this Sample's Class

            S[C] += Scores[Idx] # Score of each class in this Cluster

        print('---------------------------')
        Results += '---------------------------\n'

        print('Score of documents in '+str(Cluster)+' = ',Sum)
        Results += 'Score of documents in ' + str(Cluster) + ' = ' + str(Sum) + '\n'

        print('Score of label in '+str(Cluster)+' = ',S)
        Results += 'Score of label in ' + str(Cluster) + ' = ' + str(S) + '\n'

        print('Entropy '+str(Cluster)+' = ')
        Results += 'Entropy ' + str(Cluster) + ' = ' + '\n'

        # Claculate the Total Entropy
        for s in S:
            if S[s] != 0:
                H += S[s] * log((S[s]/Sum),2)
                print('('+str(S[s])+'/'+str(Sum)+') * log(('+str(S[s])+'/'+str(Sum)+'),2)')
                Results += '(' + str(S[s]) + '/' + str(Sum) + ') * log((' + str(S[s]) + '/' + str(Sum)+'),2)' + '\n'
            else:
                # H += 0
                print('(0/'+str(Sum)+') * log((0/'+str(Sum)+'),2)')
                Results += '(0/' + str(Sum) + ') * log((0/' + str(Sum) + '),2)' + '\n'
        print('H = ' + str(H))
        Results += 'H = ' + str(H) + '\n'
        # because the p (Probability of each Class in Cluster) is less than 1 so the log(p) is negative and as a result H is negative so multiplied it to 0
        HO += -1 * H

    # Number of Samples
    N = len(Samples)

    print('=====================')
    Results += '=====================' + '\n'

    #Compute Total Entropy
    HO = HO / N
    print('HO = ' + str(HO))
    Results += 'HO = ' + str(HO) + '\n'

    return HO





def Entropy1(Samples,Evals, Desires):#(Data, Data_W, Data_C):

    HO = 0 #Total Entropy
    Classes = dict()  # a Dictionary collection that store all Desires with samples belonging to each
    Clusters = dict() # a Dictionary collection that store all Evals   with samples belonging to each

    Scores = list()
    for i in range(len(Samples)):

        #Calculate Score of each Point
        Scores.append(1 / len(Evals[i]))

        # Initiate Classes & Clusters
        # region Classes
        for D in Desires[i]:
            if D in Classes:
                 Classes[D].append(i) # The Index of Samples
                # Classes[D].append(Samples[i]) # age moshkeli nadare Idx ezafe beshe
            else:
                Classes.update({D : [i]})
                # Classes.update({D : [Samples[i]]})
         # endregion

        # region Clusters
        for E in Evals[i]:
            if E in Clusters:
                Clusters[E].append(i)
                # Clusters[E].append(Samples[i])
            else:
                Clusters.update({E : [i]})
        # endregion

    # Calculates total Score of each Cluster  & Score of each Class in Cluster & computes the Entropy of each Cluster
    for Cluster in Clusters:

        Sum = 0 # The total Score of each Cluster
        H = 0   #Entropy of each Cluster

        # Initiate The score of each class in this Cluster
        S = dict()
        for Class in Classes:
            S.update({Class:0})

        SC = dict() # all of samples that exists in this Cluster
        FC = dict() #Frequency of each Class in this Cluster

        # The number of accurance of this Clusters' Samples in each Class
        for Idx in Clusters[Cluster]:
            # Idx = Samples.index(S)
            for c in Desires[Idx]:
                if c in FC:
                    FC[c] += 1
                else:
                    FC.update({c : 1})
            SC.update({Idx : Desires[Idx].copy()}) # A copy of Samples of this Cluster and thiers Classes

        # Find Class of each Sample in this Cluster based on the freq of Classes in this Cluster
        while len(FC) != 0:
            MaxD = max(list(FC.items()), key = lambda x:x[1]) #ret = (a,b): a->key, b -> value
            if MaxD[1] == 0: # There is no sample left that it's class is not specified
                break
            m = MaxD[0] # The Class that has maximum number of samples of this Cluster
            for s in SC:
                if m in SC[s]:
                    for i in SC[s]:
                        FC[i] -= 1
                    SC[s] = [m] # Class of every samples that has m in it's classes has to be m
            FC.pop(m)

        #Cal Score of Cluster and it's Entropy
        for Idx in Clusters[Cluster]:

            Sum += Scores[Idx] # Score of Cluster

            C = SC[Idx][0] # Select this Sample's Class

            S[C] += Scores[Idx] # Score of each class in this Cluster

        # Claculate the Total Entropy
        for s in S:
            if S[s] != 0:
                H += S[s] * log((S[s]/Sum),2)
            #else:
                # H += 0
        # because the p (Probability of each Class in Cluster) is less than 1 so the log(p) is negative and as a result H is negative so multiplied it to 0
        HO += -1 * H

    # Number of Samples
    N = len(Samples)

    #Compute Total Entropy
    HO = HO / N

    return HO


def Evaluate(WindowNum,ClusterS):
    ClusterEntropy = 0
    ClassEntropy = 0

    global XDATA

    Samples, Classes, Clusters = PrepareDataForSpecialWindow(XDATA,ClusterS,WindowNum)

    ClusterEntropy = Entropy1(Samples.copy(), Clusters.copy(), Classes.copy())

    ClassEntropy  = Entropy1(Samples.copy(), Classes.copy(), Clusters.copy())

    return ClassEntropy,ClusterEntropy

if __name__ == '__main__':

    XlsFile = 'Pejman_T4.xls'

    TxtFile = 'Results(Pejman_T4).txt'

    Results = ''

    #[Samples, Classes, Clusters]
    XData = pd.read_excel(XlsFile)

    Samples, Classes, Clusters = PrepareData(XData)

    #Cluster Entropy
    print('Total Cluster Entropy:')
    Results += 'Total Cluster Entropy:' + '\n'
    ClusterEntropy = Entropy(Samples.copy(), Clusters.copy(), Classes.copy())

    print('\n +++++++++++++++++++++++++++++++++ \n')
    Results += '\n +++++++++++++++++++++++++++++++++ \n'

    #Class Entropy
    print('Total Class Entropy:')
    Results += 'Total Class Entropy:' + '\n'
    ClassEntropy = Entropy(Samples.copy(), Classes.copy(), Clusters.copy())

    Results += 'Total Results : ////////////////////////// \n'

    Results += 'Total Cluster Entropy = ' + str(ClusterEntropy) + '\n'
    Results += 'Total Class Entropy   = ' + str(ClassEntropy) + '\n'

    file1 = open(TxtFile,"w")
    file1.write(Results)
