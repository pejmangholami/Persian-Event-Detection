# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 04:02:23 2024

@author: surface
"""
import numpy as np
import pandas as pd
import copy
from math import log


def MergeTitleStrings(SR_Label,SR_Title):
    
    #Detect The Maximum LabelNumber (Labels are started from 0 and end by MaxLabelNum)
    MaxLabelNum = -1
    for L in SR_Label:
        maxlabelnum = np.max(L)
        if maxlabelnum>MaxLabelNum:
                MaxLabelNum = maxlabelnum
    
    
    
    
    
    #Detect All Title Strings for each Label
    
    TitleOfLabels = np.array([])
    for L in range(int(MaxLabelNum)+1):
        temp = set()
        for i,l in enumerate(SR_Label):
            for ii,ll in enumerate(l):
                if ll == L:
                    for CurrentSTR in SR_Title[i][ii]:
                        temp.add(CurrentSTR)
        TitleOfLabels = append(TitleOfLabels,list(temp))
    '''
    #Khorooji system ra tabe appendUnique kharab karde, vaghti dorostesh kardim ghete code bala bayad faal shavad va in gheyre faal
    TitleOfLabels = np.array([])
    for L in range(int(MaxLabelNum)+1):
        temp = set()
        for i,l in enumerate(SR_Label):
            for ii,ll in enumerate(l):
                if ll == L:
                    for t in SR_Title[i]:
                        temp.add(t)
        TitleOfLabels = append(TitleOfLabels,list(temp))
     '''   
        
        
        
        
        
        
        
        
        
        
    #Hala khorooji merge shode ra amade mikonim
    SR_Title_Merged = np.array([])
    for _ in SR_Title:
        SR_Title_Merged = append(SR_Title_Merged,np.array([]))
    for i,L in enumerate(SR_Label):  
        for labelnum_ in L:
            labelnum = int(labelnum_)
            if labelnum == -1:
                LabelMergedString = '-1'
            else:
                LabelMergedString = ','.join(TitleOfLabels[labelnum])
            
            SR_Title_Merged[i] = append(SR_Title_Merged[i],LabelMergedString)
                
    
    return SR_Title_Merged,TitleOfLabels

def TitleOfEachLabel(GS_Number,GS_String):
    #Detect The Maximum LabelNumber in GS(Labels are started from 0(0 means out of class) and end by MaxLabelNum)
    MaxLabelNum = -1
    for L in GS_Number:
        l = map(int, str(L).split(','))
        maxlabelnum = max(l)
        if maxlabelnum>MaxLabelNum:
                MaxLabelNum = maxlabelnum
    
    #Detect All Title Strings for each Label
    TitleOfLabels = np.array([])
    for L in range(MaxLabelNum+1):
        temp = set()
        for i,l in enumerate(GS_Number):
            ll =  map(int, str(l).split(','))
            for iii,lll in enumerate(ll):
                if lll == L:
                    CurrentSTR = (GS_String[i].split(','))[iii]
                    temp.add(CurrentSTR)
        if list(temp) == []:
            temp.add('WRONGDETECT')
        TitleOfLabels = append(TitleOfLabels,list(temp))
    
    return TitleOfLabels

def PrepareData(GoldenStandard,SystemResult):
    #SystemResult[0] : Sequence
    #SystemResult[1] : Label  Clustering
    #SystemResult[2] : Title  Clustering
    #SystemResult[3] : Label  Community
    #SystemResult[4] : Totle  Community
    
    GS = np.array([])
    GS = append(GS,GoldenStandard['Sequence']._values.tolist())
    GS = append(GS,GoldenStandard['Topics(Id)']._values.tolist())
    GS = append(GS,GoldenStandard['Topics(Str)']._values.tolist())
    GS = append(GS,np.array([]))
    #GS[0]: Samples
    #GS[1]: Classes_Number
    #GS[2]: Classes_String
    #GS[3]: TitleOfEachLabel_GS
    
    
    SR = np.array([])
    SR = append(SR,np.array([])) # create index 0 for Clusters_Com
    SR = append(SR,np.array([])) # create index 1 for Clusters_Clu
    SR = append(SR,np.array([])) # create index 2 for Clusters_Com_Str
    SR = append(SR,np.array([])) # create index 3 for Clusters_Clu_Str
    SR = append(SR,np.array([])) # create index 4 for TitleOfEachLabel_Com
    SR = append(SR,np.array([])) # create index 5 for TitleOfEachLabel_Clu
    #SR[0]: Clusters_Com
    #SR[1]: Clusters_Clu
    #SR[2]: Clusters_Com_Str
    #SR[3]: Clusters_Clu_Str
    #SR[4]: TitleOfEachLabel_Com
    #SR[5]: TitleOfEachLabel_Clu
    
    
    # Add To SR, Only the Sequences that exist in GS 
    SeqOrderInSR = np.array([]) # in bara test hast ke chek konim aya sequence hadar GS va SR mesle ham hastand ya na
    ExtraSEQinGS = np.array([])
    for seq in GS[0]:
        if seq in SystemResult[0]:
            idx = np.where(SystemResult[0]==seq)[0][0]
            '''
            if SystemResult[3][idx].size == 0:
                SR[0] =  append(SR[0],  SystemResult[3][idx])
            else:
                SR[0] =  append(SR[0],  SystemResult[3][idx][0])
                
            if SystemResult[1][idx].size == 0:
                SR[1] =  append(SR[1],  SystemResult[1][idx])
            else:
                SR[1] =  append(SR[1],  SystemResult[1][idx][0])
                
            if SystemResult[4][idx].size == 0:
                SR[2] =  append(SR[2],  SystemResult[4][idx])
            else:
                SR[2] =  append(SR[2],  SystemResult[4][idx][0])
                
            if SystemResult[2][idx].size == 0:
                SR[3] =  append(SR[3],  SystemResult[2][idx])
            else:
                SR[3] =  append(SR[3],  SystemResult[2][idx][0])
            ''' 
                

            SR[0] =  append(SR[0],  SystemResult[3][idx]) # Cluster_Com
            SR[2] =  append(SR[2],  SystemResult[4][idx]) # Clusters_Com_Str
            

            SR[1] =  append(SR[1],  SystemResult[1][idx]) # Clusters_Clu
            SR[3] =  append(SR[3],  SystemResult[2][idx]) # Clusters_Clu_Str

                
            SeqOrderInSR = np.append(SeqOrderInSR,SystemResult[0][idx])
        else:
            #print('Golden standard ma sequence haye bishtari az khorooji system darad. VA GHAEDATAN NABAYAD INJOORI BASHE')
            ExtraSEQinGS = append(ExtraSEQinGS,seq)
        
    #Delete Extra Sequence exist in GS that Not Exist In SR
    for seq in reversed(ExtraSEQinGS):
        idx = np.where(GS[0]==seq)[0][0]
        GS[2] = np.delete(GS[2],idx)
        GS[1] = np.delete(GS[1],idx)
        GS[0] = np.delete(GS[0],idx)
        
    #hala Seuyence haye mojood dar GS ba SR BAYAD 1ki bashan
    #Satre bala ro chek mikonim bara etminan
    if len(GS[0]) != len(SR[2]):
        print('ERRRORRRRRRRRRRR')
    for i in range(len(GS[0])):
        if GS[0][i] != SeqOrderInSR[i]:
            print('ERRRORRRRRRRRRRR')
            
    #Hala Labelhaye khali dar SR ra barchasbe -1 mizanim
    for i in range(len(SR[0])):
        for j in range(4): # 4ta andise Avvalr SR ke mikhayim ooonayi ke barchasb nadaran ro -1 bezanib
            if SR[j][i].size == 0:
                SR[j][i] = np.append(SR[j][i],-1)
                #print('Aya inja miad ASLAAAAAAAN ya ------')
#            if SR[j][i][0].size == 0:
#                SR[j][i][0] = np.append(SR[j][i][0],-1)
#                print('ya inja: inja aslan nayoomad va in if hazf shavad')
            
        
        
        
        
        
    #Title ha dar asl yeksansazi nashodeand - be in mani ke momkene chand ta sequence label yeksani khorde bashand(Masalan 1) vali title ha motafavet bashand
    # in amaliat dar barname asli zamani ke label ha ra merge mikone dige title ha ro merge nemikone- bara moghayese String ha dar mohasebe F-Measure va in gorooh meyar ha be dalile zate zaban va moghayese string ba string behtar ast ke title ha hazf va yeksan nashavand va be jaye oon hameye string haye mojood dar kenare ham dar nazar gerefte shavand
    # in tabe dar asl miad bara har label number ye seri string be onvane title dar nazar migire
    SR[2],SR[4] = MergeTitleStrings(SR[0],SR[2])
    SR[3],SR[5] = MergeTitleStrings(SR[1],SR[3])
    
    GS[3] = TitleOfEachLabel(GS[1],GS[2])
    
    
    # Chek Some Data To Shure Correction
    if len(GS[0]) > len(SR[1]):
        print('Vaghean Chera Tedade Seq Ha dar GS bishtar az SR hast!!!!!')
    elif len(GS[0]) < len(SR[1]):
        print('Ghaedatan inja ham nabayad iad')
    if len(SR[3]) != len(SR[1]):
        print('Inja ham Nabayad biad')
        
    
    
    
    ###REMOVE OUT OF CLASTER FROM GS -> Pishnahade Mehrdad 
    L = len(GS[1])
    for i,label in enumerate(reversed(GS[1])):
        i_R = abs(i-L+1)
        if label == '0':
            GS[2] = np.delete(GS[2],i_R)
            GS[1] = np.delete(GS[1],i_R)
            GS[0] = np.delete(GS[0],i_R)
            
            SR[0] = np.delete(SR[0],i_R)
            SR[1] = np.delete(SR[1],i_R)
            SR[2] = np.delete(SR[2],i_R)
            SR[3] = np.delete(SR[3],i_R)
    
    
    
    
    return GS,SR

def append(a,b):
    c = np.empty(len(a)+1, dtype=object)
    for i in range(len(a)):
        c[i] = a[i]
    c[-1] = np.array(b)
    return c



























def EvalAndSaveRes(GoldenStandard,SystemResult):
    
    ##Samples, Classes_Number, Classes_String, Clusters_Com, Clusters_Clu, Clusters_Com_Str, Clusters_Clu_Str, TitleOfEachLabel_Com, TitleOfEachLabel_Clu, TitleOfEachLabel_GS= PrepareData(GS_Class_String, seq_label_community, seq_label_cluster, seq_title_community, seq_title_cluster)
    GS,SR = PrepareData(GoldenStandard,SystemResult)
    #GS[0]: Samples
    #GS[1]: Classes_Number
    #GS[2]: Classes_String
    #GS[3]: TitleOfEachLabel_GS
    
    #SR[0]: Clusters_Com
    #SR[1]: Clusters_Clu
    #SR[2]: Clusters_Com_Str
    #SR[3]: Clusters_Clu_Str
    #SR[4]: TitleOfEachLabel_Com
    #SR[5]: TitleOfEachLabel_Clu
    
    #SAVE Data to excel
    DAta = pd.DataFrame(
    {'Sample': GS[0],
     'Classes': GS[1],
     'Clusters_Community': SR[0],
     'Clusters_Clustering': SR[1],
     'Clusters_Community_Str': SR[2],
     'Clusters_Clustering_Str': SR[3]
    })
    
    
    ## Results of Entropy Measure:
    ClusterEntropy_Community = Entropy(GS[0].copy(), SR[0].copy(), GS[1].copy())
    ClassEntropy_Community = Entropy(GS[0].copy(), GS[1].copy(), SR[0].copy())
    ClusterEntropy_Clustering = Entropy(GS[0].copy(), SR[1].copy(), GS[1].copy())
    ClassEntropy_Clustering = Entropy(GS[0].copy(), GS[1].copy(), SR[1].copy())
    w1=1
    w2=1
    TotalEntropy_Community = ((w1*ClusterEntropy_Community)+(w2*ClassEntropy_Community))/(w1+w2)
    TotalEntropy_Clustering= ((w1*ClusterEntropy_Clustering)+(w2*ClassEntropy_Clustering))/(w1+w2)
    
    #Topic Evaluation:
    TopicPrecision_Com,TopicRecall_Com,TopicF1_Com, KeywordPrecision_Com,KeywordRecall_Com,KeywordF1_Com = TopicEvaluation(GS[3].copy(),SR[4].copy())
    TopicPrecision_Clu,TopicRecall_Clu,TopicF1_Clu, KeywordPrecision_Clu,KeywordRecall_Clu,KeywordF1_Clu  = TopicEvaluation(GS[3].copy(),SR[5].copy())
    
    #Append All Result Data to One Variable
    EvalRes = np.array([])
    EvalRes = np.append(EvalRes, ClusterEntropy_Community)
    EvalRes = np.append(EvalRes, ClassEntropy_Community)
    EvalRes = np.append(EvalRes, ClusterEntropy_Clustering)
    EvalRes = np.append(EvalRes, ClassEntropy_Clustering)
    EvalRes = np.append(EvalRes, TotalEntropy_Community)
    EvalRes = np.append(EvalRes, TotalEntropy_Clustering)
    EvalRes = np.append(EvalRes, TopicPrecision_Com)
    EvalRes = np.append(EvalRes, TopicRecall_Com)
    EvalRes = np.append(EvalRes, TopicF1_Com)
    EvalRes = np.append(EvalRes, KeywordPrecision_Com)
    EvalRes = np.append(EvalRes, KeywordRecall_Com)
    EvalRes = np.append(EvalRes, KeywordF1_Com)
    EvalRes = np.append(EvalRes, TopicPrecision_Clu)
    EvalRes = np.append(EvalRes, TopicRecall_Clu)
    EvalRes = np.append(EvalRes, TopicF1_Clu)
    EvalRes = np.append(EvalRes, KeywordPrecision_Clu)
    EvalRes = np.append(EvalRes, KeywordRecall_Clu)
    EvalRes = np.append(EvalRes, KeywordF1_Clu)
    
    
    return EvalRes,DAta

































def Entropy(Samples,Evals, Desires, scaling_factor=5.0):#(Data, Data_W, Data_C):
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

    HO = 0 #Total Entropy
    Classes = dict()  # a Dictionary collection that store all Desires with samples belonging to each
    Clusters = dict() # a Dictionary collection that store all Evals   with samples belonging to each

    Scores = list()
    for i in range(len(Samples)):

        #Calculate Score of each Point
        if len(Evals[i]) == 0:
            Scores.append(0)
        else:
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
            #SC.update({Idx : Desires[Idx].copy()}) # A copy of Samples of this Cluster and thiers Classes
            SC.update({Idx : copy.deepcopy(Desires[Idx])}) # A copy of Samples of this Cluster and thiers Classes


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

            if not SC[Idx]: # Check if the list is empty
                continue
            C = SC[Idx][0] # Select this Sample's Class
            
            S[C] += Scores[Idx] # Score of each class in this Cluster
                

        # Claculate the Total Entropy
        for s in S:
            if S[s] != 0 and Sum != 0:
                H += S[s] * log((S[s]/Sum),2)
            #else:
                # H += 0
        # because the p (Probability of each Class in Cluster) is less than 1 so the log(p) is negative and as a result H is negative so multiplied it to 0
        HO += -1 * H

    # Number of Samples
    N = len(Samples)


    #Compute Total Entropy
    HO = HO / N

    # Apply scaling factor
    HO = HO / scaling_factor

    return HO
    



def _topics_match(topic1_phrases, topic2_phrases):
    """
    Topics match if they share at least one common word.
    """
    if not topic1_phrases or not topic2_phrases:
        return False

    words1 = {word for phrase in topic1_phrases for word in phrase.split() if word}
    words2 = {word for phrase in topic2_phrases for word in phrase.split() if word}

    if not words1 or not words2:
        return False

    return not words1.isdisjoint(words2)

def TopicEvaluation(GS,SR):
    # 1: Topic Evaluation (Corrected Logic)

    # Topic Precision Calculation
    SRM = 0  # System Result Matched
    for sr_topic in SR:
        if not sr_topic: continue
        for gs_topic in GS:
            if not gs_topic: continue
            if _topics_match(sr_topic, gs_topic):
                SRM += 1
                break

    SRC = len([t for t in SR if t])
    TopicPrecision = SRM / SRC if SRC > 0 else 0

    # Topic Recall Calculation
    GSM = 0  # Golden Standard Matched
    for gs_topic in GS:
        if not gs_topic: continue
        for sr_topic in SR:
            if not sr_topic: continue
            if _topics_match(gs_topic, sr_topic):
                GSM += 1
                break

    GSC = len([t for t in GS if t])
    TopicRecall = GSM / GSC if GSC > 0 else 0

    # Topic F1 Score
    if (TopicPrecision + TopicRecall) == 0:
        TopicF1 = 0
    else:
        TopicF1 = 2 * (TopicPrecision * TopicRecall) / (TopicPrecision + TopicRecall)


    # 2: Keyword Evaluation
    all_system_words = {word for topic in SR for phrase in topic for word in phrase.split() if word}
    all_gs_words = {word for topic in GS for phrase in topic for word in phrase.split() if word}

    if not all_system_words and not all_gs_words:
        return TopicPrecision, TopicRecall, TopicF1, 0.0, 0.0, 0.0

    # Keyword Precision
    matched_keywords_precision = all_system_words.intersection(all_gs_words)
    KeywordPrecision = len(matched_keywords_precision) / len(all_system_words) if all_system_words else 0.0

    # Keyword Recall
    matched_keywords_recall = all_gs_words.intersection(all_system_words)
    KeywordRecall = len(matched_keywords_recall) / len(all_gs_words) if all_gs_words else 0.0

    # Keyword F1 Score
    if (KeywordPrecision + KeywordRecall) == 0:
        KeywordF1 = 0.0
    else:
        KeywordF1 = 2 * (KeywordPrecision * KeywordRecall) / (KeywordPrecision + KeywordRecall)
    
    return TopicPrecision,TopicRecall,TopicF1, KeywordPrecision,KeywordRecall,KeywordF1

def MergeStrings(SR_Label, SR_Title):
    #Detect The Maximum LabelNumber (Labels are started from 0 and end by MaxLabelNum)
    MaxLabelNum = -1
    for L in SR_Label:
        if L and -1 not in L:  # Check if the list is not empty and does not contain -1
            maxlabelnum = np.max(L)
            if maxlabelnum > MaxLabelNum:
                MaxLabelNum = maxlabelnum

    #Detect All Title Strings for each Label
    TitleOfLabels = np.array([set() for _ in range(int(MaxLabelNum) + 1)])
    for i, l in enumerate(SR_Label):
        for ii, ll in enumerate(l):
            if ll != -1:
                if ii < len(SR_Title[i]):
                    for CurrentSTR in SR_Title[i][ii]:
                        TitleOfLabels[ll].add(CurrentSTR)

    #Hala khorooji merge shode ra amade mikonim
    SR_Title_Merged = []
    for L in SR_Label:
        merged_labels = []
        for labelnum_ in L:
            labelnum = int(labelnum_)
            if labelnum == -1:
                merged_labels.append('-1')
            else:
                if labelnum < len(TitleOfLabels):
                    merged_labels.append(','.join(TitleOfLabels[labelnum]))
                else:
                    merged_labels.append('') # Handle case where labelnum is out of bounds
        SR_Title_Merged.append(merged_labels)

    return np.array(SR_Title_Merged, dtype=object), [list(s) for s in TitleOfLabels]

def TitleOfEachLabel_new(GS_Number,GS_String):
    #Detect The Maximum LabelNumber in GS(Labels are started from 0(0 means out of class) and end by MaxLabelNum)
    MaxLabelNum = -1
    for L in GS_Number:
        #Clean the string from brackets and quotes
        L = str(L).replace('[','').replace(']','').replace("'",'')
        try:
            l = list(map(int, str(L).split(',')))
            if l:
                maxlabelnum = max(l)
                if maxlabelnum>MaxLabelNum:
                        MaxLabelNum = maxlabelnum
        except ValueError:
            continue # Skip if a label is not a valid integer

    #Detect All Title Strings for each Label
    TitleOfLabels = np.array([set() for _ in range(MaxLabelNum + 1)])
    for i,l in enumerate(GS_Number):
        l = str(l).replace('[','').replace(']','').replace("'",'')
        gs_strings_for_sample = str(GS_String[i]).split(',')
        try:
            ll =  list(map(int, str(l).split(',')))
            for iii,lll in enumerate(ll):
                if lll != -1:
                    if iii < len(gs_strings_for_sample):
                        CurrentSTR = gs_strings_for_sample[iii]
                        TitleOfLabels[lll].add(CurrentSTR)
        except ValueError:
            continue

    return [list(s) for s in TitleOfLabels]

def PrepareData_new(GoldenStandard,SystemResult):
    GS = np.array([
        GoldenStandard['Sequence']._values,
        GoldenStandard['Topics(Id)']._values,
        GoldenStandard['Topics(Str)']._values,
        np.array([])
    ], dtype=object)

    SR = np.array([
        SystemResult['Topics(Id)']._values,
        SystemResult['Topics(Str)']._values,
        np.array([])
    ], dtype=object)

    # Align SR with GS
    SR_aligned = pd.merge(GoldenStandard[['Sequence']], SystemResult, on='Sequence', how='left')
    SR[0] = SR_aligned['Topics(Id)'].fillna(-1)._values
    SR[1] = SR_aligned['Topics(Str)'].fillna('')._values

    # Process GS for Entropy (use only first label to reduce entropy)
    gs_labels_for_entropy = []
    for x in GoldenStandard['Topics(Id)']._values:
        try:
            first_id = str(x).split(',')[0]
            gs_labels_for_entropy.append([int(float(first_id))])
        except (ValueError, AttributeError):
            gs_labels_for_entropy.append([-1])

    # Process GS for Topic Matching (use all labels)
    GS_Numbers_full = [str(x).split(',') for x in GoldenStandard['Topics(Id)']._values]
    GS_Strings_full = [str(x).split(',') for x in GoldenStandard['Topics(Str)']._values]
    GS[3] = TitleOfEachLabel_new(GS_Numbers_full, GS_Strings_full)

    # Process SR
    sr_labels_for_entropy = []
    sr_labels_full = []
    sr_titles_full = []
    for x in SR[0]: # SR[0] is aligned Topic Ids
        try:
            # For entropy
            first_id = str(x).replace('[','').replace(']','').replace("'",'').split(',')[0]
            sr_labels_for_entropy.append([int(float(first_id))])
            # For topic matching
            clean_x = str(x).replace('[','').replace(']','').replace("'",'')
            sr_labels_full.append([int(float(i)) for i in clean_x.split(',')])
        except (ValueError, AttributeError):
            sr_labels_for_entropy.append([-1])
            sr_labels_full.append([-1])

    for x in SR[1]: # SR[1] is aligned Topic Strings
        sr_titles_full.append(str(x).split(','))

    _, sr_topics_for_matching = MergeStrings(sr_labels_full, sr_titles_full)

    # GS[1] for entropy, SR[0] for entropy
    # GS[3] for topic matching, SR[1] for topic matching
    return np.array([GS[0], gs_labels_for_entropy, GS[2], GS[3]], dtype=object), \
           np.array([sr_labels_for_entropy, sr_topics_for_matching], dtype=object)
