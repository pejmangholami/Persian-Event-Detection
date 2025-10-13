from __future__ import unicode_literals
import os
from datetime import datetime
import pickle
from streamgraph import MyGraphClass
from scipy import spatial
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def append(a,b):
    c = np.empty(len(a)+1, dtype=object)
    for i in range(len(a)):
        c[i] = a[i]
    c[-1] = np.array(b)
    return c

def DefineLabelNumberForResult(label_vector, vectors_in_result, ExistVec_Weight ,DistanceTereshold): 
    # Vazife in tabe in ast ke chek konad label_vector ke jadid ast aya yeki az vectorhaye ghabli dar vectors_in_result ast ya yek label jadid ast
    # Tavajoh shavad le khode Label haman andise vector ast
    
    DistanceToVectors = np.array([])
    

    for ExistVec in vectors_in_result:
        DistanceToVectors = append(DistanceToVectors,spatial.distance.cosine(label_vector, ExistVec))
    
    mindistanse = 1000
    if DistanceToVectors.size != 0:
        mindistanse = np.min(DistanceToVectors)
    
    if mindistanse < DistanceTereshold:
        #print('LabelMearge: Yes')
        #Yani Label Jadidi ke darim baresi mikonim dar asl ghablan dar khorooji boode va alan bayad:
            #1: shomare Label ra peyda konim
            #2: Bordare Label Ra Update konim
            #3: Weight ra Update Konim Baraye Dafaate Badi
        #1:
        LabelNum = np.where(DistanceToVectors==mindistanse)[0][0]
        FinalLabel = LabelNum
        
        #2:
        vectors_in_result[LabelNum] = np.average([vectors_in_result[LabelNum], label_vector], axis=0, weights=[ExistVec_Weight[LabelNum],1])#len(self.NodeVectors[i])])
        
        #3:
        ExistVec_Weight[LabelNum] += 1
        
    else:
        #print('LabelMearge: NO')
        #Yani ke labele Jadid ke darim baresi mikonim vaghean yel Label Jadid Ast va shomare Jadid bayad ekhtesas bedim:
            #1: shomare ii bara Label ekhtesas bedim
            #2: Bordare Label Ra Zakhire konim
            #3: Weight ra ijad Konim Baraye Dafaate Badi
        #1:
        FinalLabel = len(DistanceToVectors)
        
        #2:
        vectors_in_result = append(vectors_in_result,label_vector)
        
        #3:
        ExistVec_Weight = np.append(ExistVec_Weight,1)
    
    
    return FinalLabel,vectors_in_result,ExistVec_Weight
             

def SyncResults(SubjectGraph,SequenceStream,SeqLebClu,SeqTitClu,SeqLebCom,SeqTitCom,LabelVectorsClu,LabelVectorsCom,LabelVectorsClu_Weight,LabelVectorsCom_Weight):

    
    for seq in SequenceStream:
        for ii,sg in enumerate(SubjectGraph):
            DistanceTereshold = sg.DistanseTresholdForMergeLavelsVector
            if seq in sg.Seq2Label_Community.keys():                    
                for idx,label in enumerate(sg.Seq2Label_Community[seq]):
                    label_vector = sg.LabelAverageVector_Community[label]
                    vectors_in_result = LabelVectorsCom
                    FinalLabel,LabelVectorsCom,LabelVectorsCom_Weight = DefineLabelNumberForResult(label_vector,vectors_in_result, LabelVectorsCom_Weight ,DistanceTereshold) # shomare label dar har geraph motafavet az shomare moadelash dar khorooji hast choon dar har geraph az 1 shoroo mishan be bala va dar khorooji ham mikhayim az 1 shoroo beshan be bala(Shayadam az 0 shoroo mishan be bala) -> va label shomare 1 dar graph avval lozooman hamoon label shomare 1 dar graphe dovvom nist
                    # in tabe chek mikone ke age label_vector nazdike vector hayi ke ta be hal dar khorooji hast bood haman shomare label ra bedahad va agar nabood ye shomare bad az label akhar ra ve onvane shomare label be khorooji bedahad
                    
                    if seq in SeqLebCom:
                        if FinalLabel not in SeqLebCom[seq]:
                            SeqLebCom[seq] = np.append(SeqLebCom[seq],FinalLabel)
                            SeqTitCom[seq] = np.append(SeqTitCom[seq],sg.Seq2Title_Community[seq][idx])
                            
                            if -1 in SeqLebCom[seq]:
                                SeqTitCom[seq] = np.delete(SeqTitCom[seq],np.where(SeqLebCom[seq]==-1)[0][0])
                                SeqLebCom[seq] = np.delete(SeqLebCom[seq],np.where(SeqLebCom[seq]==-1)[0][0])
                                
                    else:
                        SeqLebCom[seq] = np.array([FinalLabel])
                        SeqTitCom[seq] = np.array([sg.Seq2Title_Community[seq][idx]])
        if seq not in SeqLebCom:
            SeqLebCom[seq] = np.array([-1],dtype=int)
            SeqTitCom[seq] = np.array(['-1'])
            
        
        for ii,sg in enumerate(SubjectGraph):
            if seq in sg.Seq2Label_Clustering.keys():                    
                for idx,label in enumerate(sg.Seq2Label_Clustering[seq]):
                    label_vector = sg.LabelAverageVector_Clustering[label]
                    vectors_in_result = LabelVectorsClu
                    FinalLabel,LabelVectorsClu,LabelVectorsClu_Weight = DefineLabelNumberForResult(label_vector,vectors_in_result, LabelVectorsClu_Weight ,DistanceTereshold) # shomare label dar har geraph motafavet az shomare moadelash dar khorooji hast choon dar har geraph az 1 shoroo mishan be bala va dar khorooji ham mikhayim az 1 shoroo beshan be bala(Shayadam az 0 shoroo mishan be bala) -> va label shomare 1 dar graph avval lozooman hamoon label shomare 1 dar graphe dovvom nist
                                 # in tabe chek mikone ke age label_vector nazdihe vector hayi ke ta be hal dar khorooji hast bood haman shomare label ra bedahad va agar nabood ye shomare bad az label akhar ra ve onvane shomare label be khorooji bedahad

                    if seq in SeqLebClu:
                        if FinalLabel not in SeqLebClu[seq]:
                            SeqLebClu[seq] = np.append(SeqLebClu[seq],FinalLabel)
                            SeqTitClu[seq] = np.append(SeqTitClu[seq],sg.Seq2Title_Clustering[seq][idx])
                            
                            if -1 in SeqLebClu[seq]:
                                SeqTitClu[seq] = np.delete(SeqTitClu[seq], np.where(SeqLebClu[seq]==-1)[0][0])
                                SeqLebClu[seq] = np.delete(SeqLebClu[seq], np.where(SeqLebClu[seq]==-1)[0][0])
                                
                    else:
                        SeqLebClu[seq] = np.array([FinalLabel])
                        SeqTitClu[seq] = np.array([sg.Seq2Title_Clustering[seq][idx]])
        if seq not in SeqLebClu:
            SeqLebClu[seq] = np.array([-1],dtype=int)
            SeqTitClu[seq] = np.array(['-1'])
       
    return SeqLebClu,SeqTitClu,SeqLebCom,SeqTitCom,LabelVectorsClu,LabelVectorsCom,LabelVectorsClu_Weight,LabelVectorsCom_Weight
    
'''
def RawOutput(Seq_Stream):
    Output = np.array([])
    for tmp in range(5):
        Output = append(Output,np.array([]))
    #Output[0] : Sequence
    #Output[1] : Label  Clustering
    #Output[2] : Title  Clustering
    #Output[3] : Label  Community
    #Output[4] : Totle  Community
    Output[0] = Seq_Stream
    print(len(Seq_Stream))
    for tmp in range(len(Seq_Stream)):
        for i in range(1,5):
            Output[i] = append(Output[i],np.array([]))
    LabelsVec = np.array([])
    LabelsVec = append(LabelsVec,np.array([]))
    LabelsVec = append(LabelsVec,np.array([]))
    LabelsVec = append(LabelsVec,np.array([]))
    LabelsVec = append(LabelsVec,np.array([]))
    #LabelsVec[0] : Vector Clustering
    #LabelsVec[1] : Vector Clustering-weight
    #LabelsVec[2] : Vector Community
    #LabelsVec[3] : Vector Community-weight
    
    return Output, LabelsVec    
'''  




def main(MergeLavelsTreshold, DeleteNodeTreshold, TimestepToCalculateScoreAndFreqFORGRAPHInSec, EventDetectionTimeInDay,ForgettingInDay, TimestepInPulseVecInSec, EdgeMergeTresh, NodeMergeTresh):
    
    #khandane Ettelaat PostHaye Telegram Az File
    StartTime = StartTime = datetime(2017, 1,  1, 00, 00, 00)
    EndTime   = datetime(2017, 1, 31, 23, 59, 59)  
    print('npy file Dataset AllData loading ...')
    path = 'DataSet/StreamContainKGs-ParsBertEmbed-SubjectIndex' + str(StartTime.date()) +' to '+ str(EndTime.date()) + '.npy'
    AllDataa = np.load(path, allow_pickle=True)
    path = 'DataSet/RawOutput' + str(StartTime.date()) +' to '+ str(EndTime.date()) + '.npy'    
    Output = np.load(path, allow_pickle=True)
    path = 'DataSet/RawLabelsVec' + str(StartTime.date()) +' to '+ str(EndTime.date()) + '.npy'
    LabelsVec = np.load(path, allow_pickle=True)
    print('file loaded')
    #AllDataa[0] Seq
    #AllDataa[1] Posts Tokens
    #AllDataa[2] Types Tokens
    #AllDataa[3] Deleted Tag
    #AllDataa[4] DateSend
    #AllDataa[5] User
    #AllDataa[6] Post String
    #AllDataa[7] Knowledge Graph Triples
    #AllDataa[8] Graph Triple Embed By Parsbert
    #AllDataa[9] SubjectIndex of each Post
    
    #Output[0] : Sequence
    #Output[1] : Label  Clustering
    #Output[2] : Title  Clustering
    #Output[3] : Label  Community
    #Output[4] : Totle  Community
    
    #LabelsVec[0] : Vector Clustering
    #LabelsVec[1] : Vector Clustering-Weight
    #LabelsVec[2] : Vector Community
    #LabelsVec[3] : Vector Community-Weight
    
    
    Label_Category = {0: 'اجتماعی',
    1: 'اقتصادی',
    2: 'بین الملل',
    3: 'سیاسی',
    4: 'علمی فناوری',
    5: 'فرهنگی هنری',
    6: 'ورزشی',
    7: 'پزشکی'}
    
    
    SubjectDataStream = np.empty(len(Label_Category), dtype=object)
    for i in range(len(Label_Category)):
      SubjectDataStream[i] = np.array([])
    
    
    SubjectGraph = np.empty(len(Label_Category), dtype=object)
    for i in range(len(Label_Category)):
      SubjectGraph[i] = MyGraphClass(MergeLavelsTreshold, DeleteNodeTreshold, TimestepToCalculateScoreAndFreqFORGRAPHInSec, EventDetectionTimeInDay,ForgettingInDay, TimestepInPulseVecInSec, EdgeMergeTresh, NodeMergeTresh)
      
    
    
    
    EventExtraction_TimeWindow = 1
    '''
    # Variable For Results:
    seq_label_cluster = {}
    seq_title_cluster = {}
    seq_label_community = {}
    seq_title_community = {}
    LabelVectorsClu = np.array([])
    LabelVectorsCom = np.array([])
    LabelVectorsClu_Weight = np.array([])
    LabelVectorsCom_Weight = np.array([])
    '''
    i = 0
    start_time = datetime.now()
    print('DateTime: '+str(start_time))
    while True:
        CurrentData = np.array([AllDataa[0][i],AllDataa[1][i],AllDataa[2][i],AllDataa[3][i],AllDataa[4][i],AllDataa[5][i],AllDataa[6][i],AllDataa[7][i],AllDataa[8][i],AllDataa[9][i]],dtype=object)
        for ii in AllDataa[9][i]:
            SubjectDataStream[ii] = np.append(SubjectDataStream[ii],CurrentData)
            SubjectGraph[ii].AddTriple(CurrentData)
            
            if((CurrentData[4] - SubjectGraph[ii].LastEventDetectionTimeInGraph).days) == EventExtraction_TimeWindow or (AllDataa[0][-1] == CurrentData[0]):#(AllDataa[4][-1] == CurrentData[4]):
                
                #SubjectGraph[ii].DetectEvents_ok(SubjectGraph[ii].LastTimeInGraph)
                Output, LabelsVec = SubjectGraph[ii].DetectFinalEvents(SubjectGraph[ii].LastTimeInGraph,Output,LabelsVec)
                
                SubjectGraph[ii].LastEventDetectionTimeInGraph = CurrentData[4]
                
                #######################seq_label_cluster,seq_title_cluster,seq_label_community,seq_title_community, LabelVectorsClu,LabelVectorsCom,LabelVectorsClu_Weight,LabelVectorsCom_Weight = SyncResults(SubjectGraph,AllDataa[0],seq_label_cluster,seq_title_cluster,seq_label_community,seq_title_community, LabelVectorsClu,LabelVectorsCom,LabelVectorsClu_Weight,LabelVectorsCom_Weight)
                
            # Forgetting mechanism and deleting node:
            if (SubjectGraph[ii].LastTimeInGraph - SubjectGraph[ii].LastDeleteNodeTimeInGraph).days >= SubjectGraph[ii].ForgetingCheck:
                #print('Forgetting Mechanism Started till Post: '+str(i)+' & Subject: ' + str(ii))
                SubjectGraph[ii].ForgetNodes(SubjectGraph[ii].LastTimeInGraph)
                SubjectGraph[ii].LastDeleteNodeTimeInGraph =  CurrentData[4]
                
        
        # ba dastoore rooberoo mishe graf ra rasm kard => "SubjectGraph[X].plot_graph()"  node ha va yalha andis hastand
        i=i+1
        if(i%1000)==0:
            end_time = datetime.now()
            print(str(i) + '/' + str(len(AllDataa[0])) + ' - Duration: ' + str(end_time - start_time) + ' on: ' + str(datetime.now()))
            start_time = datetime.now()
            
        # Ta Inja SubjectDataStream va hamin tor Subject Data Graph Ta Zamane Feli Amade Shode Va Bayad Be Algurithm Event Detection Dade Beshe
        
        
        #desicion = input('Continue?(Y/N) : ')
        #if desicion != 'y':
        if i==len(AllDataa[0]):
            # Wait untill new data arrive
            
            ResultPath = 'EvaluationResults_numpy_VecCluster/DistanceTereshold_' + str(SubjectGraph[1].DistanseTresholdForMergeLavelsVector) + '/ForgetigTreshold_' + str(SubjectGraph[1].ScoreTereshold4Delete)
            if not os.path.exists(ResultPath):
                os.makedirs(ResultPath)
                
            ##Saving SubjectDataStream
            #np.save(ResultPath + '/SubjectDataStream' + str(StartTime.date()) +' to '+ str(EndTime.date()) + '.npy', SubjectDataStream)
            #print('SubjectDataStream Saved in a file.')
            
            #Saving Output, LabelsVec
            np.save(ResultPath + '/Output' + str(StartTime.date()) +' to '+ str(EndTime.date())  + '.npy', Output)
            print('Output Saved in a file.')
            
            np.save(ResultPath + '/LabelsVec' + str(StartTime.date()) +' to '+ str(EndTime.date()) + '.npy', LabelsVec)
            print('LabelsVec Saved in a file.')
                 
            
            print('FINISHEEEEDDDDDDDDDDDDDdddd')
            
            break 
        
    return Output

   
# UnComment Next Line For Run:
#Output = main(0.04, 0.8, 2*60*60, 1, 1, 60, 0.032, 0.05)
#Output = main(0.3, 1, 2*60*60, 1, 1, 60, 0.032, 0.05)
### main(MergeLavelsTreshold, DeleteNodeTreshold, TimestepToCalculateScoreAndFreqFORGRAPHInSec, EventDetectionTimeInDay,ForgettingInDay, TimestepInPulseVecInSec, EdgeMergeTresh, NodeMergeTresh)
##8: Trereshold4Nodes = 0.05 # ebarati ke faselashoon kamtar az in meghdar bashe be onvane 1 node dar nazar gerefte mishe 
##7: Trereshold4Edge = 0.032 # ebarati ke faselashoon kamtar az in meghdar bashe be onvane 1 edge dar nazar gerefte mishe 
##6: TimeStepInPulseVector = 60 #second
##5: ForgetingCheck = 1#4 # Day 
##4: EventDetectionTime = 1 # Day (in bayad kamtar az balayi bashe hamishe) 
##2: ScoreTereshold4Delete = 0.8#1 #!!!# Score az 0 ta binahayat hast, 1 be mani yeknavakgtite tekrar va kamtar az an be mani nozooli boodan va bishtar az 1 be mani soudi boodan hast 
##3: TimeStepInGraph4AllPosts = 2*60*60 #second (Equal 2Houres): mohasebate ferekans va score bara Kolle Graph 2saat be 2saat be 2saat anjam mishe(dar baze haye 2saati) -> IN FELAN EMAL NASHODE VA FEREQUANCY VA SCORE BARA KOLLE GERAPH MOHASEBE NEMISHE
##1: DistanseTresholdForMergeLavelsVector = 0.04 #!!!# IN PARAMETR HATMAN TUNE SHAVAD ## Moshakhas mikone ke bordathaye label ha az che meghdar nazdiktar be ham bashand bayad 1ki dar nazar gerefte shavand

#Output[0] : Sequence
#Output[1] : Label  Clustering
#Output[2] : Title  Clustering
#Output[3] : Label  Community
#Output[4] : Totle  Community

    