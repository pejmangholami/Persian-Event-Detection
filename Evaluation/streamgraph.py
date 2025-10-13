from scipy import spatial
import numpy as np
import networkx as nx
#from community import community
#import community
from community import community_louvain
from sklearn.cluster import SpectralClustering
#import arabic_reshaper
#from bidi.algorithm import get_display
#import matplotlib.pyplot as plt
from datetime import datetime
import math
import copy
from hdbscan import HDBSCAN
#from louvain import louvain
import scipy.cluster.hierarchy as hcluster


class MyGraphClass:  
    def __init__(self,MergeLavelsTreshold, DeleteNodeTreshold,TimestepToCalculateScoreAndFreqFORGRAPHInSec,EventDetectionTimeInDay,ForgettingInDay,TimestepInPulseVecInSec,EdgeMergeTresh, NodeMergeTresh): 
        
        #Data Store in a NODE
        self.NodeStrings    = np.array([]) # Store All Strings exist in a single node
        self.NodeVectors    = np.array([]) # Store All Vectors exist in a single node
        self.NodeAerageVector=np.array([])      # Store Average of all Vectors
        self.NodeDateTimes  = np.array([]) # Store List Of DateTime of Node Came in Main DataStream
        self.NodeUsers      = np.array([]) # Store List Of Users That Publish The Content in Main DataStream
        self.NodeActivityPulse = np.array([]) # Store an Array than contain how much this node happen in each time(har khane az araye yek baze zamani ra shamel mishe ke dar motaghayere TimeStepInPulseVector moshakhas shode)
        self.NodeFrequency = np.array([])
        self.NodeActiveScore = np.array([])
        self.NodeSequence = np.array([]) # in moshakhas mikone ke har Node az kodam post ya postha dorost shode
        
        #Data Store in an EDGE
        self.EdgeRelationStrings     = np.array([])    # Store All Relation Strings
        self.EdgeRelationVectors     = np.array([])    # Store All Relation Vectors
        self.EdgeAverageRelationVector=np.array([])       # Store Aberage Relation Vector
        self.EdgeDateTimes  = np.array([]) #
        self.EdgeActivityPulse = np.array([]) # Store an Array than contain how much this Edge happen in each time(har khane az araye yek baze zamani ra shamel mishe ke dar motaghayere TimeStepInPulseVector moshakhas shode)
        self.EdgeFrequency = np.array([])  #
        self.EdgeActiveScore = np.array([])  #
        self.EdgeSequence = np.array([]) # in moshakhas mikone ke har Edge az kodam post ya postha dorost shode
        
        self.RelationIndex4CreatingGraph = np.array([]) # inja 3tayi hayi hastan ke indexe gre ha va node ha ra moshakhas mikonan
        
        self.FirstTimeInGraph = datetime(2000, 1, 1, 00, 00)
        self.LastTimeInGraph = datetime(2000, 1, 1, 00, 00)
        self.LastDeleteNodeTimeInGraph = datetime(2000, 1, 1, 00, 00)
        self.LastEventDetectionTimeInGraph = datetime(2000, 1, 1, 00, 00)
        self.GraphFrequency = np.array([]) # in araye tedad rokhdade post ha (hame post ha) dar graph ra neshan midahad
        self.GraphActiveScore = np.array([]) 
         
        self.Trereshold4Nodes = NodeMergeTresh#0.05 # ebarati ke faselashoon kamtar az in meghdar bashe be onvane 1 node dar nazar gerefte mishe 
        self.Trereshold4Edge = EdgeMergeTresh#0.032 # ebarati ke faselashoon kamtar az in meghdar bashe be onvane 1 edge dar nazar gerefte mishe 
        self.TimeStepInPulseVector = TimestepInPulseVecInSec#60 #second
        self.ForgetingCheck = ForgettingInDay#1#4 # Day 
        self.EventDetectionTime = EventDetectionTimeInDay#1 # Day (in bayad kamtar az balayi bashe hamishe) 
        self.ScoreTereshold4Delete = DeleteNodeTreshold#0.8#1 ## Score az 0 ta binahayat hast, 1 be mani yeknavakgtite tekrar va kamtar az an be mani nozooli boodan va bishtar az 1 be mani soudi boodan hast 
        self.TimeStepInGraph4AllPosts = TimestepToCalculateScoreAndFreqFORGRAPHInSec#2*60*60 #second (Equal 2Houres): mohasebate ferekans va score bara Kolle Graph 2saat be 2saat be 2saat anjam mishe(dar baze haye 2saati) -> IN FELAN EMAL NASHODE VA FEREQUANCY VA SCORE BARA KOLLE GERAPH MOHASEBE NEMISHE
        self.DistanseTresholdForMergeLavelsVector = MergeLavelsTreshold#0.04 ## IN PARAMETR HATMAN TUNE SHAVAD ## Moshakhas mikone ke bordathaye label ha az che meghdar nazdiktar be ham bashand bayad 1ki dar nazar gerefte shavand
        self.KtopEntity = 2 # number of Nearest vector to label vector for drtrvt label Title
        self.ForcetoForgetOldNode = 2 # in Day for Nodes - if a node have more than this variable age in day this will be delete - in meghdar manteghan bayad bishtar az baze chek kardane forgetting bashad
        
        
    def append(self,a,b):
        c = np.empty(len(a)+1, dtype=object)
        for i in range(len(a)):
            c[i] = a[i]
        c[-1] = np.array(b)
        return c
    
    def appendUNQ(self,a,b):
        if b in a:
            return a
        return np.append(a,b)
    
    def appendUnique(self,L,T,l,t):
        #L and T is List of Labels And Title
        #l,t is list of label and title that should be append to L and T
        
        #L: [1,4,10, ...]
        #T: [ ['title1 of L1','title2 of L1',...] , ['title1 of L2','title2 of L2',...] , ...]
        ## t va l ham sakhtari moshabehe T va L darand 
        
        # avval baresi mishe ke agar har kodoom az l ha dakhele L nabood add beshe va titlesh ham add beshe
        # har kodoom ke bood faghat title marvoot be hamoon l Update beshe
        
        for i,label in enumerate(l):
            if label not in L:
                L=np.append(L,label)
                T=self.append(T,t[i]) 
            else:
                title_i = int(np.where(L==label)[0][0]) # 
                for title_str in t[i]:
                    if title_str not in T[title_i]:
                        T[title_i]=np.append(T[title_i],title_str)
        return L,T
    
    
    def appendUniqueLabelandCalcTitle(self,L,T,l,LabelsVec):
        #L and T is List of Labels And Title
        #l is list of label that should be append to L
        
        #L: [1,4,10, ...]
        #T: [ ['title1 of L1','title2 of L1',...] , ['title1 of L2','title2 of L2',...] , ...]
        ## l ham sakhtari moshabehe L darand 
        
        # avval baresi mishe ke agar har kodoom az l ha dakhele L nabood add beshe va titlesh ham add beshe
        # har kodoom ke bood faghat title marvoot be hamoon l Update beshe
        
        for i,label in enumerate(l):
            Title = self.NearestKEntityToVector(LabelsVec[int(label)],self.KtopEntity)
            if label not in L:
                L=np.append(L,label)
                T=self.append(T,Title) # 
            else:
                Label_i = int(np.where(L==label)[0][0]) 
                T[Label_i]=Title
        return L,T
    
    
    def AddTriple(self,CurrentData):
        #CurrentData[0] Seq
        #CurrentData[1] Posts Tokens
        #CurrentData[2] Types Tokens
        #CurrentData[3] Deleted Tag
        #CurrentData[4] DateSend
        #CurrentData[5] User
        #CurrentData[6] Post String
        #CurrentData[7] Knowledge Graph Triples
        #CurrentData[8] Graph Triple Embed By Parsbert
        
        if self.FirstTimeInGraph == datetime(2000, 1, 1, 00, 00):
            self.FirstTimeInGraph = CurrentData[4]
            self.LastDeleteNodeTimeInGraph = CurrentData[4]
            self.LastEventDetectionTimeInGraph = CurrentData[4]
        self.LastTimeInGraph = CurrentData[4]
            
        #self.UpdateGraphMainData() #!!!#  this fungtion update The frequency and score of graph  
        
        for i,TripleVector in enumerate(CurrentData[8]):
            ###################################################################
            # Adding Node1
            index = self.AddNode(TripleVector[0])
            if index == -1:
                # Be in mani ke bayad Node Jadid Add shavad
                self.NodeStrings = self.append(self.NodeStrings, [CurrentData[7][i][0]])
                self.NodeVectors = self.append(self.NodeVectors, [TripleVector[0]])
                self.NodeAerageVector = self.append( self.NodeAerageVector,TripleVector[0])
                self.NodeDateTimes = self.append(self.NodeDateTimes,[CurrentData[4]])
                self.NodeUsers = self.append(self.NodeUsers, [CurrentData[5]])
                self.NodeActivityPulse = self.append(self.NodeActivityPulse, [1])
                self.NodeSequence = self.append(self.NodeSequence, [CurrentData[0]])
                
                Node1_index = len(self.NodeAerageVector)-1
                
            else:
                # Be in mani ke bayad Node Mojood dar [index] Update Shavad
                self.NodeStrings[index] = np.append(self.NodeStrings[index], CurrentData[7][i][0])
                self.NodeVectors[index] = self.append(self.NodeVectors[index],TripleVector[0])
                self.NodeAerageVector[index] = np.average([self.NodeAerageVector[index], TripleVector[0]], axis=0, weights=[len(self.NodeVectors[index]), 1])
                self.NodeDateTimes[index] = np.append(self.NodeDateTimes[index],CurrentData[4])
                self.NodeUsers[index] = np.append(self.NodeUsers[index],CurrentData[5])  
                self.NodeActivityPulse[index] = self.CalculatePulse(self.NodeActivityPulse[index],self.NodeDateTimes[index][0],self.NodeDateTimes[index][-1])
                self.NodeSequence[index] = np.append(self.NodeSequence[index],CurrentData[0])
                
                Node1_index = index
            
            ###########################################################
            # Adding Edge
            index = self.AddEdge(TripleVector[1])
            if index == -1:
                # Be in mani ke bayad Edge Jadid Add shavad
                self.EdgeRelationStrings = self.append(self.EdgeRelationStrings,[CurrentData[7][i][1]])
                self.EdgeRelationVectors = self.append(self.EdgeRelationVectors,[TripleVector[1]])
                self.EdgeAverageRelationVector = self.append(self.EdgeAverageRelationVector,TripleVector[1])
                self.EdgeDateTimes = self.append(self.EdgeDateTimes,[CurrentData[4]])  #------------------------------------------------------------------------------------------
                self.EdgeActivityPulse = self.append(self.EdgeActivityPulse,[1]) #------------------------------------------------------------------------------------------
                self.EdgeSequence = self.append(self.EdgeSequence,[CurrentData[0]])
                
                Edge_index = len(self.EdgeAverageRelationVector)-1
                
            else:
                # Be in mani ke bayad Node Mojood dar [index] Update Shavad
                self.EdgeRelationStrings[index] = np.append(self.EdgeRelationStrings[index],CurrentData[7][i][1])
                self.EdgeRelationVectors[index] = self.append(self.EdgeRelationVectors[index],TripleVector[1])
                self.EdgeAverageRelationVector[index] = np.average([self.EdgeAverageRelationVector[index], TripleVector[1]], axis=0, weights=[len(self.EdgeAverageRelationVector[index]), 1])
                self.EdgeDateTimes[index] = np.append(self.EdgeDateTimes[index],CurrentData[4])  #------------------------------------------------------------------------------------------
                self.EdgeActivityPulse[index] = self.CalculatePulse(self.EdgeActivityPulse[index],self.EdgeDateTimes[index][0],self.EdgeDateTimes[index][-1]) #------------------------------------------------------------------------------------------
                self.EdgeSequence[index] = np.append(self.EdgeSequence[index],CurrentData[0]) 
                
                Edge_index = index     
            
            ##############################################################
            # Adding Node2
            index = self.AddNode(TripleVector[2])
            if index == -1:
                # Be in mani ke bayad Node Jadid Add shavad
                self.NodeStrings = self.append(self.NodeStrings,[CurrentData[7][i][2]])
                self.NodeVectors = self.append(self.NodeVectors,[TripleVector[2]])
                self.NodeAerageVector = self.append(self.NodeAerageVector,TripleVector[2])
                self.NodeDateTimes = self.append(self.NodeDateTimes,[CurrentData[4]])
                self.NodeUsers = self.append(self.NodeUsers,[CurrentData[5]])
                self.NodeActivityPulse = self.append(self.NodeActivityPulse,[1])
                self.NodeSequence = self.append(self.NodeSequence,[CurrentData[0]])
                
                Node2_index = len(self.NodeAerageVector)-1
                
            else:
                # Be in mani ke bayad Node Mojood dar [index] Update Shavad
                self.NodeStrings[index] = np.append(self.NodeStrings[index],CurrentData[7][i][2])
                self.NodeVectors[index] = self.append(self.NodeVectors[index],TripleVector[2])
                self.NodeAerageVector[index] = np.average([self.NodeAerageVector[index], TripleVector[2]], axis=0, weights=[len(self.NodeVectors[index]), 1])
                self.NodeDateTimes[index] = np.append(self.NodeDateTimes[index],CurrentData[4])
                self.NodeUsers[index] = np.append(self.NodeUsers[index],CurrentData[5])  
                self.NodeActivityPulse[index] = self.CalculatePulse(self.NodeActivityPulse[index],self.NodeDateTimes[index][0],self.NodeDateTimes[index][-1])
                self.NodeSequence[index] = np.append(self.NodeSequence[index],CurrentData[0])  
                
                Node2_index = index
                
            self.RelationIndex4CreatingGraph = self.append(self.RelationIndex4CreatingGraph,[Node1_index,Edge_index,Node2_index])
         
                
    def UpdateGraphMainData(self):
        now = self.LastTimeInGraph
        first = self.FirstTimeInGraph
        GraphLife = (now-first).total_seconds()
        if GraphLife == 0:
            GraphLife = 0.5
        Bordar_YeksanSaz = np.zeros((math.ceil(GraphLife/self.TimeStepInGraph4AllPosts)-(len(self.GraphFrequency))))
        self.GraphFrequency = np.append(self.GraphFrequency,Bordar_YeksanSaz)
        self.GraphFrequency[-1] = self.GraphFrequency[-1] + 1
        
        L = len(self.GraphActiveScore)
        DiffrenceLen = len(self.GraphFrequency)-L
        if DiffrenceLen == 0: #yani dar haman timestep ye poste dige oomade va akharin score bayad update beshe
            self.GraphActiveScore[-1] = sum(self.GraphFrequency)/GraphLife
        else:
            for i in range(DiffrenceLen):
                #CurrentFrequency = self.GraphFrequency[i+L]
                if len(self.GraphFrequency) == len(self.GraphActiveScore):
                    CurrentGraphLife = GraphLife
                else:
                    CurrentGraphLife = self.TimeStepInGraph4AllPosts * (i+L+1)
                self.GraphActiveScore = np.append(self.GraphActiveScore,sum(self.GraphFrequency[0:i+L+1])/CurrentGraphLife)        
            

    def AddNode(self,EmbeddingVector):
        distance = np.array([])
        for avgvec in self.NodeAerageVector:
            if len(avgvec.shape) != 1 or len(EmbeddingVector.shape) != 1:
                print('!!!!!!!!!!! In Ettefagh Nabayad Biofteeeeeeeeeeeeeeee')
            distance = np.append(distance,spatial.distance.cosine(avgvec, EmbeddingVector))
        
        if len(distance) == 0:
            return -1 # yani kollan ta hala hich nodii add nadhode
        
        minDistance = min(distance)
        if minDistance <= self.Trereshold4Nodes:
            #Add The Data To an existing Node
            index = np.where(distance==minDistance)[0][0] # data dar in index bayad add shavad
        else:
            index = -1 # be in mani ke node jadid bayad add shavad
        return index
    
    def AddEdge(self,EmbeddingVector):
        distance = np.array([])
        for avgvec in self.EdgeAverageRelationVector:
            if len(avgvec.shape) != 1 or len(EmbeddingVector.shape) != 1:
                print('!!!!!!!!!!! In Ettefagh Nabayad Biofteeeeeeeeeeeeeeee')
            distance = np.append(distance,spatial.distance.cosine(avgvec, EmbeddingVector))
        
        if len(distance) == 0:
            return -1 # yani ollan ta hala hich Edgeii add nadhode
        
        minDistance = min(distance)
        if minDistance <= self.Trereshold4Edge:
            #Add The Data To an existing Edge
            index = np.where(distance==minDistance)[0][0] # data dar in index bayad add shavad
        else:
            index = -1 # be in mani ke node jadid bayad add shavad
        return index
                    
    def CalculatePulse(self,CurrentPulseVector,t1,t2):
        #t1 avvalin date time va t2 akharin date time ast(t2 taze add shode)
        diff = (t2-t1).total_seconds()
        size_nahayi_pulse_vector = math.ceil(diff/self.TimeStepInPulseVector)
        size_feli_pulse_vector = len(CurrentPulseVector)
        if diff == 0 or size_feli_pulse_vector == size_nahayi_pulse_vector:
            CurrentPulseVector[-1] += 1
            return CurrentPulseVector
        elif size_feli_pulse_vector < size_nahayi_pulse_vector:
            for iiii in range(size_nahayi_pulse_vector-size_feli_pulse_vector):
                CurrentPulseVector = np.append(CurrentPulseVector,0)
            CurrentPulseVector[-1] += 1
            return CurrentPulseVector
        else:
            print('!!!!!!!!!!!!!!!!!! ASLAN NABAAAAYAAAAAAD IN PRINT SHAVAAAAAAAAAAAAAAD!!!!!!!!!!!!!!!!')
    
    def GraphClusteringMethods(self,G):
        
        # Clustering Graph:
        ##Raveshe 1:Community Detection:
        # Calculate node communities
        CommunityDetectionMethod = 3 #1:Community Detection   2:Louvain Community Detection   3:ClusterVectorMethod2
        if CommunityDetectionMethod==1:
            #partition = community.best_partition(G)
            partition_dic = community_louvain.best_partition(G)
            #Converting dict to list
            partition_List = np.array([])
            for key, value in partition_dic.items():
                #key is the index on graph and list(G.nodes)[key] is that index on Node list 
                if len(G.nodes) <= key or key < 0:
                    print('ye jaye kar eshtebah shode:')
                    print('G.Nodes: ' + str(list(G.nodes)))
                    print('LEN G.Nodes: ' + str(len(G.nodes)))                   
                    print('Key: ' + str(key))
                inx = list(G.nodes)[key]
                if len(partition_List) == inx:
                    partition_List = self.append(partition_List,[value])
                elif len(partition_List) < inx:
                    while len(partition_List) < inx:
                        partition_List = self.append(partition_List,np.array([]))
                    partition_List = self.append(partition_List,[value])
                elif len(partition_List) > inx:
                    if len(partition_List[inx]) != 0:
                        print('Ye Jayi EshtebahKardiiiiiim|||||||||||||||Albate age Multi Label bashe inja ham momkene biad')
                    partition_List[inx] = np.append(partition_List[inx],np.array(value))#value
            
            # Analyze and visualize communities
            
        elif CommunityDetectionMethod==3:
            vectors4cluster = np.array([self.NodeAerageVector[0]])
            for triple in self.RelationIndex4CreatingGraph:
                Node1AvgVec = self.NodeAerageVector[triple[0]]
                #EdgeAvgVec  = self.EdgeAverageRelationVector[triple[1]]
                Node2AvgVec = self.NodeAerageVector[triple[2]]
                
                vectors4cluster = np.append(vectors4cluster, [Node1AvgVec],axis=0)
                vectors4cluster = np.append(vectors4cluster, [Node2AvgVec],axis=0)
            vectors4cluster = np.delete(vectors4cluster,0,axis=0)     
            # clustering
            thresh = 23#20
            clusters = hcluster.fclusterdata(vectors4cluster, thresh, criterion="distance")
            #print('NumberOfCluster111:')
            #print(max(clusters))
            partition_List = np.array([])
            for _ in range(len(self.NodeSequence)):
                partition_List = self.append(partition_List,np.array([]))
            Real_i = 0
            for i in range(0, len(clusters), 2):
                l1 = clusters[i]
                l2 = clusters[i+1]
                #i is the Triple index on RelationIndex4CreatingGraph and l1 is the label of first node in triple and l2 is the label for 2nd node in teiple
                i1_in_SG = self.RelationIndex4CreatingGraph[Real_i][0]
                i2_in_SG = self.RelationIndex4CreatingGraph[Real_i][2]
                #i_in_SG_Edge = self.RelationIndex4CreatingGraph[i][1]
                
                partition_List[i1_in_SG] = self.appendUNQ(partition_List[i1_in_SG],l1)
                partition_List[i2_in_SG] = self.appendUNQ(partition_List[i2_in_SG],l2)
                
                Real_i+=1
                
            
            #labels should be start from 0, not 1:
            for i in range(len(partition_List)):
                partition_List[i] = partition_List[i]-1
                
               
        
        ##Ravesh 2:Spectral Clustering or HDBSCAN:
        # Cluster nodes based on graph structure
        ClusteringMethod = 3 #1:Spectral Clustering   2:HDBSCAN   3: Average Vector Clustering 
        if ClusteringMethod==1:
            
            k = int(np.ceil(len(G.nodes)/10)) # k is number of cluster if not affinity="precomputed"
            #print('NumberOfCluster:' + str(k))
            clustering = SpectralClustering(affinity="precomputed")#(n_clusters=k)#
            cluster_labels = clustering.fit_predict(nx.adjacency_matrix(G))
            # index in cluster_labels indicate node index in networkx graph but iy should be indicate the index on NodeList
            labels = np.array([])
            for i, l in enumerate(cluster_labels):
                #i is the index on graph and list(G.nodes)[i] is that index on Node list 
                inx = list(G.nodes)[i]
                if len(labels) == inx:
                    labels = self.append(labels,[l])
                elif len(labels) < inx:
                    while len(labels) < inx:
                        labels = self.append(labels,np.array([]))
                    labels = self.append(labels,[l])
                elif len(labels) > inx:
                    if len(labels[inx]) != 0:
                        print('Ye Jayi EshtebahKardiiiiiim||||||||||||| Albate age Multi Label bashe inja ham momkene biad')
                    labels[inx] = np.append(labels[inx],np.array(l))#l
            # Analyze and visualize clusters
        
        elif ClusteringMethod==2:
            
            # Assuming you have your graph as a list of edges
            edges = list(G.edges()) # ex.:[(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)]
            
            # Create HDBSCAN object
            clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
            
            # Fit HDBSCAN to the graph
            clusterer.fit(edges)
            
            labels = np.array(clusterer.labels_)
            ###In ravesh Edge ha ra label mizane vali ma node ha ra mikhayim ke label bezanim. bayad agar khastim azash estefade konim ye fekri be halesh bokonim:
            #   1: Masalan mishe aslan be kol ba edge ha label bezanim choon har edge ham mitoone ye sequence dasgte bashe     
        
        elif ClusteringMethod==3:
                
            vectors4cluster = np.array([np.concatenate((self.NodeAerageVector[0], self.NodeAerageVector[0],self.NodeAerageVector[0]), axis=None)])
            for triple in self.RelationIndex4CreatingGraph:
                Node1AvgVec = self.NodeAerageVector[triple[0]]
                EdgeAvgVec  = self.EdgeAverageRelationVector[triple[1]]
                Node2AvgVec = self.NodeAerageVector[triple[2]]
                
                vectors4cluster = np.append(vectors4cluster, [np.concatenate((Node1AvgVec, EdgeAvgVec,Node2AvgVec), axis=None)],axis=0)
            vectors4cluster = np.delete(vectors4cluster,0,axis=0)    
            # clustering
            thresh = 38#35
            clusters = hcluster.fclusterdata(vectors4cluster, thresh, criterion="distance")
            #print('NumberOfCluster222:')
            #print(max(clusters))
            labels = np.array([])
            for _ in range(len(self.NodeSequence)):
                labels = self.append(labels,np.array([]))
            for i, l in enumerate(clusters):
                #i is the Triple index on RelationIndex4CreatingGraph and l is the label of that triple
                i1_in_SG = self.RelationIndex4CreatingGraph[i][0]
                i2_in_SG = self.RelationIndex4CreatingGraph[i][2]
                #i_in_SG_Edge = self.RelationIndex4CreatingGraph[i][1]
                
                labels[i1_in_SG] = self.appendUNQ(labels[i1_in_SG],l)
                labels[i2_in_SG] = self.appendUNQ(labels[i2_in_SG],l)
        
            #labels should be start from 0, not 1:
            for i in range(len(labels)):
                labels[i] = labels[i]-1
        
        
        
        # index of [labels] and [partition_List] is the Node index in MailNraph
        return labels,partition_List
    
    def DetectFinalEvents(self,now,FinalEvents,LabelsVec):
        ### dar in tabe graph mojood cluster mishe va bad har cluster moadele yek event dar nazar gerefte mishe va bad node haye ba score bala dar har cluster be onvane reshteii be onvane namayande event dar nazar gerefte mishe
        ### In Tabe Dar Har zamani ke gharare event ha moshakhas beshe farakhani mishe
        
        ##### in tabe bayad motaghayere FinalEvents ro takmil kone va labelhayi ke tashkhis dade ro bezare toosh
        
        #FinalEvents[0] : Sequence
        #FinalEvents[1] : Label  Clustering
        #FinalEvents[2] : Title  Clustering
        #FinalEvents[3] : Label  Community
        #FinalEvents[4] : Totle  Community
        
        #LabelsVec[0] : Vector Clustering
        #LabelsVec[1] : Vector Clustering-Weight
        #LabelsVec[2] : Vector Community
        #LabelsVec[3] : Vector Community-Weight
        
        #Calculate ActiveScore and Frequency for All Nodes and Edges till now
        self.Calc_Score_Freq(now)
        # Ta inja ba ejraye dastoore fogh, NodeActiveScore va NodeFrequency Bara hameye NodeHaye Mojood dar graph (va hamcheninbara hameye Edge ha) Mohasebe shode va zakhire shode
        
        # Create a weighted Graph(weight is the Score of edge/node)
        ## RelationIndex4CreatingGraph : [[0,2,4][2,3,8][5,0,8]...]
        ## NodeActiveScore : [.4,.5,045,...]
        ## EdgeActiveScore : [.34,.5,.8,...]
        
        # Create the graph
        G = nx.Graph()
        
        # Add nodes and edges with weights
        for triple in self.RelationIndex4CreatingGraph:
            source, target, weight = triple[0], triple[2], self.EdgeActiveScore[triple[1]]
            G.add_edge(source, target, weight=weight)
        #choon Dar Delete kardan emkane vojoode Nodi ke hich yali nadashte bashad hast pas in mored dar clustering moshkel saz mishe, choon ma ba andis kar mikonim va vaghti andisha peyvaste nabashan moshkel saz mishe. pas ya oon node ha ro ham bayad hazf konim va ya inja ye fekri bokonim ke graph tori dorost beshe ke an node ham biad dakhelesh(choon an andis nemiad dakhele graph pas dar natije clustering ham ghaedatan nemiad pas andis ha eshtebah mishan)
        ##Dar Morede Moshkeli Ke too khatte ghabl zekr shod, Ma Node haye tanha ra pak mikonim alan(Single NodeHa ra Delete mikonim) ->> Baresi shavad ke yek gerafe connected darim ya baz ham momkene Graph shamele chand ta graph e mojaza az ham beshe
        
        
        
        labels_Clustering,labels_Community = self.GraphClusteringMethods(G)
        
        
        
        #In Tabe Bayad EventHaye Tashkhis dade shode ra be ezaye har Seq Zakhire kone               
        #In Maghadir Yek bar be soorate Label bayad moshakhas beshe ke har Sequence che Labeli darad 
        #Yek Bar Ham Onvane Event Bara Har Sequence Bayad Moshakhas Beshe
        
        FinalEvents, LabelsVec = self.AddFinalLabels(labels_Clustering, labels_Community, FinalEvents, LabelsVec)
        
        
        return FinalEvents, LabelsVec
    
    
    
    def AddFinalLabels(self, labels_Clustering, labels_Community, FinalEvents, FinalLabelsVec):
        #FinalEvents[0] : Sequence
        #FinalEvents[1] : Label  Clustering
        #FinalEvents[2] : Title  Clustering
        #FinalEvents[3] : Label  Community
        #FinalEvents[4] : Title  Community
        
        #FinalLabelsVec[0] : Vector Clustering
        #FinalLabelsVec[1] : Vector Clustering-Weight
        #FinalLabelsVec[2] : Vector Community
        #FinalLabelsVec[3] : Vector Community-Weight
        
        # index in FinalLabelsVec -> label number
        #FinalLabelsVec[0] : Vector Clustering
        #FinalLabelsVec[1] : Vector Community
        
        
        # inja in amaliat ra baraye natije clustering (labels_Clustering) va natije Community Detection (labels_Community) Bayad anjam bedim:
        #1: LabelHa Bar Asase andise node graph hastan avval Sequence moadele har label estekhraj mishe -> Arayeii az sequence ha be tartibi ke dar arayeye labelha oomadan ke Vectore har Seq ham shamel mishe -> Esme Motaghayer: SeqOfNode
        #   Motaghayere SeqOfNode ham sequence ha ra darad ham stringe Namayande an node
        #2: Hala motaghayere SeqOfNode peymayesh mishe va har seq az an dakhele FinalEvents[0] jostojoo mishe agar mojood bood andise marboot be oon too araye SeqOfNode dar araye Labelha meghdarash bardashte mishe -> in maghadir labelhaye oon Sequence hastand:
        #   Bayad in maghadir ro dar andis haye marboote dar FinalEvent Jaygozari konim amma bayad ghablesh chek beshe bebinim Aya LabelHaye ghabli agar be in shabih hastand shomare label beshe shomare ooni ke behesh shabihe 
        #2-1: Bordare LabelHa ba tamamiye bordarhayi ke ta be hal too motaghayere FinalOutput oomade Moghayese mishe:
        #       Agar bordari nazdikesh peyda shod Shomare Labelii ke gharare darj beshe oon shomare mishe va ham label darj mishe ham title va bordar ham Average gerefte mishe
        #       Agar peyda nashod: shomare label yek shomare az akharin label mojood dar FinalEvent bishtar dar nazar gerefte mishe va hame mavared darj mishan       
        
        
        ### Mohasebat Bara Natije Clustering:
        #1:
        SeqOfNode = self.DetectSeqsPOfNode(labels_Clustering) #-> [ [122300,13345.95803] , [1345,676754] , [Seqs of node3] , [Srqs  of Node4] ... ]
        labelsvec,labels_tit = self.Calc_labelsvec(labels_Clustering) #-> [ [label 0 vec] , [label 1 vec] , ...]
        
        #2:
        for node_i,seqS in enumerate(SeqOfNode):
            first_LabelS_of_node = labels_Clustering[int(node_i)]
            
            for seq in seqS:
                search = np.where(FinalEvents[0]==seq)
                if len(search[0]) == 0 or len(search[0]) > 1:
                    print('GhatAn Eshtebah kardim !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                else:
                    final_i = search[0][0]
                    final_labels,FinalLabelsVec[0:2] = self.MergeLabel(first_LabelS_of_node,labelsvec,FinalLabelsVec[0:2]) # in tabe bodrarhaye marboot be Labels ha ra chek mikonad dar FinalLabelVec va ham FinalLabelVec update mishe ham shomare label hayi ke bayad jaygozine LabelS beshan ra be khorooji mide
                    final_titles = labels_tit

                    #FinalEvents[1][final_i],FinalEvents[2][final_i] = self.appendUnique(FinalEvents[1][final_i],FinalEvents[2][final_i],final_labels,final_titles)
                    FinalEvents[1][final_i],FinalEvents[2][final_i] = self.appendUniqueLabelandCalcTitle(FinalEvents[1][final_i],FinalEvents[2][final_i],final_labels,FinalLabelsVec[0])
                    
        ### Mohasebat Bara Natije Community:
        #1:
        SeqOfNode = self.DetectSeqsPOfNode(labels_Community) #-> [ [[122300,13345.95803][title,title,...]] , [[1345,676754][title,title,...]] , [Seqs,title of node3] , [Srqs,title of Node4] ... ]
        labelsvec,labels_tit = self.Calc_labelsvec(labels_Community) #-> [ [label 0 vec] , [label 1 vec] , ...]
        
        #2:
        for node_i,seqS in enumerate(SeqOfNode):
            first_LabelS_of_node = labels_Community[int(node_i)]
            
            for seq in seqS:
                search = np.where(FinalEvents[0]==seq)
                if len(search[0]) == 0 or len(search[0]) > 1:
                    print('!!GhatAn Eshtebah kardim !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                else:
                    final_i = search[0][0]
                    final_labels,FinalLabelsVec[2:4] = self.MergeLabel(first_LabelS_of_node,labelsvec,FinalLabelsVec[2:4]) # in tabe bodrarhaye marboot be Labels ha ra chek mikonad dar FinalLabelVec va ham FinalLabelVec update mishe ham shomare label hayi ke bayad jaygozine LabelS beshan ra be khorooji mide
                    final_titles = labels_tit
                    
                    #FinalEvents[3][final_i],FinalEvents[4][final_i] = self.appendUnique(FinalEvents[3][final_i],FinalEvents[4][final_i],final_labels,final_titles)
                    FinalEvents[3][final_i],FinalEvents[4][final_i] = self.appendUniqueLabelandCalcTitle(FinalEvents[3][final_i],FinalEvents[4][final_i],final_labels,FinalLabelsVec[2])
        return FinalEvents, FinalLabelsVec
    
    
    def DetectSeqsPOfNode(self,labels):
        #-out> [ [[122300,13345.95803][title,title,...]] , [[1345,676754][title,title,...]] , [Seqs,title of node3] , [Srqs,title of Node4] ... ]
        SeqOfNode = np.array([])
        
        for node_i,label in enumerate(labels):
            seqS  = self.NodeSequence[node_i]
            SeqOfNode = self.append(SeqOfNode,seqS)
        
        return SeqOfNode
    

    def Calc_labelsvec(self,labels):
        #-out1> [ [label 0 vec] , [label 1 vec] , ...]
        #-out2> [ [label 0 title] , [label 1 title] , ...]
        labelsvec = np.array([])
        
        max_label_number = -2
        for node_labels in labels:
            tmp = np.max(node_labels)
            if max_label_number < tmp:
                max_label_number = int(tmp)
        
        NodeVecS_of_label = np.array([])
        for tmp in range(int(max_label_number)+1):
            NodeVecS_of_label = self.append(NodeVecS_of_label,np.array([]))
            
            
        for node_i,node_labels in enumerate(labels):
            for label_num in node_labels:
                NodeVecS_of_label[int(label_num)] = self.append(NodeVecS_of_label[int(label_num)],self.NodeAerageVector[node_i])
            
                
        for label_num in range(max_label_number+1):
            avg_vec = np.average(NodeVecS_of_label[label_num], axis=0)
            labelsvec = self.append(labelsvec,avg_vec)
            
        labelTitle = np.array([])
        for lv in labelsvec:
            lt = self.NearestKEntityToVector(lv,self.KtopEntity)
            labelTitle = self.append(labelTitle,lt)
        
        return labelsvec,labelTitle
    
    
    def NearestKEntityToVector(self,vec,k):
        DistanceToVec = np.array([])
        for VEC in self.NodeAerageVector:
            DistanceToVec = np.append(DistanceToVec,spatial.distance.cosine(VEC, vec))
        
        TopK_index = np.argsort(DistanceToVec)[:k]
        
        Titles = np.array([])
        for i in TopK_index:
            Best_Title = self.Best_Edge_of_Node(i)
            Titles = np.append(Titles, Best_Title)
            #print('____________________________')
            #print(Titles[-1])
        return Titles
    
    def Best_Edge_of_Node(self,Node_i):
        # detect best edge from/to node_i and return the related triple string 
        MaxEdgeScore = -1000
        for triple in self.RelationIndex4CreatingGraph:
            if triple[0] == Node_i or triple[2] == Node_i:
                if MaxEdgeScore < self.EdgeActiveScore[triple[1]]:
                    MaxEdgeScore = self.EdgeActiveScore[triple[1]]
                    Best_Title = self.NodeStrings[triple[0]][0] + ' ' + self.EdgeRelationStrings[triple[1]][0] + ' ' + self.NodeStrings[triple[2]][0]
            
        return Best_Title
    
    
    def MergeLabel(self,first_LabelS_of_node,first_labelsvec,FinalVecOfLabel):
        # in tabe borarhaye marboot be Labels ha ra chek mikonad dar FinalVecOfLabel[0] va ham FinalVecOfLabel update mishe ham shomare label hayi ke bayad jaygozine LabelS beshan ra be khorooji mide
        #FinalVecOfLabel[0] -> bordare labelha
        #FinalVecOfLabel[1] -> vazn ya haman weight ya dar asl tedad seq hayi ke dar an label vojood darad        
        #FinalLabelNumber az 0 ta len(FinalVecOfLabel[0])-1 hast
        # shomare Label dar asl andise FinalVecOfLabel[0] ast
        
        
        final_labels = np.array([])
        for label in first_LabelS_of_node:
            #label: ooniye ke clustering ya community ekhtesas dade
            vec = first_labelsvec[int(label)] # inam bordariye ke az natije khooshebandi hesab shode na natije nahayi
            
            # Hala faseleye vec ra ba tamame natayeje nahaei hesab mikonim:
            DistanceToFinalVec = np.array([])
            for VEC in FinalVecOfLabel[0]:
                DistanceToFinalVec = np.append(DistanceToFinalVec,spatial.distance.cosine(VEC, vec))
            
            mindistanse = 1000
            if DistanceToFinalVec.size != 0:
                mindistanse = np.min(DistanceToFinalVec)
            
            if mindistanse < self.DistanseTresholdForMergeLavelsVector:
                LABEL = np.argmin(DistanceToFinalVec)
                #hala bayad FinalVecOfLabel[LABEL] update shavad - ke niaz be bordare vazn darim
                FinalVecOfLabel[0][LABEL] = np.average([FinalVecOfLabel[0][LABEL], vec], axis=0, weights=[FinalVecOfLabel[1][LABEL],1])
                FinalVecOfLabel[1][LABEL] += 1
            else:
                LABEL = len(FinalVecOfLabel[0])
                #Hala inja bayad first_labelsvec[label] append shavad dar FinalVecOfLabel -->> if LABEL != len(FinalVecOfLabel[0]) ERROR, dar in soorat ye while bayad inja bezarim
                if LABEL != len(FinalVecOfLabel[0]):
                    print('ERROR: Bayad te while inja ezafe konim ke labelhayi ke ja moondan , jashoon ro khali bezarim')
                FinalVecOfLabel[0] = self.append(FinalVecOfLabel[0],vec)
                FinalVecOfLabel[1] = np.append(FinalVecOfLabel[1],1)
            
            final_labels = np.append(final_labels,int(LABEL))
        
        return final_labels,FinalVecOfLabel
    
    
        
    def DetectEvents_ok(self,now):
        ### dar in tabe graph mojood cluster mishe va bad har cluster moadele yek event dar nazar gerefte mishe va bad node haye ba score bala dar har cluster be onvane reshteii be onvane namayande event dar nazar gerefte mishe
        ### In Tabe Dar Har zamani ke gharare event ha moshakhas beshe farakhani mishe
        
        #####in tabe ham bayad listi az event ha dar ghalebe string baraye har zaman bede va HAM IN KE clusterii az sequence ha bede ke har sequence moshakhas bashe ke ozve chand cluster hast (dar asl har post ch barchasbe rooydadi mikhore)
        
        #Calculate ActiveScore and Frequency for All Nodes and Edges till now
        self.Calc_Score_Freq(now)
        # Ta inja ba ejraye dastoore fogh, NodeActiveScore va NodeFrequency Bara hameye NodeHaye Mojood dar graph (va hamcheninbara hameye Edge ha) Mohasebe shode va zakhire shode
        
        
        # Create a weighted Graph(weight is the Score of edge/node)
        ## RelationIndex4CreatingGraph : [[0,2,4][2,3,8][5,0,8]...]
        ## NodeActiveScore : [.4,.5,045,...]
        ## EdgeActiveScore : [.34,.5,.8,...]
        
        # Create the graph
        G = nx.Graph()
        
        # Add nodes and edges with weights
        for triple in self.RelationIndex4CreatingGraph:
            source, target, weight = triple[0], triple[2], self.EdgeActiveScore[triple[1]]
            G.add_edge(source, target, weight=weight)
        #choon Dar Delete kardan emkane vojoode Nodi ke hich yali nadashte bashad hast pas in mored dar clustering moshkel saz mishe, choon ma ba andis kar mikonim va vaghti andisha peyvaste nabashan moshkel saz mishe. pas ya oon node ha ro ham bayad hazf konim va ya inja ye fekri bokonim ke graph tori dorost beshe ke an node ham biad dakhelesh(choon an andis nemiad dakhele graph pas dar natije clustering ham ghaedatan nemiad pas andis ha eshtebah mishan)
        ##Dar Morede Moshkeli Ke too khatte ghabl zekr shod, Ma Node haye tanha ra pak mikonim alan(Single NodeHa ra Delete mikonim) ->> Baresi shavad ke yek gerafe connected darim ya baz ham momkene Graph shamele chand ta graph e mojaza az ham beshe
        
        
        # Clustering Graph:
        ##Raveshe 1:Community Detection:
        # Calculate node communities
        CommunityDetectionMethod = 3 #1:Community Detection   2:Louvain Community Detection 3:ClusterVectorMethod2
        if CommunityDetectionMethod==1:
            #partition = community.best_partition(G)
            partition_dic = community_louvain.best_partition(G)
            #Converting dict to list
            partition_List = np.array([])
            for key, value in partition_dic.items():
                #key is the index on graph and list(G.nodes)[key] is that index on Node list 
                if len(G.nodes) <= key or key < 0:
                    print('ye jaye kar eshtebah shode:')
                    print('G.Nodes: ' + str(list(G.nodes)))
                    print('LEN G.Nodes: ' + str(len(G.nodes)))                   
                    print('Key: ' + str(key))
                inx = list(G.nodes)[key]
                if len(partition_List) == inx:
                    partition_List = np.append(partition_List,value)
                elif len(partition_List) < inx:
                    while len(partition_List) < inx:
                        partition_List = np.append(partition_List,0)
                    partition_List = np.append(partition_List,value)
                elif len(partition_List) > inx:
                    if partition_List[inx] != 0:
                        print('Ye Jayi EshtebahKardiiiiiim')
                    partition_List[inx] = value
            
            # Analyze and visualize communities
            
            
        elif CommunityDetectionMethod==3:
            vectors4cluster = np.array([])
            for triple in self.RelationIndex4CreatingGraph:
                Node1AvgVec = self.NodeAerageVector[triple[0]]
                EdgeAvgVec  = self.EdgeAverageRelationVector[triple[1]]
                Node2AvgVec = self.NodeAerageVector[triple[2]]
                
                vectors4cluster = self.append(vectors4cluster,Node1AvgVec)
                vectors4cluster = self.append(vectors4cluster,Node2AvgVec)
                
            # clustering
            thresh = 1.5
            clusters = hcluster.fclusterdata(vectors4cluster, thresh, criterion="distance")
            
            partition_List = np.array([])
            for _ in range(len(self.NodeAerageVector[triple[0]])):
                partition_List = self.append(partition_List,np.array([]))
            for i in range(0, len(clusters), 2):
                l1 = clusters[i]
                l2 = clusters[i+1]
                #i is the Triple index on RelationIndex4CreatingGraph and l1 is the label of first node in triple and l2 is the label for 2nd node in teiple
                i1_in_SG = self.RelationIndex4CreatingGraph[i][0]
                i2_in_SG = self.RelationIndex4CreatingGraph[i][2]
                #i_in_SG_Edge = self.RelationIndex4CreatingGraph[i][1]
                
                partition_List[i1_in_SG] = np.appendUNQ(partition_List[i1_in_SG],l1)
                partition_List[i2_in_SG] = np.appendUNQ(partition_List[i2_in_SG],l2)
                
                
        
        ##Ravesh 2:Spectral Clustering or HDBSCAN:
        # Cluster nodes based on graph structure
        ClusteringMethod = 2 #1:Spectral Clustering     2. Average Vector Clustering    3:HDBSCAN
        if ClusteringMethod==1:
            
            k = int(np.ceil(len(G.nodes)/10)) # k is number of cluster if not affinity="precomputed"
            #print('NumberOfCluster:' + str(k))
            clustering = SpectralClustering(n_clusters=k)#(affinity="precomputed")#
            cluster_labels = clustering.fit_predict(nx.adjacency_matrix(G))
            # index in cluster_labels indicate node index in networkx graf but iy should be indicate the index on NodeList abd etc
            labels = np.array([])
            for i, l in enumerate(cluster_labels):
                #i is the index on graph and list(G.nodes)[i] is that index on Node list 
                inx = list(G.nodes)[i]
                if len(labels) == inx:
                    labels = np.append(labels,l)
                elif len(labels) < inx:
                    while len(labels) < inx:
                        labels = np.append(labels,0)
                elif len(labels) > inx:
                    labels[inx] = l
            # Analyze and visualize clusters
        
        elif ClusteringMethod==2:
                
            vectors4cluster = np.array([])
            for triple in self.RelationIndex4CreatingGraph:
                Node1AvgVec = self.NodeAerageVector[triple[0]]
                EdgeAvgVec  = self.EdgeAverageRelationVector[triple[1]]
                Node2AvgVec = self.NodeAerageVector[triple[2]]
                
                vectors4cluster = self.append(vectors4cluster, np.concatenate((Node1AvgVec, EdgeAvgVec,Node2AvgVec), axis=None))
                
            # clustering
            thresh = 1.5
            clusters = hcluster.fclusterdata(vectors4cluster, thresh, criterion="distance")
            
            labels = np.array([])
            for _ in range(len(self.NodeAerageVector[triple[0]])):
                labels = self.append(labels,np.array([]))
            for i, l in enumerate(clusters):
                #i is the Triple index on RelationIndex4CreatingGraph and l is the label of that triple
                i1_in_SG = self.RelationIndex4CreatingGraph[i][0]
                i2_in_SG = self.RelationIndex4CreatingGraph[i][2]
                #i_in_SG_Edge = self.RelationIndex4CreatingGraph[i][1]
                
                labels[i1_in_SG] = np.appendUNQ(labels[i1_in_SG],l)
                labels[i2_in_SG] = np.appendUNQ(labels[i2_in_SG],l)
                
                
        
        elif ClusteringMethod==3:
            
            # Assuming you have your graph as a list of edges
            edges = list(G.edges()) # ex.:[(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)]
            
            # Create HDBSCAN object
            clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
            
            # Fit HDBSCAN to the graph
            clusterer.fit(edges)
            
            labels = np.array(clusterer.labels_)
            ###In ravesh Edge ha ra label mizane vali ma node ha ra mikhayim ke label bezanim. bayad agar khastim azash estefade konim ye fekri be halesh bokonim:
            #   1: Masalan mishe aslan be kol ba edge ha label bezanim choon har edge ham mitoone ye sequence dasgte bashe     
        
        
        
        
        
        
        
        
        
        #In Tabe Bayad EventHaye Tashkhis dade shode ra be ezaye har Seq Zakhire kone               
        #In Maghadir Yek bar be soorate Label bayad moshakhas beshe ke har Sequence che Labeli darad : esme motaghayer: Seq2Label_Community , Seq2Label_Clustering
        #Yek Bar Ham Onvane Event Bara Har Sequence Bayad Moshakhas Beshe: Esme Motaghayer: Seq2Title_Community , Seq2Title_Clustering (inha dar asl haman motaghayerhaye ghabli hastan ba in tafavit ke be jaye label cluster yek onvan motanazer ba an neveshte mishavad)
        
        #Update self.Seq2Label_Community & self.Seq2Title_Community
        self.Seq2Title_Community, self.Seq2Label_Community, self.LabelAverageVector_Community = self.AddLabel2Seq(self.Seq2Title_Community,self.Seq2Label_Community,self.LabelAverageVector_Community,partition_List)
        
        #Update self.Seq2Label_Clustering & self.Seq2Title_Clustering
        self.Seq2Title_Clustering, self.Seq2Label_Clustering, self.LabelAverageVector_Clustering = self.AddLabel2Seq(self.Seq2Title_Clustering,self.Seq2Label_Clustering,self.LabelAverageVector_Clustering,labels)
        
        
        
        #Khorooji in fungtion bayad in mavared bashe:
        #1. SequenceEventClassterLabel, SequenceEventCommunityLabel:  ye list ke neshoon mide har Sequence az matne asli che label bara clustering khorde va che label bara community detection
        #2. EventsTriple_Community, EventsTriple_Cluster: Neshoon mide tamamiye triple haye har cluster ya community chiya hastan
        #3. Events_Community, Events_Cluster:       ye string hast ke neshoon mide har cluster va ya har community daghighan chi hast
        
    def Calc_Score_Freq(self,now):
        #1. Calculate ActiveScore and Frequency For All Nodes till time=now
        self.NodeActiveScore = np.array([])
        self.NodeFrequency = np.array([])
        for node_i,node_activity_pulse in enumerate(self.NodeActivityPulse):
            NodeLife = (now-self.NodeDateTimes[node_i][0]).total_seconds()
            if NodeLife == 0:
                NodeLife = 0.5
            Bordar_YeksanSaz = np.zeros((math.ceil(NodeLife/self.TimeStepInPulseVector)-(len(node_activity_pulse))))
            NAP = np.append(node_activity_pulse,Bordar_YeksanSaz) #Bordare NodeActivityPalse Baraye node_i ke yeksanSazi shode(yani ta zamane hazer(now) andis dare)
            
            NAP_Weight = np.array([self.WeightFungtion(i,len(NAP)) for i in range(len(NAP))])
            
            WeightedNode_Repete         = sum([v*w for v,w in zip(NAP,NAP_Weight)])
            Reverce_WeightedNode_Repete = sum([v*w for v,w in zip(NAP,reversed(NAP_Weight))])
            
            NodeFrequency_in_Second = WeightedNode_Repete/NodeLife
            NodeReverceFrequency_in_Second = Reverce_WeightedNode_Repete/NodeLife
            
            self.NodeFrequency = np.append(self.NodeFrequency,NodeFrequency_in_Second)
            self.NodeActiveScore = np.append(self.NodeActiveScore,NodeFrequency_in_Second/NodeReverceFrequency_in_Second)
            
        # 2. Calculate ActiveScore and Frequency For All Edges till time=now
        self.EdgeActiveScore = np.array([]) #------------------------------------------------------------------------------------------
        self.EdgeFrequency = np.array([]) #------------------------------------------------------------------------------------------
        for edge_i,edge_activity_pulse in enumerate(self.EdgeActivityPulse): #------------------------------------------------------------------------------------------
            EdgeLife = (now-self.EdgeDateTimes[edge_i][0]).total_seconds() #------------------------------------------------------------------------------------------
            if EdgeLife == 0: #-----------------------------------------------------------------------------------------
                EdgeLife = 0.5 #-----------------------------------------------------------------------------------------
            Bordar_YeksanSaz = np.zeros((math.ceil(EdgeLife/self.TimeStepInPulseVector)-(len(edge_activity_pulse)))) #------------------------------------------------------------------------------------------
            EAP = np.append(edge_activity_pulse,Bordar_YeksanSaz) #Bordare EdgeActivityPalse Baraye edge_i ke yeksanSazi shode(yani ta zamane hazer(now) andis dare)  #------------------------------------------------------------------------------------------
            
            EAP_Weight = np.array([self.WeightFungtion(i,len(EAP)) for i in range(len(EAP))])  #------------------------------------------------------------------------------------------
            
            WeightedEdge_Repete         = sum([v*w for v,w in zip(EAP,EAP_Weight)])  #-----------------------------------------------------------------------------------------
            Reverce_WeightedEdge_Repete = sum([v*w for v,w in zip(EAP,reversed(EAP_Weight))])  #------------------------------------------------------------------------------------------
            
            EdgeFrequency_in_Second = WeightedEdge_Repete/EdgeLife  #------------------------------------------------------------------------------------------
            EdgeReverceFrequency_in_Second = Reverce_WeightedEdge_Repete/EdgeLife  #-----------------------------------------------------------------------------------------
            
            self.EdgeFrequency = np.append(self.EdgeFrequency, EdgeFrequency_in_Second)  #------------------------------------------------------------------------------------------
            self.EdgeActiveScore = np.append(self.EdgeActiveScore, EdgeFrequency_in_Second/EdgeReverceFrequency_in_Second)  #------------------------------------------------------------------------------------------

    def WeightFungtion(self,x,MaxX,MaxY=1,b=0.1,func='Linear'):
        if func == 'Linear':
            y=((MaxY-b)/MaxX)*x + b

        return y
    
    def AddLabel2Seq(self,Seq2Title,OldSeq2Label,OldLabelAvgVec,NewLabel):
        NewUniqClass = np.array(list(set(NewLabel)))
        AvgVec = self.CalculateAverageVector(NewLabel,NewUniqClass)
        
            
        
        for l in NewUniqClass:
            DistanceToAverage = np.array([])
            LA = np.array([])
            
            new_class_vec = AvgVec[l]
            Label_title = self.Label2Title(l,NewLabel,new_class_vec)
            for L in OldLabelAvgVec.keys():
                old_class_vec = OldLabelAvgVec[L]
                DistanceToAverage = np.append(DistanceToAverage,spatial.distance.cosine(new_class_vec, old_class_vec))
                LA = self.append(LA,L)
                
            mindistanse = 1000
            if DistanceToAverage.size != 0:
                mindistanse = np.min(DistanceToAverage)
            
            
            if mindistanse < self.DistanseTresholdForMergeLavelsVector: #0.1: # in adad Tune Shavad HATTMAN
                L = LA[np.argmin(DistanceToAverage)]
                #Be in mani le class e jadide l haman class e ghabli L hast ke tashkhis dade shode boode
                #dar inja Bayad:
                #    1. Seq hayi ke classeshoon l hast haman L khorde beshan
                #    2. Vector marboot be label L ham Update shavad
                
                OldSeq2Label_TEMP = copy.deepcopy(OldSeq2Label)
                #1:
                for i,clas in enumerate(NewLabel):
                    if clas == l:
                        for seq in self.NodeSequence[i]:
                            if seq in OldSeq2Label.keys():
                                if L not in OldSeq2Label[seq]:
                                    OldSeq2Label[seq] = np.append(OldSeq2Label[seq],L)
                                    Seq2Title[seq] = np.append(Seq2Title[seq],Label_title)
                            else:
                                OldSeq2Label[seq] = np.array([L])
                                Seq2Title[seq] = np.array([Label_title])
                
                #2:
                currentVEC = new_class_vec
                sumWeight = 0
                for ll in OldSeq2Label_TEMP.values():
                    if L in ll: # ll == L:
                        sumWeight += 1
                OldLabelAvgVec[int(L)] = np.average([OldLabelAvgVec[int(L)], currentVEC], axis=0, weights=[sumWeight,1])#len(self.NodeVectors[i])])
                
                
            else:
                #Be in mani ke class e jadide l yek class e jadid hast ke tashkhis dade shode
                #dar inja Bayad:
                #    1. Seq hayi ke classeshoon l hast class e X bokhoran be soorati ke X ye vahed be akharin shomare classike az ghabl boode ezaf mishe (X = Max(Previous Label)+1)
                #    2. Vector marboot be label X ham Zakhire shavad (in haman new_class_vec ast va bayad dar OldLabelAvgVec zakhire beshe)
                if LA.size == 0:
                    X=0
                else:
                    X = np.max(LA) + 1
                #1:
                for i,clas in enumerate(NewLabel):
                    if clas == l:
                        for seq in self.NodeSequence[i]:
                            if seq in OldSeq2Label.keys():
                                if X not in OldSeq2Label[seq]:
                                    OldSeq2Label[seq] = np.append(OldSeq2Label[seq],X)
                                    Seq2Title[seq] = np.append(Seq2Title[seq],Label_title)
                            else:
                                OldSeq2Label[seq] = np.array([X])
                                Seq2Title[seq] = np.array([Label_title])
                                
                #2:
                OldLabelAvgVec[X] = new_class_vec
                
            
        return Seq2Title,OldSeq2Label,OldLabelAvgVec
    
    def CalculateAverageVector(self,Label,UniqClass):
        #Hesab Kardane Average Vector Bara Har Label
        #UniqClass =  np.array(list(set(Label)))
        AvgVec_Label = {}
        
        for l in UniqClass:
            AvgVEC = np.array([])
            sumWeight = 0
            for i in range(len(Label)):
                if Label[i]==l:
                    currentVEC = self.NodeAerageVector[i]
                    if AvgVEC.size == 0:
                        AvgVEC = currentVEC
                        sumWeight = 1#len(self.NodeVectors[i])
                    else:
                        AvgVEC = np.average([AvgVEC, currentVEC], axis=0, weights=[sumWeight,1])#len(self.NodeVectors[i])])
                        sumWeight += 1#len(self.NodeVectors[i])
            AvgVec_Label[l] = AvgVEC
        return AvgVec_Label
    
    
    def Label2Title(self,L,Label,AvgVec_L):
        #This Fungtion return the Title of label L
        #rahkarha:
        #1: Aan Nodi Ke Bishtarin Emtiyaz ra darad
        #2: Aan Edgi Ke Bishtarin Emtiyaz ra darad -> Bayad az dakhele self.RelationIndex4CreatingGraph ertebatha va edge ha estekhraj beshe
        #3: Aan Nodi ke nazdiktarin vector ta be AverageVector an label dare -> niyazmande Bordare Label ha dare ke be onvane voroodi tabe bayad bashad
        #
        #
        
        #3:
        DistanceToAverage = np.array([])
        Strings = np.array([])
        for i,l in enumerate(Label):
            if l == L:
                for index_str,vec in enumerate(self.NodeVectors[i]):
                    DistanceToAverage = np.append(DistanceToAverage,spatial.distance.cosine(vec, AvgVec_L))
                    Strings = np.append(Strings,self.NodeStrings[i][index_str])
        return Strings[np.argmin(DistanceToAverage)]    
    

    def ForgetNodes(self,now): ################Dar in tabe tori shavad ke now ra zamanhaye ghabl tar az feli ra bedim ham kar kone be in soorat ke dar yeksanSazi betoone ezafi ha ra hazf kone va node hayi ke ta an zaman naboodan ra ham dar nazar nagire
        #This Fungtion remove Nodes And Related Edges That Not Happen For A long and cant be An Event Candidate
        #Calculate NodeActiveScore & Delete Low Score
        
        #Calculate NodeActiveScore and NodeFrequency for All Nodes till now
        self.Calc_Score_Freq(now)
        # Ta inja ba ejraye dastoore fogh, NodeActiveScore va NodeFrequency Bara hameye NodeHaye Mojood dar graph Mohasebe shode va zakhire shode
        
        
        # Hala bayad Eghdam be Hazfe Nod ha konim
        L = len(self.NodeActiveScore)
        deleted = 0
        for node_i,node_score in enumerate(reversed(self.NodeActiveScore)):
            R_node_i = abs(node_i-L+1)
            if(node_score < self.ScoreTereshold4Delete) or ( (now - self.NodeDateTimes[R_node_i][0]).days >= self.ForcetoForgetOldNode):
                self.DeleteNode(R_node_i)
                deleted+=1
        
        #Marhale 5 dar marahele tabe DeleteNode bayad injaejra mishod
        DelSingleNode = self.DeleteSingleNode()
        
        #print('Forgeted: ' + str(deleted) + ' SingleNodeDeleted:' + str(DelSingleNode) + ' L:' + str(L) + ' CurrentNodes:' + str(len(self.NodeActiveScore)))
                 
    def DeleteNode(self,index):
        #1.Delete Node
        self.NodeStrings = np.delete(self.NodeStrings,index)
        self.NodeVectors = np.delete(self.NodeVectors,index)
        self.NodeAerageVector = np.delete(self.NodeAerageVector,index)
        self.NodeDateTimes = np.delete(self.NodeDateTimes,index)
        self.NodeUsers = np.delete(self.NodeUsers,index)
        self.NodeActivityPulse = np.delete(self.NodeActivityPulse,index)
        self.NodeFrequency = np.delete(self.NodeFrequency,index)
        self.NodeActiveScore = np.delete(self.NodeActiveScore,index)
        self.NodeSequence = np.delete(self.NodeSequence,index)
        
        #2. Delete Relations of Node
        LENGTH = len(self.RelationIndex4CreatingGraph)
        for i,triples in enumerate(reversed(self.RelationIndex4CreatingGraph)):
            r_i = abs(i-LENGTH+1)
            if triples[0] == index or triples[2] == index:
                self.RelationIndex4CreatingGraph = np.delete(self.RelationIndex4CreatingGraph,r_i)
        
        #3. Correct Node Index in Relations List
        for i,triples in enumerate(self.RelationIndex4CreatingGraph):
            if triples[0] > index:
                self.RelationIndex4CreatingGraph[i][0] -= 1
            if triples[2] > index:
                self.RelationIndex4CreatingGraph[i][2] -= 1
        
        #4. Delete Extra Edges
        #4.1. Detect Extra Edges Indexes(EEI)
        EEI = np.zeros(len(self.EdgeRelationStrings))
        for triples in self.RelationIndex4CreatingGraph:
            EEI[triples[1]] += 1
            
        #4.2. Delete Detected Extra Edge(Not Use Edge) in EEI from Edges Lists ->> EEI[i] == 0 meanse Edge index i not use and must be delete
        L = len(EEI)
        for i,val in enumerate(reversed(EEI)):
            reversed_i = abs(i-L+1)
            if val != EEI[reversed_i]:
                print('!!!!!!!!!!!!!!!!!! INAJ Eshtebah Kardim !!!!!!!!!!!!!!!!!!!!!!!!!!')
            if val == 0:
                self.EdgeRelationStrings = np.delete(self.EdgeRelationStrings,reversed_i)
                self.EdgeRelationVectors = np.delete(self.EdgeRelationVectors,reversed_i)
                self.EdgeAverageRelationVector = np.delete(self.EdgeAverageRelationVector,reversed_i)
                self.EdgeDateTimes = np.delete(self.EdgeDateTimes,reversed_i) #------------------------------------------------------------------------------------------
                self.EdgeActivityPulse = np.delete(self.EdgeActivityPulse,reversed_i) #------------------------------------------------------------------------------------------
                self.EdgeFrequency = np.delete(self.EdgeFrequency,reversed_i) #------------------------------------------------------------------------------------------
                self.EdgeActiveScore = np.delete(self.EdgeActiveScore,reversed_i) #------------------------------------------------------------------------------------------
                self.EdgeSequence = np.delete(self.EdgeSequence,reversed_i)
                for ii,triples in enumerate(self.RelationIndex4CreatingGraph):
                    if triples[1] == reversed_i:
                        print('!!!!!!!!!!!!!!!!!! INAJ Ham Eshtebah Kardim !!!!!!!!!!!!!!!!!!!!!!!!!!')
                    if triples[1] > reversed_i:
                        self.RelationIndex4CreatingGraph[ii][1] -= 1                
        
        #5. Delete Single Nodes
        ##### in marhale dar yek tabe neveshte shod va dar akhar ejra mishe(zamani ke hameye Node Ha va Edge Ha Delete shodand)
        
                
        #6. Update FirstTimeInGraph and LastTimeInGraph
        if len(self.NodeStrings) != 0:
            self.FirstTimeInGraph = self.NodeDateTimes[0][0]
            self.LastTimeInGraph = self.NodeDateTimes[0][0]
            for NodeTimeS in self.NodeDateTimes:
                if NodeTimeS[-1] < self.FirstTimeInGraph:
                    self.FirstTimeInGraph = NodeTimeS[-1]
                if NodeTimeS[-1] > self.LastTimeInGraph:
                    self.LastTimeInGraph = NodeTimeS[-1]
        else:
            self.FirstTimeInGraph = datetime(2000, 1, 1, 00, 00)
            self.LastTimeInGraph = datetime(2000, 1, 1, 00, 00)


    def DeleteSingleNode(self):
        #5. Delete Single Nodes
        NumberOfDeletedSingleNode = 0
        #5.1. Detect&Delete Single Node:
        L = len(self.NodeStrings)
        for INDEX,tmp in enumerate(reversed(self.NodeStrings)):
            R_INDEX = abs(INDEX-L+1)
            #Detect SingleNode
            IsSingke = True
            for I,T in enumerate(self.RelationIndex4CreatingGraph):
                if T[0] == R_INDEX or T[2] == R_INDEX:
                    IsSingke = False
                    break
            if IsSingke:
                #Delete Node
                self.NodeStrings = np.delete(self.NodeStrings,R_INDEX)
                self.NodeVectors = np.delete(self.NodeVectors,R_INDEX)
                self.NodeAerageVector = np.delete(self.NodeAerageVector,R_INDEX)
                self.NodeDateTimes = np.delete(self.NodeDateTimes,R_INDEX)
                self.NodeUsers = np.delete(self.NodeUsers,R_INDEX)
                self.NodeActivityPulse = np.delete(self.NodeActivityPulse,R_INDEX)
                self.NodeFrequency = np.delete(self.NodeFrequency,R_INDEX)
                self.NodeActiveScore = np.delete(self.NodeActiveScore,R_INDEX)
                self.NodeSequence = np.delete(self.NodeSequence,R_INDEX)
                
                NumberOfDeletedSingleNode += 1
                #5.2 Correct Indexes in Relations List
                for i,triples in enumerate(self.RelationIndex4CreatingGraph):
                    if triples[0] == R_INDEX or triples[2] == R_INDEX:
                        print('Chon SingleNod Bode Pas inja nabayad biyayad va age biad inja ye jaye kar eshtebah kardim ..........')
                    if triples[0] > R_INDEX:
                        self.RelationIndex4CreatingGraph[i][0] -= 1
                    if triples[2] > R_INDEX:
                        self.RelationIndex4CreatingGraph[i][2] -= 1
        return NumberOfDeletedSingleNode
