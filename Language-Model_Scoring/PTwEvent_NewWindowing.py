# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import networkx as nx
import math
import re
#import pyodbc
import numpy as np
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datetime import datetime
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer  # if not work try: ""pip uninstall scipy"" and then ""pip install scipy""
import jarvispatrick
from sklearn import metrics  # if not work try: ""pip uninstall scipy"" and then ""pip install scipy""
import copy
import xlwt
import CEval

import random
import pprint
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

import clr    # install this library by this command : pip install pythonnet   (if yoy have install clr first remove that by: pip uninstall clr)
#clr.AddReference('{}\\FastTokenizer.dll'.format(os.getcwd()))
clr.AddReference('{}\\FastTokenizer.dll'.format(os.getcwd()))
from FastTokenizer import Api 

# Choose model: 'HooshvareLab/bert-fa-base-uncased' or 'bert-base-multilingual-cased'
LM_MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'
lm_scorer = None

class LanguageModelScorer:
    def __init__(self, model_name='HooshvareLab/bert-fa-base-uncased'):
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model {model_name}...")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded.")

    def get_score(self, text):
        if not text:
            return 0.0

        # Tokenize the input text
        tokenized_text = self.tokenizer.tokenize(text)
        if not tokenized_text:
            return 0.0

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        total_log_prob = 0

        # Iterate through each token to calculate pseudo-log-likelihood
        for i in range(len(tokenized_text)):
            # Create a copy of the tokenized text and mask the current token
            temp_tokenized_text = tokenized_text.copy()
            temp_tokenized_text[i] = self.tokenizer.mask_token

            # Convert tokens to IDs
            masked_tokens_ids = self.tokenizer.convert_tokens_to_ids(temp_tokenized_text)
            tokens_tensor_masked = torch.tensor([masked_tokens_ids])

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(tokens_tensor_masked)
                predictions = outputs[0]

            # Get the log probability of the original token at the masked position
            log_probs = torch.nn.functional.log_softmax(predictions[0, i], dim=0)
            token_log_prob = log_probs[indexed_tokens[i]].item()

            total_log_prob += token_log_prob

        # Normalize by length and convert back from log space
        avg_log_prob = total_log_prob / len(tokenized_text)
        score = math.exp(avg_log_prob)

        return score

def initialize_lm_scorer():
    global lm_scorer
    if lm_scorer is None:
        print(f"Initializing language model: {LM_MODEL_NAME}...")
        lm_scorer = LanguageModelScorer(LM_MODEL_NAME)
        print("Language model initialized.")

global nof
nof = 1
#============================================================================
#============================================================================
class Buffering:
    def __init__(self):
        self.BufferSize = 50000000
        self.Buffer = {}
    
    def ValueInBuffer(self,Segment):
        #Return -1 if not exist and the value if exist
        Segment = " ".join(Segment)
        Temp = self.Buffer.get(Segment)
        if Temp == None:
            return -1
        else:
            Temp[-1] += 1
            self.Buffer[Segment] = Temp
            return Temp[0]
    
    def AddToBuffer(self,Segment,Values):
        #
        Segment = " ".join(Segment)
        Values = [Values]
        Values.append(1)
        self.Buffer[Segment] = Values
        self.DellIfTooBig()
        
    def DellIfTooBig(self):
        #Delete one from Buffer (if exceed BufferSize) depend on Buffer strategis
        # heare Delete the ithem that less appear untill now
        if self.BufferSize < len(self.Buffer):
            ValuesArray =  np.array(list (self.Buffer.values()))
            IndexToRemove = np.argmin(ValuesArray[:,-1]) # Find minimum of last column
            
            self.Buffer = {key:val for key, val in self.Buffer.items() if val != list(ValuesArray[IndexToRemove])}
#============================================================================
NgramBuffer = Buffering()
WikiPediaBuffer = Buffering()
#============================================================================


def Stickiness(segment, flag):
    if not segment:
        print('ERROR in Stickiness Calculation: Segment cannot be empty')
        return 0.0
    
    segment_text = " ".join(segment)
    
    # Get score from language model
    # This score replaces both n-gram (c) and Wikipedia (Q) scores.
    C = lm_scorer.get_score(segment_text)
    
    # The original paper's length normalization
    if len(segment) == 1:
        l = 1.0
    else:
        l = (len(segment) - 1) / len(segment)
        
    L = l * C
    
    return L



def PrepareEdge(Segments,Tweets):
    edges = {}
    for node1 in Segments:
        for node2 in Segments:
            if node1==node2:
                edges[(node2,node1)] =0  ##### inja az maghale chek shavad
                continue
            Eshterak = 0 #Eshterak = Number_Of_Tweets_Include_Node1_And_Node2
            Ejtema = 0   #Ejtema = Number_Of_Tweets_Include_Node1_or_Node2_or_Both
            
            TweetContainNode1 = set() 
            #TweetContainNode1.remove("")
            TweetContainNode2 = set()
            #TweetContainNode2.remove("")
            
            for i,tweet in enumerate(Tweets):
                if True:    #tweet != CurrentTweet: ##### in behboode khodam hast va felan shart Ra TRUE kardam ke raveshe maghale ejra shavad
                    tweetSTR = " ".join(tweet)
                    Node1Nist = [re.search(node1,tweetSTR)] == [None]
                    Node2Nist = [re.search(node2,tweetSTR)] == [None]
                
                
                    if Node1Nist==False:
                        #TweetContainNode1.add("{}".format(i))
                        TweetContainNode1.add(i)
                    
                    if Node2Nist==False:
                        #TweetContainNode2.add("{}".format(i))
                        TweetContainNode2.add(i)

            Ejtema = len(TweetContainNode1.union(TweetContainNode2))
            Eshterak = len(TweetContainNode1.intersection(TweetContainNode2))  
            if False: #Ejtema==0:   ## agar behboode khodam emal nashavad hich moghe in halat pish nemiad va felan shart false shode
                edges[(node2,node1)] =0
            #elif Eshterak == 0:    #######in ham taghire khodam hast
                #edges[(node2,node1)] = Ejtema * 0.1
            else:
                edges[(node2,node1)] = Eshterak/Ejtema
            
    return edges

def RandomWalk_point_P(G,a,b):
    Wab = G.get_edge_data(a,b)[0]['weight']
    Wac = 0
    for c in G.nodes:
        Wac += G.get_edge_data(a,c)[0]['weight']
    if Wac == 0:
        return 0
    else:
        return Wab/Wac

def RandomWalk_point_e(G,s):
    # s : segment (haman node grapg)
    QpS = math.pow(math.e,Q(s))
    QpSj = 0
    for j in G.nodes:
        QpSj += math.pow(math.e,Q(j))
    return QpS/QpSj



def RandomWalk_P(G):
    n = len(G.nodes)
    P = np.zeros((n,n))
    i=0
        
    for a in G.nodes:
        j=0
        for b in G.nodes:
            P[i][j] = RandomWalk_point_P(G,a,b)
            j += 1
        i += 1
    return P


def RandomWalk_e(G):
    n = len(G.nodes)
    e = (np.zeros((n,1)))
    i=0
    for a in G.nodes:
        e[i][0] = RandomWalk_point_e(G,a)
        i += 1
    return e

def ApplyEffectOf_e_and_P(P,e,gama):
    Ones = np.ones( (1,len(e)) )
    tempP = gama*(P.transpose())
    tempE = (1-gama)*(np.dot(e,Ones))
    PP = tempP + tempE
    return PP

def CalculatePi(PP):
#    w, v = LA.eig(PP)
#    eigenvector1 = v[:,0] # this means first eigenvector and also v[:,1] is 2nd eigenvector
    FirstPi = np.zeros(len(PP),)+1/len(PP)

    AD = [] #AbsoluteDeviation
    
    ## in raveshi bood ke bahs shod
    
    pii = []
    iteration = 500
    pii.append(np.transpose(FirstPi))
    for i in range(iteration):
        NewPi = np.dot(PP,pii[-1])
        pii.append(NewPi)
        
        ad = AbsoluteDeviation(pii[-1]-pii[-2])
        AD.append(ad)
#        if i%50 == 0 and i != StartPlot:
#            plt.plot(AD[StartPlot:i])
#            plt.ylabel('Absolute Deviation Value')
#            plt.xlabel('Iteration: {} to {}'.format(StartPlot,i))
#            plt.show()
#            StartPlot = i
    #return np.sum(pii[-1])
    return pii[-1]
 
def AbsoluteDeviation(Vec):
    AbsVec = list(map(abs, Vec))
    return sum(AbsVec)/len(AbsVec)
    
def AssignNodeNameTo_y(G,y):
    Y = []
    i=0
    for node in G.nodes:
        Y.append([y[i][0], node])
        i += 1
    return Y




def RemoveSTOPword(text):
    PatternRemoveStopWord = codecs.open( "C:\\Users\\Pejman\\Desktop\\PhD\\PatternRemoveStopWord.txt", "r", "utf-8" )
    DeletedPatternWord = PatternRemoveStopWord.read()
    DeletedPatternWord = Api.Normalize(DeletedPatternWord)
    
    PatternRemoveStopMark = codecs.open( "C:\\Users\\Pejman\\Desktop\\PhD\\PatternRemoveStopMark.txt", "r", "utf-8" )
    DeletedPatternMark = PatternRemoveStopMark.read()   
    DeletedPatternMark = Api.Normalize(DeletedPatternMark)
    
    text = re.sub(DeletedPatternWord,"",text)
    text = re.sub(DeletedPatternMark," ",text)
    return text

def DynamicAlg(t,u,e,flag):
    # Flag Can Be pmi or scp
    l = len(t)
    S = [] # ham segment va ham stikiness zakhire mishavad  [ [S1] [S2] [S3] [S4] ...  ] (S1 = ([['segment1','C1'],['segment2','C2']...])
    for i in range(l): # inja tamame kalamat ra peymatesh mikonad
        S.append([])
        si = t[0:i+1]
        
        if i < u:
            #do not Spilit
            c=Stickiness(si,flag)
            S[i].append([si] + [c])   #s[i] be onvane yek segmente motabar entekhab mishavad
        # baghiyeye halate momken barate segmente motabar
        start = i-u
        if start <0:
            start = 0
        for j in range(start,i):
            #if (i-j) <= u: # j>=i-u
            #s1 = t[0:j+1]
            s2 = t[j+1:i+1]
            c=Stickiness(s2,flag)
            for jj in range(len(S[j])):
                sj = S[j][jj]
                if sj != []:
                    cc = c+sj[-1]
                    S[i].append(sj[0:-1] + [s2] + [cc])
        #sort S and store top e to S
        S[i].sort(key=lambda index: index[-1],reverse=True)
        #S[i] = DeleteDuplicate(S[i])   ########### Chek shavad ke in laze ast ya na , nemidoonam chera ino avordam
        S[i] = S[i][0:e]
            
    return S[-1][0]  
# Dar inja bayad faghat 1 reshte az segment ha ba bishtarin emtiaz(bishtarin C) ra return konad? fek nakonam

def DeleteDuplicate(S):
    Result = []
    Result.append(S[0])
    for i in range(len(S)):
        if S[i][0:-1] != Result[-1][0:-1]:
            Result.append(S[i])
    if Result != S:
        print("Inja Nabayad Biyayad. va agar inja nayayad in tabe aslan lazem nist")
    return Result
       
    
def NoiseFilter(SegmentedSentence):
    # Input is a list like this: SS=[ ['W1'],['W2'],['W3' , 'W4'],['W5', 'W6', 'W7'],['W8', 'W9'], 5.1 ]
    SS_NoiseFiltered = []
    Pattern = ('^[۰-۹0-9 -_)(*&^%$#@!~]+$|^.*خخ+.*$|^.*(ههه )+.*$|^.*\#\#+.*$|^WORDorEXP$|^WORDorEXP$|^WORDorEXP$')
    for segment in SegmentedSentence[0:-1]:
        SegmentString = " ".join(segment)
        
        SegmentString = re.sub(Pattern,"",SegmentString)
        if(SegmentString != ""):
            SS_NoiseFiltered.append(SegmentString)

    SS_NoiseFiltered = list(filter(None,SS_NoiseFiltered))    
    return SS_NoiseFiltered

def CreateGraph(Segments,Tweets):
    
    G = nx.MultiDiGraph()
    print('Adding Nodes To Graph ...')
    G.add_nodes_from(Segments)
    
    print('Preparing the Edges ...')
    edges_wts = PrepareEdge(Segments,Tweets)
    
    print('Adding Edges To Grapg ...')
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
  
    '''
    print(f'Edges:')
    pprint(list(G.edges(data=True)))

    pos= nx.spring_layout(G)
    nx.draw(G, pos, font_size=16, with_labels=False)
    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G, pos)
    plt.show()
    '''   
    
    return G


def takeFirst(elem):
    return elem[0]
def SelectBest(Y,k):
    Y.sort(key=takeFirst,reverse=True)
    Resault = []
    for i in range(min(k,len(Y))):
        Resault.append(Y[i][1])
    return Resault

def FastTokenize(text):
    
    TokenizeTXT = Api.Tokenize(text)
    
    S = []
    for sentence in TokenizeTXT.Sentences:
        for i in range(len(sentence.Tokens)):
            if(sentence.Tokens[i].Type == "PersianWord" or sentence.Tokens[i].Type == "CompoundWord" or sentence.Tokens[i].Type == "Integer"):
                if(sentence.Tokens[i].Token != ''):
                    S.append(sentence.Tokens[i].Token)
    return S


def TaggedBIOfromSegments(NER,AllSegnment):
    Tags = []
    TagScore = []
    for i in range(len(AllSegnment)):
        DetectNER = False
        for ts,ner in enumerate(NER):
            if " ".join(AllSegnment[i]) == ner:
                DetectNER = True
                Tags.append('B')
                TagScore.append(ts+1)
                
                for j in range(1,len(AllSegnment[i])):
                    Tags.append('I')
                    TagScore.append(ts+1)
                break
        if DetectNER == False:
            for j in range(len(AllSegnment[i])):
                Tags.append('O')
                TagScore.append(0)
    return Tags,TagScore


def ReadBatch():
    path = 'C:\\Users\\Pejman\\Desktop\\PhD\\Data4PersianNLP\\Persian-NER-master\\Batch\\Big_Complete_TelegramPost\\Entekhabat(Top100)\\Entekhabat(Top100).txt'
    #files = glob.glob(path)
    #Tweets = []
    
    InputStream = codecs.open( path, "r", "utf-8" )
    Posts = InputStream.readlines() # Returns a Unicode string from the UTF-8 bytes in the file  
    
    for i in range(len(Posts)):
        
        Posts[i] = Api.Normalize(Posts[i])
        Posts[i] = RemoveSTOPword(Posts[i])
        Posts[i] = FastTokenize(Posts[i])
        
    
    
    InputStream.close()

    
    return Posts


#====================================================================================
    


    




def ReadData(Start,End):
    
    
    # SqlConnString = ('Driver={SQL Server};'
    #               'Server=(local);'
    #               'Database=Posts4Pejman;'
    #               'Trusted_Connection=yes;')

    # Query = ("SELECT [Seq],[SourceText],[Deleted],[DateSent],[FromId],[WindowNum] FROM [Posts4Pejman].[dbo].[Posts] Where SourceText IS NOT NULL and [DateSent] BETWEEN '{}' and '{}' order by DateSent ASC, Seq DESC".format(Start,End))

    
    
    SqlConnString = ('Driver={SQL Server};'
                  'Server=(local);'
                  'Database=Posts;'
                  'Trusted_Connection=yes;')

    Query = ("SELECT [Seq],[SourceText],[Deleted],[DateSent],[FromId],[WindowNum] FROM [Posts].[dbo].[Posts] Where SourceText IS NOT NULL and [DateSent] BETWEEN '{}' and '{}' order by DateSent ASC, Seq DESC".format(Start,End))

    Sql_Conn = pyodbc.connect(SqlConnString)
    cursor = Sql_Conn.cursor()
    cursor.execute(Query)
    #QueryRes = cursor.fetchall()
    
    ii=0
    Posts = []
    Types = []
    Seq = []
    Deleted = []
    DateSend = []
    User = []
    
    Posts_Windowing = []
    Types_Windowing = []
    Seq_Windowing = []
    Deleted_Windowing = []
    DateSend_Windowing = []
    User_Windowing = []
    WindowNum = []
    CurrentWindowNum = -1
    
    for Row in cursor:
        #Row[0] Seq
        #Row[1] Jsone String
        #Row[2] Deleted Tag
        #Row[3] Date Send
        #Row[4] User
        #Row[5] WindowNum
        
        Post = json.loads(Row[1]) #processed Text (Json)
        if Row[0] == Post['DocumentId']:
            if CurrentWindowNum == -1:
                CurrentWindowNum = Row[5]
            if CurrentWindowNum != Row[5]:
                Posts_Windowing.append(Posts)
                Types_Windowing.append(Types)
                Seq_Windowing.append(Seq)
                Deleted_Windowing.append(Deleted)
                DateSend_Windowing.append(DateSend)
                User_Windowing.append(User)
                WindowNum.append(CurrentWindowNum)
                
                
                Posts = []
                Types = []
                Seq = []
                Deleted = []
                DateSend = []
                User = []
                
                CurrentWindowNum = Row[5]
                
            Posts.append([])
            Types.append([])
            Seq.append(Row[0])
            if(Row[2]==None):
                Deleted.append("")
            else:
                Deleted.append(Row[2])
            DateSend.append(Row[3])
            User.append(Row[4])
            for Sentence in Post['Sentences']:
                for Token in Sentence['Tokens']:
                    if (Token['Type'] == "PersianWord" or Token['Type'] == "Integer" or Token['Type'] == "EnglishWord" or Token['Type'] == "CompoundWord") and (Token['IsStopWord'] == False) :
                        Posts[-1].append(Token['Token'])
                        Types[-1].append(Token['Type'])
    
            if ii%100==0:
                print("{} Readed".format(ii))
            ii += 1
    cursor.close()
    Sql_Conn.commit()
    Sql_Conn.close()         
    
    if(len(Posts) != 0):
        Posts_Windowing.append(Posts)
        Types_Windowing.append(Types)
        Seq_Windowing.append(Seq)
        Deleted_Windowing.append(Deleted)
        DateSend_Windowing.append(DateSend)
        User_Windowing.append(User)
        WindowNum.append(CurrentWindowNum)

    AllData = Seq_Windowing,Posts_Windowing,Types_Windowing,Deleted_Windowing,DateSend_Windowing,User_Windowing,WindowNum
    return AllData
    


def Segmentation(Tweets):
    
    SegmentationResults_Windowing = []    

    for WinNum in range(len(Tweets)): 
        SegmentationResults_Windowing.append([])

        for i in range(len(Tweets[WinNum])): 
    
        
            CurrentTweet = Tweets[WinNum][i]
            
            print('Target Tweet[{}/{}]'.format(i,len(Tweets[WinNum])))
            if CurrentTweet !=[]:
                print('DynamicAlg Running.')
                #Marhale Avval Dynamic Alg...
                PosibbleSegments = DynamicAlg(CurrentTweet,5,20,'scp') # Flag Can Be pmi or scp
        
            print('ValidSegment Colculated.')
            
            TempSegment = PosibbleSegments[0:-1] ###############################################################################################
            SegmentationResults_Windowing[-1].append(TempSegment)
            
    
    return SegmentationResults_Windowing

def P(PostSegments,CurrentSegment,CurrentWinNum):
    #Article :->  where ps is the expected probability of tweets that contain segment s in a random time window
    
    MovingWindowStart = 0  #Should Be CurrentWinNum-WindowSize
    MovingWindowEnd = len(PostSegments) #Should Be CurrentWinNum
    
    NumberOfTweetContainSegment = 0
    TotalNumberOfTweet = 0
    
    for WinNum in range(MovingWindowStart,MovingWindowEnd):
        for Post in PostSegments[WinNum]:
            if CurrentSegment in Post:
                NumberOfTweetContainSegment +=1
        TotalNumberOfTweet += len(PostSegments[WinNum])
    
    return NumberOfTweetContainSegment/TotalNumberOfTweet

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def U(AllData,PostSegments,WinNum,CurrentSegment):
    Users = set()
    for PostNum in range(len(PostSegments[WinNum])):
        if CurrentSegment in PostSegments[WinNum][PostNum]:
            Users.add(AllData[5][WinNum][PostNum])   
    return len(Users) 

def FST(segment,Window):
    res = 0
    for tweet in Window:
        if segment in tweet:
            res+=1
    return res

def FST1(node,Window):
    res = 0
    for tweet in Window:
        TweetSTR = []
        for Segment in tweet:
            TweetSTR.append(" ".join(Segment))
        if node in TweetSTR:
            res+=1
    return res






def DetectBurstyNLastWindow(AllData,PostSegments,n):
#    AllData Contain :
#        Seq_Windowing,
#        Posts_Windowing,
#        Types_Windowing,
#        Deleted_Windowing,
#        DateSend_Windowing,
#        User_Windowing,
#        WindowNum
    
    EventSegment_Windowing = []
    EventSegmentWeight_Windowing = []
    CurentPostBurstySegment = []
    CurentPostBurstyWeight = []
    BurstySegment = []
    BurstyWeight = []

    for WinNum in range(len(PostSegments)-n,len(PostSegments)):
        BurstySegment.append([])
        BurstyWeight.append([])
        for PostNum in range(len(PostSegments[WinNum])):
            for SegmentNum in range(len(PostSegments[WinNum][PostNum])):
                CurrentSegment = PostSegments[WinNum][PostNum][SegmentNum]
                Ps = P(PostSegments,CurrentSegment,WinNum)
                Nt = len(PostSegments[WinNum])
                Est = Ps*Nt
                Fst = FST(CurrentSegment,PostSegments[WinNum])
                
                if Fst > Est :
                    #Segment Is Bursty
                    CurentPostBurstySegment.append(CurrentSegment)
                    Dst = math.sqrt(Nt*Ps*(1-Ps)) #Enheraf meyar
                    if Fst >= (Est+2*Dst):
                        PBst = 1
                    else:
                        PBst = sigmoid(10*(Fst-(Est+Dst))/Dst)
                    
                    Ust = U(AllData,PostSegments,WinNum,CurrentSegment)
                    
                    if Ust == 0:
                        WBst = 0
                    else:
                        WBst = PBst*math.log(Ust)
                    
                    CurentPostBurstyWeight.append(WBst)
            BurstySegment[-1].append(CurentPostBurstySegment)   
            BurstyWeight[-1].append(CurentPostBurstyWeight)
            CurentPostBurstySegment = []
            CurentPostBurstyWeight = []
             
    for WinNum in range(len(BurstySegment)):
        EventSegment_Windowing.append([])
        EventSegmentWeight_Windowing.append([])
        for PostNum in range(len(BurstySegment[WinNum])):
            if len(BurstySegment[WinNum][PostNum]) != 0:
                Nt = len(BurstySegment[WinNum])
                K = math.ceil( math.sqrt(Nt) )
                
                #Soarting Process
                zipped_lists = zip(BurstyWeight[WinNum][PostNum], BurstySegment[WinNum][PostNum])
                sorted_pairs = sorted(zipped_lists,reverse=True)
                tuples = zip(*sorted_pairs)
                Wei, Seg = [ list(tuple) for tuple in tuples]
                
                EventSegment_Windowing[-1].append(Seg[0:K])
                EventSegmentWeight_Windowing[-1].append(Wei[0:K])
        if len(EventSegment_Windowing[-1]) == 0:
            del EventSegment_Windowing[-1]
            del EventSegmentWeight_Windowing[-1]
            
    return EventSegment_Windowing,EventSegmentWeight_Windowing






def DetectBursty(AllData,PostSegments):
#    AllData Contain :
#        Seq_Windowing,
#        Posts_Windowing,
#        Types_Windowing,
#        Deleted_Windowing,
#        DateSend_Windowing,
#        User_Windowing,
#        WindowNum
    
    EventSegment_Windowing = []
    EventSegmentWeight_Windowing = []
    CurentPostBurstySegment = []
    CurentPostBurstyWeight = []
    BurstySegment = []
    BurstyWeight = []

    for WinNum in range(len(PostSegments)):
        print('DetectBursty For Win:{}/{}'.format(WinNum,len(PostSegments)))
        BurstySegment.append([])
        BurstyWeight.append([])
        for PostNum in range(len(PostSegments[WinNum])):
            for SegmentNum in range(len(PostSegments[WinNum][PostNum])):
                CurrentSegment = PostSegments[WinNum][PostNum][SegmentNum]
                Ps = P(PostSegments,CurrentSegment,WinNum)
                Nt = len(PostSegments[WinNum])
                Est = Ps*Nt
                Fst = FST(CurrentSegment,PostSegments[WinNum])
                
                if Fst > Est :
                    #Segment Is Bursty
                    CurentPostBurstySegment.append(CurrentSegment)
                    Dst = math.sqrt(Nt*Ps*(1-Ps)) #Enheraf meyar
                    if Fst >= (Est+2*Dst):
                        PBst = 1
                    else:
                        PBst = sigmoid(10*(Fst-(Est+Dst))/Dst)
                    
                    Ust = U(AllData,PostSegments,WinNum,CurrentSegment)
                    
                    if Ust == 0:
                        WBst = 0
                    else:
                        WBst = PBst*math.log(Ust)
                    
                    CurentPostBurstyWeight.append(WBst)
            BurstySegment[-1].append(CurentPostBurstySegment)   
            BurstyWeight[-1].append(CurentPostBurstyWeight)
            CurentPostBurstySegment = []
            CurentPostBurstyWeight = []
             
    for WinNum in range(len(BurstySegment)):
        EventSegment_Windowing.append([])
        EventSegmentWeight_Windowing.append([])
        for PostNum in range(len(BurstySegment[WinNum])):
            if len(BurstySegment[WinNum][PostNum]) != 0:
                Nt = len(BurstySegment[WinNum])
                K = math.ceil( math.sqrt(Nt) )
                
                #Soarting Process
                zipped_lists = zip(BurstyWeight[WinNum][PostNum], BurstySegment[WinNum][PostNum])
                sorted_pairs = sorted(zipped_lists,reverse=True)
                tuples = zip(*sorted_pairs)
                Wei, Seg = [ list(tuple) for tuple in tuples]
                
                EventSegment_Windowing[-1].append(Seg[0:K])
                EventSegmentWeight_Windowing[-1].append(Wei[0:K])
        if len(EventSegment_Windowing[-1]) == 0:
            del EventSegment_Windowing[-1]
            del EventSegmentWeight_Windowing[-1]
            
    return EventSegment_Windowing,EventSegmentWeight_Windowing

def TweetContain(node,CurrentSubWindow):
    res = []
    for tweet in CurrentSubWindow:
        TweetSTR = []
        for Segment in tweet:
            TweetSTR.append(" ".join(Segment))
        if node in TweetSTR:
            res.append(TweetSTR)
    return res


def WT(node,m,CurrentSubWindowing):
    Soorat = FST1(node,CurrentSubWindowing[m])
    Makhraj = 0
    for CurrentSubWindow in CurrentSubWindowing:
        Makhraj += FST1(node,CurrentSubWindow)
    
    if Makhraj == 0:
        return 0
    else:
        return Soorat/Makhraj

def SIM(T1,T2):
    if(T1=='' or T2==''):
        return 0
    else:
#        vect = TfidfVectorizer(min_df=1)                                                                                                                                                                                                   
#        tfidf = vect.fit_transform([T1,T2])                                                                                                                                                                                                                       
#        pairwise_similarity = tfidf * tfidf.T 
#        return pairwise_similarity[0,1]
        
        AllWords = set()
        for w in T1.split(" "):
            AllWords.add(w)
        for w in T2.split(" "):
            AllWords.add(w)
        
        AllWords = list(AllWords)
        
        T1_tfidf = []
        T2_tfidf = []
        for w in AllWords:
            T1_w_tf = T1.split(" ").count(w)
            T2_w_tf = T2.split(" ").count(w)
            if T1_w_tf != 0 and T2_w_tf != 0:
                T1_tfidf.append(T1_w_tf/2)
                T2_tfidf.append(T2_w_tf/2)
            else:
                T1_tfidf.append(T1_w_tf)
                T2_tfidf.append(T2_w_tf)
        
        cos_sim = dot(T1_tfidf, T2_tfidf)/(norm(T1_tfidf)*norm(T2_tfidf))
        return cos_sim

def CreatePseudoDoc(T):
    TempList = []
    for post in T:
        TempList.append(" ".join(post))
    return " ".join(TempList)


def CalculateSim(node2,node1,CurrentSubWindowing):
    #formoole 9 maghale
    Sim = 0 
    for m in range(len(CurrentSubWindowing)):
        CurrentSubWindow = CurrentSubWindowing[m]
        T1 = TweetContain(node2,CurrentSubWindow)
        T2 = TweetContain(node1,CurrentSubWindow)
        Temp1 = WT(node2,m,CurrentSubWindowing)*WT(node1,m,CurrentSubWindowing)
        
        TT1 = CreatePseudoDoc(T1)
        TT2 = CreatePseudoDoc(T2)

        Temp2 = SIM(TT1,TT2)
        Sim += Temp1 * Temp2

    return Sim

def PrepareSimilarityEdge(NodeList,CurrentSubWindowing):
    edges = {}
    for node1 in NodeList:
        for node2 in NodeList:
            if node1==node2:
                edges[(node2,node1)] = 0  ##### inja az maghale chek shavad
                break
            else:
                edges[(node2,node1)] = CalculateSim(node2,node1,CurrentSubWindowing)
                edges[(node1,node2)] = edges[(node2,node1)]
    return edges


def CreateSimilarityGraph(CurrentWindow,CurrentSubWindowing):
    
    StartTime = datetime.now()
    print(StartTime)
    print('current window size: {}'.format(len(CurrentWindow)))
    G = nx.MultiDiGraph()
    print('Adding Nodes To Graph ...')
    NodeList = []
    for Tweet in CurrentWindow:
        for Segment in Tweet:
            NodeList.append(" ".join(Segment))
    G.add_nodes_from(NodeList)
    
    print('Node Number: {}'.format(len(NodeList)))
    
    print('Preparing the Edges ...')
    edges_wts = PrepareSimilarityEdge(NodeList,CurrentSubWindowing)
    
    print('Adding Edges To Grapg ...')
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, key='w', weight=v, label=v)
   
    print('Time Taken: {}'.format(datetime.now()-StartTime))
#    nx.draw(G,with_labels=True)
#    plt.savefig('labels.png')
    
#    nx.draw(G,with_labels=True)
#    plt.draw()
#    plt.show()

#    options = {
#        'node_color': 'blue',
#        'node_size': 100,
#        'width': 3,
#        'arrowstyle': '-|>',
#        'arrowsize': 12,
#    }
#    nx.draw_networkx(G, arrows=True, **options)

#    val_map = {'A': 1.0,
#               'D': 0.5714285714285714,
#               'H': 0.0}
#    values = [val_map.get(node, 0.25) for node in G.nodes()]     
#    nx.draw(G, cmap = plt.get_cmap('jet'), node_color = values)
#    plt.show()
    
  
    return G

def ClusteringAlgByParametrTunning(Graph):
    clusters = []
    n=len(Graph.nodes)
    DistanceMatrix = np.zeros([n,n])
    WeightMatrix = np.zeros([n,n])
    def weight_func(node1, node2):
        return Graph[node1][node2]['w']['weight']
        #or return G.edge[node1][node2]['weight']
        #or return G.get_edge_data(a,c)[0]['weight']
    
    def distance_func(node1, node2):
        return DistanceMatrix[int(node1[0])][int(node2[0])]
        #return WeightMatrix[int(node1[0])][int(node2[0])]
    
        
    cluster_gen = jarvispatrick.JarvisPatrick(Graph.nodes, weight_func)
    
#    K=round(len(Graph.nodes)/5)#15 # “K”, the number of nearest neighbors to consider OR  'number_of_neighbors'
#    Step1 = max(1,round(K/50))
#    K_MIN=round(len(Graph.nodes)/10)#6 #  K_min, the number of minimum shared neighbors OR 'threshold_number_of_common_neighbors'
#    Step2 = max(1,round(K_MIN/20))
#    print('Total Node is:{} , K and K_MIN is: {}&{}'.format(len(Graph.nodes),K,K_MIN))
#    CLUSTERS = cluster_gen(K, K_MIN) # initialize clusters
    
    CLUSTERS = cluster_gen(10, 5) # initialize clusters
    K = 5
    Max_K = 50

    Silhouette_Coefficient = -1
    KList=[]
    KMinList=[]
    S_Score=[]
    for k in range(K,Max_K):
        for k_min in range(1,k):
            
            clusters = cluster_gen(k, k_min)
            
            NoiseClusterKey = []
            for key,cluster in clusters.items():
                if len(cluster) <= 1: # inja bayad <=1 bashad vali choon hame dar yek cluster miravand felan injoorish kardam ############################################3
                    NoiseClusterKey.append(key)
            for i in NoiseClusterKey:
                clusters[-1].append(*clusters[i])
                del clusters[i]
            
            
            if len(clusters) != 1 and len(clusters) != len(Graph.nodes):
                X = []
                X_Num = []
                xnum=0
                
                labels = []
                for key,val in clusters.items():
                    for node in val:
                        labels.append(key)
                        X.append(node)
                        X_Num.append(xnum)
                        xnum+=1
                
                #WeightMatrix = np.zeros([n,n])
                MaxWeight = 0
                for i,n1 in enumerate(X):
                    for j,n2 in enumerate(X):
                        WeightMatrix[i][j] = weight_func(n1, n2)
                        if WeightMatrix[i][j] > MaxWeight:
                            MaxWeight = WeightMatrix[i][j]
                
                for i in range(len(X)):
                    for j in range(len(X)):
                        DistanceMatrix[i][j] = abs(WeightMatrix[i][j]-MaxWeight)
                
                X2 = np.array(X_Num)
                X2=np.reshape(X2,(-1,1))
                
                l=np.array(labels)
                l=np.reshape(l,(-1,))
            
            
                Temp = metrics.silhouette_score(X2, l, metric=distance_func)
                KList.append(k)
                KMinList.append(k_min)
                S_Score.append(Temp)
                
                if Temp > Silhouette_Coefficient:
                    Silhouette_Coefficient = Temp
                    CLUSTERS = clusters
                
                print('Clustering with K={} and K-Min={}, Number Of TrurCluster={} and Silhouette={}'.format(k,k_min,len(clusters),Temp))
    
    clusters = CLUSTERS
    
#    MaxCluster = max(cluster_gen.cluster.values())
#    for _ in range(MaxCluster+1):
#        clusters.append([])
#    
#    for Node,ClusterNum in cluster_gen.cluster.items():
#        clusters[ClusterNum].append(Node)
    
    #RemoveNoiseClusters
    if -1 in clusters.keys():
        del clusters[-1]
        
    #Remove Keys of deleted cluster
    clusters = {i:clusters[i] for i in clusters if cluster[i]!=[]}        
    
    #ClusterS = [clusters[k] for k in clusters.keys()]
    return clusters,KList,KMinList,S_Score


def ClusteringAlg(Graph,k,k_min):
    clusters = []
    noises = {}
    def weight_func(node1, node2):
        return Graph[node1][node2]['w']['weight']
        #or return G.edge[node1][node2]['weight']
        #or return G.get_edge_data(a,c)[0]['weight']
        
    cluster_gen = jarvispatrick.JarvisPatrick(Graph.nodes, weight_func)
    
            
    clusters = cluster_gen(k, k_min)
    
    NoiseClusterKey = []
    for key,cluster in clusters.items():
        if len(cluster) <= 1: 
            NoiseClusterKey.append(key)
    for i in NoiseClusterKey:
        clusters[-1].append(*clusters[i])
        del clusters[i]
            
    
    #RemoveNoiseClusters
    if -1 in clusters.keys():
        noises = {-1:clusters[-1]}
        del clusters[-1]
    
    #Remove Keys of deleted cluster
    clusters = {i:clusters[i] for i in clusters if clusters[i]!=[]}
    
    #ClusterS = [clusters[k] for k in clusters.keys()]
    return clusters,noises




def EventSegmentClustering_Similarity_NLastWindow(EventSegment,n):
    
    # Calculate Similarity Between EventSegments for Clustering
    SubWindowPosts = []
    CurrentSubWindowPosts = []
    for WinNum in range(len(EventSegment)):
        # Spliting Each Window to M SubWindow
        SubWindowPosts.append([])
        SizeOfCurrentWindow = len(EventSegment[WinNum])
        M = round(min(10,max(1,SizeOfCurrentWindow/5))) # Felan Har Panjere ro be tedade postHaye dakhelash subWindow Mikonim
        NumberOfPostInEachSubWindow = round(len(EventSegment[WinNum])/M)
        if NumberOfPostInEachSubWindow*M<len(EventSegment[WinNum]):
            NumberOfPostInEachSubWindow+=1
            
        for SubWinNum in range(M):
            #CurrentSubWindowPosts.append([])
            for PostNum in range(SubWinNum*NumberOfPostInEachSubWindow,min(SubWinNum*NumberOfPostInEachSubWindow+NumberOfPostInEachSubWindow,SizeOfCurrentWindow)):
                CurrentSubWindowPosts.append(EventSegment[WinNum][PostNum])
            SubWindowPosts[-1].append(CurrentSubWindowPosts)
            CurrentSubWindowPosts = []
    
    # CAlculating Similariti between Each pair of segment in eact Window and Create A graph
    SimilarityGraph = []
    for WinNum in range(len(EventSegment)-n,len(EventSegment)):
        SimilarityGraph.append([])
        
        CurrentWindow = EventSegment[WinNum]
        CurrentSubWindowing = SubWindowPosts[WinNum]
        
        SimilarityGraph[-1] = CreateSimilarityGraph(CurrentWindow,CurrentSubWindowing)
    
    return SimilarityGraph

def EventSegmentClustering_Similarity(AllData,EventSegment,StartTime,StepTime):
    
    # Calculate Similarity Between EventSegments for Clustering
    SubWindowPosts = []
    CurrentSubWindowPosts = []
    for WinNum in range(len(EventSegment)):
        # Spliting Each Window to M SubWindow
        SubWindowPosts.append([])
        
#        SizeOfCurrentWindow = len(EventSegment[WinNum])
#        M = round(min(10,max(1,SizeOfCurrentWindow/5))) # Felan Har Panjere ro be tedade postHaye dakhelash subWindow Mikonim
#        NumberOfPostInEachSubWindow = round(len(EventSegment[WinNum])/M)
#        if NumberOfPostInEachSubWindow*M<len(EventSegment[WinNum]):
#            NumberOfPostInEachSubWindow+=1
#            
#        for SubWinNum in range(M):
#            #CurrentSubWindowPosts.append([])
#            for PostNum in range(SubWinNum*NumberOfPostInEachSubWindow,min(SubWinNum*NumberOfPostInEachSubWindow+NumberOfPostInEachSubWindow,SizeOfCurrentWindow)):
#                CurrentSubWindowPosts.append(EventSegment[WinNum][PostNum])
#            SubWindowPosts[-1].append(CurrentSubWindowPosts)
#            CurrentSubWindowPosts = []
            
        for PostNum in range(len(EventSegment[WinNum])):    
            if AllData[4][WinNum][PostNum] >= StartTime and AllData[4][WinNum][PostNum] < StartTime+StepTime:
                CurrentSubWindowPosts.append(EventSegment[WinNum][PostNum])
                    
            elif AllData[4][WinNum][PostNum] >= StartTime and AllData[4][WinNum][PostNum] >= StartTime+StepTime:
                SubWindowPosts[-1].append(CurrentSubWindowPosts)
                CurrentSubWindowPosts = []
                
                StartTime = StartTime+StepTime
                
                CurrentSubWindowPosts.append(EventSegment[WinNum][PostNum])
                
            else:
                print('NABAYAD INJA BIAD')
                
        SubWindowPosts[-1].append(CurrentSubWindowPosts)
    
    # CAlculating Similariti between Each pair of segment in eact Window and Create A graph
    SimilarityGraph = []
    for WinNum in range(len(EventSegment)):
        SimilarityGraph.append([])
        
        CurrentWindow = EventSegment[WinNum]
        CurrentSubWindowing = SubWindowPosts[WinNum]
        
        SimilarityGraph[-1] = CreateSimilarityGraph(CurrentWindow,CurrentSubWindowing)
    
    return SimilarityGraph

def EventSegmentClusteringByParametrTunning(SimilarityGraph,EventSegment):
    
    EventSegment_Clusters = []
    KList=[]
    KMinList=[]
    S_Score=[]
    for WinNum in range(len(EventSegment)):
        #EventSegment_Clusters.append([])
        clusters,current_KList,current_KMinList,current_S_Score = ClusteringAlgByParametrTunning(SimilarityGraph[WinNum])
        EventSegment_Clusters.append(clusters)
        KList.append(current_KList)
        KMinList.append(current_KMinList)
        S_Score.append(current_S_Score)    
    return EventSegment_Clusters,KList,KMinList,S_Score

def EventSegmentClustering(SimilarityGraph,EventSegment,k,k_min):    
    EventSegment_Clusters = []
    Noises = []
    
    for WinNum in range(len(EventSegment)):
        #EventSegment_Clusters.append([])
        clusters,noise = ClusteringAlg(SimilarityGraph[WinNum],k,k_min)
        EventSegment_Clusters.append(clusters)
        Noises.append(noise)
    return EventSegment_Clusters,Noises

def MiuS(EventSegment):
    # The language model score is used as a direct replacement for the original MiuS calculation.
    # The original implementation had a complex dependency on the Q function which queried a database.
    return lm_scorer.get_score(EventSegment)

def EventNewsWorthy_NLastWindow(CondidateEvents,SimilarityGraph,n):
    MiuE = []
    for WinNum in range(len(CondidateEvents)-n,len(CondidateEvents)):
        Window = CondidateEvents[WinNum]
        MiuE.append([])
        WINDOW = list(Window.values())
        for ClusterNum,Cluster in enumerate(WINDOW):
            SooratKasrAvval = 0
            
            ClusterEventSegmentString = []
            for i,eventsegment in enumerate(Cluster):
                SooratKasrAvval += MiuS(eventsegment)
                ClusterEventSegmentString.append(eventsegment)
                print('WinNum:{}/{}##ClusterNum:{}/{}##EventSegment:{}/{}'.format(WinNum,len(CondidateEvents),ClusterNum,len(WINDOW),i,len(Cluster)))
                
            SooratKasrDovvom= 0    
            Graph = SimilarityGraph[WinNum]
            for node1 in ClusterEventSegmentString:
                for node2 in ClusterEventSegmentString:
                    SooratKasrDovvom += Graph[node1][node2]['w']['weight']  #or G.edge[node1][node2]['weight']
            
            MakhrajKasrHa = len(Cluster)#CondidateEvents[WinNum][ClusterNum])
            if(MakhrajKasrHa==0):
                print('TASMIMGIRI SHAVAD') ##################################################################################################################
                MiuE[-1].append(0) ##########################################################################################################################
            else:
                MiuE[-1].append((SooratKasrAvval/MakhrajKasrHa)*(SooratKasrDovvom/MakhrajKasrHa))
            
    return MiuE  

def EventNewsWorthy(CondidateEvents,SimilarityGraph):
    MiuE = []
    print('Start MiyE Calculating')
    for WinNum,Window in enumerate(CondidateEvents):
        MiuE.append([])
        WINDOW = list(Window.values())
        print('\n-----------------------------------------\n')
        print('WinNum:{}/{}-#-TotalClusters{}->'.format(WinNum,len(CondidateEvents),len(WINDOW)))
        
        for ClusterNum,Cluster in enumerate(WINDOW):
            print('{}-'.format(ClusterNum))
            SooratKasrAvval = 0
            
#            SooratKasrAvval = random.random()############################################
#            MakhrajKasrHa = random.random()######################################
#            SooratKasrDovvom = random.random()#######################################
            
            ClusterEventSegmentString = []
            for i,eventsegment in enumerate(Cluster):
                SooratKasrAvval += MiuS(eventsegment)
                ClusterEventSegmentString.append(eventsegment)
                
            SooratKasrDovvom= 0    
            Graph = SimilarityGraph[WinNum]
            for node1 in ClusterEventSegmentString:
                for node2 in ClusterEventSegmentString:
                    SooratKasrDovvom += Graph[node1][node2]['w']['weight']  #or G.edge[node1][node2]['weight']
            
            MakhrajKasrHa = len(Cluster)#CondidateEvents[WinNum][ClusterNum])
            if(MakhrajKasrHa==0):
                print('TASMIMGIRI SHAVAD') ##################################################################################################################
                MiuE[-1].append(0) ##########################################################################################################################
            else:
                MiuE[-1].append((SooratKasrAvval/MakhrajKasrHa)*(SooratKasrDovvom/MakhrajKasrHa))
            
    return MiuE  


def HighestNewsWorthy_NLastWindow(MiuE,CondidateEvents,n):
    MiuX = []
    for WinNum in range(len(CondidateEvents)-n,len(CondidateEvents)):
        MiuX.append([])
        MaxMiuValue = max(MiuE[WinNum])
        MiuX[-1]=MaxMiuValue
    return MiuX

def HighestNewsWorthy(MiuE,CondidateEvents):
    MiuX = []
    for WinNum in range(len(CondidateEvents)):
        MiuX.append([])
        MaxMiuValue = max(MiuE[WinNum])
        MiuX[-1]=MaxMiuValue
    return MiuX


def DetectRealisticEvents_2LastWindow(MiuX,MiuE,Tereshold,CondidateEvents):
    RealisticEvents = []
    for WinNum in range(len(CondidateEvents)-2,len(CondidateEvents)):
        RealisticEvents.append([])
        for ClusterNum,Cluster in enumerate(CondidateEvents[WinNum].values()):
            if MiuE[WinNum][ClusterNum] == 0:
                Ratio = Tereshold
            else:
                Ratio = MiuX[WinNum]/MiuE[WinNum][ClusterNum]
            if Ratio<=Tereshold:
                RealisticEvents[-1].append(Cluster)
    return RealisticEvents

def DetectRealisticEvents(MiuX,MiuE,Tereshold,CondidateEvents):
    RealisticEvents = []
    for WinNum in range(len(CondidateEvents)):
        RealisticEvents.append([])
        for ClusterNum,Cluster in enumerate(CondidateEvents[WinNum].values()):
            if MiuE[WinNum][ClusterNum] == 0:
                Ratio = Tereshold
            else:
                Ratio = MiuX[WinNum]/MiuE[WinNum][ClusterNum]
#                print(Ratio)
            if Ratio<Tereshold:
                RealisticEvents[-1].append(Cluster)
    return RealisticEvents


def DetectRealisticEventsTopK(MiuX,MiuE,K_Value,Tereshold,CondidateEvents):
    RealisticEvents = []
    SelectedMiuE = []
    for WinNum in range(len(CondidateEvents)):
        RealisticEvents.append([])
        SelectedMiuE.append([])
        for ClusterNum,Cluster in enumerate(CondidateEvents[WinNum].values()):
            if MiuE[WinNum][ClusterNum] == 0:
                Ratio = Tereshold
            else:
                Ratio = MiuX[WinNum]/MiuE[WinNum][ClusterNum]
#                print(Ratio)
            if Ratio<Tereshold:
                RealisticEvents[-1].append(Cluster)
                SelectedMiuE[-1].append(MiuE[WinNum][ClusterNum])
        
        while len(RealisticEvents[-1])>K_Value:
            MinIndex = SelectedMiuE[-1].index(min(SelectedMiuE[-1]))
            del RealisticEvents[-1][MinIndex]
            del SelectedMiuE[-1][MinIndex]
            
            
    return RealisticEvents




def Top5Rank(Cluster):
    
    AllWordRank = []
    AllWord = []
    for i,Segment in enumerate(Cluster):
        #print('Segment:{}/{}'.format(i,len(Cluster)))
        CurrentRank = lm_scorer.get_score(Segment)
        AllWordRank.append(CurrentRank)
        AllWord.append(Segment)
    
    #Soarting Process
    zipped_lists = zip(AllWordRank, AllWord)
    sorted_pairs = sorted(zipped_lists,reverse=True)
    tuples = zip(*sorted_pairs)
    Rnk, Wrd = [ list(tuple) for tuple in tuples]    
    
    CurrentTitle = []
    for wrd in Wrd[0:5]:
        CurrentTitle.append(wrd)
    
    return CurrentTitle


def DescribeEvents_2LastWindow(RealisticEvents):
    TitleToDescribeEvents = []
    for WinNum in range(len(RealisticEvents)-2,len(RealisticEvents)):
        TitleToDescribeEvents.append([])
        for ClusterNum,Cluster in enumerate(RealisticEvents[WinNum]):
            print('WinNum:{}/{}##ClusterNum:{}/{}'.format(WinNum,len(RealisticEvents),ClusterNum,len(RealisticEvents[WinNum])))
            CurrentTitle = Top5Rank(Cluster)
            TitleToDescribeEvents[-1].append(CurrentTitle)
    return TitleToDescribeEvents

def DescribeEvents(RealisticEvents):
    TitleToDescribeEvents = []
    for WinNum in range(len(RealisticEvents)):
        TitleToDescribeEvents.append([])
        for ClusterNum,Cluster in enumerate(RealisticEvents[WinNum]):
            print('WinNum:{}/{}##ClusterNum:{}/{}'.format(WinNum,len(RealisticEvents),ClusterNum,len(RealisticEvents[WinNum])))
            CurrentTitle = Top5Rank(Cluster)
            TitleToDescribeEvents[-1].append(CurrentTitle)
    return TitleToDescribeEvents




def WindowingLikeTwiner(AllData):

    #AllData Contain :
    #        Seq_Windowing,
    #        Posts_Windowing,
    #        Types_Windowing,
    #        Deleted_Windowing,
    #        DateSend_Windowing,
    #        User_Windowing,
    #        WindowNum
    
    
    Seq_Windowing=[]
    Posts_Windowing=[]
    Types_Windowing=[]
    Deleted_Windowing=[]
    DateSend_Windowing=[]
    User_Windowing=[]
    WindowNum=[]
    
    Current_Seq_Windowing=[]
    Current_Posts_Windowing=[]
    Current_Types_Windowing=[]
    Current_Deleted_Windowing=[]
    Current_DateSend_Windowing=[]
    Current_User_Windowing=[]
    Current_WindowNum=[]
    
    
    PostAdd = 0
    # datetime(year, month, day, hour, minute, second, microsecond)
    HashemiStartTime = datetime(2017, 1, 8, 18, 30, 00, 00)
    PelaskoStartTime = datetime(2017, 1, 19, 8, 00, 00, 00)
    EventStart = HashemiStartTime


    PostNumber = -1
    PostPerWindow = 50#3 #########################50##########50##50##50##50##50#############################################################################################################################################################################
    
    TotalPostsCount = 0
    for Posts in AllData[1]:
        TotalPostsCount += len(Posts)
        
    for OldWindowNum,Posts in enumerate(AllData[1]):
        for ii,Post in enumerate(Posts):
            PostNumber += 1
            if PostNumber>=100 and PostNumber<=(TotalPostsCount-100) and AllData[3][OldWindowNum][ii]=="" and PostAdd < 150 and AllData[4][OldWindowNum][ii]>=EventStart :
                Current_Posts_Windowing.append([])
                Current_Types_Windowing.append([])

                for i,Token in enumerate(Post):
                    Current_Posts_Windowing[-1].append(Token)
                    Current_Types_Windowing[-1].append(AllData[2][OldWindowNum][ii][i])
                PostAdd += 1
                Current_Seq_Windowing.append( AllData[0][OldWindowNum])
                
                
                Current_Deleted_Windowing.append(AllData[3][OldWindowNum][ii])
                Current_DateSend_Windowing.append(AllData[4][OldWindowNum][ii])
                Current_User_Windowing.append(AllData[5][OldWindowNum][ii])
                
                
                
                if(PostAdd % PostPerWindow == 0):
                    
                    Current_WindowNum.append(AllData[6][OldWindowNum])
                    
                    Posts_Windowing.append(Current_Posts_Windowing)
                    Seq_Windowing.append(Current_Seq_Windowing)
                    Types_Windowing.append(Current_Types_Windowing)
                    Deleted_Windowing.append(Current_Deleted_Windowing)
                    DateSend_Windowing.append(Current_DateSend_Windowing)
                    User_Windowing.append(Current_User_Windowing)
                    WindowNum.append(Current_WindowNum)
                    
                    Current_Seq_Windowing=[]
                    Current_Posts_Windowing=[]
                    Current_Types_Windowing=[]
                    Current_Deleted_Windowing=[]
                    Current_DateSend_Windowing=[]
                    Current_User_Windowing=[]
                    Current_WindowNum=[]
                    if(PostAdd == 150 and EventStart == HashemiStartTime):
                        PostAdd = 0
                        EventStart = PelaskoStartTime    
    
    
    
    NewAllData = Seq_Windowing,Posts_Windowing,Types_Windowing,Deleted_Windowing,DateSend_Windowing,User_Windowing,WindowNum     
    return NewAllData


def NewWindowing(AllData,PostsSegments_Windowing,StartTime,StepTime):
    #AllData Contain :
    #        Seq_Windowing,
    #        Posts_Windowing,
    #        Types_Windowing,
    #        Deleted_Windowing,
    #        DateSend_Windowing,
    #        User_Windowing,
    #        WindowNum
    
    SeqInWindow  = []
    PostInWindow  = []
    TypesInWindow  = []
    DeletedInWindow  = []
    DateSendInWindow  = []
    UserInWindow  = []

    
    NewAllData = []
    for i in range(len(AllData)):
       NewAllData.append([]) 
    NewPostsSegments_Windowing = []
    
    PostSegmentInWindow  = []
    
    NewWinNum = 1
    
    
    for OldWinNum in range(len(AllData[1])):
        for OldPostNum in range(len(AllData[1][OldWinNum])):
            
            if AllData[4][OldWinNum][OldPostNum] >= StartTime and AllData[4][OldWinNum][OldPostNum] < StartTime+StepTime:
                SeqInWindow.append(AllData[0][OldWinNum][OldPostNum])
                PostInWindow.append(AllData[1][OldWinNum][OldPostNum])
                TypesInWindow.append(AllData[2][OldWinNum][OldPostNum])
                DeletedInWindow.append(AllData[3][OldWinNum][OldPostNum])
                DateSendInWindow.append(AllData[4][OldWinNum][OldPostNum])
                UserInWindow.append(AllData[5][OldWinNum][OldPostNum])
                #WinNumInWindow.append(AllData[0][OldWinNum][OldPostNum])
                
                PostSegmentInWindow.append(PostsSegments_Windowing[OldWinNum][OldPostNum])
            
            elif  AllData[4][OldWinNum][OldPostNum] >= StartTime and AllData[4][OldWinNum][OldPostNum] >= StartTime+StepTime:
                StartTime = StartTime+StepTime
                
                NewAllData[0].append(SeqInWindow)
                NewAllData[1].append(PostInWindow)
                NewAllData[2].append(TypesInWindow)
                NewAllData[3].append(DeletedInWindow)
                NewAllData[4].append(DateSendInWindow)
                NewAllData[5].append(UserInWindow)
                NewAllData[6] = [NewAllData[6] , NewWinNum]
                NewWinNum+=1
                
                NewPostsSegments_Windowing.append(PostSegmentInWindow)
                
                SeqInWindow  = []
                PostInWindow  = []
                TypesInWindow  = []
                DeletedInWindow  = []
                DateSendInWindow  = []
                UserInWindow  = []
                
                PostSegmentInWindow  = []
                
                SeqInWindow.append(AllData[0][OldWinNum][OldPostNum])
                PostInWindow.append(AllData[1][OldWinNum][OldPostNum])
                TypesInWindow.append(AllData[2][OldWinNum][OldPostNum])
                DeletedInWindow.append(AllData[3][OldWinNum][OldPostNum])
                DateSendInWindow.append(AllData[4][OldWinNum][OldPostNum])
                UserInWindow.append(AllData[5][OldWinNum][OldPostNum])
                #WinNumInWindow.append(AllData[0][OldWinNum][OldPostNum])
                
                PostSegmentInWindow.append(PostsSegments_Windowing[OldWinNum][OldPostNum])     
                
            else:
                print('NNAABBAAYYAADD IINNJJAA Biad CHEK SHAVAD')
    
    
    NewAllData[0].append(SeqInWindow)
    NewAllData[1].append(PostInWindow)
    NewAllData[2].append(TypesInWindow)
    NewAllData[3].append(DeletedInWindow)
    NewAllData[4].append(DateSendInWindow)
    NewAllData[5].append(UserInWindow)
    NewAllData[6] = [NewAllData[6] , NewWinNum]
    
    NewPostsSegments_Windowing.append(PostSegmentInWindow)
    
    return NewAllData,NewPostsSegments_Windowing





def DetectRelated(AllDocumentInWindow,AllSequenceInWindow,Segment):
    Segment = Segment.split(' ')
    IndexToDelete = []
    RelatedDoc = []
    RelatedSeq = []
    for i,post in enumerate(AllDocumentInWindow):
        if Segment in post:
            RelatedDoc.append(post)
            RelatedSeq.append(AllSequenceInWindow[i])
            IndexToDelete.append(i)
            
    for i in reversed(IndexToDelete):
        del AllDocumentInWindow[i]
        del AllSequenceInWindow[i]
    
    return AllDocumentInWindow,AllSequenceInWindow,RelatedDoc,RelatedSeq

def DetectRelatedDoc(AllData,PostsSegments_Windowing,RealisticEvents):
    
    RelatedDocuments = []
    RelatedSequence = []
    
    for WinNum in range(len(RealisticEvents)):
        RelatedDocuments.append([])
        RelatedSequence.append([])
        for EventClusterNum in range(len(RealisticEvents[WinNum])):
            AllDocumentInWindow = copy.deepcopy(PostsSegments_Windowing[WinNum])
            AllSequenceInWindow = copy.deepcopy(AllData[0][WinNum])
            CurentRelatedDocuments = []
            CurentRelatedSequence = []
            for Segment in RealisticEvents[WinNum][EventClusterNum]:
                AllDocumentInWindow,AllSequenceInWindow,RelatedDoc,RelatedSeq = DetectRelated(AllDocumentInWindow,AllSequenceInWindow,Segment)
                CurentRelatedDocuments = [*CurentRelatedDocuments,*RelatedDoc]
                CurentRelatedSequence = [*CurentRelatedSequence,*RelatedSeq]
            RelatedDocuments[-1].append(CurentRelatedDocuments)
            RelatedSequence[-1].append(CurentRelatedSequence)
    
    return RelatedDocuments,RelatedSequence








def JoinSegments(RelatedDocuments):
    RelatedDocumentsString = []
    RelatedDocumentsStringInEvent = []
    RelatedDocumentsStringInCluster = []
    
    for WinNum in range(len(RelatedDocuments)):
        for EventClusterNum in range(len(RelatedDocuments[WinNum])):
            for RelatedDocNum in range(len(RelatedDocuments[WinNum][EventClusterNum])):
                CurrentPostSTR = ' '.join(str(item) for innerlist in RelatedDocuments[WinNum][EventClusterNum][RelatedDocNum] for item in innerlist)
                RelatedDocumentsStringInEvent.append(CurrentPostSTR)
            RelatedDocumentsStringInCluster.append(RelatedDocumentsStringInEvent)
            RelatedDocumentsStringInEvent = []
        RelatedDocumentsString.append(RelatedDocumentsStringInCluster) 
        RelatedDocumentsStringInCluster = []
    return RelatedDocumentsString


def DetectEventOfEachPost(RelatedSequence):
    EventNumberOfEachSequence = []
    
    
    for WinNum in range(len(RelatedSequence)):
        EventNumberOfEachSequence.append({})
        
        for ClusterNum,SequenceSS in enumerate(RelatedSequence[WinNum]):
            for Seq in SequenceSS:
                if Seq  in EventNumberOfEachSequence[-1].keys():
                    EventNumberOfEachSequence[-1][Seq].append(ClusterNum+1)
                else:
                    EventNumberOfEachSequence[-1][Seq] = [ClusterNum+1]
    
    return EventNumberOfEachSequence


def ListTiCSVstring(List):
    CSV = '' 
    for item in List:
        CSV = CSV + ',' + str(item)
    
    return CSV[1:]


def ReadyToWriteToExcell(AllData,EventNumberOfEachSequence):
    AllRelatedEventsOfSequence = []
    
    
    for WinNum in range(len(AllData[1])):
        AllRelatedEventsOfSequence.append([])
        
        for Seq in AllData[0][WinNum]:
            if Seq  in EventNumberOfEachSequence[WinNum].keys():
                CSVformatOfEventNumver = ListTiCSVstring(EventNumberOfEachSequence[WinNum][Seq])
                AllRelatedEventsOfSequence[-1].append(CSVformatOfEventNumver) 
            else:
                AllRelatedEventsOfSequence[-1].append('0')
    
    
    return AllRelatedEventsOfSequence




def SaveToExcel(Path,AllSequence,AllEventNumber):
    
    book = xlwt.Workbook(encoding="utf-8")
    
    for WinNum in range(len(AllSequence)):
        CurrentSheet = book.add_sheet("Window-{}".format(WinNum+1))
        CurrentSheet.write(0, 0, "Sequence")
        CurrentSheet.write(0, 1, "EventNumber")
        
        Row = 1 #RowToAddInCurrentSheet
        for i,seq in enumerate(AllSequence[WinNum]):
            CurrentSheet.write(Row, 0, seq)
            CurrentSheet.write(Row, 1,AllEventNumber[WinNum][i])
            Row+=1

    book.save(Path)
    print("File Saved")


def SaveToExcellK_value(Path,KList,KMinList,S_Score,Column3Label):
    book = xlwt.Workbook(encoding="utf-8")
    
    for WinNum in range(len(KList)):
        CurrentSheet = book.add_sheet("Window-{}".format(WinNum+1))
        CurrentSheet.write(0, 0, "Radif")
        CurrentSheet.write(0, 1, "K")
        CurrentSheet.write(0, 2, "K_Min")
        CurrentSheet.write(0, 3, Column3Label)
        
        Row=1
        for i in range(len(KList[WinNum])):
            CurrentSheet.write(Row, 0, i+1)
            CurrentSheet.write(Row, 1,KList[WinNum][i])
            CurrentSheet.write(Row, 2,KMinList[WinNum][i])
            CurrentSheet.write(Row, 3,S_Score[WinNum][i])
            Row+=1
    book.save(Path)
    print("File Saved")




def CalculateSiluhet(clusters,Graph):
    

    n=len(Graph.nodes)
    DistanceMatrix = np.zeros([n,n])
    WeightMatrix = np.zeros([n,n])
    
    def weight_func(node1, node2):
        return Graph[node1][node2]['w']['weight']
    
    def distance_func(node1, node2):
        return DistanceMatrix[int(node1[0])][int(node2[0])]

    Silhouette_Coefficient = -1
            
    if len(clusters) != 1 and len(clusters) != len(Graph.nodes):
        X = []
        X_Num = []
        xnum=0
        
        labels = []
        for Num,Clus in enumerate(clusters):
            for node in Clus:
                labels.append(Num+1)
                X.append(node)
                X_Num.append(xnum)
                xnum+=1
        
        #WeightMatrix = np.zeros([n,n])
        MaxWeight = 0
        for i,n1 in enumerate(X):
            for j,n2 in enumerate(X):
                WeightMatrix[i][j] = weight_func(n1, n2)
                if WeightMatrix[i][j] > MaxWeight:
                    MaxWeight = WeightMatrix[i][j]
        
        for i in range(len(X)):
            for j in range(len(X)):
                DistanceMatrix[i][j] = abs(WeightMatrix[i][j]-MaxWeight)
        
        X2 = np.array(X_Num)
        X2=np.reshape(X2,(-1,1))
        
        l=np.array(labels)
        l=np.reshape(l,(-1,))
    
    
        Silhouette_Coefficient = metrics.silhouette_score(X2, l, metric=distance_func)
    
    return Silhouette_Coefficient





def CalculateEntropy(AllSequenceAndRelatedEvents,WinNum):
    
    classE,clusterE = CEval.Evaluate(WinNum,AllSequenceAndRelatedEvents[0])
    
    return classE,clusterE





def SaveSystemResultTopic(Events,WindowNum,Path):

    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Topic_Systemresult")

    col = 0
    for i,Topics in enumerate(Events):
        sheet1.write(0,col, "Window {}".format(WindowNum[i]))
        for row,topic in enumerate(Topics):
            sheet1.write(row+1,col, topic)
        col = col+1
     
    book.save(Path + r"\Topic_Systemresult.xls")
    print('Result for compute partiale saved in :')
    print(Path + r"\Topic_Systemresult.xls")












##=============================================================================
##============================TwiEvent Main Body===============================
##=============================================================================


print("Start Program")

# The Path variable is now relative to the script's location.
Path = '.'

# The database connection is no longer needed and has been removed.
# global c

# Initialize the language model scorer
initialize_lm_scorer()


# The script now loads only the initial raw data.
print('Loading AllData ...')
# Make sure the AllData.npy file is in the same directory as the script.
tempNumpyArray=np.load(os.path.join(Path, 'AllData.npy'), allow_pickle=True)
AllData = tempNumpyArray.tolist()


# AllData Contain :
#        Seq_Windowing,
#        Posts_Windowing,
#        Types_Windowing,
#        Deleted_Windowing,
#        DateSend_Windowing,
#        User_Windowing,
#        WindowNum


# The following sections are now uncommented to generate data on the fly,
# instead of loading from pre-computed files.

#Segment kardane post ha (Az SCP Estefade shode)
print("Segmenting posts...")
PostsSegments_Windowing = Segmentation(AllData[1])
#AllData[1] Means the Posts in TimeWindowing
np.save(os.path.join(Path, 'PostsSegments_Windowing.npy'), PostsSegments_Windowing)
print('\n PostsSegments_Windowing Saved')
'''
#OR
print('Loading PostsSegments_Windowing ...')
tempNumpyArray=np.load(Path+r'\PostsSegments_Windowing.npy',allow_pickle=True)
PostsSegments_Windowing = tempNumpyArray.tolist()
'''

####If Load AllData(Not Only Window14-17) this two Line Should be execute
#StepTime = timedelta(hours=12) # timedelta(days=1)
#AllData,PostsSegments_Windowing = NewWindowing(AllData,PostsSegments_Windowing,StartTime,StepTime)


#Detect Bursty Segment
print("Detecting bursty segments...")
EventSegment_Windowing,EventSegmentWeight_Windowing = DetectBursty(AllData,PostsSegments_Windowing)
#EventSegment is TweetsBurstySegments
np.save(os.path.join(Path, 'EventSegment_Windowing.npy'), EventSegment_Windowing)
print('\n EventSegment_Windowing Saved')
np.save(os.path.join(Path, 'EventSegmentWeight_Windowing.npy'), EventSegmentWeight_Windowing)
print('\n EventSegmentWeight_Windowing Saved')
'''
#OR
print('Loading EventSegment ...')
tempNumpyArray=np.load(Path+r'\EventSegment_Windowing.npy',allow_pickle=True)
EventSegment_Windowing = tempNumpyArray.tolist()
tempNumpyArray=np.load(Path+r'\EventSegmentWeight_Windowing.npy',allow_pickle=True)
EventSegmentWeight_Windowing = tempNumpyArray.tolist()
'''

#
#################################################################################################################
#
#EventSegment_Windowing_2LastWindow,EventSegmentWeight_Windowing_2LastWindow = DetectBurstyNLastWindow(AllData,PostsSegments_Windowing,13)
#
#EventSegment_Windowing = [*EventSegment_Windowing ,*EventSegment_Windowing_2LastWindow ]
#EventSegmentWeight_Windowing = [*EventSegmentWeight_Windowing ,*EventSegmentWeight_Windowing_2LastWindow ]
#
#np.save(Path+r'\EventSegment_Windowing.npy',EventSegment_Windowing)
#print('\n EventSegment_Windowing Saved')
#np.save(Path+r'\EventSegmentWeight_Windowing.npy',EventSegmentWeight_Windowing)
#print('\n EventSegmentWeight_Windowing Saved')
#
#################################################################################################################
#

# This part requires a StartTime variable which was defined in the commented out SQL block.
# I will define it here.
StartTime = datetime(2017, 1,  1, 00, 00, 00)

print("Clustering event segments...")
StepTime = timedelta(hours=4) # timedelta(days=1)
SimilarityGraph = EventSegmentClustering_Similarity(AllData,EventSegment_Windowing,StartTime,StepTime)
# Condedate Evente is clysters of segment in each time window
np.save(os.path.join(Path, 'SimilarityGraph.npy'), SimilarityGraph)
print('\n SimilarityGraph Saved')
'''
#OR
print('Loading SimilarityGraph ...')
tempNumpyArray=np.load(Path+r'\SimilarityGraph.npy',allow_pickle=True)
SimilarityGraph = tempNumpyArray.tolist()
'''

#
#################################################################################################################
#SimilarityGraph_2LastWindow = EventSegmentClustering_Similarity_NLastWindow(EventSegment_Windowing,13)
#
#SimilarityGraph = [*SimilarityGraph ,*SimilarityGraph_2LastWindow ]
#
#np.save(Path+r'\SimilarityGraph.npy',SimilarityGraph)
#print('\n SimilarityGraph Saved')
#
#################################################################################################################
#


#############EvaluateAfterClusteringComplete
print("Clustering started...")
CondidateEvents,NoiseS = EventSegmentClustering(SimilarityGraph,EventSegment_Windowing,15,6)      # CondidateEvents,KList,KMinList,S_Score = EventSegmentClusteringByParametrTunning(SimilarityGraph,EventSegment_Windowing)
np.save(os.path.join(Path, 'CondidateEvents.npy'), CondidateEvents)
print('\n CondidateEvents Saved')
'''
#OR
print('Loading CondidateEvents ...')
tempNumpyArray=np.load(Path+r'\CondidateEvents.npy',allow_pickle=True)
CondidateEvents = tempNumpyArray.tolist()
'''


##SaveToExcellK_value(Path+r'\KvalueBySiluhette.xls',KList,KMinList,S_Score,"Silh")   # if run Bt Parametr tuning in line 2519


"""##############################################################################









#############EvaluateAfterDetectRealisticEvent
KList=[]
KMinList=[]
S_Score_After=[]
Class_E=[]
Cluster_E=[]
Entropy_Measure=[]

K=50
MaxK = 118


####################Only Window 14 for Speed test###################################3
WINDOWNUMBER = 15
start=1
end=2

AllData[0] = AllData[0][start:end]
AllData[1] = AllData[1][start:end]
AllData[2] = AllData[2][start:end]
AllData[3] = AllData[3][start:end]
AllData[4] = AllData[4][start:end]
AllData[5] = AllData[5][start:end]
AllData[6] = AllData[6][start:end]
PostsSegments_Windowing = PostsSegments_Windowing[start:end]
EventSegment_Windowing = EventSegment_Windowing[start:end]
EventSegmentWeight_Windowing = EventSegmentWeight_Windowing[start:end]
SimilarityGraph = SimilarityGraph[start:end]
#######################################################3


for k in range(K,MaxK):
    print("Process Start By K:{}".format(k))
    for k_min in range(1,k):

        CondidateEvents,NoiseS = EventSegmentClustering(SimilarityGraph,EventSegment_Windowing,k,k_min)
        
        MiuE = EventNewsWorthy(CondidateEvents,SimilarityGraph)
        MiuX = HighestNewsWorthy(MiuE,CondidateEvents)
        
        Tereshold = 5##################################################################################.5
        RealisticEvents = DetectRealisticEvents(MiuX,MiuE,Tereshold,CondidateEvents)
        
        RelatedDocuments,RelatedSequence = DetectRelatedDoc(AllData,PostsSegments_Windowing,RealisticEvents)
        
        EventNumberOfEachSequence = DetectEventOfEachPost(RelatedSequence)
        AllSequenceAndRelatedEvents = ReadyToWriteToExcell(AllData,EventNumberOfEachSequence)
        
        
        for WinNum in range(len(MiuX)):
            if len(KList) == WinNum:
                KList.append([])
                KMinList.append([])
                S_Score_After.append([])
                Class_E.append([])
                Cluster_E.append([])
                Entropy_Measure.append([])
                
            KList[WinNum].append(k)
            KMinList[WinNum].append(k_min)
            S_Score_After[WinNum].append(CalculateSiluhet(RealisticEvents[WinNum],SimilarityGraph[WinNum]))
            classE,clusterE = CalculateEntropy(AllSequenceAndRelatedEvents,WINDOWNUMBER) #Or WinNum for total process
            
            Class_E[WinNum].append(classE)
            Cluster_E[WinNum].append(clusterE)
            Entropy_Measure[WinNum].append((2*classE*clusterE)/(classE+clusterE))
        
        
        
        
SaveToExcellK_value(Path+r'\KvalueBySiluhetteAfterAllProcesss_'+str(WINDOWNUMBER)+'.xls',KList,KMinList,S_Score_After,'SiluhetteAfter')
SaveToExcellK_value(Path+r'\KvalueByClassEntropy_'+str(WINDOWNUMBER)+'.xls',KList,KMinList,Class_E,'ClassEntropy')
SaveToExcellK_value(Path+r'\KvalueByClusterEntropy_'+str(WINDOWNUMBER)+'.xls',KList,KMinList,Cluster_E,'ClusterEntropy')
SaveToExcellK_value(Path+r'\KvalueByTotalEntropyMeasure_'+str(WINDOWNUMBER)+'.xls',KList,KMinList,Entropy_Measure,'EntropyMeasure')
        
        
    
        
        
"""##############################################################################     
     
        








"""##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
'''        
print("Clustering Start")
CondidateEvents = EventSegmentClustering(SimilarityGraph,EventSegment_Windowing,k,k_min)

np.save(Path+r'\CondidateEvents.npy',CondidateEvents)
print('\n CondidateEvents Saved')

#OR
'''
print('Loading CondidateEvents ...')
tempNumpyArray=np.load(Path+r'\CondidateEvents.npy',allow_pickle=True)
CondidateEvents = tempNumpyArray.tolist()
"""##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55


print("Calculating event newsworthiness...")
MiuE = EventNewsWorthy(CondidateEvents,SimilarityGraph)
np.save(os.path.join(Path, 'MiuE.npy'), MiuE)
print('\n MiuE Saved')
'''
#OR
print('Loading MiuE ...')
tempNumpyArray=np.load(Path+r'\MiuE.npy',allow_pickle=True)
MiuE = tempNumpyArray.tolist()
'''


#################################################################################################################
#MiuE_2LastWindow = EventNewsWorthy_NLastWindow(CondidateEvents,SimilarityGraph,13)
#
#MiuE = [*MiuE ,*MiuE_2LastWindow ]
#
#np.save(Path+r'\MiuE.npy',MiuE)
#print('\n MiuE Saved')
#################################################################################################################


print("Calculating highest newsworthiness...")
MiuX = HighestNewsWorthy(MiuE,CondidateEvents)
np.save(os.path.join(Path, 'MiuX.npy'), MiuX)
print('\n MiuX Saved')
'''
#OR
print('Loading MiuX ...')
tempNumpyArray=np.load(Path+r'\MiuX.npy',allow_pickle=True)
MiuX = tempNumpyArray.tolist()
'''



##################################################################################################################
#MiuX_2LastWindow = HighestNewsWorthy_NLastWindow(MiuE,CondidateEvents,13)
#
#MiuX = [*MiuX ,*MiuX_2LastWindow ]
#
#np.save(Path+r'\MiuX.npy',MiuX)
#print('\n MiuX Saved')
##################################################################################################################
#









##### Select And Save Special Window  ####
##
##
#AllData[0] = AllData[0][13:17]
#AllData[1] = AllData[1][13:17]
#AllData[2] = AllData[2][13:17]
#AllData[3] = AllData[3][13:17]
#AllData[4] = AllData[4][13:17]
#AllData[5] = AllData[5][13:17]
#AllData[6] = AllData[6][13:17]
#PostsSegments_Windowing = PostsSegments_Windowing[13:17]
#EventSegment_Windowing = EventSegment_Windowing[13:17]
#EventSegmentWeight_Windowing = EventSegmentWeight_Windowing[13:17]
#SimilarityGraph = SimilarityGraph[13:17]
#CondidateEvents = CondidateEvents[13:17]
#MiuE = MiuE[13:17]
#MiuX = MiuX[13:17]
#
#Path = Path+r'\Window14_17'
#np.save(Path+r'\AllData.npy',AllData)
#print('\n AllData Saved')
#np.save(Path+r'\PostsSegments_Windowing.npy',PostsSegments_Windowing)
#print('\n PostsSegments_Windowing Saved')
#np.save(Path+r'\EventSegment_Windowing.npy',EventSegment_Windowing)
#print('\n EventSegment_Windowing Saved')
#np.save(Path+r'\EventSegmentWeight_Windowing.npy',EventSegmentWeight_Windowing)
#print('\n EventSegmentWeight_Windowing Saved')
#np.save(Path+r'\SimilarityGraph.npy',SimilarityGraph)
#print('\n SimilarityGraph Saved')
#np.save(Path+r'\CondidateEvents.npy',CondidateEvents)
#print('\n CondidateEvents Saved')
#np.save(Path+r'\MiuE.npy',MiuE)
#print('\n MiuE Saved')
#np.save(Path+r'\MiuX.npy',MiuX)
#print('\n MiuX Saved')
####
####
####
####
####
####
####
####









#->>>>>>      AZ INJA BE BAD HATMAN ESME File Hayi Ke SAVE Mishe Cheack Shavad

print("Detecting realistic events...")
Tereshold = 15##################################################################################.5
RealisticEvents = DetectRealisticEvents(MiuX,MiuE,Tereshold,CondidateEvents)
np.save(os.path.join(Path, 'RealisticEvents_tereshold15.npy'), RealisticEvents)
print('\n RealisticEvents Saved')
'''
#OR
print('Loading RealisticEvents ...')
tempNumpyArray=np.load(Path+r'\RealisticEvents_tereshold15.npy',allow_pickle=True)
RealisticEvents = tempNumpyArray.tolist()
'''




print("Detecting top K realistic events...")
Tereshold = 15
K_Value = 5
RealisticEventsTopK = DetectRealisticEventsTopK(MiuX,MiuE,K_Value,Tereshold,CondidateEvents)
np.save(os.path.join(Path, 'RealisticEvents_tereshold15_TopK5.npy'), RealisticEventsTopK)
print('\n RealisticEventsTopK Saved')
'''
#OR
print('Loading RealisticEventsTopK ...')
tempNumpyArray=np.load(Path+r'\RealisticEvents_tereshold15_TopK5.npy',allow_pickle=True)
RealisticEventsTopK = tempNumpyArray.tolist()
'''











#################################################################################################################
#Tereshold = 500##################################################################################.5
#RealisticEvents_2LastWindow = DetectRealisticEvents_2LastWindow(MiuX,MiuE,Tereshold,CondidateEvents)
#
#RealisticEvents = [*RealisticEvents ,*RealisticEvents_2LastWindow ]
#
#np.save(Path+r'\RealisticEvents.npy',RealisticEvents)
#print('\n RealisticEvents Saved')
#################################################################################################################


print("Describing events...")
TitleToDescribeEventsSTR = DescribeEvents(RealisticEvents)
np.save(os.path.join(Path, 'TitleToDescribeEventsSTR_tereshold15.npy'), TitleToDescribeEventsSTR)
print('\n TitleToDescribeEvents Saved')
'''
#OR
print('Loading TitleToDescribeEventsSTR ...')
tempNumpyArray=np.load(Path+r'\TitleToDescribeEventsSTR_tereshold15.npy',allow_pickle=True)
TitleToDescribeEventsSTR = tempNumpyArray.tolist()
'''







#################################################################################################################
#TitleToDescribeEventsSTR_2LastWindow = DescribeEvents_2LastWindow(RealisticEvents)
#
#TitleToDescribeEventsSTR = [*TitleToDescribeEventsSTR ,*TitleToDescribeEventsSTR_2LastWindow ]
#
#np.save(Path+r'\TitleToDescribeEventsSTR.npy',TitleToDescribeEventsSTR)
#print('\n TitleToDescribeEvents Saved')
#################################################################################################################
#






print("Detecting related documents...")
RelatedDocuments,RelatedSequence = DetectRelatedDoc(AllData,PostsSegments_Windowing,RealisticEvents)
np.save(os.path.join(Path, 'RelatedDocuments_tereshold15.npy'), RelatedDocuments)
print('\n RelatedDocuments Saved')
np.save(os.path.join(Path, 'RelatedSequence_tereshold15.npy'), RelatedSequence)
print('\n RelatedSequence Saved')
'''
#OR
print('Loading RelatedDocuments ...')
tempNumpyArray=np.load(Path+r'\RelatedDocuments_tereshold15.npy',allow_pickle=True)
RelatedDocuments = tempNumpyArray.tolist()
print('Loading RelatedSequence ...')
tempNumpyArray=np.load(Path+r'\RelatedSequence_tereshold15.npy',allow_pickle=True)
RelatedSequence = tempNumpyArray.tolist()
'''




print("Joining segments...")
RelatedDocumentsString = JoinSegments(RelatedDocuments)
np.save(os.path.join(Path, 'RelatedDocumentsString_tereshold15.npy'), RelatedDocumentsString)
print('\n RelatedDocumentsString Saved')
'''
#OR
print('Loading RelatedDocumentsString ...')
tempNumpyArray=np.load(Path+r'\RelatedDocumentsString_tereshold15.npy',allow_pickle=True)
RelatedDocumentsString = tempNumpyArray.tolist()
'''






EventNumberOfEachSequence = DetectEventOfEachPost(RelatedSequence)


AllSequenceAndRelatedEvents = ReadyToWriteToExcell(AllData,EventNumberOfEachSequence)



SaveToExcel(os.path.join(Path, 'ResultsToCompaire_Tereshold15.xls'), AllData[0], AllSequenceAndRelatedEvents)


WindowNum = [1,2,14,15,16,17,18,19,26,31,32,37,38,39,40]
SaveSystemResultTopic(RealisticEventsTopK,WindowNum,Path)








#"""##############################################################################