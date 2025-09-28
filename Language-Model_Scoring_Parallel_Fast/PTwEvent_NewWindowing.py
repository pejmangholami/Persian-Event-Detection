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
import multiprocessing
from datetime import timedelta
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer  # if not work try: ""pip uninstall scipy"" and then ""pip install scipy""
import jarvispatrick
from sklearn import metrics  # if not work try: ""pip uninstall scipy"" and then ""pip install scipy""
import copy
import xlwt
import CEval
import itertools
import random
import pprint
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

# clr and FastTokenizer are loaded lazily to support multiprocessing.

# Choose model: 'HooshvareLab/bert-fa-base-uncased' or 'bert-base-multilingual-cased'
LM_MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'
lm_scorer = None

class LanguageModelScorer:
    def __init__(self, model_name='HooshvareLab/bert-fa-base-uncased', local_files_only=False):
        print(f"Loading tokenizer for {model_name} (offline: {local_files_only})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        print(f"Loading model {model_name} (offline: {local_files_only})...")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.eval()
        self.cache = {}
        print("Model loaded.")

    def get_score(self, text):
        # Return cached score if available to avoid re-computation
        if text in self.cache:
            return self.cache[text]
        if not text:
            return 0.0

        # Tokenize the input text
        tokenized_text = self.tokenizer.tokenize(text)
        if not tokenized_text or len(tokenized_text) == 0:
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

            # Get model predictions by explicitly passing the input tensor as `input_ids`
            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensor_masked)
                predictions = outputs.logits

            # Get the log probability of the original token at the masked position
            log_probs = torch.nn.functional.log_softmax(predictions[0, i], dim=0)
            token_log_prob = log_probs[indexed_tokens[i]].item()

            total_log_prob += token_log_prob

        # Normalize by length and convert back from log space
        avg_log_prob = total_log_prob / len(tokenized_text)
        score = math.exp(avg_log_prob)

        # Cache the result before returning
        self.cache[text] = score
        return score

def initialize_lm_scorer(offline_mode=False):
    global lm_scorer
    if lm_scorer is None:
        print(f"Initializing language model for main process (offline: {offline_mode})...")
        lm_scorer = LanguageModelScorer(LM_MODEL_NAME, local_files_only=offline_mode)
        print("Language model initialized for main process.")

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

_fast_tokenizer_api = None

def get_fast_tokenizer_api():
    """Initializes and returns the FastTokenizer API, ensuring it's loaded only once."""
    global _fast_tokenizer_api
    if _fast_tokenizer_api is None:
        print("Initializing FastTokenizer API...")
        import clr
        import os
        clr.AddReference(os.path.join(os.getcwd(), 'FastTokenizer.dll'))
        from FastTokenizer import Api
        _fast_tokenizer_api = Api
    return _fast_tokenizer_api


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
    Api = get_fast_tokenizer_api()
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
    Pattern = r'^[۰-۹0-9 -_)(*&^%$#@!~]+$|^.*خخ+.*$|^.*(ههه )+.*$|^.*\#\#+.*$|^WORDorEXP$|^WORDorEXP$|^WORDorEXP$'
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
    Api = get_fast_tokenizer_api()
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



def init_worker(offline_mode=False):
    global lm_scorer
    # This function will be called by each worker process.
    # It initializes the language model scorer for that process.
    print(f"Initializing language model for worker (pid: {os.getpid()})...")
    lm_scorer = LanguageModelScorer(LM_MODEL_NAME, local_files_only=offline_mode)

def segment_tweet_worker(tweet, u, e):
    if not tweet:
        return []

    # Unpack the tweet and its index
    i, CurrentTweet = tweet

    print('Target Tweet[{}]'.format(i))
    if CurrentTweet != []:
        print('DynamicAlg Running.')
        #Marhale Avval Dynamic Alg...
        PosibbleSegments = DynamicAlg(CurrentTweet, u, e, 'scp') # Flag Can Be pmi or scp
        print('ValidSegment Colculated.')
        TempSegment = PosibbleSegments[0:-1]
        return TempSegment
    return []

def Segmentation(Tweets, u, e, offline_mode=False):
    SegmentationResults_Windowing = []

    # Determine the number of processes to use
    num_processes = min(os.cpu_count(), 8) # Limit to 8 processes to avoid excessive memory usage

    # Initialize the language model scorer in the main process
    initialize_lm_scorer(offline_mode=offline_mode)

    # Create a partial function to pass the fixed u and e parameters to the worker
    worker_func = partial(segment_tweet_worker, u=u, e=e)


    # Use a multiprocessing pool to parallelize the segmentation
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(PARAM_FORCE_OFFLINE,)) as pool:
        for WinNum in range(len(Tweets)):
            print(f"--- Processing Window {WinNum+1}/{len(Tweets)} ---")

            # Prepare the data for the current window with indices
            window_tweets = list(enumerate(Tweets[WinNum]))

            # Process tweets in parallel
            results = pool.map(worker_func, window_tweets)

            SegmentationResults_Windowing.append(results)

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

    return EventSegment_Windowing,EventSegmentWeight_Windowing






def DetectBursty(AllData, PostSegments):
    EventSegment_Windowing = []
    EventSegmentWeight_Windowing = []

    for WinNum in range(len(PostSegments)):
        print('DetectBursty For Win:{}/{}'.format(WinNum, len(PostSegments)))

        Window_BurstySegments = []
        Window_BurstyWeights = []

        for PostNum in range(len(PostSegments[WinNum])):
            CurentPostBurstySegment = []
            CurentPostBurstyWeight = []

            for SegmentNum in range(len(PostSegments[WinNum][PostNum])):
                CurrentSegment = PostSegments[WinNum][PostNum][SegmentNum]
                Ps = P(PostSegments, CurrentSegment, WinNum)
                Nt = len(PostSegments[WinNum])
                Est = Ps * Nt
                Fst = FST(CurrentSegment, PostSegments[WinNum])

                if Fst > Est:
                    CurentPostBurstySegment.append(CurrentSegment)
                    Dst = math.sqrt(Nt * Ps * (1 - Ps))
                    if Fst >= (Est + 2 * Dst):
                        PBst = 1
                    else:
                        PBst = sigmoid(10 * (Fst - (Est + Dst)) / Dst)
                    Ust = U(AllData, PostSegments, WinNum, CurrentSegment)
                    WBst = PBst * math.log(Ust) if Ust > 0 else 0
                    CurentPostBurstyWeight.append(WBst)

            if CurentPostBurstySegment:
                Nt = len(PostSegments[WinNum])
                K = math.ceil(math.sqrt(Nt))

                zipped_lists = zip(CurentPostBurstyWeight, CurentPostBurstySegment)
                sorted_pairs = sorted(zipped_lists, reverse=True)

                if sorted_pairs:
                    tuples = zip(*sorted_pairs)
                    Wei, Seg = [list(t) for t in tuples]
                    Window_BurstySegments.append(Seg[0:K])
                    Window_BurstyWeights.append(Wei[0:K])
                else:
                    Window_BurstySegments.append([])
                    Window_BurstyWeights.append([])
            else:
                Window_BurstySegments.append([])
                Window_BurstyWeights.append([])

        EventSegment_Windowing.append(Window_BurstySegments)
        EventSegmentWeight_Windowing.append(Window_BurstyWeights)

    return EventSegment_Windowing, EventSegmentWeight_Windowing

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


def calculate_sim_for_subwindow(m, node1, node2, CurrentSubWindowing):
    CurrentSubWindow = CurrentSubWindowing[m]
    T1 = TweetContain(node2, CurrentSubWindow)
    T2 = TweetContain(node1, CurrentSubWindow)
    Temp1 = WT(node2, m, CurrentSubWindowing) * WT(node1, m, CurrentSubWindowing)
    TT1 = CreatePseudoDoc(T1)
    TT2 = CreatePseudoDoc(T2)
    Temp2 = SIM(TT1, TT2)
    return Temp1 * Temp2

def CalculateSim(node_pair, CurrentSubWindowing):
    node1, node2 = node_pair
    total_similarity = 0
    for m in range(len(CurrentSubWindowing)):
        total_similarity += calculate_sim_for_subwindow(m, node1, node2, CurrentSubWindowing)
    return (node1, node2, total_similarity)

def CreateSimilarityGraph(CurrentWindow, CurrentSubWindowing):
    StartTime = datetime.now()
    print(StartTime)
    print('current window size: {}'.format(len(CurrentWindow)))
    G = nx.MultiDiGraph()
    print('Adding Nodes To Graph ...')
    NodeList = []
    for Tweet in CurrentWindow:
        for Segment in Tweet:
            NodeList.append(" ".join(Segment))

    # Remove duplicates
    NodeList = list(dict.fromkeys(NodeList))
    G.add_nodes_from(NodeList)

    print('Node Number: {}'.format(len(NodeList)))

    print('Preparing and Adding Edges...')
    from itertools import combinations

    # Use an iterator instead of a list to save memory
    node_pairs = combinations(NodeList, 2)

    # Limit the number of processes to avoid overwhelming the system
    num_processes = min(os.cpu_count(), 8)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a partial function to pass the CurrentSubWindowing argument
        worker_func = partial(CalculateSim, CurrentSubWindowing=CurrentSubWindowing)

        # Use imap_unordered for memory efficiency, it processes results as they complete
        results_iterator = pool.imap_unordered(worker_func, node_pairs)

        print('Adding Edges To Grapg ...')
        for node1, node2, similarity in results_iterator:
            if similarity > 0:
                G.add_edge(node1, node2, key='w', weight=similarity, label=similarity)
                G.add_edge(node2, node1, key='w', weight=similarity, label=similarity)

    print('Time Taken: {}'.format(datetime.now() - StartTime))
    return G

def ClusteringAlgByParametrTunning(Graph):
    clusters = []
    n=len(Graph.nodes)
    DistanceMatrix = np.zeros([n,n])
    WeightMatrix = np.zeros([n,n])
    def weight_func(node1, node2):
        if Graph.has_edge(node1, node2):
            return Graph[node1][node2]['w']['weight']
        else:
            return 0.0
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
        if Graph.has_edge(node1, node2):
            return Graph[node1][node2]['w']['weight']
        else:
            return 0.0
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
                    if Graph.has_edge(node1, node2):
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
                    if Graph.has_edge(node1, node2):
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
        if MiuE[WinNum]:
            MaxMiuValue = max(MiuE[WinNum])
            MiuX.append(MaxMiuValue)
        else:
            MiuX.append(0)
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
    AllWord = Cluster
    if not AllWord:
        return []

    AllWordRank = [MiuS(segment) for segment in AllWord]

    # Soarting Process
    zipped_lists = zip(AllWordRank, AllWord)
    sorted_pairs = sorted(zipped_lists, reverse=True)

    if not sorted_pairs:
        return []

    tuples = zip(*sorted_pairs)
    Rnk, Wrd = [list(t) for t in tuples]

    CurrentTitle = []
    for wrd in Wrd[0:5]:
        CurrentTitle.append(wrd)

    return CurrentTitle


def DescribeEvents_2LastWindow(RealisticEvents):
    TitleToDescribeEvents = []
    num_processes = min(os.cpu_count(), 4)
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        for WinNum in range(len(RealisticEvents) - 2, len(RealisticEvents)):
            print(f"--- Describing Events for Window {WinNum} ---")
            clusters_to_process = RealisticEvents[WinNum]
            titles = pool.map(Top5Rank, clusters_to_process)
            TitleToDescribeEvents.append(titles)
    return TitleToDescribeEvents


def DescribeEvents(RealisticEvents, offline_mode=False):
    TitleToDescribeEvents = []
    # It's more efficient to create the pool once
    num_processes = min(os.cpu_count(), 4)
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(offline_mode,)) as pool:
        for WinNum in range(len(RealisticEvents)):
            print(f"--- Describing Events for Window {WinNum+1}/{len(RealisticEvents)} ---")
            clusters_to_process = RealisticEvents[WinNum]

            # Map the sequential Top5Rank function across the clusters
            titles = pool.map(Top5Rank, clusters_to_process)

            TitleToDescribeEvents.append(titles)

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





def get_param_stamped_filename(base_name, param_dict, extension):
    """
    Generates a filename with a stamp of the given parameters.
    Example: RealisticEvents_u-5_k-15.npy
    """
    stamp = ""
    for key, value in param_dict.items():
        # Sanitize key for filename
        sanitized_key = key.replace('PARAM_', '').replace('_VALUES', '').lower()
        stamp += f"_{sanitized_key}-{value}"

    return f"{base_name}{stamp}.{extension}"

def SaveSystemResultTopic(Events,WindowNum,FullPath):

    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Topic_Systemresult")

    # Write headers
    sheet1.write(0, 0, "Window Number")
    sheet1.write(0, 1, "Topic")

    row_idx = 1 # Start writing from the second row
    for i, Topics in enumerate(Events):
        # Use the actual window number
        # Handle cases where WindowNum might be nested due to previous buggy code.
        current_window_num = WindowNum[i]
        if isinstance(current_window_num, list):
            # Take the last element if it's a list, assuming it's the intended number
            current_window_num = current_window_num[-1]

        if not Topics: # Handle cases with no topics for a given window
            sheet1.write(row_idx, 0, current_window_num)
            sheet1.write(row_idx, 1, "") # Write empty string for topic
            row_idx += 1
        else:
            for topic in Topics:
                sheet1.write(row_idx, 0, current_window_num)
                # topic is a list of segments, join them into a string
                topic_str = " | ".join(topic)
                sheet1.write(row_idx, 1, topic_str)
                row_idx += 1

    book.save(FullPath)
    print('Result for compute partiale saved in :')
    print(FullPath)












##=============================================================================
##============================TwiEvent Main Body===============================
##=============================================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    print("Start Program")

    # The Path variable is now relative to the script's location.
    Path = '.'

    # =============================================================================
    # == Configuration Parameters for the Experiment ==
    # =============================================================================
    # -- Define lists of values for parameters to be tuned --
    # -- To run multiple experiments, add more values to these lists --
    PARAM_U_VALUES = [5]  # Example: [3, 5, 7]
    PARAM_E_VALUES = [20]  # Example: [10, 20]
    PARAM_K_VALUES = [15]  # Example: [10, 15, 20]
    PARAM_K_MIN_VALUES = [6]  # Example: [4, 6]
    PARAM_TERESHOLD_VALUES = [15.0]  # Example: [5.0, 10.0, 15.0]
    PARAM_K_VALUE_VALUES = [5]  # Example: [5, 10]

    # -- Single-value parameters (not iterated over in this setup) --
    PARAM_STEP_TIME_HOURS = 4  # Duration of each processing window in hours
    PARAM_FORCE_OFFLINE = False # Set to True to force offline mode after initial download
    # =============================================================================

    # --- Create all combinations of parameters ---
    param_combinations = list(itertools.product(
        PARAM_U_VALUES,
        PARAM_E_VALUES,
        PARAM_K_VALUES,
        PARAM_K_MIN_VALUES,
        PARAM_TERESHOLD_VALUES,
        PARAM_K_VALUE_VALUES
    ))
    print(f"Total number of experiments to run: {len(param_combinations)}")

    # --- Load data and initialize model once before starting the loop ---
    print('Loading AllData ...')
    tempNumpyArray = np.load(os.path.join('../Language-Model_Scoring', 'AllData.npy'), allow_pickle=True)
    AllData_original = tempNumpyArray.tolist()
    initialize_lm_scorer(offline_mode=PARAM_FORCE_OFFLINE)


    # --- Main loop to run the pipeline for each parameter combination ---
    for i, params in enumerate(param_combinations):
        # Make a deep copy of the original data to ensure each run is isolated
        AllData = copy.deepcopy(AllData_original)

        # Unpack the current parameter combination
        PARAM_U, PARAM_E, PARAM_K, PARAM_K_MIN, PARAM_TERESHOLD, PARAM_K_VALUE = params

        print("\n" + "="*80)
        print(f"Running Experiment {i+1}/{len(param_combinations)}")
        print(f"PARAMETERS: u={PARAM_U}, e={PARAM_E}, k={PARAM_K}, k_min={PARAM_K_MIN}, Tereshold={PARAM_TERESHOLD}, K_Value={PARAM_K_VALUE}")
        print("="*80 + "\n")

        # Define parameter dictionaries for each stage to ensure correct caching
        segmentation_params = {'u': PARAM_U, 'e': PARAM_E}
        bursty_params = {'u': PARAM_U, 'e': PARAM_E}
        similarity_params = {'u': PARAM_U, 'e': PARAM_E, 'step_time_hours': PARAM_STEP_TIME_HOURS}
        clustering_params = {'u': PARAM_U, 'e': PARAM_E, 'step_time_hours': PARAM_STEP_TIME_HOURS, 'k': PARAM_K, 'k_min': PARAM_K_MIN}
        miu_params = {'u': PARAM_U, 'e': PARAM_E, 'step_time_hours': PARAM_STEP_TIME_HOURS, 'k': PARAM_K, 'k_min': PARAM_K_MIN}
        realistic_events_params = {'u': PARAM_U, 'e': PARAM_E, 'step_time_hours': PARAM_STEP_TIME_HOURS, 'k': PARAM_K, 'k_min': PARAM_K_MIN, 'tereshold': PARAM_TERESHOLD}
        final_params = {'u': PARAM_U, 'e': PARAM_E, 'step_time_hours': PARAM_STEP_TIME_HOURS, 'k': PARAM_K, 'k_min': PARAM_K_MIN, 'tereshold': PARAM_TERESHOLD, 'k_value': PARAM_K_VALUE}

        #Segment kardane post ha
        posts_segments_path = os.path.join(Path, get_param_stamped_filename('PostsSegments_Windowing', segmentation_params, 'npy'))
        if os.path.exists(posts_segments_path):
            print(f'Loading PostsSegments_Windowing from file: {posts_segments_path}')
            PostsSegments_Windowing = np.load(posts_segments_path, allow_pickle=True).tolist()
        else:
            print("Segmenting posts...")
            PostsSegments_Windowing = Segmentation(AllData[1], u=PARAM_U, e=PARAM_E, offline_mode=PARAM_FORCE_OFFLINE)
            np.save(posts_segments_path, np.array(PostsSegments_Windowing, dtype=object), allow_pickle=True)
            print(f'\n PostsSegments_Windowing Saved to {posts_segments_path}')

        #Detect Bursty Segment
        event_segment_path = os.path.join(Path, get_param_stamped_filename('EventSegment_Windowing', bursty_params, 'npy'))
        event_weight_path = os.path.join(Path, get_param_stamped_filename('EventSegmentWeight_Windowing', bursty_params, 'npy'))
        if os.path.exists(event_segment_path) and os.path.exists(event_weight_path):
            print(f'Loading EventSegment and weights from files...')
            EventSegment_Windowing = np.load(event_segment_path, allow_pickle=True).tolist()
            EventSegmentWeight_Windowing = np.load(event_weight_path, allow_pickle=True).tolist()
        else:
            print("Detecting bursty segments...")
            EventSegment_Windowing, EventSegmentWeight_Windowing = DetectBursty(AllData, PostsSegments_Windowing)
            np.save(event_segment_path, np.array(EventSegment_Windowing, dtype=object), allow_pickle=True)
            np.save(event_weight_path, np.array(EventSegmentWeight_Windowing, dtype=object), allow_pickle=True)
            print(f'\nSaved EventSegment and weights to {event_segment_path} and {event_weight_path}')

        # This part requires a StartTime variable.
        StartTime = datetime(2017, 1,  1, 00, 00, 00)

        #Similarity Graph
        similarity_graph_path = os.path.join(Path, get_param_stamped_filename('SimilarityGraph', similarity_params, 'npy'))
        if os.path.exists(similarity_graph_path):
            print(f'Loading SimilarityGraph from file: {similarity_graph_path}')
            SimilarityGraph = np.load(similarity_graph_path, allow_pickle=True).tolist()
        else:
            print("Clustering event segments...")
            StepTime = timedelta(hours=PARAM_STEP_TIME_HOURS)
            SimilarityGraph = EventSegmentClustering_Similarity(AllData,EventSegment_Windowing,StartTime,StepTime)
            np.save(similarity_graph_path, np.array(SimilarityGraph, dtype=object), allow_pickle=True)
            print(f'\n SimilarityGraph Saved to {similarity_graph_path}')

        #Clustering
        condidate_events_path = os.path.join(Path, get_param_stamped_filename('CondidateEvents', clustering_params, 'npy'))
        if os.path.exists(condidate_events_path):
            print(f'Loading CondidateEvents from file: {condidate_events_path}')
            CondidateEvents, NoiseS = np.load(condidate_events_path, allow_pickle=True)
        else:
            print("Clustering started...")
            CondidateEvents,NoiseS = EventSegmentClustering(SimilarityGraph,EventSegment_Windowing,PARAM_K,PARAM_K_MIN)
            np.save(condidate_events_path, np.array((CondidateEvents, NoiseS), dtype=object), allow_pickle=True)
            print(f'\n CondidateEvents Saved to {condidate_events_path}')

        #Newsworthiness
        miue_path = os.path.join(Path, get_param_stamped_filename('MiuE', miu_params, 'npy'))
        if os.path.exists(miue_path):
            print(f'Loading MiuE from file: {miue_path}')
            MiuE = np.load(miue_path, allow_pickle=True).tolist()
        else:
            print("Calculating event newsworthiness...")
            MiuE = EventNewsWorthy(CondidateEvents,SimilarityGraph)
            np.save(miue_path, np.array(MiuE, dtype=object), allow_pickle=True)
            print(f'\n MiuE Saved to {miue_path}')

        miux_path = os.path.join(Path, get_param_stamped_filename('MiuX', miu_params, 'npy'))
        if os.path.exists(miux_path):
            print(f'Loading MiuX from file: {miux_path}')
            MiuX = np.load(miux_path, allow_pickle=True).tolist()
        else:
            print("Calculating highest newsworthiness...")
            MiuX = HighestNewsWorthy(MiuE,CondidateEvents)
            np.save(miux_path, np.array(MiuX, dtype=object), allow_pickle=True)
            print(f'\n MiuX Saved to {miux_path}')

        #Realistic Events
        realistic_events_path = os.path.join(Path, get_param_stamped_filename('RealisticEvents', realistic_events_params, 'npy'))
        if os.path.exists(realistic_events_path):
            print(f'Loading RealisticEvents from file: {realistic_events_path}')
            RealisticEvents = np.load(realistic_events_path, allow_pickle=True).tolist()
        else:
            print("Detecting realistic events...")
            RealisticEvents = DetectRealisticEvents(MiuX,MiuE,PARAM_TERESHOLD,CondidateEvents)
            np.save(realistic_events_path, np.array(RealisticEvents, dtype=object), allow_pickle=True)
            print(f'\n RealisticEvents Saved to {realistic_events_path}')

        realistic_events_topk_path = os.path.join(Path, get_param_stamped_filename('RealisticEventsTopK', final_params, 'npy'))
        if os.path.exists(realistic_events_topk_path):
            print(f'Loading RealisticEventsTopK from file: {realistic_events_topk_path}')
            RealisticEventsTopK = np.load(realistic_events_topk_path, allow_pickle=True).tolist()
        else:
            print("Detecting top K realistic events...")
            RealisticEventsTopK = DetectRealisticEventsTopK(MiuX,MiuE,PARAM_K_VALUE,PARAM_TERESHOLD,CondidateEvents)
            np.save(realistic_events_topk_path, np.array(RealisticEventsTopK, dtype=object), allow_pickle=True)
            print(f'\n RealisticEventsTopK Saved to {realistic_events_topk_path}')

        #Final Reports
        title_to_describe_path = os.path.join(Path, get_param_stamped_filename('TitleToDescribeEventsSTR', realistic_events_params, 'npy'))
        if os.path.exists(title_to_describe_path):
            print(f'Loading TitleToDescribeEventsSTR from file: {title_to_describe_path}')
            TitleToDescribeEventsSTR = np.load(title_to_describe_path, allow_pickle=True).tolist()
        else:
            print("Describing events...")
            TitleToDescribeEventsSTR = DescribeEvents(RealisticEvents, offline_mode=PARAM_FORCE_OFFLINE)
            np.save(title_to_describe_path, np.array(TitleToDescribeEventsSTR, dtype=object), allow_pickle=True)
            print(f'\n TitleToDescribeEvents Saved to {title_to_describe_path}')

        related_docs_path = os.path.join(Path, get_param_stamped_filename('RelatedDocuments', realistic_events_params, 'npy'))
        related_seq_path = os.path.join(Path, get_param_stamped_filename('RelatedSequence', realistic_events_params, 'npy'))
        if os.path.exists(related_docs_path) and os.path.exists(related_seq_path):
            print(f"Loading related documents from {related_docs_path}")
            RelatedDocuments = np.load(related_docs_path, allow_pickle=True).tolist()
            RelatedSequence = np.load(related_seq_path, allow_pickle=True).tolist()
        else:
            print("Detecting related documents...")
            RelatedDocuments, RelatedSequence = DetectRelatedDoc(AllData, PostsSegments_Windowing, RealisticEvents)
            np.save(related_docs_path, np.array(RelatedDocuments, dtype=object), allow_pickle=True)
            np.save(related_seq_path, np.array(RelatedSequence, dtype=object), allow_pickle=True)
            print(f"Saved related documents and sequences.")

        related_docs_string_path = os.path.join(Path, get_param_stamped_filename('RelatedDocumentsString', realistic_events_params, 'npy'))
        if os.path.exists(related_docs_string_path):
            print(f'Loading RelatedDocumentsString from file: {related_docs_string_path}')
            RelatedDocumentsString = np.load(related_docs_string_path, allow_pickle=True).tolist()
        else:
            print("Joining segments...")
            RelatedDocumentsString = JoinSegments(RelatedDocuments)
            np.save(related_docs_string_path, np.array(RelatedDocumentsString, dtype=object), allow_pickle=True)
            print(f'\n RelatedDocumentsString Saved to {related_docs_string_path}')

        EventNumberOfEachSequence = DetectEventOfEachPost(RelatedSequence)
        AllSequenceAndRelatedEvents = ReadyToWriteToExcell(AllData,EventNumberOfEachSequence)

        results_excel_path = os.path.join(Path, get_param_stamped_filename('ResultsToCompaire', final_params, 'xls'))
        SaveToExcel(results_excel_path, AllData[0], AllSequenceAndRelatedEvents)

        WindowNum = AllData[6]
        topic_excel_path = os.path.join(Path, get_param_stamped_filename('Topic_Systemresult', final_params, 'xls'))
        SaveSystemResultTopic(RealisticEventsTopK,WindowNum,topic_excel_path)