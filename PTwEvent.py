# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import networkx as nx
import math
import re
import pyodbc
import numpy as np
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import jarvispatrick
import random
from sklearn import metrics
import copy 

import pprint
import matplotlib.pyplot as plt

import clr    # install this library by this command : pip install pythonnet   (if yoy have install clr first remove that by: pip uninstall clr)
#clr.AddReference('{}\\FastTokenizer.dll'.format(os.getcwd()))
clr.AddReference('{}\\FastTokenizer.dll'.format(os.getcwd()))
from FastTokenizer import Api 

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


def Stickiness(segment,flag):
# Flag Can Be pmi or scp 
    if len(segment)==0:
        print('ERROR in Stikiness Calculation: Segment bayad hadde aghal 1 kalame bashad')
        return 0
    else:
        if flag=='pmi':
            #c = PMI(segment)       #use Exact Formulla in Article
            c = PMISimple(segment)  #use Simple Formulla  
        else:
            #c = SCP(segment)        #use Exact Formulla in Article
            c = SCPSimple(segment)   #use Simple Formulla
            
        #c: Chasbandegi bedoone dar nazar gereftane wikipedia

        C = c * math.pow(math.e,Q(segment))  
        #C: Ba dakhil kardane wikipedia
        
        if len(segment) == 1:
            l = 1#1/3
        else:
            l = (len(segment) - 1)/len(segment) 
        # l: Parametre Normal sazi tool (formul 12 dar maghale)
        
        L = l * C
        
        return L ###########################################################################################################################################



def PMISimple(sg1):
#    #Calculate C By PMI Simple Formulla:
#    if len(sg1) == 1:
#        tempp = SimplePr(sg1)
#        c = tempp/(1+tempp)
#    else:
#        m=0
#        for i in range(len(sg1)-1):
#            m += SimplePr(sg1[0:i+1])*SimplePr(sg1[i+1:])
#        Makhraj=m/(len(sg1)-1)
#        Soorat = SimplePr(sg1)
#
#        if (Soorat + Makhraj) == 0 :      
#            c = 1
#        else: 
#            c = Soorat/(Soorat+Makhraj)        
#    return c 

    #Calculate C By PMI Simple Formulla By Count rather than Probability:
    N = 99603396 # this is Sum([Count]) OneGram
    if len(sg1) == 1:
        tempp = NgramCount(sg1)
        c = tempp/(N+tempp)
    else:
        m=0
        for i in range(len(sg1)-1):
            m += NgramCount(sg1[0:i+1])*NgramCount(sg1[i+1:])
        Makhraj=m/(len(sg1)-1)
        Soorat = NgramCount(sg1) * N

        if (Soorat + Makhraj) == 0 :      
            c = 1
        else: 
            c = Soorat/(Soorat+Makhraj)        
    return c       


def SCPSimple(sg2):######### Sade saziye in formool hattman barresi shavad ke ghalat nabashad
#    #Calculate C By SCP Simple Formulla:
#    if len(sg2) == 1:         ##### in ghesmat hatman chek shavad
#        tempp = math.pow(SimplePr(sg2),2)
#        c = (2*tempp)/(tempp+1)
#    else:
#        m=0
#        for i in range(len(sg2)-1):
#            m += SimplePr(sg2[0:i+1])*SimplePr(sg2[i+1:])
#        Makhraj=m/(len(sg2)-1)
#        Soorat = math.pow(Pr(sg2),2)
#
#        if (Soorat + Makhraj) == 0 :   
#            c = 2 ##########################
#        else: 
#            c = (2*Soorat)/(Soorat+Makhraj)        
#    return c
    
    #Calculate C By SCP Simple Formulla By Count rather than Probability:
    N = 99603396 # this is Sum([Count]) OneGram
    if len(sg2) == 1:     ##### in gheesmat hatttman chek shavad
        tempp = NgramCount(sg2)
        c = 2*math.pow(tempp,2)/(math.pow(tempp,2)+math.pow(N,2))
    else:
        m=0
        for i in range(len(sg2)-1):
            m += NgramCount(sg2[0:i+1])*NgramCount(sg2[i+1:])
        Makhraj=m/(len(sg2)-1)
        Soorat = math.pow(NgramCount(sg2),2)

        if (Soorat + Makhraj) == 0 :      
            c = 2 #############################
        else: 
            c = (2*Soorat)/(Soorat+Makhraj)       
    return c       

def PMI(sg1):
    #Calculate C By PMI (Exact formula in Article)
    if len(sg1) == 1:
        tempp = Pr(sg1)
        if tempp == 0:
            PMI = 0
        else:
            PMI = math.log(tempp,math.e)
    else:
        m=0
        for i in range(len(sg1)-1):
            m += Pr(sg1[0:i+1])*Pr(sg1[i+1:])
        M=m/(len(sg1))
        Prr = Pr(sg1)
        if Prr == 0  :
            PMI = -math.inf
        elif M == 0 and Prr != 0:
            PMI = math.inf
        else: 
            PMI = math.log((Prr/M),math.e)
     
    c = (1/(1+math.pow(math.e,-PMI)))  
    return c               

def SCP(sg2):
    if len(sg2) == 1:
        tempp = Pr(sg2)
        if tempp == 0:
            SCP = 0
        else:
            SCP = 2*math.log(tempp,math.e)
    else:
        m=0
        for i in range(len(sg2)-1):
            m += Pr(sg2[0:i+1])*Pr(sg2[i+1:])
        M=m/(len(sg2)-1)
        Prr = Pr(sg2)

        if M == 0 or Prr == 0:
            SCP = 0
        else: 
            SCP = math.log((math.pow(Prr,2)/M),math.e)
    
    c = 2/(1+math.pow(math.e,-SCP))
    return c
    

def NgramCount(sg3):
    # Count Segment in Ngram
    #
    global NgramBuffer

    global nof
    
    Res = NgramBuffer.ValueInBuffer(sg3)
    
    if Res == -1:
    
        query = CreateQueryForExtractNGram_Count(sg3)

        if query == 'EmptySegment':
            Res = 0
        elif query == 'Above5Segment':
            Res = 0
        else:
            global c
        
            try:
                c.execute(query)
                NGRAM = c.fetchall()[0][0]
                nof += 1
            except:
                print('One Excepo Accureeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                NGRAM = 0
            
            if NGRAM == None:
                NGRAM = 0
            Res = NGRAM
        NgramBuffer.AddToBuffer(sg3,Res)
    
    return Res


def SimplePr(sg3):
    # Ehtemale vogho dar Web N-Gram (N-Gram Probability hast yani ehtemalate sharti hast)
    #
    global NgramBuffer

    global nof
    
    Res = NgramBuffer.ValueInBuffer(sg3)
    
    if Res == -1:
    
        query = CreateQueryForExtractNGram_SimplePr(sg3)

        if query == 'EmptySegment':
            Res = 0
        elif query == 'Above5Segment':
            Res = 0
        else:
            global c
        
            try:
                c.execute(query)
                NGRAM = c.fetchall()[0][0]
                nof += 1
            except:
                print('One Excepo Accureeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                NGRAM = 0
            
            if NGRAM == None:
                NGRAM = 0
            Res = NGRAM
        NgramBuffer.AddToBuffer(sg3,Res)
    
    return Res


def Pr(sg3):
    # Ehtemale vogho dar Web N-Gram (N-Gram Probability hast yani ehtemalate sharti hast)
    #
    global NgramBuffer

    global nof
    
    Res = NgramBuffer.ValueInBuffer(sg3)
    
    if Res == -1:
    
        query = CreateQueryForExtractNGram(sg3)

        if query == 'EmptySegment':
            Res = 0
        elif query == 'Above5Segment':
            Res = 0
        else:
            global c
        
            try:
                c.execute(query)
                NGRAM = c.fetchall()[0][0]
                nof += 1
            except:
                print('One Excepo Accureeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
                NGRAM = 0
            
            if NGRAM == None:
                NGRAM = 0
            Res = NGRAM
        NgramBuffer.AddToBuffer(sg3,Res)
    
    return Res

def CreateQueryForExtractNGram(SEG):
    if type(SEG) == str:
        SEG = FastTokenize(SEG)
    Ngram = len(SEG)
    
    #for i in range(Ngram):
        #SEG[i] = unicode(SEG[i], "utf-8")    ######################################################################################################
    
    if Ngram == 0:
        #reshte tohi amade va hich kalameii nayamade
        #
        print('!!!!!!!!!!!!!!!!!!!!Cant Calculate Pr of empty segment( Pr[W] where W = [] )')
        query = 'EmptySegment'
        #
        #
    elif Ngram == 1:
        #1 kalame hast va 1-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract1Gram N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0])
        #
        #
    elif Ngram == 2:
        #2 kalame hast va 2-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract2Gram N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1])
        #
        #
    elif Ngram ==3:
        #3 kalame hast va 3-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract3Gram N\'{}\', N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1], SEG[2])
        #
        #
    elif Ngram ==4:
        #4 kalame hast va 4-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract4Gram N\'{}\', N\'{}\', N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1], SEG[2], SEG[3])
        #
        #
    elif Ngram ==5:
        #5 kalame hast va 5-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract5Gram N\'{}\', N\'{}\', N\'{}\', N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1], SEG[2], SEG[3], SEG[4])
        #
        #
    else:
        #Bishtar az 6 kalame amade va nabayad in halat ettefagh bioftad
        #
        print('!!!!!!!!!!!!!!!!!!!!Cant Calculate Pr of a Segment with More Than 5 Word')
        query = 'Above5Segment'
        #
        #
    return query
def CreateQueryForExtractNGram_SimplePr(SEG):
    if type(SEG) == str:
        SEG = FastTokenize(SEG)
    Ngram = len(SEG)
    
    #for i in range(Ngram):
        #SEG[i] = unicode(SEG[i], "utf-8")    ######################################################################################################
    
    if Ngram == 0:
        #reshte tohi amade va hich kalameii nayamade
        #
        print('!!!!!!!!!!!!!!!!!!!!Cant Calculate Pr of empty segment( Pr[W] where W = [] )')
        query = 'EmptySegment'
        #
        #
    elif Ngram == 1:
        #1 kalame hast va 1-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract1GramSimpleProbability N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0])
        #
        #
    elif Ngram == 2:
        #2 kalame hast va 2-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract2GramSimpleProbability N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1])
        #
        #
    elif Ngram ==3:
        #3 kalame hast va 3-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract3GramSimpleProbability N\'{}\', N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1], SEG[2])
        #
        #
    elif Ngram ==4:
        #4 kalame hast va 4-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract4GramSimpleProbability N\'{}\', N\'{}\', N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1], SEG[2], SEG[3])
        #
        #
    elif Ngram ==5:
        #5 kalame hast va 5-Gram bayad hesab shavad
        #
        query = """declare @ngr float;
                    exec Extract5GramSimpleProbability N\'{}\', N\'{}\', N\'{}\', N\'{}\', N\'{}\', @ngram=@ngr output;
                    SELECT @ngr as NGR """.format(SEG[0], SEG[1], SEG[2], SEG[3], SEG[4])
        #
        #
    else:
        #Bishtar az 6 kalame amade va nabayad in halat ettefagh bioftad
        #
        print('!!!!!!!!!!!!!!!!!!!!Cant Calculate Pr of a Segment with More Than 5 Word')
        query = 'Above5Segment'
        #
        #
    return query
def CreateQueryForExtractNGram_Count(SEG):
    if type(SEG) == str:
        SEG = FastTokenize(SEG)
    Ngram = len(SEG)
    
    #for i in range(Ngram):
        #SEG[i] = unicode(SEG[i], "utf-8")    ######################################################################################################
    
    if Ngram == 0:
        #reshte tohi amade va hich kalameii nayamade
        #
        print('!!!!!!!!!!!!!!!!!!!!Cant Calculate Pr of empty segment( Pr[W] where W = [] )')
        query = 'EmptySegment'
        #
        #
    elif Ngram == 1:
        #1 kalame hast va 1-Gram bayad hesab shavad
        #
        query = """declare @cnt float;
                    exec Extract1GramCount N\'{}\', @ngramcount=@cnt output;
                    SELECT @cnt as CNT """.format(SEG[0])
        #
        #
    elif Ngram == 2:
        #2 kalame hast va 2-Gram bayad hesab shavad
        #
        query = """declare @cnt float;
                    exec Extract2GramCount N\'{}\', N\'{}\', @ngramcount=@cnt output;
                    SELECT @cnt as CNT """.format(SEG[0], SEG[1])
        #
        #
    elif Ngram ==3:
        #3 kalame hast va 3-Gram bayad hesab shavad
        #
        query = """declare @cnt float;
                    exec Extract3GramCount N\'{}\', N\'{}\', N\'{}\', @ngramcount=@cnt output;
                    SELECT @cnt as CNT """.format(SEG[0], SEG[1], SEG[2])
        #
        #
    elif Ngram ==4:
        #4 kalame hast va 4-Gram bayad hesab shavad
        #
        query = """declare @cnt float;
                    exec Extract4GramCount N\'{}\', N\'{}\', N\'{}\', N\'{}\', @ngramcount=@cnt output;
                    SELECT @cnt as CNT """.format(SEG[0], SEG[1], SEG[2], SEG[3])
        #
        #
    elif Ngram ==5:
        #5 kalame hast va 5-Gram bayad hesab shavad
        #
        query = """declare @cnt float;
                    exec Extract5GramCount N\'{}\', N\'{}\', N\'{}\', N\'{}\', N\'{}\', @ngramcount=@cnt output;
                    SELECT @cnt as CNT """.format(SEG[0], SEG[1], SEG[2], SEG[3], SEG[4])
        #
        #
    else:
        #Bishtar az 6 kalame amade va nabayad in halat ettefagh bioftad
        #
        print('!!!!!!!!!!!!!!!!!!!!Cant Calculate Pr of a Segment with More Than 5 Word')
        query = 'Above5Segment'
        #
        #
    return query
        
def Q_Title(sg4):
    # Ehtemale vogho dar Wikipedia titles
    
    global nof
    
    global WikiPediaBuffer

    if type(sg4) == str:
        sg4 = FastTokenize(sg4)
    

    Res = WikiPediaBuffer.ValueInBuffer(sg4)
    if Res == -1:
        segmentstringforQuery = ''
        for i in range(10):
            if(i<len(sg4)):
                segmentstringforQuery = '{} N\'{}\','.format(segmentstringforQuery, sg4[i])
            else:
                segmentstringforQuery = '{} N\'\','.format(segmentstringforQuery)
    
    
        query = """declare @output float;
                    exec ExtractNameEntityContaining {} @output output;
                    SELECT @output as NER""".format(segmentstringforQuery)
    
        global c

        try:
            c.execute(query)
            NameEntityP = c.fetchall()[0][0]
            nof += 1
        except: # Zamani ke Query Shamele Emoji ya .... bashad error khahad dad
            print('One Excepo Accureeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            NameEntityP = 0
        if NameEntityP == None:
            NameEntityP = 0
        Res = NameEntityP
        
        WikiPediaBuffer.AddToBuffer(sg4,Res)
        
    return Res
    
def Q_Anchor(sg4):
    # Ehtemale vogho dar Wikipedia AncherDF
    
    global nof
    
    global WikiPediaBuffer

    if type(sg4) == str:
        sg4 = FastTokenize(sg4)
    

    Res = WikiPediaBuffer.ValueInBuffer(sg4)
    if Res == -1:
        segmentstringforQuery = ''
        for i in range(10):
            if(i<len(sg4)):
                segmentstringforQuery = '{} N\'{}\','.format(segmentstringforQuery, sg4[i])
            else:
                segmentstringforQuery = '{} N\'\','.format(segmentstringforQuery)
    
    
        query = """declare @output float;
                    exec ExtractAnchorDFRatioContaining {} @output output;
                    SELECT @output as AnchorDF_Ratio""".format(segmentstringforQuery)
    
        global c

        try:
            c.execute(query)
            AncgerDFRatio = c.fetchall()[0][0]
            nof += 1
        except: # Zamani ke Query Shamele Emoji ya .... bashad error khahad dad
            print('One Excepo Accureeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            AncgerDFRatio = 0
        if AncgerDFRatio == None:
            AncgerDFRatio = 0
        Res = AncgerDFRatio
        
        WikiPediaBuffer.AddToBuffer(sg4,Res)
        
    return Res

def Q(sg4):
    # Ehtemale vogho dar Wikipedia AncherDF
    
    global nof
    
    global WikiPediaBuffer

    if type(sg4) == str:
        sg4 = FastTokenize(sg4)
    

    Res = WikiPediaBuffer.ValueInBuffer(sg4)
    if Res == -1:
        segmentstringforQuery = ''
        for i in range(10):
            if(i<len(sg4)):
                segmentstringforQuery = '{} N\'{}\','.format(segmentstringforQuery, sg4[i])
            else:
                segmentstringforQuery = '{} N\'\','.format(segmentstringforQuery)
    
    
        query = """declare @output float;
                    exec ExtractAnchorDFRatioContaining {} @output output;
                    SELECT @output as AnchorDF_Ratio""".format(segmentstringforQuery)
    
        global c

        try:
            c.execute(query)
            AncgerDFRatio = c.fetchall()[0][0]
            nof += 1
        except: # Zamani ke Query Shamele Emoji ya .... bashad error khahad dad
            print('One Excepo Accureeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            AncgerDFRatio = 0
        if AncgerDFRatio == None:
            AncgerDFRatio = 0
        Res = AncgerDFRatio
        
        WikiPediaBuffer.AddToBuffer(sg4,Res)
        
    return Res



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
        vect = TfidfVectorizer(min_df=1)                                                                                                                                                                                                   
        tfidf = vect.fit_transform([T1,T2])                                                                                                                                                                                                                       
        pairwise_similarity = tfidf * tfidf.T 
        return pairwise_similarity[0,1]

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
    
    print('current window size: {}'.format(len(CurrentWindow)))
    G = nx.MultiDiGraph()
    print('Adding Nodes To Graph ...')
    NodeList = []
    for Tweet in CurrentWindow:
        for Segment in Tweet:
            NodeList.append(" ".join(Segment))
    G.add_nodes_from(NodeList)
    
    print('Preparing the Edges ...')
    edges_wts = PrepareSimilarityEdge(NodeList,CurrentSubWindowing)
    
    print('Adding Edges To Grapg ...')
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, key='w', weight=v, label=v)
   
    
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

def ClusteringAlg(Graph):
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
    
    K=round(len(Graph.nodes)/5)#15 # “K”, the number of nearest neighbors to consider OR  'number_of_neighbors'
    Step1 = max(1,round(K/50))
    K_MIN=round(len(Graph.nodes)/10)#6 #  K_min, the number of minimum shared neighbors OR 'threshold_number_of_common_neighbors'
    Step2 = max(1,round(K_MIN/20))
    print('Total Node is:{} , K and K_MIN is: {}&{}'.format(len(Graph.nodes),K,K_MIN))
    CLUSTERS = cluster_gen(K, K_MIN) # initialize clusters
    
    
    Silhouette_Coefficient = -1
    for k in range(1,K,Step1):
        for k_min in range(1,min(k+1,K_MIN),Step2):
            
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
    
    #ClusterS = [clusters[k] for k in clusters.keys()]
    return clusters




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

def EventSegmentClustering_Similarity(EventSegment):
    
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
    for WinNum in range(len(EventSegment)):
        SimilarityGraph.append([])
        
        CurrentWindow = EventSegment[WinNum]
        CurrentSubWindowing = SubWindowPosts[WinNum]
        
        SimilarityGraph[-1] = CreateSimilarityGraph(CurrentWindow,CurrentSubWindowing)
    
    return SimilarityGraph

def EventSegmentClustering(SimilarityGraph,EventSegment):
    
    EventSegment_Clusters = []
    for WinNum in range(len(EventSegment)):
        #EventSegment_Clusters.append([])
        
        EventSegment_Clusters.append(ClusteringAlg(SimilarityGraph[WinNum]))
    
    return EventSegment_Clusters

def MiuS(EventSegment):
    MiuS=0
    EventSegment=EventSegment.split(" ")
    for Segment in EventSegment:
        TempMiuS = math.pow(math.e,Q(Segment))
        MiuS = max(MiuS,TempMiuS)
    return MiuS-1
    

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
            
            MakhrajKasrHa = len(CondidateEvents[WinNum][ClusterNum])
            if(MakhrajKasrHa==0):
                print('TASMIMGIRI SHAVAD') ##################################################################################################################
                MiuE[-1].append(0) ##########################################################################################################################
            else:
                MiuE[-1].append((SooratKasrAvval/MakhrajKasrHa)*(SooratKasrDovvom/MakhrajKasrHa))
            
    return MiuE  

def EventNewsWorthy(CondidateEvents,SimilarityGraph):
    MiuE = []
    for WinNum,Window in enumerate(CondidateEvents):
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
            
            MakhrajKasrHa = len(CondidateEvents[WinNum][ClusterNum])
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


def Top5Rank(Cluster):
    
    AllWordRank = []
    AllWord = []
    for i,Segment in enumerate(Cluster):
        #print('Segment:{}/{}'.format(i,len(Cluster)))
        CurrentRank = math.pow(math.e,Q(Segment))
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

def DetectRelated(AllDocumentInWindow,Segment):
    Segment = Segment.split(' ')
    IndexToDelete = []
    RelatedDoc = []
    for i,post in enumerate(AllDocumentInWindow):
        if Segment in post:
            RelatedDoc.append(post)
            IndexToDelete.append(i)
            
    for i in reversed(IndexToDelete):
        del AllDocumentInWindow[i]
    
    return AllDocumentInWindow,RelatedDoc

def DetectRelatedDoc(PostsSegments_Windowing,RealisticEvents):
    
    RelatedDocuments = []

    for WinNum in range(len(RealisticEvents)):
        RelatedDocuments.append([])
        for EventClusterNum in range(len(RealisticEvents[WinNum])):
            AllDocumentInWindow = copy.deepcopy(PostsSegments_Windowing[WinNum])
            CurentRelatedDocuments = []
            for Segment in RealisticEvents[WinNum][EventClusterNum]:
                AllDocumentInWindow,RelatedDoc = DetectRelated(AllDocumentInWindow,Segment)
                CurentRelatedDocuments = [*CurentRelatedDocuments,*RelatedDoc]
            RelatedDocuments[-1].append(CurentRelatedDocuments)
    
    return RelatedDocuments








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











##=============================================================================
##============================TwiEvent Main Body===============================
##=============================================================================


print("Start Program")

Path = r'C:\Users\Pejman\Desktop\PhD\Python Programming\PtweventOKOK'

conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                              'Server=(local);'
                              'Database=NGram;'
                              'Trusted_Connection=yes;'
                         )      
global c
c = conn.cursor()







'''
#khandane Ettelaat PostHaye Telegram Az SQL
StartTime = datetime(2017, 1,  1, 00, 00, 00)
EndTime   = datetime(2017, 1, 31, 23, 59, 59)   ############datetime(2017, 1, 31, 23, 59, 59)################ ta 31 om bayad bashad
AllData = ReadData(StartTime,EndTime)


AllData = WindowingLikeTwiner(AllData)


np.save(Path+r'\AllData.npy',AllData)
print('\n AllData Saved')
#OR
'''
tempNumpyArray=np.load(Path+r'\AllData.npy',allow_pickle=True)
AllData = tempNumpyArray.tolist()


#AllData Contain :
#        Seq_Windowing,
#        Posts_Windowing,
#        Types_Windowing,
#        Deleted_Windowing,
#        DateSend_Windowing,
#        User_Windowing,
#        WindowNum










'''
#Segment kardane post ha (Az SCP Estefade shode)
PostsSegments_Windowing = Segmentation(AllData[1])
#AllData[1] Means the Posts in TimeWindowing

np.save(Path+r'\PostsSegments_Windowing.npy',PostsSegments_Windowing)
print('\n PostsSegments_Windowing Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\PostsSegments_Windowing.npy',allow_pickle=True)
PostsSegments_Windowing = tempNumpyArray.tolist()






AllData[0] = AllData[0][0:30]
AllData[1] = AllData[1][0:30]
AllData[2] = AllData[2][0:30]
AllData[3] = AllData[3][0:30]
AllData[4] = AllData[4][0:30]
AllData[5] = AllData[5][0:30]
AllData[6] = AllData[6][0:30]

PostsSegments_Windowing = PostsSegments_Windowing[0:30]





'''
#Detect Bursty Segment
EventSegment_Windowing,EventSegmentWeight_Windowing = DetectBursty(AllData,PostsSegments_Windowing)
#EventSegment is TweetsBurstySegments

np.save(Path+r'\EventSegment_Windowing.npy',EventSegment_Windowing)
print('\n EventSegment_Windowing Saved')
np.save(Path+r'\EventSegmentWeight_Windowing.npy',EventSegmentWeight_Windowing)
print('\n EventSegmentWeight_Windowing Saved')
#OR
'''
tempNumpyArray=np.load(Path+r'\EventSegment_Windowing.npy',allow_pickle=True)
EventSegment_Windowing = tempNumpyArray.tolist()
tempNumpyArray=np.load(Path+r'\EventSegmentWeight_Windowing.npy',allow_pickle=True)
EventSegmentWeight_Windowing = tempNumpyArray.tolist()







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









'''
SimilarityGraph = EventSegmentClustering_Similarity(EventSegment_Windowing)
# Condedate Evente is clysters of segment in each time window

np.save(Path+r'\SimilarityGraph.npy',SimilarityGraph)
print('\n SimilarityGraph Saved')
'''
#OR
tempNumpyArray=np.load(Path+r'\SimilarityGraph.npy',allow_pickle=True)
SimilarityGraph = tempNumpyArray.tolist()










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




















'''
CondidateEvents = EventSegmentClustering(SimilarityGraph,EventSegment_Windowing)
np.save(Path+r'\CondidateEvents.npy',CondidateEvents)
print('\n CondidateEvents Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\CondidateEvents.npy',allow_pickle=True)
CondidateEvents = tempNumpyArray.tolist()








'''
MiuE = EventNewsWorthy(CondidateEvents,SimilarityGraph)

np.save(Path+r'\MiuE.npy',MiuE)
print('\n MiuE Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\MiuE.npy',allow_pickle=True)
MiuE = tempNumpyArray.tolist()



#################################################################################################################
#MiuE_2LastWindow = EventNewsWorthy_NLastWindow(CondidateEvents,SimilarityGraph,13)
#
#MiuE = [*MiuE ,*MiuE_2LastWindow ]
#
#np.save(Path+r'\MiuE.npy',MiuE)
#print('\n MiuE Saved')
#################################################################################################################














'''
MiuX = HighestNewsWorthy(MiuE,CondidateEvents)

np.save(Path+r'\MiuX.npy',MiuX)
print('\n MiuX Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\MiuX.npy',allow_pickle=True)
MiuX = tempNumpyArray.tolist()




##################################################################################################################
#MiuX_2LastWindow = HighestNewsWorthy_NLastWindow(MiuE,CondidateEvents,13)
#
#MiuX = [*MiuX ,*MiuX_2LastWindow ]
#
#np.save(Path+r'\MiuX.npy',MiuX)
#print('\n MiuX Saved')
##################################################################################################################
#









'''
Tereshold = 40##################################################################################.5
RealisticEvents = DetectRealisticEvents(MiuX,MiuE,Tereshold,CondidateEvents)

np.save(Path+r'\RealisticEvents.npy',RealisticEvents)
print('\n RealisticEvents Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\RealisticEvents.npy',allow_pickle=True)
RealisticEvents = tempNumpyArray.tolist()




#################################################################################################################
#Tereshold = 500##################################################################################.5
#RealisticEvents_2LastWindow = DetectRealisticEvents_2LastWindow(MiuX,MiuE,Tereshold,CondidateEvents)
#
#RealisticEvents = [*RealisticEvents ,*RealisticEvents_2LastWindow ]
#
#np.save(Path+r'\RealisticEvents.npy',RealisticEvents)
#print('\n RealisticEvents Saved')
#################################################################################################################



















'''
TitleToDescribeEventsSTR = DescribeEvents(RealisticEvents)

np.save(Path+r'\TitleToDescribeEventsSTR.npy',TitleToDescribeEventsSTR)
print('\n TitleToDescribeEvents Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\TitleToDescribeEventsSTR.npy',allow_pickle=True)
TitleToDescribeEventsSTR = tempNumpyArray.tolist()



#################################################################################################################
#TitleToDescribeEventsSTR_2LastWindow = DescribeEvents_2LastWindow(RealisticEvents)
#
#TitleToDescribeEventsSTR = [*TitleToDescribeEventsSTR ,*TitleToDescribeEventsSTR_2LastWindow ]
#
#np.save(Path+r'\TitleToDescribeEventsSTR.npy',TitleToDescribeEventsSTR)
#print('\n TitleToDescribeEvents Saved')
#################################################################################################################
#



















'''
RelatedDocuments = DetectRelatedDoc(PostsSegments_Windowing,RealisticEvents)
np.save(Path+r'\RelatedDocuments.npy',RelatedDocuments)
print('\n RelatedDocuments Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\RelatedDocuments.npy',allow_pickle=True)
RelatedDocuments = tempNumpyArray.tolist()





'''
RelatedDocumentsString = JoinSegments(RelatedDocuments)
np.save(Path+r'\RelatedDocumentsString.npy',RelatedDocumentsString)
print('\n RelatedDocumentsString Saved')

#OR
'''
tempNumpyArray=np.load(Path+r'\RelatedDocumentsString.npy',allow_pickle=True)
RelatedDocumentsString = tempNumpyArray.tolist()




