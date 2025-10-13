import Main1Fast as EventDetection
import EvaluateFungtional as Eval
import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import pickle
import copy
from xlrd import open_workbook
import xlwt
from xlutils.copy import copy
import os

def AddTitlesToExcellsheet(Sheet,i,j,EvalName,A,B):
    Sheet.write(i, j, EvalName)
    Sheet.write(i, j+1, "Distance to Merge")
    Sheet.write(i+1, j, "Forget Tereshold")
    Sheet.write(i+1, j+1, "###")
    
    for ii,a in enumerate(A):
        Sheet.write(i+1, j+ii+2, a)
    #Sheet.write(i+1, j+2, "0.5")
    #Sheet.write(i+1, j+3, "1")
    #Sheet.write(i+1, j+4, "1.5")
    #Sheet.write(i+1, j+5, "2")
    
    for jj,b in enumerate(B):
        Sheet.write(i+jj+2, j+1, b)
    #Sheet.write(i+2, j+1, "0.8")
    #Sheet.write(i+3, j+1, "0.9")
    #Sheet.write(i+4, j+1, "1")
    #Sheet.write(i+5, j+1, "1.1")
    #Sheet.write(i+6, j+1, "1.2")
    
def AddDataToExcell(a_idx,b_idx,A_len,B_len,ClusterEntropy_Community,ClassEntropy_Community,ClusterEntropy_Clustering,ClassEntropy_Clustering,TotalEntropy_Community,TotalEntropy_Clustering,TopicPrecision_Com,TopicRecall_Com,TopicF1_Com, KeywordPrecision_Com,KeywordRecall_Com,KeywordF1_Com,TopicPrecision_Clu,TopicRecall_Clu,TopicF1_Clu, KeywordPrecision_Clu,KeywordRecall_Clu,KeywordF1_Clu):
    sheet1.write(0*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, TotalEntropy_Community)
    sheet1.write(0*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, TotalEntropy_Clustering)
    sheet1.write(1*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, TopicF1_Com)
    sheet1.write(1*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, TopicF1_Clu)
    sheet1.write(2*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, KeywordF1_Com)
    sheet1.write(2*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, KeywordF1_Clu)
    
    sheet2.write(0*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, ClassEntropy_Community)
    sheet2.write(0*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, ClassEntropy_Clustering)
    sheet2.write(1*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, ClusterEntropy_Community)
    sheet2.write(1*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, ClusterEntropy_Clustering)
    sheet2.write(2*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, TotalEntropy_Community)
    sheet2.write(2*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, TotalEntropy_Clustering)
    
    sheet3.write(0*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, TopicPrecision_Com)
    sheet3.write(0*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, TopicPrecision_Clu)
    sheet3.write(1*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, TopicRecall_Com)
    sheet3.write(1*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, TopicRecall_Clu)
    sheet3.write(2*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, TopicF1_Com)
    sheet3.write(2*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, TopicF1_Clu)
    
    sheet4.write(0*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, KeywordPrecision_Com)
    sheet4.write(0*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, KeywordPrecision_Clu)
    sheet4.write(1*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, KeywordRecall_Com)
    sheet4.write(1*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, KeywordRecall_Clu)
    sheet4.write(2*(B_len+4)+2+b_idx,0*(A_len+4)+2+a_idx, KeywordF1_Com)
    sheet4.write(2*(B_len+4)+2+b_idx,1*(A_len+4)+2+a_idx, KeywordF1_Clu)




StartTime = datetime(2017, 1,  1, 00, 00, 00)
EndTime   = datetime(2017, 1, 31, 23, 59, 59)   ############datetime(2017, 1, 31, 23, 59, 59)################ ta 31 om bayad bashad

XlsFile = 'DataSet/GoldenStandard_TopicID_and_TopicString.xlsx'
GoldenStandard = pd.read_excel(XlsFile)

A=[0.001,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,1,1.5,2,2.5,3]
B=[0.01,0.8,0.9,1,1.1,1.2,1.3,1.5,2,2.5,3]
A_len = len(A)
B_len = len(B)


print('Start Running And Evaluating For parameters A:'+str(A)+' and B:'+str(B))

EvalResFolder = 'EvaluationResults_numpy'
FinalEvalFileXlsfilePath = EvalResFolder + '/FinalEvaluationRes-A_'+str(A)+'-B_'+str(B)+'.xls'

if os.path.exists(FinalEvalFileXlsfilePath):
    FinalResFileExist = True
    rb = open_workbook(FinalEvalFileXlsfilePath)
    book = copy(rb)
else:
    FinalResFileExist = False
    
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("CompairAllEvaluation")
    AddTitlesToExcellsheet(sheet1,0*(B_len+4),0*(A_len+4),'Entropy - Community',A,B)
    AddTitlesToExcellsheet(sheet1,0*(B_len+4),1*(A_len+4),'Entropy - Clustering',A,B)
    AddTitlesToExcellsheet(sheet1,1*(B_len+4),0*(A_len+4),'TopicF1 - Community',A,B)
    AddTitlesToExcellsheet(sheet1,1*(B_len+4),1*(A_len+4),'TopicF1 - Clustering',A,B)
    AddTitlesToExcellsheet(sheet1,2*(B_len+4),0*(A_len+4),'KeywordF1 - Community',A,B)
    AddTitlesToExcellsheet(sheet1,2*(B_len+4),1*(A_len+4),'KeywordF1 - Clustering',A,B)

    sheet2 = book.add_sheet("Entropy")
    AddTitlesToExcellsheet(sheet2,0*(B_len+4),0*(A_len+4),'ClassEntropy - Community',A,B)
    AddTitlesToExcellsheet(sheet2,0*(B_len+4),1*(A_len+4),'ClassEntropy - Clustering',A,B)
    AddTitlesToExcellsheet(sheet2,1*(B_len+4),0*(A_len+4),'ClusterEntropy - Community',A,B)
    AddTitlesToExcellsheet(sheet2,1*(B_len+4),1*(A_len+4),'ClusterEntropy - Clustering',A,B)
    AddTitlesToExcellsheet(sheet2,2*(B_len+4),0*(A_len+4),'TotalEntropy - Community',A,B)
    AddTitlesToExcellsheet(sheet2,2*(B_len+4),1*(A_len+4),'TotalEntropy - Clustering',A,B)

    sheet3 = book.add_sheet("Topic Evaluation")
    AddTitlesToExcellsheet(sheet3,0*(B_len+4),0*(A_len+4),'Topic Precision - Community',A,B)
    AddTitlesToExcellsheet(sheet3,0*(B_len+4),1*(A_len+4),'Topic Precision - Clustering',A,B)
    AddTitlesToExcellsheet(sheet3,1*(B_len+4),0*(A_len+4),'Topic Recall - Community',A,B)
    AddTitlesToExcellsheet(sheet3,1*(B_len+4),1*(A_len+4),'Topic Recall - Clustering',A,B)
    AddTitlesToExcellsheet(sheet3,2*(B_len+4),0*(A_len+4),'Topic F1 - Community',A,B)
    AddTitlesToExcellsheet(sheet3,2*(B_len+4),1*(A_len+4),'Topic F1 - Clustering',A,B)

    sheet4 = book.add_sheet("Keyword Evaluation")
    AddTitlesToExcellsheet(sheet4,0*(B_len+4),0*(A_len+4),'Keyword Precision - Community',A,B)
    AddTitlesToExcellsheet(sheet4,0*(B_len+4),1*(A_len+4),'Keyword Precision - Clustering',A,B)
    AddTitlesToExcellsheet(sheet4,1*(B_len+4),0*(A_len+4),'Keyword Recall - Community',A,B)
    AddTitlesToExcellsheet(sheet4,1*(B_len+4),1*(A_len+4),'Keyword Recall - Clustering',A,B)
    AddTitlesToExcellsheet(sheet4,2*(B_len+4),0*(A_len+4),'Keyword F1 - Community',A,B)
    AddTitlesToExcellsheet(sheet4,2*(B_len+4),1*(A_len+4),'Keyword F1 - Clustering',A,B)
    
    
for ai,a in enumerate(A):
    for bi,b in enumerate(B):
        start_time_parameter = datetime.now()
        VariablePath = EvalResFolder + '/DistanceTereshold_' + str(a) + '/ForgetigTreshold_' + str(b)
        if FinalResFileExist:
            print('In Halat Hatman Trace Shavad Ehtemale Eshtebah Hast')
            sheet1 = rb.sheet_by_index(0)
            row_b = 2+bi
            col_a = 2+ai
            row = sheet1.row_values(row_b)
            data4param = row[col_a] # dar asl dadeii ke dar shite avval bara entropy community sabt shode ra estekhraj kardim ke agar meghdar dashte bashad yani baraye in parametr ha hesab kardimesh ghablan
            if data4param == []:
                print('Final Excell file Exist But Cell of Parameter_'+ str(a)+' - ' + str(b) + 'is empty and should be calculate')
            else:
                print('Final Excell file Exist And Cell of Parameter_'+ str(a)+' - ' + str(b) + 'Have Value, then go to next parameter')
                continue
                
        if not os.path.exists(VariablePath):
            print('Model by variable ' + str(a) + ' - ' + str(b) + 'Not Run Yet.')
            #SystemResults = EventDetection.main(a, b, 2*60*60, 1, 1, 60, 0.032, 0.05)
        else:
            print('Model by variable ' + str(a) + ' - ' + str(b) + 'Has Runned Before. so loading Result of that')
            SystemResults = np.load(VariablePath + '/Output' + str(StartTime.date()) +' to '+ str(EndTime.date())  + '.npy', allow_pickle=True)
            print('LoadED Results File')
            print('TimeTaken for parameter A_'+str(a)+' B_'+str(b) + ' is: '+str(datetime.now()-start_time_parameter))
            start_time_parameter = datetime.now()
            print('Start Evaluating...')
            EVAL_RES, DAta = Eval.EvalAndSaveRes(GoldenStandard,SystemResults)
        
            DAta.to_excel(VariablePath + '/DAta4SeeFinalResult-' + str(a) + '-' + str(b) + '.xlsx')
        
            print('TimeTaken for Evaluating is: '+str(datetime.now()-start_time_parameter))  
        
            print('__________Eval Res_____________')
            print('Antropy       : '+str(EVAL_RES[4]))
            print('Topic Recall  : '+str(EVAL_RES[7]))
            print('K.W. Precision: '+str(EVAL_RES[9]))
            print('K.W. Recall   : '+str(EVAL_RES[10]))
            print('K.W F-1       : '+str(EVAL_RES[11]))
            print('__________Eval Res_____________')
        
        
            print('Evaluate Done. ')
        
            #AddDataToExcell(ai,bi, A_len, B_len,ClusterEntropy_Community,ClassEntropy_Community,ClusterEntropy_Clustering,ClassEntropy_Clustering,TotalEntropy_Community,TotalEntropy_Clustering,TopicPrecision_Com,TopicRecall_Com,TopicF1_Com, KeywordPrecision_Com,KeywordRecall_Com,KeywordF1_Com,TopicPrecision_Clu,TopicRecall_Clu,TopicF1_Clu,  KeywordPrecision_Clu,KeywordRecall_Clu,KeywordF1_Clu)
            AddDataToExcell(ai, bi, A_len, B_len,    EVAL_RES[0],              EVAL_RES[1],                EVAL_RES[2],            EVAL_RES[3],           EVAL_RES[4],          EVAL_RES[5],          EVAL_RES[6],     EVAL_RES[7],  EVAL_RES[8],      EVAL_RES[9],     EVAL_RES[10],    EVAL_RES[11],    EVAL_RES[12],    EVAL_RES[13],  EVAL_RES[14],    EVAL_RES[15],        EVAL_RES[16],   EVAL_RES[17])
      
book.save(FinalEvalFileXlsfilePath)
print('Final Result Excell File SavED')     