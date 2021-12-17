# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 23:08:05 2021

@author: ASUS
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from ts2vg import NaturalVG,HorizontalVG
import math
import glob
import os
from pyentrp import entropy as ent2

path="D:\\Project\\"

Method=[]
Subject=[]
Axis=[]
Avg_deg=[]
Nt_dia=[]
Signal_Length=[]
Avg_path=[]
Activity=[]
Entropy=[]
avg_d={}
net_d={}
Dimension=[]
Delay=[]
avg_d['Walking']=[]
avg_d['Running']=[]
avg_d['Climbing Up']=[]
avg_d['Climbing Down']=[]
net_d['Walking']=[]
net_d['Running']=[]
net_d['Climbing Up']=[]
net_d['Climbing Down']=[]


def plotter_VG(X,type=0,label='X',typeOf='Walking',subject='Subject 1',IsPlot=0):
    if type==1:
        g = HorizontalVG()
    else:
        g = NaturalVG()
    g.build(X, only_degrees=True)
    ks, ps = g.degree_distribution

    g.build(X)
    nxg = g.as_networkx()
    if IsPlot==1:
        fig, [ax0, ax1, ax2] = plt.subplots(ncols=3, figsize=(12, 3.5))
        ax0.plot(X)
        ax0.set_title(label+' Time Series')

        graph_plot_options = {
            'with_labels': False,
            'node_size': 2,
            'node_color': [(0, 0, 0, 1)],
            'edge_color': [(0, 0, 0, 0.15)],
        }
    
        nx.draw_networkx(nxg, ax=ax1, pos=g.node_positions(), **graph_plot_options)
        ax1.tick_params(bottom=True, labelbottom=True)
        ax1.plot(X)
        if type==0:
            ax1.set_title(label+' Natural Visibility Graph')
        else:
            ax1.set_title(label+' Horizontal Visibility Graph')
        nx.draw_networkx(nxg, ax=ax2, pos=nx.kamada_kawai_layout(nxg), **graph_plot_options)
    if type==0:
        if IsPlot==1:
            ax2.set_title(label+' Natural Visibility Graph')
        Method.append('NVG')
    else:
        if IsPlot==1:
            ax2.set_title(label+' Horizontal Visibility Graph')
        Method.append('HVG')
    Entropy.append(ent2.sample_entropy(X, 4, 0.2 * np.std(X)))        
    Subject.append(subject)
    Axis.append(label)
    Avg_deg.append(math.degrees(mean(ps)))
    Nt_dia.append(nx.algorithms.distance_measures.diameter(nxg))
    Avg_path.append(nx.average_shortest_path_length(nxg))
    Activity.append(typeOf)
    avg_d[typeOf].append(math.degrees(mean(ps)))
    net_d[typeOf].append(nx.algorithms.distance_measures.diameter(nxg))
    Signal_Length.append(324)
    Dimension.append(3)
    Delay.append(1)
    #return {'Average degree':mean(ps),'Diameter':nx.algorithms.distance_measures.diameter(nxg),'Average path length':nx.average_shortest_path_length(nxg)}

def do_all(data_csv,typeOf='Walking',subject='Subject 1',IsPlot=0):
    data=pd.read_csv(data_csv)
    X=data['attr_x']
    Y=data['attr_y']
    Z=data['attr_z']
    
    X=X.values
    #X=X[1000:2024]
    X=X[1000:1324]
    Y=Y.values
    #Y=Y[1000:2024]
    Y=Y[1000:1324]
    Z=Z.values
    #Z=Z[1000:2024]
    Z=Z[1000:1324]
    
    
    plotter_VG(X,0,'X',typeOf,subject,IsPlot)
    plotter_VG(Y,0,'Y',typeOf,subject,IsPlot)
    plotter_VG(Z,0,'Z',typeOf,subject,IsPlot)

    plotter_VG(X,1,'X',typeOf,subject,IsPlot)
    plotter_VG(Y,1,'Y',typeOf,subject,IsPlot)
    plotter_VG(Z,1,'Z',typeOf,subject,IsPlot)
import time
#i=1
for i in range(1,16):
    sub='subject'+str(i)
    loc=path+"sub"+str(i)+"\\acc_walking_csv\\"
    csv_files = glob.glob(os.path.join(loc, "*.csv"))
    time.sleep(0.01)
    #print(len(csv_files))    
    IsPlot=0
    for f in csv_files:    
        do_all(f,'Walking',sub,IsPlot)
    #print("Done")
    
    loc=path+"sub"+str(i)+"\\acc_running_csv\\"
    csv_files = glob.glob(os.path.join(loc, "*.csv"))
    time.sleep(0.01)    
    for f in csv_files:    
        do_all(f,'Running',sub,IsPlot)
    #print("Done")
    
    loc=path+"sub"+str(i)+"\\acc_climbingup_csv\\"
    csv_files = glob.glob(os.path.join(loc, "*.csv"))
    time.sleep(0.01)    
    for f in csv_files:    
        do_all(f,'Climbing Up',sub,IsPlot)
    #print("Done")
    
    loc=path+"sub"+str(i)+"\\acc_climbingdown_csv\\"
    csv_files = glob.glob(os.path.join(loc, "*.csv"))
    time.sleep(0.01)    
    for f in csv_files:    
        do_all(f,'Climbing Down',sub,IsPlot)
    print("Done======================="+str(i))

df = pd.DataFrame()
df.loc[:,'Method']=pd.Series(Method)
df.loc[:,'Subject']=pd.Series(Subject)
df.loc[:,'Accelerometer Axis']=pd.Series(Axis)
df.loc[:,'Average Degree']=pd.Series(Avg_deg)
df.loc[:,'Network Diameter']=pd.Series(Nt_dia)
df.loc[:,'Average path length']=pd.Series(Avg_path)
df.loc[:,'Activity']=pd.Series(Activity)
df.to_csv( path+'subject_Data.csv')

df = pd.DataFrame()
df.loc[:,'Subject']=pd.Series(Subject)
df.loc[:,'Accelerometer Axis']=pd.Series(Axis)
df.loc[:,'Signal_Length']=pd.Series(Signal_Length)
df.loc[:,'Dimension']=pd.Series(Dimension)
df.loc[:,'Delay']=pd.Series(Delay)
df.loc[:,'Entropy']=pd.Series(Entropy)
df.loc[:,'Activity']=pd.Series(Activity)
df.to_csv( path+'Entropy.csv')


#Avg degree v/s net dia
plt.scatter(avg_d['Walking'], net_d['Walking'])
plt.scatter(avg_d['Running'], net_d['Running'],color='Red')
plt.show()

plt.scatter(avg_d['Climbing Up'], net_d['Climbing Up'])
plt.scatter(avg_d['Climbing Down'], net_d['Climbing Down'],color='Red')
plt.show()



