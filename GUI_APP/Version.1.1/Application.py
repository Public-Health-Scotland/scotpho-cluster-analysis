# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:26:36 2019

@author: panagn01
"""


import tkinter as tk
from tkinter import ttk
import matplotlib
import pandas as pd
import numpy as np
from functions_ import *
from time import sleep
from cluster_methods import *
from tkinter import font
import webbrowser



import sys 
matplotlib.use("TkAgg")


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.animation as animation
from matplotlib import style

LARGE_FONT = ('Verdana', 12)
NORM_FONT = ('Verdana', 10)
SMALL_FONT = ('Verdana', 8)
style.use("ggplot")

dataset_all = pd.read_csv("scotpho_data_extract.csv")
area_types = dataset_all['area_type'].unique()[:-1]
l_area_types = list(area_types)
        
dataset_all_full = pd.read_csv('dataset_all.csv')
dataset_all_full.drop(columns=['Unnamed: 0'],inplace= True)


def popupmsg(msg):
    popup = tk.Tk()
    
    def leavemini():
        popup.destroy()
    
    popup.wm_title('!')
    label = ttk.Label(popup, text = msg , font = NORM_FONT)
    label.pack(side = "top", fill = "x", pady=10)
    button1= ttk.Button(popup, text = 'OKAY', command = leavemini)
    button1.pack()
    popup.mainloop()
    
class mainapp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        
        #tk.Tk.iconbitmap(self,default = "favicon.bmp" )
        tk.Tk.wm_title(self,'Group IT')
        
        width = tk.Tk.winfo_screenwidth(self)
        height = tk.Tk.winfo_screenheight(self)

        self.geometry('%dx%d+0+0'%(width,height))
        
        container = tk.Frame(self)
        container.pack(side = 'top', fill = 'both', expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        menubar = tk.Menu(container)
        filemenu  = tk.Menu(menubar, tearoff = 0)
        filemenu.add_command(label ='Save settings', command = lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label = 'Exit', command = self.destroy)
        menubar.add_cascade(label = 'File', menu = filemenu)
        
        tk.Tk.config(self, menu = menubar)
        
        self.frames = { }
        for F in (HomePage, Clustering, TopK):
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row = 0, column =0, sticky = 'nsew')
        
        self.show_frame(HomePage)
        
    def show_frame(self, cont):
            frame = self.frames[cont]
            frame.tkraise()
        

def default(param):
    print(param)

class HomePage(tk.Frame):
    
    def __init__(self,parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text = 'Home Page', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        button1 = ttk.Button(self, text = 'Clustering',command= lambda: controller.show_frame(Clustering))  
        button1.pack()
        
        button2 = ttk.Button(self, text = 'TopK+',command= lambda: controller.show_frame(TopK))  
        button2.pack()
        

        
       
        
        
        text2 = tk.Text(self, height=40, width=162)
        scroll = tk.Scrollbar(self, command=text2.yview)
        text2.configure(yscrollcommand=scroll.set)
        text2.tag_configure('bold_italics', font=('Constantia', 12, 'bold', 'italic'))
        text2.tag_configure('big', font=('Calibri', 20, 'bold'))
        text2.tag_configure('color',
                            foreground='#476042',
                            font=('Arial', 12, 'bold'))
       #print(tk.font.families())

        text2.insert(tk.END,'Abstract:\n\n', 'big')
        abstr = 'An analysis and exploration of the public dataset of ScotPHO’s online profile tool.\nAn automatic way of clustering/grouping indicators of Scotpho’s online profile tool based on identifying similar objects at some specific features.\nA tool that finds the topK indicators for each area in Scotland.\nA tool that finds similar areas in terms of indicator values.\nA Python Tkinter application to combine everything and depict the results.\n'
        text2.insert(tk.END, abstr, 'color')
        
        text2.insert(tk.END,'\nFeatures:\n\n', 'big')
        features = 'Some of The new features that were used for the clustering method are described bellow:\n\n\tThe indicator’s value distance between Scotland and a particular area.\n\t Percentage of areas at this area type that are Statistical Different for this Indicator.\n\t Percentage of Indicators that are Different at this area.\n\t Flunctuations through 1,2,3 years.\n'
        text2.insert(tk.END, features, 'color')
        
        text2.insert(tk.END,'\nUsefulness:\n\n', 'big')
        usefulness  = 'Τhe indicators that belong to the same cluster may correlate with each other.\nAn identification of ONE bad indicator will lead to the identification of many other bad indicators.\nAs a result, a strategy to alleviate one indicator may lead to the alleviation of many more.\nThe topK tool will try to summarise how an area performs but also it may used by the users to identify really bad or good indicators so to use them at the clustering results.\nThe similarity tool can provide to the user deep understanding of similarities between areas.'
        text2.insert(tk.END, usefulness, 'color')
       
        text2.pack(side=tk.TOP, anchor = tk.CENTER)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        def callback(url):
            webbrowser.open_new(url)
            
        link1 = tk.Label(self, text="ScotPHO Hyperlink", fg="blue", cursor="hand2")
        link1.pack()
        link1.bind("<Button-1>", lambda e: callback("https://www.scotpho.org.uk/"))

        link2 = tk.Label(self, text="CODE Hyperlink", fg="blue", cursor="hand2")
        link2.pack()
        link2.bind("<Button-1>", lambda e: callback("https://github.com/ScotPHO/Python_code/tree/master/GUI_APP"))
        
        link3 = tk.Label(self, text="Paper Hyperlink", fg="blue", cursor="hand2")
        link3.pack()
# =============================================================================
#         #NEED TO ADD THE LINK FOR THE PAPER.
# =============================================================================
        link3.bind("<Button-1>", lambda e: callback("https://github.com/ScotPHO/Python_code/tree/master/GUI_APP"))
        

class Clustering(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self,parent)
        label = ttk.Label(self, text = 'Clustering', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        
# =============================================================================
# FUNCTIONS
# =============================================================================
        
        def selective_clustering_kmeans_(self,dataset,area_type, area_name, year,option):
            input_ = select_df(dataset,area_type, area_name, year)
            print(input_.shape)
            input_ = input_.reset_index(drop=True)
            points_ = input_.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
            scaler = MinMaxScaler()
            points_scaled_all = scaler.fit_transform(points_)
            if(input_.shape[0] > 2 ):
                    points_scaled_all =np.nan_to_num(points_scaled_all)
                    res_pca_all = PCA_for_kmeans(points_scaled_all,2)
                    clusters = kmeans_kneed(res_pca_all.values, input_.shape[0])
                    results_kmeans_pca = PCA_Kmeans(points_scaled_all,clusters)
                    df_pca = results_kmeans_pca['DF']
                    points_pca = df_pca[['first feature','second feature']]
                    labels_pca = df_pca['labels']
                    centroids_pca = results_kmeans_pca['centroids']
                    to_Export = display_indicators_per_cluster_1(input_, labels_pca)
                    Visualise_Clusters_(self,points_pca.values, centroids_pca, labels_pca, 'PCA and Kmeans at '+ area_type+ ' '+area_name+' '+str(year), input_)
                    #save clusters?
                    if(option == 'yes'):
                        to_Export = display_indicators_per_cluster_1(input_, labels_pca)
                        to_Export.to_excel('Kmeans_'+area_type+' '+area_name+' '+str(year)+'.xlsx')
            else:
                print('sorry not enough data')
                
        
        def selective_clustering_dbscan_(self,dataset,area_type, area_name, year,option):
            #DBSCAN
            input_ = select_df(dataset,area_type, area_name, year)
            print(input_.shape)
            input_ = input_.reset_index(drop=True)
            points_ = input_.loc[:,['difference', 'Stat diff at indicator', 'Better at Indicator', 'Not Sure at Indicator', 'Worse at Indicator', 'Stat diff at area', 'Not Sure at area', 'Better at area', 'Worse at area', 'difference from next year', 'difference from next 2 years', 'difference from next 3 years']].values
            scaler = MinMaxScaler()
            points_scaled_all = scaler.fit_transform(points_)
            if(input_.shape[0] > 2 ):
                points_scaled_all =np.nan_to_num(points_scaled_all)
                res_pca_all = PCA_for_kmeans(points_scaled_all,2)
                dbscan = DBSCAN(eps=0.123, min_samples = 2)
                clusters = dbscan.fit_predict(res_pca_all.values)
                Visualise_Clusters_dbscan_(self,res_pca_all.values, clusters , 'PCA and DBSCAN at '+ area_type+ ' '+area_name+' '+str(year) )
                if(option == 'yes'):
                    to_Export = display_indicators_per_cluster_1(input_, clusters)
                    to_Export.to_excel('DBSCAN_'+area_type+' '+area_name+' '+str(year)+'.xlsx')

            else:
                print('sorry not enough Data')
            

        def Visualise_Clusters_(self,points, center, labels, title, df):
          
# =============================================================================
             self.kmeans_.cla()
             #self.kmeans_.scatter(points[:, 0], points[:, 1], c = labels, s=60, cmap='viridis')
             self.kmeans_.scatter(center[:, 0], center[:, 1], c='black', s=250, alpha=0.6)
#             
             k = 0
            
             for row in center:
                 print(row)
                 self.kmeans_.annotate(k, 
                 (row[0], row[1]),
                  horizontalalignment='center',
                  verticalalignment='center',
                  size=20, weight='bold',
                  color='r')
                 k= k+1
             
             self.kmeans_.set_title('groups of indicators: Clustering Result ( '+title+' )', fontsize = 12.0,fontweight="bold")
             self.kmeans_.set_xticks([])
             self.kmeans_.set_yticks([])
             self.kmeans_.set_xlabel('First Feature', fontsize = 12.0)
             self.kmeans_.set_ylabel('Second Feature', fontsize = 12.0)
         
             if(self.skiniko != None):
                 self.skiniko.pack_forget()
    
             x= points[:,0]
             y = points[:,1]
             self.names = labels
             cmap = plt.cm.RdYlGn
                
             #self.fig, self.ax = plt.subplots()
             self.sc = self.kmeans_.scatter(x,y,c=labels, s=100, cmap= cmap)
             self.annot = self.kmeans_.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
             self.annot.set_visible(False)
                
             self.names = df['indicator'].tolist()
                
        def update_annot(self,ind):
             pos = self.sc.get_offsets()[ind["ind"][0]]
             self.annot.xy = pos
             text = ([self.names[n] for n in ind["ind"]])
             print(text)
             self.annot.set_text(text)
             self.annot.get_bbox_patch().set_facecolor('r')
             self.annot.get_bbox_patch().set_alpha(0.4)
                
        def hover(self,event):
             print(event)
             vis = self.annot.get_visible()
             if event.inaxes == self.kmeans_:
                cont, ind = self.sc.contains(event)
                if cont:
                    update_annot(ind)
                    self.annot.set_visible(True)
                    self.f.canvas.draw_idle()
                else:
                    if vis:
                        self.annot.set_visible(False)
                        self.f.canvas.draw_idle()
            
        
            #plt.show()    
                
            
        def Visualise_Clusters_dbscan_(self,points,clusters, title):
          
            
            
            self.dbscan_.scatter(points[:, 0], points[:, 1], c = clusters, s=60, cmap = "plasma")
            self.dbscan_.set_title('groups of indicators: Clustering Result ( '+title+' )', fontsize = 12.0,fontweight="bold" )
            self.dbscan_.set_xticks([])
            self.dbscan_.set_yticks([])
            self.dbscan_.set_xlabel('First Feature', fontsize = 12.0)
            self.dbscan_.set_ylabel('Second Feature', fontsize = 12.0)
            
            if(self.skiniko != None):
                self.skiniko.pack_forget()
            


        Final = pd.read_csv('13_07_19_full.csv')
        Final.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'],inplace= True)
            
        
        
        Final = Final.rename(columns = { 'Percentage of areas at this area type that are Statistical Diff for this Indicator' : 'Stat diff at indicator' })
        Final = Final.rename(columns = { 'Percentage of areas at this area type that are Better for this Indicator': 'Better at Indicator' } )
        Final = Final.rename(columns = { 'Percentage of areas at this area type that we are not sure if its Stat Diff for this Indicator': 'Not Sure at Indicator' } )
        Final = Final.rename(columns = { 'Percentage of areas at this area type that are Worse for this Indicator': 'Worse at Indicator' } )
        Final = Final.rename(columns = { 'Percentage of Indicators that are Different at this area': 'Stat diff at area' } )
        Final = Final.rename(columns = { 'Percentage of Indicators that are worse at this area': 'Worse at area' } )
        Final = Final.rename(columns = { 'Percentage of Indicators that  are better at this area' : 'Better at area'} )
        Final = Final.rename(columns = { 'Percentage of Indicators that we are not sure at this area': 'Not Sure at area' } )
        
        Final.loc[Final['difference from next year'] == 'no data', 'difference from next year'] = 0.0
        Final.loc[Final['difference from next 2 years'] == 'no data', 'difference from next 2 years'] = 0.0
        Final.loc[Final['difference from next 3 years'] == 'no data', 'difference from next 3 years'] = 0.0


        With_Intervals = (Final.loc[(Final['has_intervals']== True)])
# =============================================================================
# menu
# =============================================================================
        
        menu_bar = tk.Frame(self)
        
        button1 = ttk.Button(menu_bar, text = 'Home',command= lambda: controller.show_frame(HomePage))
        button1.pack(side = 'left', padx = 5, pady=5)
        
        button2 = ttk.Button(menu_bar, text = 'TOPK+',command= lambda: controller.show_frame(TopK)) 
        button2.pack(side = 'left')
        
        menu_bar.pack()
        
# =============================================================================
#  option list       
# =============================================================================
       
        #remove the estimation since we can't compare the values of an area to the sum of population.
        
        
        def pick_area(event):
            area_name_list['values'] =[]
            area_name_list.set('select')
            indicator_list['values'] = []
            indicator_list.set('select')   
            area_level_selected = geography_level.get()
            area_n = dataset_all[dataset_all['area_type'] == area_level_selected]['area_name'].unique()
            area_val = list(area_n)
            area_name_list['values'] = area_val
            year_list.set('select')
        
        
        option_bar = tk.Frame(self)
        
        geography_level_l = ttk.Label(option_bar, text= 'Geography Level')
        geography_level_l.pack(side = 'left', padx = 3)
        geography_level = ttk.Combobox(option_bar, values = l_area_types, width =15, state="readonly")
        geography_level.set('select')
        geography_level.pack(side = 'left', padx = 3)
        geography_level.bind("<<ComboboxSelected>>", pick_area)
        
        def pick_indicator(event):
            area_selected = area_name_list.get()
            indicators = dataset_all[dataset_all['area_name'] == area_selected]['indicator'].unique()
            indicators_val = list(indicators)
            indicator_list['values'] = indicators_val
        
        def pick_year(event):
             area_selected = area_name_list.get()
             years = dataset_all[dataset_all['area_name'] == area_selected]['year'].unique()
             years_val = list(years)
             year_list['values'] = years_val
        
        area_name_l = ttk.Label(option_bar, text= 'Area Name')
        area_name_l.pack(side = 'left', padx = 3)
        area_name_list = ttk.Combobox(option_bar, width =15, state="readonly")
        area_name_list.set('select')
        area_name_list.pack(side = 'left', padx = 3)
        area_name_list.bind("<<ComboboxSelected>>", pick_year)
        
        indicator_l = ttk.Label(option_bar, text= 'Indicator')
        #indicator_l.pack(side = 'left')
        indicator_list = ttk.Combobox(option_bar, width =15, state="readonly")
        indicator_list.set('select')
        #indicator_list.pack(side = 'left')
       
            
        
        year_ = ttk.Label(option_bar, text = 'Year')
        year_list = ttk.Combobox(option_bar, width =15, state="readonly")
        year_list.set('select')
        year_.pack(side = 'left', padx = 3)
        year_list.pack(side = 'left', padx = 3)
        
        
        self.CheckVar = tk.IntVar(self, value=1)
        checkaki = ttk.Checkbutton(option_bar, variable= self.CheckVar, text="Save results")
        checkaki.pack(side ='left', padx=3)
        self.canvas = None
        self.skiniko = None
        self.f = Figure(figsize =(5,8), dpi=100)
        self.kmeans_= self.f.add_subplot(211)
        self.dbscan_ =self.f.add_subplot(212)
     
        def cluster():
           # self.canvas.delete(self,'all')
            area = area_name_list.get()
            geography = geography_level.get()
            year = year_list.get()
                
        
            if(geography == 'select' or area == 'select' or year =='select'):
                print('Select Valid Values')
                popupmsg('Enter Valid Values')
            else:
                print('we are ok')
                
                
           
            
           
            
            
            save_option = (self.CheckVar.get())
            text_option = 'yes'
            if(save_option == 1):
                text_option = 'yes'
            else:
                text_option = 'no'
            
          
            dfaki__ = select_df(With_Intervals, str(geography), str(area), int(year))
            
           
            
            self.f = Figure(figsize =(5,8), dpi=100)
            self.kmeans_= self.f.add_subplot(211)
            self.dbscan_ =self.f.add_subplot(212)  
            
            selective_clustering_kmeans_(self,With_Intervals,geography, area, int(year),text_option)    
            selective_clustering_dbscan_(self,With_Intervals,geography, area, int(year),text_option)
            
          
        
            self.canvas = FigureCanvasTkAgg(self.f,self)
            self.canvas.draw()
            self.skiniko= self.canvas.get_tk_widget()
            
            self.skiniko.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            self.f.canvas.mpl_connect("motion_notify_event", hover(self))
        
         
        
        def clear():
            print('clear')
            self.f.clf()
            self.canvas.draw()
            self.skiniko.pack_forget()
            
        def explain():
            popupmsg('This page depicts the clustering results. Each datapoint is an indicator. The indicators that belong to the same cluster are similar in term of some features (see Homepage). \n If you want to see the indicators of each cluster please select the save option; the app will create an excel workbook.\nPlease select the values from the left to the right.')
            
            
        
        search_button = ttk.Button(option_bar, text= 'SEARCH', command = cluster)
        search_button.pack(side = 'left', padx = 3)
        clear_button = ttk.Button(option_bar, text= 'CLEAR', command = clear)
        clear_button.pack(side = 'left', padx = 3)
        help_button = ttk.Button(option_bar, text= 'HELP', command = explain)
        help_button.pack(side = 'left', padx = 3)
        option_bar.pack()

class TopK(tk.Frame):
    
    def __init__(self, parent, controller):
        
        
        
        #building the frame
        
        tk.Frame.__init__(self,parent)
        
        
      

        
        label = ttk.Label(self, text = 'TopK+', font = LARGE_FONT)
        #label.grid(row=1, column = 1)
        label.pack()
        
        menu_bar = tk.Frame(self)
        
        button1 = ttk.Button(menu_bar, text = 'Home',command= lambda: controller.show_frame(HomePage))
        #button1.grid(row=2, column = 0, sticky='W')
        button1.pack(side = 'left', padx = 5, pady=5)
        
        button2 = ttk.Button(menu_bar, text = 'Clustering',command= lambda: controller.show_frame(Clustering)) 
        #button2.grid(row=2, column = 1, sticky= 'WNS')
        button2.pack(side = 'left')
        menu_bar.pack()
        #creating the lists
    
      
        
        #remove the estimation since we can't compare the values of an area to the sum of population.
        dataset_all_ = dataset_all_full[(dataset_all_full.indicator !='Mid-year population estimate - all ages') & (dataset_all_full.indicator != 'Quit attempts')]
        
        prof_ind = pd.read_excel('Profiles to indicators.xlsx')
        
        
        def pick_area(event):
            area_name_list['values'] =[]
            area_name_list.set('select')
            indicator_list['values'] = []
            indicator_list.set('select')
            listBox.delete(*listBox.get_children())
            listBox1.delete(*listBox1.get_children())
            listBox2.delete(*listBox2.get_children())
            year_list.set('select')
            
            area_level_selected = geography_level.get()
            area_n = dataset_all[dataset_all['area_type'] == area_level_selected]['area_name'].unique()
            area_val = list(area_n)
            area_name_list['values'] = area_val
        
        
        option_bar = tk.Frame(self)
        
        geography_level_l = ttk.Label(option_bar, text= 'Geography Level')
        #geography_level_l.grid(row=3, column =0, sticky = 'SW')
        geography_level_l.pack(side = 'left', padx = 3)
        geography_level = ttk.Combobox(option_bar, values = l_area_types, width =15, state="readonly")
        geography_level.set('select')
        #geography_level.grid(row = 3, column =1, sticky = 'SE',)
        geography_level.pack(side = 'left', padx = 3)
        geography_level.bind("<<ComboboxSelected>>", pick_area)
        
        def pick_indicator(event):
            area_selected = area_name_list.get()
            indicators = dataset_all[dataset_all['area_name'] == area_selected]['indicator'].unique()
            indicators_val = list(indicators)
            print(len(indicators_val))
            indicator_list['values'] = indicators_val
        
        def pick_year(event):
             area_selected = area_name_list.get()
             years = dataset_all[dataset_all['area_name'] == area_selected]['year'].unique()
             years_val = list(years)
             year_list['values'] = years_val
        
        area_name_l = ttk.Label(option_bar, text= 'Area Name')
        #area_name_l.grid(row=4, column =0, sticky = 'W')
        area_name_l.pack(side = 'left', padx = 3)
        area_name_list = ttk.Combobox(option_bar, width =15, state="readonly")
        area_name_list.set('select')
        #area_name_list.grid(row = 4, column =1, sticky = 'W')
        area_name_list.pack(side = 'left', padx = 3)
        area_name_list.bind("<<ComboboxSelected>>", pick_year)
        
        indicator_l = ttk.Label(option_bar, text= 'Indicator')
        #indicator_l.grid(row=5, column =0, sticky = 'W')
        #indicator_l.pack(side = 'left')
        indicator_list = ttk.Combobox(option_bar, width =15, state="readonly")
        indicator_list.set('select')
        #indicator_list.grid(row = 5, column =1, sticky = 'E')
        #indicator_list.pack(side = 'left')
       
            
        
        year_ = ttk.Label(option_bar, text = 'Year')
        year_list = ttk.Combobox(option_bar, width =15, state="readonly")
        year_list.set('select')
        #year_.grid(row=5, column =0, sticky = 'W')
        #year_list.grid(row=5, column =1)
        year_.pack(side = 'left', padx = 3)
        year_list.pack(side = 'left', padx = 3)
        
        table_frame = tk.Frame(self)
        
        
        cols = ('index','indicator', 'area_name', 'area_type','actual_diff','year', 'definition'  )
        listBox = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        cols1= ['index','indicator', 'profile', 'actual_diff', 'area_name', 'area_type','definition', 'year']
        listBox1 = ttk.Treeview(table_frame, columns=cols1, show='headings')
        
        cols2 = ('index','indicator', 'area_name', 	'area_type', 	'year', 	'measure', 'sim_diff' )
        listBox2 = ttk.Treeview(table_frame, columns=cols2, show='headings')
        label_top_k = tk.Label(table_frame, font=LARGE_FONT)
        label_top_profile = tk.Label(table_frame, font =LARGE_FONT)
        def search_m():
            
            geo = geography_level.get()
            area = area_name_list.get()
            indi = indicator_list.get()
            k = k_l_list.get()
            option = option_list.get()
            year = year_list.get()
            print(k)
            listBox.delete(*listBox.get_children())
            listBox1.delete(*listBox1.get_children())
            listBox2.delete(*listBox2.get_children())
            
            if(geo == 'select' or area == 'select'  or k =='select' or option == 'select' or year =='select'):
                print('Select Valid Values')
                popupmsg('Enter Valid Values')
            else:
                print('we are ok')
             
# =============================================================================
# TOP K    
# =============================================================================
                
                
                
                
                label_top_k['text']= 'TOP'+k+' '+option
                #label.grid(row=10, column =0)
                label_top_k.pack(pady=2)
                
                for col in cols:
                    listBox.heading(col, text=col)    
                #listBox.grid(row=10, column=0, columnspan=1)
                #listBox.grid(row = 11, column =0)
                listBox.pack(pady=2)
                
                df_ = select_df(dataset_all_, str(geo), str(area), int(year))
                dfaki = top_k(df_, int(k), str(option))
                dfaki = dfaki.values.tolist()
                for i, (indicator, area_name, area_type, year, actual_diff, definition) in enumerate(dfaki, start=1):
                    listBox.insert("", "end", values=(i, indicator, area_name, area_type, year, actual_diff, definition))
                
               
# =============================================================================
# PER PROFILE              
# =============================================================================
               
                
 
                
                #label1.grid(row =12,column =0)
                label_top_profile['text'] = 'TOP'+str(k)+' '+str(option)+' PER PROFILE'
                label_top_profile.pack(pady=2)
              
                print(option)
                
                for col in cols1:
                     listBox1.heading(col, text=col)   
                     
                dfaki1 = rank_by_profiles(df_,prof_ind,option)
                listaki = []
                for l in dfaki1:
                    top_ = 1
                    for l_ in l:
                        listaki.append(l_)
                        top_ = top_ + 1
                        if(top_ == int(k)):
                            break
                dfakos = pd.DataFrame(listaki)
                dfakos_ = dfakos[['Indicator', 'Profile', 'actual_diff', 'area_name', 'area_type',
       'definition', 'year']].values.tolist()
              
            
                for i, (indicator, profile_name, actual_diff, area_name, area_type, definition, year) in enumerate(dfakos_, start=1):
                     listBox1.insert("", "end", values=(i,indicator, profile_name, actual_diff, area_name, area_type, definition, year))
                     
                listBox1.pack(pady=2)
# =============================================================================
# =============================================================================
# Similarity             
# =============================================================================
                label2 = tk.Label(table_frame, text="TOP"+str(k)+' SIMILAR', font=LARGE_FONT)
                #label2.grid(row=14, column =0)
                label2.pack(pady=2)
      
          
                for col in cols2:
                    listBox2.heading(col, text=col)    
                #listBox2.grid(row=15, column =0)
                listBox2.pack(pady=2)

                _dfaki =  find_the_most_Similar(df_,int(k))
    
                list_dfaki = _dfaki[['indicator', 'area_name', 	'area_type', 	'year', 	'measure', 'sim_diff']].values.tolist()
          
                for i, (indicator, area_name, 	area_type, year, 	measure, sim_diff) in enumerate(list_dfaki, start=1):
                    listBox2.insert("", "end", values=(i, indicator, area_name, area_type, year, measure, sim_diff))


        
       
        def printk(event):
            print(k_l_list.get())
        
        #K
        k_l = ttk.Label(option_bar, text= 'K')
       # k_l.grid(row=6, column =0, sticky = 'W')
        k_l.pack(side = 'left', padx = 3)
        k_l_list = ttk.Combobox(option_bar, values = list(range(1,21)), width =15, state="readonly")
        #k_l_list.grid(row=6, column =1,)
        k_l_list.pack(side = 'left', padx = 3)
        k_l_list.set('select')
        
        k_l_list.bind("<<ComboboxSelected>>", printk)
        
        #option
        option_l = ttk.Label(option_bar, text= 'Option')
        #option_l.grid(row=7, column =0, sticky = 'W')
        option_l.pack(side = 'left', padx = 3)
        option_list = ttk.Combobox(option_bar, values = ['better','worse'], width =15)
        #option_list.grid(row=7, column =1)
        option_list.pack(side = 'left', padx = 3)
        option_list.set('select')
        def explain():
            popupmsg('This page depicts the TOPK and similarities results.\nThe second table depicts the TOPK indicators per profile(Health...tobacco..etc).\nPlease select the values from the left to the right. ')
        
        search_button = ttk.Button(option_bar, text= 'SEARCH', command = search_m)
        #search_button.grid(row= 8, column = 1, rowspan=2, sticky = 'W')
        search_button.pack(side = 'left', padx = 3)
        help_button = ttk.Button(option_bar, text= 'HELP', command = explain)
        help_button.pack()
        option_bar.pack()
        table_frame.pack()
        
        
        


app = mainapp()

app.mainloop()