# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:26:36 2019

@author: Panagiotis (Panos) Ntoulos

Msc DataScience University of Glasgow industrial placement - Dissertation
"""

# =============================================================================
# Libraries
# =============================================================================
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
import matplotlib.patches as mpatches
# =============================================================================
# Datasets
# =============================================================================
LARGE_FONT = ('Verdana', 12)
NORM_FONT = ('Verdana', 10)
SMALL_FONT = ('Verdana', 8)
style.use("ggplot")

dataset_all = pd.read_csv("scotpho_data_extract.csv")
area_types = dataset_all['area_type'].unique()[:-1]
l_area_types = list(area_types)
        
dataset_all_full = pd.read_csv('dataset_all.csv')
dataset_all_full.drop(columns=['Unnamed: 0'],inplace= True)
checkaki = pd.read_csv('techdoc_backup.csv')
checkaki = checkaki.replace(np.nan, 'no profile assigned', regex=True)
checkaki = checkaki.rename(columns = {'indicator_name': 'indicator'})
dataset_all_m = dataset_all_full.merge(checkaki[['indicator', 'interpretation']], on ='indicator', how ='left')
dataset_all_m['actual_diff'][dataset_all_m.interpretation == 'Higher numbers are better'] = -dataset_all_m['actual_diff'][dataset_all_m.interpretation == 'Higher numbers are better']

dataset_all_ = dataset_all_m[(dataset_all_m.indicator !='Mid-year population estimate - all ages') & (dataset_all_m.indicator != 'Quit attempts')]

# =============================================================================
# Help functions
# =============================================================================
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

# =============================================================================
# Main app controler
# This class creates the main frame and it is responsible to load the pages.    
# =============================================================================
class mainapp(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        
        #tk.Tk.iconbitmap(self,default = "favicon.bmp" )
        tk.Tk.wm_title(self,'Group IT')
        
        width = tk.Tk.winfo_screenwidth(self)
        height = tk.Tk.winfo_screenheight(self)
        #setting the display to almost fullscreen.
        self.geometry('%dx%d+0+0'%(width,height))
        
        container = tk.Frame(self)
        container.pack(side = 'top', fill = 'both', expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
# =============================================================================
#         dropdown menu
# =============================================================================
        menubar = tk.Menu(container)
        filemenu  = tk.Menu(menubar, tearoff = 0)
        filemenu.add_command(label ='Save settings', command = lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label = 'Exit', command = self.destroy)
        menubar.add_cascade(label = 'File', menu = filemenu)
        
        tk.Tk.config(self, menu = menubar)
        
# =============================================================================
#         adds all the pages at a dictionary. This is the easiest and also efficient way to iterate through pages.
# =============================================================================
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
    
    
# =============================================================================
# HomePage: each frame is divided to smaller frames so to be easier the depictions of the widgets(buttons,text,label etc)
# Each page has links to iterate through the other pages.
# HomePage describes the basic concepts of the project.

# =============================================================================

class HomePage(tk.Frame):
    
    def __init__(self,parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text = 'Home Page', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        button1 = ttk.Button(self, text = 'Clustering',command= lambda: controller.show_frame(Clustering))  
        button1.pack()
        
        button2 = ttk.Button(self, text = 'TopK+',command= lambda: controller.show_frame(TopK))  
        button2.pack()
        
# =============================================================================
#         Setting the fonts for the text
# =============================================================================
        text2 = tk.Text(self, height=40, width=162)
        scroll = tk.Scrollbar(self, command=text2.yview)
        text2.configure(yscrollcommand=scroll.set)
        text2.tag_configure('bold_italics', font=('Constantia', 12, 'bold', 'italic'))
        text2.tag_configure('big', font=('Calibri', 20, 'bold'))
        text2.tag_configure('color',
                            foreground='#476042',
                            font=('Arial', 12, 'bold'))
# =============================================================================
# Inserting the text
# =============================================================================

        text2.insert(tk.END,'Abstract:\n\n', 'big')
        abstr = 'An analysis and exploration of the public dataset of ScotPHO’s online profile tool.\nAn automatic way of clustering/grouping indicators of Scotpho’s online profile tool based on identifying similar objects at some specific features.\nA tool that finds the topK indicators for each area in Scotland.\nA tool that finds similar areas in terms of indicator values.\nA Python Tkinter application to combine everything and depict the results.\n'
        text2.insert(tk.END, abstr, 'color')
        
        text2.insert(tk.END,'\nFeatures:\n\n', 'big')
        features = 'Some of The new features that were used for the clustering method are described bellow:\n\n\tThe indicator’s value distance between Scotland and a particular area.\n\t Percentage of areas at this area type that are Statistical Different for this Indicator.\n\t Percentage of Indicators that are Different at this area.\n\t Flunctuations through 1,2,3 years.\n'
        text2.insert(tk.END, features, 'color')
        
        text2.insert(tk.END,'\nUsefulness:\n\n', 'big')
        usefulness  = 'Τhe indicators that belong to the same cluster may correlate with each other.\nAn identification of ONE bad indicator will lead to the identification of many other bad indicators.\nAs a result, a strategy to alleviate one indicator may lead to the alleviation of many more.\nThe topK tool can be used to summarise how an area performs but also it may be used by the users to identify really bad or good indicators so to use them at the clustering results.\nThe similarity tool can provide to the user a deep understanding of similarity between areas.\n Please press the help buttons at each page if you are not familiar with the concepts that this project uses.'
        text2.insert(tk.END, usefulness, 'color')
       
        text2.pack(side=tk.TOP, anchor = tk.CENTER)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
# =============================================================================
#        adding and  open the links
# =============================================================================
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
        

# =============================================================================
# Clustering page
# Each page has links to iterate through the other pages.
# Clustering page has an pagemenu bar/frame, an option bar/frame and a frame that contains the 3 graphs.
# the option frame let the user to choose the values of : area_type, area_name, year 
#the first graph contains the TopK indicators 
# the second graph contains the topK indicators per profile.
#the third graph contains the areas that hava the most similar values of indicators with the area that was given as an input.
# =============================================================================
class Clustering(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self,parent)
        label = ttk.Label(self, text = 'Clustering', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        
# =============================================================================
# FUNCTIONS
        # clustering methods ( Kmeans and DBSCAN)
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
                    Visualise_Clusters_(self,points_pca.values, centroids_pca, labels_pca, 'PCA and Kmeans at '+ area_type+ ' '+area_name+' '+str(year))
                    #save clusters?
                    if(option == 'yes'):
                        to_Export = display_indicators_per_cluster_1(input_, labels_pca)
                        to_Export.to_excel('Clustering_Results\\Kmeans_'+area_type+' '+area_name+' '+str(year)+'.xlsx')
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
                    to_Export.to_excel('Clustering_Results\\DBSCAN_'+area_type+' '+area_name+' '+str(year)+'.xlsx')

            else:
                print('sorry not enough Data')
            
            
# =============================================================================
# Visualisation method of Kmeans
            # it plots the center points and all the points given each point the color of the cluster that it belongs.
# =============================================================================
        def Visualise_Clusters_(self,points, center, labels, title):
          
            self.kmeans_.cla()
            self.kmeans_.scatter(points[:, 0], points[:, 1], c = labels, s=60, cmap='viridis')
            self.kmeans_.scatter(center[:, 0], center[:, 1], c='black', s=250, alpha=0.6)
            print(points)
            print(center)
            print(labels)
            
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
            
            self.kmeans_.set_title('groups of indicators: Grouping Result ( '+title+' )', fontsize = 12.0,fontweight="bold")
            self.kmeans_.set_xticks([])
            self.kmeans_.set_yticks([])
            self.kmeans_.set_xlabel('First Feature', fontsize = 12.0)
            self.kmeans_.set_ylabel('Second Feature', fontsize = 12.0)
            
            #unpacks so to demonstrate  the new plots
            if(self.skiniko != None):
                self.skiniko.pack_forget()
           
            
# =============================================================================
# Same as the above function with the only difference that dbscan doesn't generate center points
# =============================================================================
        def Visualise_Clusters_dbscan_(self,points,clusters, title):  
            
            self.dbscan_.scatter(points[:, 0], points[:, 1], c = clusters, s=60, cmap = "tab20")
            self.dbscan_.set_title('groups of indicators: Grouping Result ( '+title+' )', fontsize = 12.0,fontweight="bold" )
            self.dbscan_.set_xticks([])
            self.dbscan_.set_yticks([])
            self.dbscan_.set_xlabel('First Feature', fontsize = 12.0)
            self.dbscan_.set_ylabel('Second Feature', fontsize = 12.0)
            blue_patch = mpatches.Patch(color='steelblue', label = 'Outlier')
            self.dbscan_.legend(handles=[blue_patch])
            if(self.skiniko != None):
                self.skiniko.pack_forget()
            
# =============================================================================
# END of functions
# =============================================================================
#***************************************************************************
                
# =============================================================================
#        load the datasets and dataframes manipulation       
# =============================================================================
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
# Pagemenu
# =============================================================================
        
        menu_bar = tk.Frame(self)
        
        button1 = ttk.Button(menu_bar, text = 'Home',command= lambda: controller.show_frame(HomePage))
        button1.pack(side = 'left', padx = 5, pady=5)
        
        button2 = ttk.Button(menu_bar, text = 'TOPK+',command= lambda: controller.show_frame(TopK)) 
        button2.pack(side = 'left')
        
        menu_bar.pack()
        
# =============================================================================
#  option list: buttons and functions. Each geography level has different areas, years, indicators.
        #       I managed to implement the functions that choose the values dynamically. 
        #       They use the datasets and the input of the user
# =============================================================================
       
# =============================================================================
#         Pick the areas that correspond to the geography level that was given (event)
#         Resets the values when a new geography level is selected.
# =============================================================================
        def pick_area(event):
            area_name_list['values'] =[]
            area_name_list.set('select')
            indicator_list['values'] = []
            indicator_list.set('select')   
            area_level_selected = geography_level.get()
            area_n = dataset_all[dataset_all['area_type'] == area_level_selected]['area_name'].unique()
            area_val = list(area_n)
            #area_val = area_val.sort()
            sort_area = sorted(area_val)
            area_name_list['values'] = sort_area
            
            year_list.set('select')
        
        
        option_bar = tk.Frame(self)
        
        geography_level_l = ttk.Label(option_bar, text= 'Geography Level')
        geography_level_l.pack(side = 'left', padx = 3)
        geography_level = ttk.Combobox(option_bar, values = l_area_types, width =15, state="readonly")
        geography_level.set('select')
        geography_level.pack(side = 'left', padx = 3)
        geography_level.bind("<<ComboboxSelected>>", pick_area)
        
# =============================================================================
#         pick the indicators that correspond to the area level that was given (i dont use this button it may be usefull in the future)
# =============================================================================
        def pick_indicator(event):
            area_selected = area_name_list.get()
            indicators = dataset_all[dataset_all['area_name'] == area_selected]['indicator'].unique()
            indicators_val = list(indicators)
            indicator_list['values'] = indicators_val
        
# =============================================================================
#         same pick the year given the area
# =============================================================================
        def pick_year(event):
             area_selected = area_name_list.get()
             years = dataset_all[dataset_all['area_name'] == area_selected]['year'].unique()
             years_val = list(years)
             years_val_sorted  = sorted(years_val)
             year_list['values'] = years_val_sorted
             
             
        
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
        
# =============================================================================
#         initialising the variables as class variables. 
#         This will allow the canvas to get updated for different areas values.
# =============================================================================
        self.CheckVar = tk.IntVar(self, value=0)
        checkaki = ttk.Checkbutton(option_bar, variable= self.CheckVar, text="Save results")
        checkaki.pack(side ='left', padx=3)
        self.canvas = None
        self.skiniko = None
        self.f = Figure(figsize =(5,8), dpi=100)
        self.kmeans_= self.f.add_subplot(211)
        self.dbscan_ =self.f.add_subplot(212)
     
# =============================================================================
# The functions that is triggered when the user presses the search button
# =============================================================================
        def cluster():
           
            area = area_name_list.get()
            geography = geography_level.get()
            year = year_list.get()
                
# =============================================================================
#             check if all the values have been selected
# =============================================================================
            if(geography == 'select' or area == 'select' or year =='select'):
                print('Select Valid Values')
                popupmsg('Enter Valid Values')
            else:
                print('we are ok')
            
            #checks the save option, if yes it saves an excell workbook with the results.
            save_option = (self.CheckVar.get())
            text_option = 'yes'
            if(save_option == 1):
                text_option = 'yes'
            else:
                text_option = 'no'
            
          
            dfaki__ = select_df(With_Intervals, str(geography), str(area), int(year))
            
           
            
            self.f = Figure(figsize =(5,8), dpi=100)
            self.kmeans_= self.f.add_subplot(211,picker=True)
            self.dbscan_ =self.f.add_subplot(212,picker=True)  
            
            selective_clustering_kmeans_(self,With_Intervals,geography, area, int(year),text_option)    
            selective_clustering_dbscan_(self,With_Intervals,geography, area, int(year),text_option)
            
          
            
            self.canvas = FigureCanvasTkAgg(self.f,self)
            self.canvas.draw()
            self.skiniko= self.canvas.get_tk_widget()
            
            self.skiniko.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
            
        
         
        #clears all canvas.
        def clear():
            print('clear')
            self.f.clf()
            self.canvas.draw()
            self.skiniko.pack_forget()
         
        #function of help button
        def explain():
            popupmsg('This page depicts the grouping results After performing two clustering Algorithms on the 13 derived features from ScotPHOS dataset; Kmeans and DBSCAN.\nPCA is a dimensionality reduction method; I used it so plot the data in 2D space.\nEach datapoint is an indicator. The indicators that belong to the same cluster are similar in term of some features (see Homepage). \nIf you want to see the indicators of each cluster please select the save option; the app will create an excel workbook at the ''Clustering_Results'' folder.\nPlease select the values from the left to the right.')
            
            
        
        search_button = ttk.Button(option_bar, text= 'SEARCH', command = cluster)
        search_button.pack(side = 'left', padx = 3)
        clear_button = ttk.Button(option_bar, text= 'CLEAR', command = clear)
        clear_button.pack(side = 'left', padx = 3)
        help_button = ttk.Button(option_bar, text= 'HELP', command = explain)
        help_button.pack(side = 'left', padx = 3)
        option_bar.pack()


# =============================================================================
# TopK page
# =============================================================================
class TopK(tk.Frame):
    
    def __init__(self, parent, controller):
        
        
        
        #building the frame
        
        tk.Frame.__init__(self,parent)
        
        
      

        
        label = ttk.Label(self, text = 'TopK+', font = LARGE_FONT)
        label.pack()
        
# =============================================================================
#         First frame
# =============================================================================
        menu_bar = tk.Frame(self)
        
        button1 = ttk.Button(menu_bar, text = 'Home',command= lambda: controller.show_frame(HomePage))
        button1.pack(side = 'left', padx = 5, pady=5)
        
        button2 = ttk.Button(menu_bar, text = 'Clustering',command= lambda: controller.show_frame(Clustering)) 
        button2.pack(side = 'left')
        menu_bar.pack()
    
        
        #remove the estimation since we can't compare the values of an area to the sum of population.
        
        
        prof_ind = pd.read_excel('Profiles to indicators.xlsx')
        
# =============================================================================
#         Pick the areas that correspond to the geography level that was given (event)
#         Resets the values when a new geography level is selected.
# =============================================================================
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
            area_val_sorted = sorted(area_val)
            area_name_list['values'] = area_val_sorted
        
# =============================================================================
#         seond frame
# =============================================================================
        option_bar = tk.Frame(self)

        geography_level_l = ttk.Label(option_bar, text= 'Geography Level')
        geography_level_l.pack(side = 'left', padx = 3)
        geography_level = ttk.Combobox(option_bar, values = l_area_types, width =15, state="readonly")
        geography_level.set('select')
        geography_level.pack(side = 'left', padx = 3)
        geography_level.bind("<<ComboboxSelected>>", pick_area)
        
        
# =============================================================================
#         pick the indicators that correspond to the area level that was given (i dont use this button it may be usefull in the future)
# =============================================================================
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
             years_val_sorted = sorted(years_val)
             year_list['values'] = years_val_sorted
        
        area_name_l = ttk.Label(option_bar, text= 'Area Name')
        area_name_l.pack(side = 'left', padx = 3)
        area_name_list = ttk.Combobox(option_bar, width =15, state="readonly")
        area_name_list.set('select')
        area_name_list.pack(side = 'left', padx = 3)
        area_name_list.bind("<<ComboboxSelected>>", pick_year)
        
# =============================================================================
#         May be used in the future
# =============================================================================
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
        
        
# =============================================================================
#         Tables frame
# =============================================================================
        table_frame = tk.Frame(self)
        
        
        cols = ('Indicator','Difference From Scotland','Year', 'Definition'  )
        listBox = ttk.Treeview(table_frame, columns=cols, show='headings')
        
        cols1= ['Indicator', 'Profile', 'Difference From Scotland','Definition', 'Year']
        listBox1 = ttk.Treeview(table_frame, columns=cols1, show='headings')
        
        cols2 = ('Indicator', 'Area Name', 	'Area Type', 	'Year', 	'Measure', 'Difference Between Areas' )
        listBox2 = ttk.Treeview(table_frame, columns=cols2, show='headings')
       
        label_top_k = tk.Label(table_frame, font=LARGE_FONT)
        label_top_profile = tk.Label(table_frame, font =LARGE_FONT)
        label2 = tk.Label(table_frame, font = LARGE_FONT)
        
# =============================================================================
#         Search function
# =============================================================================
        def search_m():
            
            geo = geography_level.get()
            area = area_name_list.get()
            indi = indicator_list.get()
            k = k_l_list.get()
            option = option_list.get()
            year = year_list.get()
            
            #deletes the previous data when a new search is done.
            listBox.delete(*listBox.get_children())
            listBox1.delete(*listBox1.get_children())
            listBox2.delete(*listBox2.get_children())
            
# =============================================================================
#             check if all the values have been selected
# =============================================================================
            if(geo == 'select' or area == 'select'  or k =='select' or option == 'select' or year =='select'):
                print('Select Valid Values')
                popupmsg('Enter Valid Values')
            else:
                print('we are ok')
                                             
# =============================================================================
# TOP K : calling the function and fill the table.    
# =============================================================================                
                
                label_top_k['text']= 'Top '+k+' '+option+ ' Indicators '+'of '+area+'@'+geo+ ' Compared to Scotland'
                label_top_k.pack(pady=2)
                
                for col in cols:
                    listBox.heading(col, text=col)    
            
            
                listBox.pack(pady=2)
                
                df_ = select_df(dataset_all_, str(geo), str(area), int(year))
                dfaki = top_k(df_, int(k), str(option))
                dfaki= dfaki.round(2)
                dfaki= dfaki[['indicator','year','actual_diff','definition']]
                dfaki = dfaki.values.tolist()
                print(dfaki[0])
                for i, (indicator, year, actual_diff, definition) in enumerate(dfaki, start=1):
                    listBox.insert("", "end", values=( indicator, actual_diff, year, definition))
                
               
# =============================================================================
# PER PROFILE   call the functions and fill the tables.           
# =============================================================================
            
                
                label_top_profile['text'] = 'Top '+str(option)+' Indicators per Profile of '+area+'@'+geo+ ' Compared to Scotland'
                label_top_profile.pack(pady=2)
              
                
                for col in cols1:
                     listBox1.heading(col, text=col)   
                     
                # rank_by_profiles returns a list of lists of dictionaries.
                #the List contains a list for every different profile.
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
                dfakos = dfakos.round(2)
                dfakos_ = dfakos[['Indicator', 'Profile', 'actual_diff', 'definition', 'year']].values.tolist()
        
        
                #dfakos_ contains a list of lists. every row of the dataframe is an 'inside' list.          
                for  i, (indicator, profile_name, actual_diff, definition, year) in enumerate(dfakos_, start=1):
                     listBox1.insert("", "end", values=(indicator, profile_name, actual_diff, definition, year))
                     
                listBox1.pack(pady=2)
# =============================================================================
# =============================================================================
# Similarity             
# =========================================================INDICATORS====================
                label2['text'] = "Top "+str(k)+' Similar indicators at '+area+'@'+geo+' Between Areas'
                label2.pack(pady=2)
      
          
                for col in cols2:
                    listBox2.heading(col, text=col)    
               

                _dfaki =  find_the_most_Similar(df_,int(k))
                _dfaki.round(3)
    
                list_dfaki = _dfaki[['indicator', 'area_name', 	'area_type', 	'year', 	'measure', 'sim_diff']].values.tolist()
          
                for i, (indicator, area_name, 	area_type, year, 	measure, sim_diff) in enumerate(list_dfaki, start=1):
                    listBox2.insert("", "end", values=( indicator, area_name, area_type, year, measure, sim_diff))
                    
                listBox2.pack(pady=2)

        
       
        def clear():
            print('clear')
            listBox.delete(*listBox.get_children())
            listBox1.delete(*listBox1.get_children())
            listBox2.delete(*listBox2.get_children())
            listBox.pack_forget()
            listBox1.pack_forget()
            listBox2.pack_forget()
            label_top_profile.pack_forget()
            label2.pack_forget()
            label_top_k.pack_forget()
            
        #K
        k_l = ttk.Label(option_bar, text= 'K')
        k_l.pack(side = 'left', padx = 3)
        k_l_list = ttk.Combobox(option_bar, values = list(range(1,21)), width =15, state="readonly")
        k_l_list.pack(side = 'left', padx = 3)
        k_l_list.set('select')
        
        
        #option
        option_l = ttk.Label(option_bar, text= 'Option')
        option_l.pack(side = 'left', padx = 3)
        option_list = ttk.Combobox(option_bar, values = ['better','worse'], width =15)
        option_list.pack(side = 'left', padx = 3)
        option_list.set('select')
        
       
        
        #help button function
        def explain():
            popupmsg('This page depicts the TOPK and similarities results.\nThe second table depicts the TOPK indicators per profile(Health...tobacco..etc).\nK depicts the number of indicators to be retrieved.\nOption means when the indicators perform better or worse than the Comparator(Scotland).\nPlease select the values from the left to the right.')
        
        search_button = ttk.Button(option_bar, text= 'SEARCH', command = search_m)
        search_button.pack(side = 'left', padx = 3)
        clear_button = ttk.Button(option_bar, text= 'CLEAR', command = clear)
        clear_button.pack(side = 'left', padx = 3)
        help_button = ttk.Button(option_bar, text= 'HELP', command = explain)
        help_button.pack(side = 'left', padx = 3)
        
        
        option_bar.pack()
        table_frame.pack()
        
        
        

#Initialise a mainapp object and run the app.
app = mainapp()
app.mainloop()