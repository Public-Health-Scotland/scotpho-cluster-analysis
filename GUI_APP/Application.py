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
        for F in (StartPage, PageOne, PageTwo, PageThree):
            frame = F(container,self)
            self.frames[F] = frame
            frame.grid(row = 0, column =0, sticky = 'nsew')
        
        self.show_frame(StartPage)
        
    def show_frame(self, cont):
            frame = self.frames[cont]
            frame.tkraise()
        

def default(param):
    print(param)

class StartPage(tk.Frame):
    
    def __init__(self,parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text = 'Start Page', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        button1 = ttk.Button(self, text = 'Clustering',command= lambda: controller.show_frame(PageOne))  
        button1.pack()
        
        button2 = ttk.Button(self, text = 'TopK',command= lambda: controller.show_frame(PageTwo))  
        button2.pack()
        
        button3 = ttk.Button(self, text = 'Graph',command= lambda: controller.show_frame(PageThree))  
        button3.pack()

class PageOne(tk.Frame):
    
    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self,parent)
        label = ttk.Label(self, text = 'Clustering', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
# =============================================================================
# menu
# =============================================================================
        menu_bar = tk.Frame(self)
        
        button1 = ttk.Button(menu_bar, text = 'Home',command= lambda: controller.show_frame(StartPage))
        button1.pack(side = 'left', padx = 5, pady=5)
        
        button2 = ttk.Button(menu_bar, text = 'TOPK',command= lambda: controller.show_frame(PageTwo)) 
        button2.pack(side = 'left')
        
        menu_bar.pack()
        
# =============================================================================
#  option list       
# =============================================================================
        dataset_all = pd.read_csv("scotpho_data_extract.csv")
        area_types = dataset_all['area_type'].unique()[:-1]
        l_area_types = list(area_types)
        
        dataset_all_full = pd.read_csv('dataset_all.csv')
        dataset_all_full.drop(columns=['Unnamed: 0'],inplace= True)
        #remove the estimation since we can't compare the values of an area to the sum of population.
        dataset_all_ = dataset_all_full[(dataset_all_full.indicator !='Mid-year population estimate - all ages') & (dataset_all_full.indicator != 'Quit attempts')]
        
        prof_ind = pd.read_excel('Profiles to indicators.xlsx')
        
        def pick_area(event):
            area_name_list['values'] =[]
            area_name_list.set('select')
            indicator_list['values'] = []
            indicator_list.set('select')   
            area_level_selected = geography_level.get()
            area_n = dataset_all[dataset_all['area_type'] == area_level_selected]['area_name'].unique()
            area_val = list(area_n)
            area_name_list['values'] = area_val
        
        
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

        def cluster():
            
        
        search_button = ttk.Button(option_bar, text= 'SEARCH', command = cluster)
        search_button.pack(side = 'left', padx = 3)
        option_bar.pack()

class PageTwo(tk.Frame):
    
    def __init__(self, parent, controller):
        
        
        
        #building the frame
        
        tk.Frame.__init__(self,parent)
        
        
      

        
        label = ttk.Label(self, text = 'TopK', font = LARGE_FONT)
        #label.grid(row=1, column = 1)
        label.pack()
        
        menu_bar = tk.Frame(self)
        
        button1 = ttk.Button(menu_bar, text = 'Home',command= lambda: controller.show_frame(StartPage))
        #button1.grid(row=2, column = 0, sticky='W')
        button1.pack(side = 'left', padx = 5, pady=5)
        
        button2 = ttk.Button(menu_bar, text = 'Clustering',command= lambda: controller.show_frame(PageOne)) 
        #button2.grid(row=2, column = 1, sticky= 'WNS')
        button2.pack(side = 'left')
        menu_bar.pack()
        #creating the lists
        
        dataset_all = pd.read_csv("scotpho_data_extract.csv")
        area_types = dataset_all['area_type'].unique()[:-1]
        l_area_types = list(area_types)
        
        dataset_all_full = pd.read_csv('dataset_all.csv')
        dataset_all_full.drop(columns=['Unnamed: 0'],inplace= True)
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
            
            if(geo == 'select' or area == 'select'  or k =='select' or option == 'select'):
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
        
        
        search_button = ttk.Button(option_bar, text= 'SEARCH', command = search_m)
        #search_button.grid(row= 8, column = 1, rowspan=2, sticky = 'W')
        search_button.pack(side = 'left', padx = 3)
        option_bar.pack()
        table_frame.pack()
        
        
        
class PageThree(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = ttk.Label(self, text = 'Graph Page', font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        
        button1 = ttk.Button(self, text = 'Home',command= lambda: controller.show_frame(StartPage))
        button1.pack()
        
        f= Figure(figsize =(5,5), dpi=100)
        a= f.add_subplot(111)
        a.plot([1,2,3,4,5],[6,7,8,9,10])
        
        canvas = FigureCanvasTkAgg(f,self)
        canvas.draw()
        canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        
        toolbar = NavigationToolbar2Tk(canvas,self)
        toolbar.update()
        canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        

app = mainapp()

app.mainloop()