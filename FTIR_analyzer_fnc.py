'''
__author__ = Dr. Giada Innocenti \
__maintainer__ = Dr. Giada Innocenti  \
__email__ = giadainnocenti.gi@gmail.com \
__Python__ = 3.8.5

'''


#resolution used for every experiment
resolution = 4
# type of computer
def computer(computer_type):
    if computer_type.casefold() == 'windows':
        computer_root='C:\\'
    else:
        computer_root='/'
    return(computer_root)

##### set of functions to read the CSV file
import os
import matplotlib.pyplot as plt 
import fnmatch
import numpy as np
import matplotlib
from matplotlib import cm
import re

from numpy.lib import index_tricks

# sort the name of files ending with a number
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]  
# select the directory of interest
def choose_directory(dr):
    for root, dirs, files in os.walk(dr):     
        for file in files:  
        # change the extension from '.mp3' to  
        # the one of your choice. 
            if file.endswith('.CSV'): 
                dr = root

    os.chdir(dr)

    if any(File.endswith(".CSV") for File in os.listdir(".")):
        print("I found CSV data")
    else:
        print("Check if the *.SPA files were converted into *.CSV")
    print(os.getcwd())
    return(dr)
# read the CSV files of interest excluding the ones ending with '_raw.CSV'    
def file_reader(namepart, dr):
    fname = []
    for filename in os.listdir(dr):
        if fnmatch.fnmatch(filename, namepart):
            if not fnmatch.fnmatch(filename, '*_raw.CSV'):
                print(filename)
                fname.append(filename)
    return(fname)

BIG_FONT = 28
MEDIUM_FONT = 24
SMALL_FONT = 20
# plts the files shwoing also the background as dashed line.
def one_cat_plot(W,A, BG = None, XMIN=None, XMAX=None, YMIN=None, YMAX=None, TLT=None, LBL=None,LGDHEIGHT=None):
    NF=np.size(W,0)
    fig,ax = plt.subplots(figsize = [10,8])
    try:
        if BG.any() != None and BG.any != False:
            plt.plot(BG[:,0], BG[:,1], ls=':', label = 'background')
    except:
        pass
    if NF < 2:
        color = ['black']
    else:
        color=cm.rainbow(np.linspace(0,1,NF))
        c_m = matplotlib.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=0,vmax=NF)
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

    plt.setp(ax.spines.values(), linewidth=1.5)
    try:
        for j,c in zip(range(0,NF,1),color):
            try:
                ax.plot(W[j,:],A[j,:], c=c, label = str(LBL[j]))
            except:
                if np.shape(W)[0] == 1:
                    ax.plot(W[0,:],A[0,:])
                elif len(LBL) == 0:
                    print(f'Please control that you inserted all the necessary information to buil a time/temperature array')
                    raise
                else:
                    print(f'absorbance and/or wavenumber information missing at the cycle{j}')
                    raise
    except:
        print('Double check the file name')
            
            
    ax.set_ylabel( "Absorbance (AU)",fontsize=BIG_FONT)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)",fontsize=BIG_FONT)
    ax.set_title(TLT,fontsize=BIG_FONT)
    if XMAX !=None and XMIN !=None:
        ax.set_xlim(XMAX,XMIN)
    if YMAX !=None and YMIN !=None:
        ax.set_ylim(YMIN, YMAX)
    ax.tick_params(labelsize=MEDIUM_FONT)
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)
    try:
        if LBL.any() != None:
            if len(LBL)> 5:
                if LGDHEIGHT == None:
                    LGDHEIGHT = -0.5
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, LGDHEIGHT), ncol = 10, fontsize=SMALL_FONT)
            else:
                 ax.legend(fontsize=SMALL_FONT)
    except:
        print('I cannot write a legend')
    
    ax.axes.ticklabel_format(axis='both', style='plain')


    plt.tight_layout()
    return plt.show()

import sys 
import pandas as pd
#read the experimental data files.
def Rexp(experiment_folder_path,  key_word_for_all_the_files, sampling=None, UOM=None, file_to_subtract=None, plot = None, subtraction = None, start_T = None):
    ### CHOOSING THE CORRECT DIRECTORY TO READ THE DATA FILES:
    ddr = choose_directory(experiment_folder_path)
    #READING THE 'BACKGROUND' FILE. IT CONTAINS THE BARE CATALYST SURFACE BEFORE THE DOSING OF ANY MOLECULE
    try:
        if (file_to_subtract != None) and (file_to_subtract != ''):
            bkgstart = '*'+file_to_subtract+'*.CSV'
            bkgname = file_reader(bkgstart, ddr)
            df = pd.read_csv(bkgname[0])
            bkg = np.array(df)
            print(np.shape(bkg))
            background = True
        else:
            background = False
    except:
        background=False
        print(f'I did not find any CSV file containing {bkgstart}')
    # READING THE EXPERIMENT DATA FILE #
    expstart = '*'+key_word_for_all_the_files+'*.CSV'
    filenamearr = list(file_reader(expstart, ddr))     
    nf=len(filenamearr)
    
    if nf == 0:
        print('CONTROL THE FILE NAME, NO FILE WAS FOUND WITH THAT KEYWORD')
        sys.exit()
    else:
        print('NUMBER OF FILES: ', nf) 
        filenamearr.sort(key=natural_sort_key)
    try:    
        if start_T == None:
            if UOM.casefold() == 'c':
                start_T = 50. 
            elif UOM.casefold() == 'k':
                start_T = 50+273.15
    except:        
        print("It is not possible to set the initial temperature since I have no information on the unit of measure desired")
            
    try:    
        if UOM.casefold() == 's':
            divisor = 1.
        elif UOM.casefold() == 'min':
            divisor = 60.
        elif UOM.casefold() == 'h':
            divisor = 3600.
        elif UOM.casefold() == 'days':
            divisor =86400.
        if UOM.casefold() != 'c' and UOM.casefold() != 'k':    
            time_array = np.arange(0,nf*float(sampling)/divisor, float(sampling)/divisor)
        elif UOM.casefold() == 'c' or UOM.casefold() == 'k':
            time_array = np.arange(float(start_T),float(start_T)+nf*float(sampling), float(sampling))
    except:
        print('something went wrong, I probably miss the sampling information to build the time array')
        time_array = []
    
        
    print ( 'I sorted your elements this way: ')
    for i in range(len(filenamearr)):
        print(filenamearr[i])
    
    wavenumbers = []
    absorbance = []
    for i in range(len(filenamearr)):
        df = pd.read_csv(filenamearr[i])
        wavenumbers.append(df.iloc[:,0])
        absorbance.append(df.iloc[:,1])

    absorbance = np.array(absorbance)
    wavenumbers = np.array(wavenumbers)
    print('wn shape: ', np.shape(wavenumbers))
    ##########PLOT TO CHECK THAT EVERYTHING IS ALRIGHT ########
    if plot != None and plot != False:
        #one_cat_plot(W,A, BG = None, XMIN=None, XMAX=None, YMIN=None, YMAX=None, TLT=None, LBL=None, LGDHEIGHT=None):
        if not background:
            bkg = background
        one_cat_plot(wavenumbers,absorbance,bkg, LBL=time_array, LGDHEIGHT=-1)

    ######### SUBTRACTION OF THE BARE CATALYST SURFACE FROM THE SPECTRA CONTAINING ALSO THE ADSORBED SPECIES
    if subtraction != None and background != False:
        for i in range(nf):
            absorbance[i,:] = absorbance[i,:]-bkg[:,1]

    print('final absorbance array shape: ', np.shape(absorbance))
    return(wavenumbers, absorbance, time_array)

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse

#### OUTLIER REMOVAL
def outlier_removal(absorbance, wavenumber, time_array, stdv_value = None, print_stdv_list = False):
    stdv = []
    time = list(time_array)
    for x, t in enumerate(time):
        stdv.append(np.std([absorbance[time.index(t),:]]))
    if print_stdv_list:
        print(stdv)
    print(f'maximum standard deviation {max(stdv)}')
    print(f'Mean standard deviation {np.mean(stdv)}')
    print(f'Median standard deviation {np.median(stdv)}')
    if type(stdv_value) == float or type(stdv_value) == int: 
        print(f'I am selecting all the data with a standard deviation higher than {stdv_value}')
        idx = np.where(np.array(stdv)>stdv_value)
    elif stdv_value == 'median':
        print(f'I am I am selecting all the data with a standard deviation higher than median of the standard deviation {np.median(stdv):.3f}')
        idx = np.where(np.array(stdv)>np.median(stdv))
    else:
        print(f'I am selecting all the data with a standard deviation higher than median of the standard deviation {np.mean(stdv):.3f}')
        idx = np.where(np.array(stdv)>np.mean(stdv))
    masked_wn = np.delete(wavenumber,idx,0)
    masked_ab = np.delete(absorbance,idx,0)
    masked_time = np.delete(time_array, idx)
    #import matplotlib.pyplot as plt
    #for x, di in enumerate(idx):
    #   plt.plot(wn[di,:],ab[di,:])
    #print(np.shape(masked_ab))
    print(f'I dropped a total of {np.size(wavenumber,0)-np.size(masked_wn,0)} spectra,\ninitial  number of spectra read = {np.size(wavenumber,0)}\nfinal number of spectra kept {np.size(masked_wn,0)}') 
    return(masked_ab, masked_wn, masked_time)

#### BASELINE CORRECTION - Asymmetric Least Squares baseline correction - ALS
#these two values should be the optimized ones for FTIR spectra
#p for asymmetry and λ for smoothness. Both have to be tuned to the data at hand. 
#Generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 
#10^2 ≤ λ ≤ 10^9 , but exceptions may occur.
# In any case one should vary λ on a grid that is approximately linear for log λ.
def baseline_als(y, lamda= None, pp = None, niter=20):
    if lamda == None:
        lamda = 10**6
    if pp == None:
        pp = 0.05
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lamda * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = pp * (y > z) + (1-pp) * (y < z)
    return z

#### SELECT ONLY A SPECIFIC RANGE OF WAVENUMBERS OF INTEREST.
def wn_selector(wavenumbers,wn_min,wn_max):
    idmin = list(np.where(np.isclose(wavenumbers[0,:], [wn_min], atol = 1.0, rtol = 0)))
    #print(idmin)
    idmax = list(np.where(np.isclose(wavenumbers[0,:], [wn_max], atol = 1.0, rtol = 0)))
    #print(idmax)
    return np.arange(idmin[0][0], idmax[0][0], 1)


from scipy.signal import find_peaks, savgol_filter
def find_peak_intensity(absorbance, distance=resolution, height_threshold=None, prominence=None, width=None, rel_height=None):
    if  height_threshold is None:
        height_threshold = 100
    if prominence is None:
        prominence = 0.01
    if width is None:
        width = 1.0
    if rel_height is None:
        rel_height = 0.3
    peaks,_ = find_peaks(absorbance, distance = distance, height = np.max(absorbance)/height_threshold, prominence = prominence,\
                         width = width,rel_height=rel_height)
    
    return peaks

def max_abs_finder(absorbance, peak_index, wavenumber):
    ix = 0
    while ix < 40:
            if peak_index >= len(absorbance)-1:
                peak_opt = peak_index
                break
            elif absorbance[peak_index] < absorbance[peak_index-1]:
                #print(f'absorbance[index] < absorbance[index-1], index =',peak_index, 'wavenumber = ',wavenumber[peak_index])
                peak_index = peak_index-1
                peak_opt = peak_index
                ix = ix+1
            elif absorbance[peak_index] < absorbance[peak_index+1]:
                #print('absorbance[index] < absorbance[index+1], index =',peak_index, 'wavenumber = ',wavenumber[peak_index])
                peak_index = peak_index+1
                peak_opt = peak_index
                ix = ix+1
            elif absorbance[peak_index] >= absorbance[peak_index-1] and absorbance[peak_index] >= absorbance[peak_index+1]:
                #print('index =',peak_index, 'wavenumber = ',wavenumber[peak_index])
                peak_opt = peak_index
                break
    return peak_opt


from datetime import datetime
def colorbarr(clr, array, title_name):
    clr.ax.get_yaxis().set_ticks(array, len(array))
    clr.ax.tick_params(labelsize =MEDIUM_FONT) 
    clr.ax.get_yaxis().labelpad = SMALL_FONT
    clr.ax.set_ylabel(title_name, rotation=270, fontsize=MEDIUM_FONT)
    return()
# function to plot mostly all the plots in the main script
def FTIR_plot(W,A, X0, X1, TLT, T, BARTL = None, ylegend = None, figname=None,  y_min = None,\
     y_max = None, X2 = None, X3 = None, text_height_p2=None, text_height_p1=None,\
     text_alignment_p1=None,text_alignment_p2=None, wn_peaks_df=None,ab_peaks_df=None):

    axs = plt.figure(figsize = [15,7]).subplots(1, 2)
    
    a = 0
    for ax in axs:
        color=cm.rainbow(np.linspace(0,1,np.size(W,0)))
        c_m = matplotlib.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin=T[0],vmax=T[len(T)-1])
        s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])

        plt.setp(ax.spines.values(), linewidth=1.5)
        if a == 1 and X0==X1:
            ax.set_axis_off()
            # it would be nice to be able to change the size of the subplot to equal the size of the figure.
            break

        for j,c in zip(range(0,np.size(W,0),1), color):               
            ax.plot(W[j,:],A[j,:], c=c)
        try:
            for col in wn_peaks_df.columns:        
                ax.plot(wn_peaks_df[col],ab_peaks_df[col], ls='none', marker='*')
        except:
            pass
        try:
            count=0
            for index in wn_peaks_df.columns:
                wavenumber = wn_peaks_df[index].median()
                ############### AX2 = 2000 - 1300 cm-1 ################
                if (wavenumber > X0 and wavenumber < 1800.) and a==1: 
                    ax.text(wavenumber, text_height_p2[count], str(np.int(wavenumber)), \
                        va ='center', ha = text_alignment_p2[count], fontsize =SMALL_FONT)   
                    count = count+1
                ############### AX1 = 4000 - 2500 cm-1 ################
                if (wavenumber > 2725. and wavenumber < 3050.) and a==0:
                        #print(wavenumber)
                    ax.text(wavenumber, text_height_p1[count], str(np.int(wavenumber)), \
                        va ='center', ha = text_alignment_p1[count], fontsize =SMALL_FONT )
                    count = count+1
        except:
            pass

        ax.set_ylabel( "Absorbance (AU)",fontsize=BIG_FONT)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)",fontsize=BIG_FONT)
        ax.set_title(TLT,fontsize=(20))
        
            
        if a == 0:
            if X3 !=None and X2 != None:
                ax.set_xlim(X3,X2)
            elif X2 != None and X3 == None:
                ax.set_xlim(W[0,-1:],X2)
            elif X3 !=None and X2 == None:
                ax.set_xlim(X3,X1)
            else:
                ax.set_xlim(W[0,-1:],X1)
        elif a == 1:
            ax.set_xlim(X1, X0)
        if y_min != None and y_max !=None:
            ax.set_ylim(y_min,y_max)
            
        ax.tick_params(labelsize=MEDIUM_FONT)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.axes.ticklabel_format(axis='both', style='plain')
        if BARTL != None:
            BARTLC=BARTL.casefold()
        else:
            BARTLC=BARTL
        if ylegend != None:
            ax.legend(T, loc='lower center', bbox_to_anchor=(0.5, ylegend), ncol = 10, fontsize=SMALL_FONT)   
        elif BARTLC == 'legend' or BARTL == None:
            ax.legend(T, loc='best', fontsize=SMALL_FONT)
        elif BARTLC != 'legend' or BARTL != None:
            if X0 == X1 and a == 0:
                cbar = plt.colorbar(s_m, ax = axs[0]) 
                colorbarr(cbar, T, BARTL)
            elif a == 1:
                cbar = plt.colorbar(s_m)
                colorbarr(cbar, T, BARTL)
        a = a+1
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
   
    try:
        figurename = figname+'.png'#now())+'.png'
        plt.savefig(figurename)
    except:
        print('I did not save this image as png')
    return plt.show()
#### PLOTTING THE intensity or area/concentration trends with respect to the time
def plot_time_trends(time,ab_df,wn_label,xlabel=None,ylabel=None,line = None, marker = None, smooth=None,ab_label=None, figname = None):
    fig,ax = plt.subplots()
    plt.setp(ax.spines.values(), linewidth=1.5)
    for index,col in enumerate(wn_label.columns):
        if ab_label:
            col_ab = ab_label[index]
        else:
            col_ab = col 
        if smooth:
            try:
                smoothed_ab = savgol_filter(ab_df[col_ab].fillna(0.),window_length =7, polyorder=1)
                print('Dataset smoothed with a window length of 7')
            except:
                print(' It was not possible to use 7 as window length, testing 5...')
                try:
                    smoothed_ab = savgol_filter(ab_df[col_ab].fillna(0.),window_length =5, polyorder=1)
                    print('Dataset smoothed with a window length of 5')
                except:
                    print(' It was not possible to use 5 as window length, testing 3...')
                    try:
                        smoothed_ab = savgol_filter(ab_df[col_ab].fillna(0.),window_length =3, polyorder=1)
                        print('Dataset smoothed with a window length of 3')
                    except:
                        print('It is not possible to smooth three or less data points')
                        smooth = None
        
        if smooth == None or smooth == False:
            smoothed_ab = ab_df[col_ab].fillna(0.)
        
        if marker == None and line == None:
            ax.plot(time, smoothed_ab,label ='{:.0f}'.format(wn_label[col].median()))
        else:
            if line == None:
                line = ' '
            ax.plot(time, smoothed_ab,label ='{:.0f}'.format(wn_label[col].median()), ls = line, marker=marker,ms = 10)
    ax.legend(fontsize=SMALL_FONT)
    if ylabel:
        ax.set_ylabel( ylabel, fontsize=BIG_FONT)
    if xlabel:
        ax.set_xlabel(xlabel,fontsize=BIG_FONT)
    ax.tick_params(labelsize=MEDIUM_FONT)
    ax.xaxis.set_tick_params(width=1.5)
    ax.yaxis.set_tick_params(width=1.5)
    ax.axes.ticklabel_format(axis='both', style='plain')
    if figname != None:
        plt.tight_layout()
        plt.savefig(figname, dpi = 300, transparent = True)
    return plt.show()


def break_without_breaking_script(string1,string2):
    try:
        sys.exit()
    except:
        print(f'The {string1} format is not correct, please double check the keyword {string2}.')
        print(f'The are two possible problems:')
        print(f'1. {string2} is set to None or is a string. The format required is float or int')
        print(f'2. {string2} is not in your time or temperature list, please double check that the inserted number is correct')
    return()

########## SUBTRACTING AND PLOTTING
#it can subtract 1 spectrum from another or 1 spectrum from a list of other spectra.
def subtract_one_spectrum(absorbance, wavenumber, subtrahend, minuends, timeframe, x_min = None, x_max = None, y_min = None, y_max = None):
    if subtrahend == None or type(subtrahend) == str or (subtrahend in timeframe) == False:
        break_without_breaking_script('subtrahend', 'time_base')
        
    elif minuends == None or type(minuends) == str or (all(x in timeframe for x in minuends)) == False:
        break_without_breaking_script('minuend', 'time_list')
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        plt.setp(ax.spines.values(), linewidth=1.5)
        plt.tick_params(axis='both',width=1.5,labelsize=MEDIUM_FONT)
        matplotlib.rcParams['lines.linewidth'] = 1.5
        timeframe = list(timeframe)
        ab_sub = []
        wn_idx = []
        for idx, t in enumerate(minuends):
            ab_sub.append(absorbance[timeframe.index(t),:]-absorbance[timeframe.index(subtrahend),:])
            wn_idx.append(timeframe.index(t))
            plt.plot(wavenumber[wn_idx[idx],:], ab_sub[idx], label = f'{t} - {subtrahend}')
        if x_max != None and x_min != None:    
            plt.xlim(x_max,x_min)
        else:
            plt.xlim(4000,1300)
        if y_min != None and y_max != None:
            plt.ylim(y_min,y_max)
        plt.xlabel('Wavenumber (cm$^{-1}$)',fontsize=BIG_FONT)
        plt.legend(fontsize=SMALL_FONT)
        plt.ylabel('Absorbance (AU)',fontsize=BIG_FONT)
        plt.show()
        return(ab_sub, wn_idx )

# performs the first and second derivative tests to find maxima and inflection points
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import argrelextrema
def derivative_tests(derivative, wavenumber, inter_points, second_derivative=None, atol = None, plot = None):
    if not atol:
        atol = np.median(derivative)/10
    y = derivative
    x = wavenumber
    #increasing the virtual resolution of the wavenumbers
    xi = np.linspace(wavenumber[0], wavenumber[-1], inter_points)
    #interpolation function
    ius = InterpolatedUnivariateSpline(x, y)
    # increasing the virtual resolution by interpolating
    yi = ius(xi)
    # finding where y is equal to zero with the increased resolution
    idx_0 = np.where(np.isclose(yi,0., atol = atol,rtol=0))
    #plotting to check the results
    if plot:
        plt.plot(x,y, ls =' ', marker = 'x')
        plt.plot(xi,yi,ls ='-', color='green')
    #saving the indexes where the derivative is zero
    wn_0 = []
    wnd_idx_0 = []
    for wn_idx_0 in xi[idx_0]:
        idx_list = list(np.where(np.isclose(x,wn_idx_0, atol = 1.0, rtol=0))[0])
        if x[idx_list[0]] not in wn_0:
            wn_0 += [x[idx_list[0]]]
            wnd_idx_0 += [idx_list[0]]
     
    #second derivative test to find only minimum or inflection points
    if second_derivative:
        # retrieving the indexes where the second derivative is in a minimum
        minima = list(argrelextrema(y, np.less)[0])
        # adding those indexes to the ones where it was zero.
        wnd_idx_0.extend(minima)

        # removing all the indexes that are bigger than the virtual zero chosen.
        wnd_idx_d2 = []
        wn_control = []
        for idx in wnd_idx_0:
            if y[idx] < atol and x[idx] not in wn_control: # atol was used to define what to consideras a zero value.
                wnd_idx_d2 += [idx]
                wn_control += [x[idx]]
        wnd_idx_0 = wnd_idx_d2

        color = 'black'
        alpha = 0.5
    else:
        color = 'gray'
        alpha=1
    if plot:
        plt.plot(x[wnd_idx_0], y[wnd_idx_0], ls = '', marker = '*', color=color, markersize=10, alpha=alpha)
    return wnd_idx_0

#converts a string to a list of numbers
def string_to_float(string):
    int_list = string.replace(',',' ').split()
    # print list
    print('list: ', int_list)
    # convert each item to float type
    for i in range(len(int_list)):
    # convert each item to int type
        int_list[i] = float(int_list[i])
    return int_list

#it is finding the peaks by using the first and second derivative tests or you can select the peaks.
def peak_finder_derivative(wavenumber,absorbance, tolerance, fit_wns, plot_title = None, user_interaction=None ):
    indici = []
    Y = wavenumber#wn[20, fit_idx]
    X1 = savgol_filter(absorbance, window_length=9, polyorder=2, deriv=1)
    X2 = savgol_filter(absorbance, window_length=9, polyorder=2, deriv=2)
    #print(X1)
    
    #derivative_tests(derivative, wavenumber, inter_points, plot = None)
    d1_test_idx = derivative_tests(X1,Y,1000000,second_derivative=False, atol = None,plot=False)
    d2_test_idx = derivative_tests(X2,Y,1000000,second_derivative=True, atol = None,plot=False)

    for idx in d1_test_idx:
        #print(Y[idx])
        if X1[idx] == X1[-2]:
            break
        if (Y[idx]> tolerance[0] and Y[idx]<tolerance[1]) and\
            (idx in d2_test_idx or\
            idx+1 in d2_test_idx or idx-1 in d2_test_idx or\
            idx+2 in d2_test_idx or idx-2 in d2_test_idx or\
            idx+3 in d2_test_idx or idx-3 in d2_test_idx or\
            idx+4 in d2_test_idx or idx-4 in d2_test_idx or\
            idx+5 in d2_test_idx or idx-5 in d2_test_idx): 
                print(f'I am saving your index at {Y[idx]} with 1st derivative = {X1[idx]} and second = {X2[idx]}')
                indici +=[idx]
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax[0].plot(Y, absorbance ,'-b', label = 'normal form')#ab[20,fit_idx]
    ax[0].plot(Y[indici], absorbance[indici], ls = 'None', marker='*', color = 'black')
    ax[1].plot(Y[indici], X1[indici], ls = 'None', marker='*', color = 'green')
    ax[1].plot(wavenumber,X1, label = '1$^{st}$ derivative', color = 'red')
    ax[1].plot(Y,X2, '-r', label = '2$^{nd}$ derivative', color = 'orange' )
    ax[1].plot(Y[indici], X2[indici], ls = 'None', marker='*',color = 'brown', alpha=0.8)
    ax[1].axhline(y=0.0, color='k', linestyle='--')
    if plot_title != None:
        ax[0].set_title(plot_title)
    ax[1].legend()
    for n in [0,1]:
        ax[n].set_xlim(fit_wns[0], fit_wns[-1])
    plt.show()
    Y_out = list(Y[indici])    
    if user_interaction:
        print("Welcome in interactive mode! I am still in development so please be patient with me.")
        print("You will be guided through the process of adding or removing peaks. Enjoy!")
        while user_interaction:
            print(f'These are the wavenumber positions I stored {Y_out}.\nWould you like to add something or to remove? ')
            switch = input()
            
            if switch[0].casefold() == 'r':
                while switch[0].casefold() == 'r':
                    rm_wn = input('Please tell me what wavenumber I should remove using all the decimals ')
                    try:
                        rm_wn = string_to_float(rm_wn)
                        if len(rm_wn)>1:
                            for wavenb in rm_wn:
                                if wavenb in Y_out:
                                    Y_out.remove(wavenb)
                    except:
                        print(f' I am sorry you typed {rm_wn} while I need a number')
                    answer = input('Do you want to remove anything else?')
                    try:
                        if answer[0] == 'y':
                            switch = 'remove'
                        else:
                            break
                    except:
                        continue
            elif switch[0].casefold() == 'a':
                add_wn = input ('what wavenumber(s) would you like to add? ')
                wn_list = string_to_float(add_wn)
                Y_out.extend(wn_list)
            else:
                print(f'These are the wavenumber positions I stored {Y_out}.\nDo you need to do anyting else?')
                user_input = input()
                if user_input == 'n' or user_input == 'no'or user_input == '':
                    user_interaction = False

    return(Y_out)

# convert the user answer into Boolean
def user_answer(usr_inpt):
    if usr_inpt[0] == 'y':
        condition = True
    else:
        condition = False
    return condition

# control of the user interaction
def user_interactive_mode(user_check, user_decision, time):
    if user_check:
            print(f'Do you want to check the time {time+1} min or do you want me to use the same wavenumber initially picked? ')
            try:
                user_decision = user_answer(input())
            except:
                print('Sorry, I did not understand your answer')
            if user_decision == False:
                print('Do you want to check any other time?')
                try:
                    user_check = user_answer(input())
                except:
                    print('Sorry, I did not understand your answer')
    return user_check, user_decision

def variable_control(variable, default_value, reference, variable_name):
    if variable == None:
        variable = [default_value]*len(reference)
    elif type(variable) == int or type(variable) == float or type(variable) == str:
        variable = [variable]*len(reference)
    elif len(variable)< len(reference):
        print(f'{variable_name} has less entries than the number of peaks you are trying to integrate, the values you entered are repeated in the order a,b,a,b,a,b, ... for all the peaks')
        variable = variable*len(reference)
    return(variable)

time_counter = 0 

def select_distribution(peak_shape, mod_prefix, gauss, idx):
    # importing the model to fit the peak shape according to the user choice   
    if str(peak_shape[idx]).casefold() == "gaussian":
        from lmfit.models import GaussianModel
        mod_prefix += ['g'+str(idx)]
        gauss += [GaussianModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == "lorentzian":
        from lmfit.models import LorentzianModel
        mod_prefix += ['l'+str(idx)]
        gauss += [LorentzianModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == "voigt" or str(peak_shape[idx]).casefold() == "voight":
        from lmfit.models import VoigtModel
        mod_prefix += ['v'+str(idx)]
        gauss += [VoigtModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == 'doniachmodel':
        from lmfit.models import DoniachModel
        mod_prefix += ['d'+str(idx)]
        gauss += [DoniachModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == 'skewedvoigtmodel':
        from lmfit.models import SkewedVoigtModel
        mod_prefix += ['sv'+str(idx)]
        gauss +=[SkewedVoigtModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == 'pseudo-voigt':
        from lmfit.models import PseudoVoigtModel
        mod_prefix += ['pv'+str(idx)]
        gauss += [PseudoVoigtModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == 'splitlorentzianmodel':
        from lmfit.models import SplitLorentzianModel
        mod_prefix += ['sl'+str(idx)]
        gauss += [SplitLorentzianModel(prefix=f'{mod_prefix[idx]}_')]
    elif str(peak_shape[idx]).casefold() == 'exponentialgaussianmodel':
        from lmfit.models import ExponentialGaussianModel
        mod_prefix += ['eg'+str(idx)]
        gauss += [ExponentialGaussianModel(prefix=f'{mod_prefix[idx]}_', nan_policy = 'omit')]
    elif str(peak_shape[idx]).casefold() == 'skewedgaussianmodel':
        from lmfit.models import SkewedGaussianModel
        mod_prefix += ['sg'+str(idx)]
        gauss += [SkewedGaussianModel(prefix=f'{mod_prefix[idx]}_')]
    return(mod_prefix, gauss)

def select_baseline(baseline_type):
    if str(baseline_type).casefold() == 'linear':
        from lmfit.models import LinearModel
        prefix = 'lin_'
        bl_mod = LinearModel(prefix=prefix)
    elif str(baseline_type).casefold() == 'polynomial':
        from lmfit.models import PolynomialModel
        prefix = 'pol_'
        bl_mod = PolynomialModel(prefix=prefix)
    elif str(baseline_type).casefold() == 'exponential':
        from lmfit.models import ExponentialModel
        prefix = 'exp_'
        bl_mod = ExponentialModel(prefix=prefix)
    return(prefix, bl_mod)

# wavenumber: range of wavenumbers to perform the peak fitting
# absorbance: experimental absorbance of the IR spectrum
# peaks: list containing the curve center to perform the fit
# peak_shape : user_defined function, if none the program will use a pseudo-voight function
# baseline_type: user_defined function, in None the program will use an exponential baseline
time_counter = 0
def peak_fitting(wavenumber, absorbance, peaks, peak_shape=None, baseline_type=None, peak_center_min = None, peak_center_max = None, area_min = None, area_max = None, sigma_min = None, sigma_max = None, variable_OI= None, time_list=None,UOM = None, resolution = resolution):
    global time_counter 
    log_file = open('fit_log.txt', mode='a')
    print(f'Peak fitting for {variable_OI} {time_list[time_counter]} {UOM}')
    print(f'Peak fitting started at {datetime.now()}')
    log_file.write(f'Peak fitting for {variable_OI} {time_list[time_counter]} {UOM}')
    log_file.write(f'Peak fitting started at {datetime.now()}')

    #controlling that all the variables are in place
    peak_shape = variable_control(peak_shape,'pseudo-voigt',peaks,'peak_shape')
    peak_center_min = variable_control(peak_center_min, resolution*4, peaks, 'peak_center_min')
    peak_center_max = variable_control(peak_center_max, resolution*4, peaks, 'peak_center_max')
    area_min = variable_control(area_min, 0, peaks, 'area_min')
    area_max = variable_control(area_max, 10000000, peaks, 'area_max')
    sigma_min = variable_control(sigma_min, 4, peaks, 'sigma_min')
    sigma_max = variable_control(sigma_max, 30, peaks, 'sigma_max')
    

    #importing the model to fit the baseline
    #defining the model prefix as well
    if baseline_type == None:
        baseline_type = 'exponential'
        
    prefix, bl_mod = select_baseline(baseline_type)
    #initialization of the initial guess
    pars = bl_mod.guess(absorbance, x=wavenumber)

    gauss = []
    # initialization of the model used for the peak fitting
   
    mod = bl_mod
    mod_prefix=[]
    gauss = []
    
    for index, peak in enumerate(peaks):
        # creating a model for each peak according to the user_choice
        mod_prefix, gauss = select_distribution(peak_shape, mod_prefix, gauss, index)

        #initializing the parameters for the curve 
        pars.update(gauss[index].make_params())
        #setting parameters boundaries.
        # value = initial guess for starting the computation
        # min = minimum value admissible
        # max = maximum value admissible
        pars[f'{mod_prefix[index]}_center'].set(value=peak, min = peak-peak_center_min[index], max=peak+peak_center_max[index]) # I am forcing the peaks to vary within the resolution of the instrument
        pars[f'{mod_prefix[index]}_amplitude'].set(value=10, min=area_min[index], max=area_max[index]) #no peak area can be lower than 0
        pars[f'{mod_prefix[index]}_sigma'].set(value=15, min=sigma_min[index], max = sigma_max[index]) # peak characteristic width 
        # adding the model for each additional curve to  the model in use
        mod = mod+gauss[index]

    #initialization of the model
    init = mod.eval(pars,x=wavenumber)
    #fitting the model to the experimental results
    out = mod.fit(absorbance, pars, x=wavenumber)
    print(out.fit_report())
    log_file.writelines(out.fit_report())
    #plotting the fit results with respect to the experimental curve
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(wavenumber, absorbance, 'b', label='experimental data')
    #axes[0].plot(wavenumber, init, 'k--', label='initial fit')
    axes[0].plot(wavenumber, out.best_fit, 'r-', label='best fit')
    axes[0].set_xlabel('Wavenumber (cm$^{-1}$)')
    axes[0].set_ylabel('Absorbance (AU)')
    axes[0].legend(loc='best')
    axes[0].set_xlim(wavenumber[-1],wavenumber[0])
    #plotting each component curve under the experimental curve.
    comps = out.eval_components(x=wavenumber)
    axes[1].plot(wavenumber, absorbance, 'b')
    color=iter(cm.rainbow(np.linspace(0,1,len(peaks))))
    # initializing empty lists for area and peak_position so that we can save the fit results necessary later on
    areas = []
    peak_positions = []
    for index, peak in enumerate(peaks):
        c=next(color)
        axes[1].plot(wavenumber, comps[f'{mod_prefix[index].casefold()}_'], \
                     '--',c=c, label=f'{peak_shape[index]} component {index}')
        areas += [out.params[f'{mod_prefix[index].casefold()}_amplitude'].value]
        peak_positions += [out.params[f'{mod_prefix[index].casefold()}_center'].value]

    axes[1].plot(wavenumber, comps[prefix], 'k--', label='baseline')
    axes[1].legend(loc='best')
    axes[1].set_xlabel('Wavenumber (cm$^{-1}$)')
    axes[1].set_ylabel('Absorbance (AU)')
    axes[1].set_xlim(wavenumber[-1],wavenumber[0])
    plt.show()
    time_counter+=1

    log_file.close()
    return(areas, peak_positions)

#convert the nested dictionaries obtained with the peak fitting function in a pandas dataframe
def from_dict_to_df(dictionary,time, resolution = 4):
    #building the empty dataframe to host the organized results
    max_len = 0
    for t in time:
        if len(dictionary[t]['Positions'])> max_len:
            max_len = len(dictionary[t]['Positions'])
    #creating the column names
    area_names = []
    peak_names = []
    for x in range( max_len):
        area_names += ["Area"+str(x)]
        peak_names += ["Peak"+str(x)]
    area_names.extend(peak_names)
    test_df = pd.DataFrame(data=None, index=time, columns = area_names)
   
    #populating the dataframe with the data in the dictionary
    for idx_t,t in enumerate(time):
        for idx_wn, wn_pos in enumerate(dictionary[t]['Positions']):
            # the first time considered is going to occupy the first row as it is without any need for modification
            if idx_t == 0:
                test_df.iloc[idx_t,idx_wn] = dictionary[t]['Integration'][idx_wn]
                pos_idx = int(idx_wn+max_len)
                test_df.iloc[idx_t,pos_idx] = dictionary[t]['Positions'][idx_wn]
            else:
                counter = 0
                for idx_control in range(0,max_len): 
                # the other peaks will be placed according to the average wavenumber of the peaks placed in the previous rows.
                # the tolerance is the resolution*3
                    if np.isclose(test_df[f'Peak{idx_control}'].mean(),dictionary[t]['Positions'][idx_wn], rtol=0, atol=resolution*3): 
                        test_df.iloc[idx_t,idx_control] = dictionary[t]['Integration'][idx_wn]
                        pos_idx = int(idx_control+max_len)
                        test_df.iloc[idx_t,pos_idx] = dictionary[t]['Positions'][idx_wn]
                        break
                    else:
                        counter +=1
                # if the peak after testing all the rows is still not added anywhere it will be added to the first 'empty' column
                        if counter == max_len:
                            max_na = 0
                            col_idx = 0
                            for col in test_df.columns:
                                if test_df[col].isna().sum() > max_na:
                                    max_col = col_idx
                                    max_na = test_df[col].isna().sum()
                                col_idx +=1
                            test_df.iloc[idx_t,max_col] = dictionary[t]['Integration'][idx_wn]
                            pos_idx = int(max_col+max_len)
                            test_df.iloc[idx_t,pos_idx] = dictionary[t]['Positions'][idx_wn]
                            counter +=1
    return test_df
#### CORRECTION OF THE MULTISCATTERING
# MSC (Multiplicative Scatter Correction) - WAVENUMBER SHIFT CORRECTION
# https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
def MSC(input_data, reference=None):
    #Perform Multiplicative scatter correction
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
    return (data_msc)



###### Simple linear regression functions to remove gas phase contributions.

#functions to check that the sampling of the gas contribuent spectrum/a is the same of the reactions spectra.
def shape_check(W,A, CH4_W,CH4_A):
    if np.shape(CH4_W)[1] == np.shape(W)[1]:
        print('Good news! your spectra are compatible, no need to interpolate!')
    while np.shape(CH4_W)[1] != np.shape(W)[1]:
        if not np.isclose(W[0,0], CH4_W[0,0], atol=1.):
            end = np.where(np.isclose(CH4_W[0,:], W[0,0],atol = 0.3))
            print(end[0])
            delete = np.arange(0, end[0], 1, dtype=np.int)
            if np.shape(CH4_W)[1]> np.shape(W)[1]:
                CH4_W = np.delete(CH4_W, delete, 1)
                CH4_A = np.delete(CH4_A,delete, 1)
            elif np.shape(CH4_W)[1]< np.shape(W)[1]:
                W = np.delete(W,delete,1)
                A = np.delete(A,delete,1)  
        elif W[0,-1:] != CH4_W[0,-1:]:
            if np.shape(CH4_W)[1]> np.shape(W)[1]:
                CH4_W = CH4_W[:,:-1]
                CH4_A = CH4_A[:,:-1]
            elif np.shape(CH4_W)[1]< np.shape(W)[1]:
                W = W[:,:-1]
                A = A[:,:-1]
        else:
            yrng = []
            xrng = []
            for i in range(np.shape(CH4_W)[1]):
                xrng.append(W[0,:])
                f = interpolate.interp1d(CH4_W[i,:],CH4_A[i,:])
                #f = interpolate.interp1d(x_rng[i,:],y_rng[i,:],kind='cubic')
                yrng.append(f(wn[0,:]))
            CH4_A = np.array(yrng)
            CH4_W = np.array(xrng)
            
    

    print('y_std shape = ',np.shape(CH4_W))
    print('wn shape = ',np.shape(W))
    return (W,A,CH4_W,CH4_A)

def convoluzione(weights, y_standard):
    return np.dot(weights, y_standard) 

def e2(weights, y_std, y_real, NF=None, SN=None):
    #weights = weights.reshape(NF,SN)
    #print(np.shape(weights), np.shape(y_std))
    conv = convoluzione(weights, y_std.reshape(1,-1))
    diff = np.sum((y_real-conv)**2)
    return diff

def build_IG(W,A,CH4_W,CH4_A, plot = None):
    igw =  np.random.rand(np.shape(W)[0],np.shape(CH4_W)[0])
    print('IGW shape =',np.shape(igw))
    conv = convoluzione(igw, CH4_A)
    x_conv = CH4_W
    print('x_conv =',np.shape(x_conv))
    print('ab shape =',np.shape(A))
    error = e2(igw, CH4_A, A, np.shape(W)[0], np.shape(CH4_W)[0]) 
    if plot:
        fig,ax = plt.subplots(figsize=(10,6))
        ax.plot(x_conv[0,:], conv[0,:], ls ='--', label = 'Initial Guess at '+str(time[100])+ ' min')
        ax.plot(W[100,:], A[0,:], label = 'Real data at '+str(time[100])+ ' min')
        ax.plot(W[100,:], A[0,:]-conv[0,:], label = 'Subtraction data at '+str(time[100])+ ' min')
        plt.ylabel( "Absorbance (AU)", fontsize=(16))
        plt.xlabel("Wavenumber (cm$^{-1}$)", fontsize=(16))
        ax.tick_params(labelsize=15)
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol = 10, fontsize=(16))
        ax.axes.ticklabel_format(axis='both', style='plain')
    return igw, x_conv

from scipy.optimize import minimize
def CH4_removal(W,A,CH4_W,CH4_A, time, IG=None, IG_plot=None, x_conv=None, plot = None):
    ############## I AM FITTING THE STANDARD MATRIX MULTIPLIED BY THE LOADING MATRIX TO THE EXPERIMENTAL RESULTS #########
    if not IG:
        IG, x_conv = build_IG(W,A,CH4_W,CH4_A, plot = IG_plot)
    
    bnds = [(0, None)]*np.shape(CH4_W)[0]
    #print(np.shape(IG),np.shape(CH4_A),np.shape(A))
    y_slz = np.empty((np.shape(A)[0],np.shape(A)[1]))
    dif = np.empty((np.shape(A)[0],np.shape(A)[1]))
    sol = []
    for x in range(np.shape(W)[0]):
        slz = minimize(e2,IG[x,:], args=(CH4_A[0,:], A[x,:], np.shape(W)[0], np.shape(CH4_W)[0]),\
                       method='SLSQP', bounds=bnds)
        sol.append(slz.x)

        y_slz[x,:] = convoluzione(sol[x].reshape(-1, np.shape(CH4_W)[0]), CH4_A)
        dif[x,:] = A[x,:] - y_slz[x,:]
        diff = e2(sol[x], CH4_A, A, np.shape(W)[0], np.shape(CH4_W)[0])
        if plot:
            ax1,ax2 = plt.figure(figsize=(10,6)).subplots(1,2)
            ax1.plot(x_conv[0,:], y_slz[x,:],  label = 'fit result', color = 'red')
            ax1.plot(x_conv[0,:], A[x,:],  label = 'experiment', color = 'black')
            ax1.set_ylabel( "Absorbance (AU)", fontsize=(28))
            ax1.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=(28))
            ax2.plot(x_conv[0,:], dif[x,:] , label = 'subtraction')
            ax2.set_ylabel( "Absorbance (AU)", fontsize=(28))
            ax2.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=(28))
            plt.gca().set_title(str(time[x])+' min')
            ax1.tick_params(labelsize=26)
            ax1.xaxis.set_tick_params(width=1.5)
            ax1.yaxis.set_tick_params(width=1.5)
            ax1.legend(fontsize=(20))
            ax1.axes.ticklabel_format(axis='both', style='plain')    
            ax2.tick_params(labelsize=26)
            ax2.xaxis.set_tick_params(width=1.5)
            ax2.yaxis.set_tick_params(width=1.5)
            ax2.legend(fontsize=(20))
            ax2.axes.ticklabel_format(axis='both', style='plain')
            plt.subplots_adjust(wspace=0.4)
            plt.tight_layout()
    return dif,sol 

from math import floor,ceil
####### PSD Functions developed in Flaherty's Lab.
def period_average(data,time_per_period,time_per_scan):
    """
    Takes data matrix with rows corresponding to the independent
    spectroscopy variable (wavelength) and columns corresponding to
    times and averages spectra taken at the same point in different
    periods.
    """

    number_wavelengths,number_scans=data.shape
    scans_per_period=floor(time_per_period/time_per_scan)
    number_periods=floor(number_scans/scans_per_period)
    remainder=time_per_period/time_per_scan-time_per_period//time_per_scan 
    if remainder < 0.5:
        sgn=-1
    else:
        sgn=1
        remainder=1.-remainder
    
    avg_data=np.zeros((number_wavelengths,scans_per_period))

    for i in range(number_periods):
        offset=round(remainder*i)
        lb=i*scans_per_period+sgn*offset
        ub=(i+1)*scans_per_period+sgn*offset
        intrvl=range(lb,ub)
        avg_data+=data[:,intrvl]

    avg_data/=number_periods
    return avg_data

def psd_transform(A,phis=np.linspace(0,2*np.pi,360),ks=np.arange(1,2)):
    """
     The provided matrix must correspond to one period.
    """
    nr,nc=A.shape
    nphis=len(phis)
    args=np.linspace(0,1,nc)*2*np.pi
    args=np.outer(args,ks)
    args=np.tile(args,reps=(nphis,1,1))+phis[:,np.newaxis,np.newaxis]
    coeffs=ntgrtn_cffcnts(nc)*2./nc
    integral=A*coeffs
    return integral @ np.sin(args) #this is a matrix multiplication

def ntgrtn_cffcnts(n, order=1):
    """
    Returns integration coefficients depending on desired order of
    accuracy. Does not normalize them.
    """
    if order == 1:
        coeffs=np.ones(n)
        coeffs[0],coeffs[-1]=0.5,0.5
        print(coeffs)
    else:
        raise Warning('Order > 1 not yet supported')
    return coeffs

from scipy.sparse.linalg import svds
def SVD(A,k_range=range(1,51)): 
    """
    Calculates singular value decomposition provided range of number
    of singular values and quantifies error in approximation with each
    number of singular components in the normalized Frobenius norm.
    """
    print(k_range)
    u,s,vt=svds(A,max(k_range))
    l=[]
    for k in k_range:
        diag=np.diag(s[:k])
        approxA=u[:,:k]@diag@vt[:k]
        E=A-approxA
        fnorm=np.linalg.norm(E) #normalize this
        l.append(fnorm)
    return (u,s,vt),l

# multivariate curve resolution packages
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import *
import logging

def MCR_fit(A_psd,IG_guess,c_l,st_l,max_iter=None):
    #multivariate curve resolution
    c_constraints=[]
    st_constraints=[]
    if max_iter == None:
        max_iter = 1000

    #likely a more elegant way exists
    if 'ConstraintNonneg' in c_l:
        c_constraints.append(ConstraintNonneg())
    if 'ConstraintCumsumNonneg' in c_l:
        c_constraints.append(ConstraintCumsumNonneg())
    if 'ConstraintNorm' in c_l:
        c_constraints.append(ConstraintNorm())
    if 'ConstraintNonneg' in st_l:
        st_constraints.append(ConstraintNonneg())
    if 'ConstraintCumsumNonneg' in st_l:
        st_constraints.append(ConstraintCumsumNonneg())
    if 'ConstraintNorm' in st_l:
        st_constraints.append(ConstraintNorm())
    #print(c_constraints,st_constraints)

    logger = logging.getLogger('pymcr')
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_format = logging.Formatter('%(message)s')
    stdout_handler.setFormatter(stdout_format)
    logger.addHandler(stdout_handler)

    mcrar=McrAR(max_iter=max_iter,st_regr=NNLS(),c_regr=OLS(),c_constraints=[],st_constraints=st_constraints)
    mcrar.fit(A_psd,ST=IG_guess,verbose=True)

    return mcrar