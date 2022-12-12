# -*- coding: utf-8 -*-

import os
import sys
cwd = os.getcwd()
parent_wd = cwd.replace('/notebooks', '')
sys.path.insert(1, parent_wd)
data_path = parent_wd + '/raw_data/'
output_path = parent_wd + '/output_figures/'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from scipy import integrate
import uncertainties.unumpy as unp
import uncertainties as unc
import string

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None

def sigmoid(x, L, x0, k):
    y = L / (1 + np.exp(-k*(x-x0))) - L/(1 + np.exp(k*x0))
    return y

def sigmoid_velocity(x, L, x0, k):
    y = L*k*np.exp(-k*(x-x0))/(1 + np.exp(-k*(x-x0)))**2
    return y

def sigmoid_unp(x, L, x0, k):
    y = L / (1 + unp.exp(-k*(x-x0))) - L/(1 + unp.exp(k*x0))
    return y

def predband(x, xd, yd, p, func, conf=0.95):
    """
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name    
    """
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

def analyze_kinematics_new(file_name, mu_surrounding, mu_cyto = 0.05, h_sheath = 25E-9, h_slip = 6E-9, 
                           D = 100E-9, R_spore = 1.3E-6, delta = 0, dt = 0.009, time_col_name = 'Time (s)', V_expansion = 5E-19):
    R = D/2 # radius of polar tube
    df_L = pd.read_excel(data_path + file_name, sheet_name = 'L')
    df_V = pd.read_excel(data_path + file_name, sheet_name = 'V')
    df_time = df_L[time_col_name]
    df_L = df_L.drop(columns='Time (s)')
    df_V = df_V.drop(columns='Time (s)')
    
    x_data = []
    y_data = []
    v_data = []
    for label, content in df_L.items():
        y_data.extend(list(content[~np.isnan(content)]))
        x_data.extend(list(df_time[~np.isnan(content)]))
    for label, content in df_V.items():
        v_data.extend(list(content[~np.isnan(content)]))
        
    E_drag_fullTube = []
    E_drag_tip = []
    E_poiseuille = []
    E_sheath_fullTube_slip = []
    E_sheath_eversion_slip = []
    E_plug_slip = []
    E_poiseuille_fullopen = []
    E_poiseuille_external_tip = []
    E_poiseuille_external_tip_cyto = []
    W_jackinthebox = []
    E_jackinthebox = []
    
    W_peak_fullTubeMoving = []
    P_peak_fullTubeMoving = []
    E_fullTubeMoving = []
    W_peak_tipMovingClosedEnd = []
    P_peak_tipMovingClosedEnd = []
    E_tipMovingClosedEnd = []
    W_peak_tipMovingOpenEnd = []
    P_peak_tipMovingOpenEnd = []
    E_tipMovingOpenEnd = []
    W_peak_tipMovingExternalEnd = []
    P_peak_tipMovingExternalEnd = []
    E_tipMovingExternalEnd = []
    W_peak_tipMovingExternalEnd_cyto = []
    P_peak_tipMovingExternalEnd_cyto = []
    E_tipMovingExternalEnd_cyto = []
    
    # compute energy dissipation of individual trace
    FTM = {'W_peak': [], 'P_peak': [], 'E': [], 'W': [], 'P': [],
           'W_drag_fulltube': [], 'P_drag_fulltube': [], 'E_drag_fulltube': [], 
           'W_lub_sheath': [], 'P_lub_sheath': [], 'E_lub_sheath': [], 
           'P_cytoplasm': [], 'E_osmotic': []}
    TMC = {'W_peak': [], 'P_peak': [], 'E': [], 'W': [], 'P': [],
           'W_drag_tip': [], 'P_drag_tip': [], 'E_drag_tip': [], 
           'W_lub_sheath': [], 'P_lub_sheath': [], 'E_lub_sheath': [],
           'W_lub_slip': [], 'P_lub_slip': [], 'E_lub_slip': [],
           'W_cytoplasm': [], 'P_cytoplasm': [], 'E_cytoplasm': [], 'E_osmotic': []}
    TMO = {'W_peak': [], 'P_peak': [], 'E': [], 'W': [], 'P': [],
           'W_drag_tip': [], 'P_drag_tip': [], 'E_drag_tip': [], 
           'W_lub_sheath': [], 'P_lub_sheath': [], 'E_lub_sheath': [],
           'W_lub_slip': [], 'P_lub_slip': [], 'E_lub_slip': [], 'E_osmotic': []}
    TME = {'W_peak': [], 'P_peak': [], 'E': [], 'W': [], 'P': [],
           'W_drag_tip': [], 'P_drag_tip': [], 'E_drag_tip': [], 
           'W_lub_sheath': [], 'P_lub_sheath': [], 'E_lub_sheath': [],
           'W_lub_slip': [], 'P_lub_slip': [], 'E_lub_slip': [],
           'W_cytoplasm': [], 'P_cytoplasm': [], 'E_cytoplasm': [],
           'W_tubecontent': [], 'P_tubecontent': [], 'E_tubecontent': [], 'E_osmotic': []}
    TMEP = {'W_peak': [], 'P_peak': [], 'E': [], 'W': [], 'P': [],
            'W_drag_tip': [], 'P_drag_tip': [], 'E_drag_tip': [], 
            'W_lub_sheath': [], 'P_lub_sheath': [], 'E_lub_sheath': [],
            'W_lub_slip': [], 'P_lub_slip': [], 'E_lub_slip': [],
            'W_lub_cell': [], 'P_lub_cell': [], 'E_lub_cell': [],
            'W_cytoplasm': [], 'P_cytoplasm': [], 'E_cytoplasm': [],
            'W_tubecontent': [], 'P_tubecontent': [], 'E_tubecontent': [], 'E_osmotic': []}
    JITB = {'W_drag_fulltube': [], 'P_drag_fulltube': [], 'E_drag_fulltube': [], 
            'W_jackinthebox': [], 'E_jackinthebox': []}
    
    for (label_L, content_L), (label_V, content_V) in zip(df_L.items(), df_V.items()):
        L = content_L[~np.isnan(content_L)]*1E-6
        L_tot = np.max(L)
        L_sheath_full = (L_tot - L)/2
        L_sheath_eversion = (L_tot - 2*L)*(L_tot - 2*L > 0)
        L_open = (2*L - L_tot)*(2*L - L_tot > 0)
        L_slip = np.minimum(L, L_tot - L)
        V = content_V[~np.isnan(content_V)]*1E-6
        V = V*(V >= 0)
        epsilon = 1/np.log(L/R)
        
        
        W_drag_FTM = 2*np.pi*mu_surrounding*V**2*L*(epsilon + 0.80685*epsilon**2 + 0.82854*epsilon**3)
        P_drag_FTM = 2*mu_surrounding*V*L*(epsilon + 0.80685*epsilon**2 + 0.82854*epsilon**3)/R**2
        E_drag_FTM = sum(W_drag_FTM)*dt
        
        W_lub_sheath_FTM = np.pi*mu_cyto*(V/(h_sheath+2*delta))**2*L_sheath_full*(2*R*h_sheath + h_sheath**2)
        P_lub_sheath_FTM = 2*mu_cyto*(V/(h_sheath+2*delta))*L_sheath_full/R
        E_lub_sheath_FTM = sum(W_lub_sheath_FTM)*dt
        
        P_cytoplasm_FTM = 2*mu_cyto*(L+L_sheath_full)*V/(1/2*(R+delta)**2 - 1/4*R**2)
        
        W_FTM = W_drag_FTM + W_lub_sheath_FTM
        P_FTM = P_drag_FTM + P_lub_sheath_FTM + P_cytoplasm_FTM
        E_osmotic = np.mean(P_FTM)*V_expansion
        E_FTM = E_drag_FTM + E_lub_sheath_FTM
        
        FTM['W_peak'].append(np.max(W_FTM))
        FTM['P_peak'].append(np.max(P_FTM))
        FTM['E'].append(E_FTM)
        FTM['E_osmotic'].append(E_osmotic)
        FTM['W'].append(W_FTM)
        FTM['P'].append(P_FTM)
        FTM['W_drag_fulltube'].append(W_drag_FTM)
        FTM['P_drag_fulltube'].append(P_drag_FTM)
        FTM['E_drag_fulltube'].append(E_drag_FTM)
        FTM['W_lub_sheath'].append(W_lub_sheath_FTM)
        FTM['P_lub_sheath'].append(P_lub_sheath_FTM)
        FTM['E_lub_sheath'].append(E_lub_sheath_FTM)
        FTM['P_cytoplasm'].append(P_cytoplasm_FTM)
        
        W_JITB = 6*np.pi*mu_surrounding*R_spore*V**2
        E_JITB = sum(W_JITB)*dt
        JITB['W_drag_fulltube'].append(W_drag_FTM)
        JITB['P_drag_fulltube'].append(P_drag_FTM)
        JITB['E_drag_fulltube'].append(E_drag_FTM)
        JITB['W_jackinthebox'].append(W_JITB)
        JITB['E_jackinthebox'].append(E_JITB)
        
        W_drag_tip = 2*np.pi*mu_surrounding*V**2*R
        P_drag_tip = 2*mu_surrounding*V/R
        E_drag_tip = sum(W_drag_tip)*dt
        
        W_lub_sheath_Ev = np.pi*mu_cyto*(2*V/(h_sheath+2*delta))**2*L_sheath_eversion*(2*R*h_sheath + h_sheath**2)
        P_lub_sheath_Ev = 2*mu_cyto*(2*V/(h_sheath+2*delta))*L_sheath_eversion/R
        E_lub_sheath_Ev = sum(W_lub_sheath_Ev)*dt
        
        W_lub_slip_Ev = np.pi*mu_cyto*(2*V/(h_slip+2*delta))**2*L_slip*(2*R*h_slip + h_slip**2)
        P_lub_slip_Ev = 2*mu_cyto*(2*V/(h_slip+2*delta))*L_slip/R
        E_lub_slip_Ev = sum(W_lub_slip_Ev)*dt
        
        W_cytoplasm_Ev = np.pi/2*mu_cyto*L_open*(2*V)**2*R**4/(1/2*(R+delta)**2 - 1/4*R**2)**2
        P_cytoplasm_Ev = 2*mu_cyto*L_open*(2*V)/(1/2*(R+delta)**2 - 1/4*R**2)
        E_cytoplasm_Ev = sum(W_cytoplasm_Ev)*dt
        
        W_TMC = W_drag_tip + W_lub_sheath_Ev + W_lub_slip_Ev + W_cytoplasm_Ev
        P_TMC = P_drag_tip + P_lub_sheath_Ev + P_lub_slip_Ev + P_cytoplasm_Ev
        E_osmotic = np.mean(P_TMC)*V_expansion
        E_TMC = E_drag_tip + E_lub_sheath_Ev + E_lub_slip_Ev + E_cytoplasm_Ev
        
        TMC['W_peak'].append(np.max(W_TMC))
        TMC['P_peak'].append(np.max(P_TMC))
        TMC['E'].append(E_TMC)
        TMC['E_osmotic'].append(E_osmotic)
        TMC['W'].append(W_TMC)
        TMC['P'].append(P_TMC)
        TMC['W_drag_tip'].append(W_drag_tip)
        TMC['P_drag_tip'].append(P_drag_tip)
        TMC['E_drag_tip'].append(E_drag_tip)
        TMC['W_lub_sheath'].append(W_lub_sheath_Ev)
        TMC['P_lub_sheath'].append(P_lub_sheath_Ev)
        TMC['E_lub_sheath'].append(E_lub_sheath_Ev)
        TMC['W_lub_slip'].append(W_lub_slip_Ev)
        TMC['P_lub_slip'].append(P_lub_slip_Ev)
        TMC['E_lub_slip'].append(E_lub_slip_Ev)
        TMC['W_cytoplasm'].append(W_cytoplasm_Ev)
        TMC['P_cytoplasm'].append(P_cytoplasm_Ev)
        TMC['E_cytoplasm'].append(E_cytoplasm_Ev)
        
        W_TMO = W_drag_tip + W_lub_sheath_Ev + W_lub_slip_Ev 
        P_TMO = P_drag_tip + P_lub_sheath_Ev + P_lub_slip_Ev
        E_osmotic = np.mean(P_TMO)*V_expansion
        E_TMO = E_drag_tip + E_lub_sheath_Ev + E_lub_slip_Ev
        
        TMO['W_peak'].append(np.max(W_TMO))
        TMO['P_peak'].append(np.max(P_TMO))
        TMO['E'].append(E_TMO)
        TMO['E_osmotic'].append(E_osmotic)
        TMO['W'].append(W_TMO)
        TMO['P'].append(P_TMO)
        TMO['W_drag_tip'].append(W_drag_tip)
        TMO['P_drag_tip'].append(P_drag_tip)
        TMO['E_drag_tip'].append(E_drag_tip)
        TMO['W_lub_sheath'].append(W_lub_sheath_Ev)
        TMO['P_lub_sheath'].append(P_lub_sheath_Ev)
        TMO['E_lub_sheath'].append(E_lub_sheath_Ev)
        TMO['W_lub_slip'].append(W_lub_slip_Ev)
        TMO['P_lub_slip'].append(P_lub_slip_Ev)
        TMO['E_lub_slip'].append(E_lub_slip_Ev)
        
        W_tubecontent = np.pi/2*mu_cyto*(L_slip + L_sheath_eversion)*(2*V)**2*R**4/(1/2*(R+delta)**2 - 1/4*R**2)**2
        P_tubecontent = 2*mu_cyto*(L_slip + L_sheath_eversion)*(2*V)/(1/2*(R+delta)**2 - 1/4*R**2)
        E_tubecontent = sum(W_tubecontent)*dt
        
        W_TME = W_drag_tip + W_lub_sheath_Ev + W_lub_slip_Ev + W_cytoplasm_Ev + W_tubecontent
        P_TME = P_drag_tip + P_lub_sheath_Ev + P_lub_slip_Ev + P_cytoplasm_Ev + P_tubecontent
        E_osmotic = np.mean(P_TME)*V_expansion
        E_TME = E_drag_tip + E_lub_sheath_Ev + E_lub_slip_Ev + E_cytoplasm_Ev + E_tubecontent
        
        TME['W_peak'].append(np.max(W_TME))
        TME['P_peak'].append(np.max(P_TME))
        TME['E'].append(E_TME)
        TME['W'].append(W_TME)
        TME['E_osmotic'].append(E_osmotic)
        TME['P'].append(P_TME)
        TME['W_drag_tip'].append(W_drag_tip)
        TME['P_drag_tip'].append(P_drag_tip)
        TME['E_drag_tip'].append(E_drag_tip)
        TME['W_lub_sheath'].append(W_lub_sheath_Ev)
        TME['P_lub_sheath'].append(P_lub_sheath_Ev)
        TME['E_lub_sheath'].append(E_lub_sheath_Ev)
        TME['W_lub_slip'].append(W_lub_slip_Ev)
        TME['P_lub_slip'].append(P_lub_slip_Ev)
        TME['E_lub_slip'].append(E_lub_slip_Ev)
        TME['W_cytoplasm'].append(W_cytoplasm_Ev)
        TME['P_cytoplasm'].append(P_cytoplasm_Ev)
        TME['E_cytoplasm'].append(E_cytoplasm_Ev)
        TME['W_tubecontent'].append(W_tubecontent)
        TME['P_tubecontent'].append(P_tubecontent)
        TME['E_tubecontent'].append(E_tubecontent)
        
        W_lub_cell = np.pi*mu_cyto*(2*V/(h_slip+2*delta))**2*L_open*(2*R*h_slip + h_slip**2)
        P_lub_cell = 2*mu_cyto*(2*V/(h_slip+2*delta))*L_open/R
        E_lub_cell = sum(W_lub_cell)*dt
        
        W_TMEP = W_drag_tip + W_lub_sheath_Ev + W_lub_slip_Ev + W_lub_cell + W_cytoplasm_Ev + W_tubecontent
        P_TMEP = P_drag_tip + P_lub_sheath_Ev + P_lub_slip_Ev + P_lub_cell + P_cytoplasm_Ev + P_tubecontent
        E_osmotic = np.mean(P_TMEP)*V_expansion
        E_TMEP = E_drag_tip + E_lub_sheath_Ev + E_lub_slip_Ev + E_lub_cell + E_cytoplasm_Ev + E_tubecontent
        
        TMEP['W_peak'].append(np.max(W_TMEP))
        TMEP['P_peak'].append(np.max(P_TMEP))
        TMEP['E'].append(E_TMEP)
        TMEP['E_osmotic'].append(E_osmotic)
        TMEP['W'].append(W_TMEP)
        TMEP['P'].append(P_TMEP)
        TMEP['W_drag_tip'].append(W_drag_tip)
        TMEP['P_drag_tip'].append(P_drag_tip)
        TMEP['E_drag_tip'].append(E_drag_tip)
        TMEP['W_lub_sheath'].append(W_lub_sheath_Ev)
        TMEP['P_lub_sheath'].append(P_lub_sheath_Ev)
        TMEP['E_lub_sheath'].append(E_lub_sheath_Ev)
        TMEP['W_lub_slip'].append(W_lub_slip_Ev)
        TMEP['P_lub_slip'].append(P_lub_slip_Ev)
        TMEP['E_lub_slip'].append(E_lub_slip_Ev)
        TMEP['W_lub_cell'].append(W_lub_cell)
        TMEP['P_lub_cell'].append(P_lub_cell)
        TMEP['E_lub_cell'].append(E_lub_cell)
        TMEP['W_cytoplasm'].append(W_cytoplasm_Ev)
        TMEP['P_cytoplasm'].append(P_cytoplasm_Ev)
        TMEP['E_cytoplasm'].append(E_cytoplasm_Ev)
        TMEP['W_tubecontent'].append(W_tubecontent)
        TMEP['P_tubecontent'].append(P_tubecontent)
        TMEP['E_tubecontent'].append(E_tubecontent)
    
    n_data = len(y_data)
    p0 = [max(y_data), np.median(x_data), 1] # initial guess
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0)
    perr = np.sqrt(np.diag(pcov))
    r2 = 1.0-(sum((y_data-sigmoid(x_data,*popt))**2)/((n_data-1.0)*np.var(y_data,ddof=1)))
    
    unp_popt = unc.correlated_values(popt, pcov)
    # calculate regression confidence interval
    px = np.copy(df_time)
    py = sigmoid_unp(px, *unp_popt)
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    lpb, upb = predband(px, np.array(x_data), np.array(y_data), popt, sigmoid, conf=0.95)
    v_lpb, v_upb = predband(px, np.array(x_data), np.array(v_data), popt, sigmoid_velocity, conf=0.95)
    
    L = sigmoid(df_time, *popt)
    V = sigmoid_velocity(df_time, *popt)
    
    result = {'time': df_time, 
              'popt': popt, 
              'pcov': pcov,
              'perr': perr,
              'unp_popt': unp_popt,
              'px': px,
              'py': py,
              'nom': nom,
              'std': std,
              'lpb': lpb,
              'upb': upb,
              'v_lpb': v_lpb,
              'v_upb': v_upb,
              'r2': r2,
              'df_L': df_L, 
              'df_V': df_V,
              'L': L, 
              'V': V,
              'FTM': FTM,
              'TMC': TMC,
              'TMO': TMO,
              'TME': TME,
              'TMEP': TMEP,
              'JITB': JITB
              }
    return result

def summary_statistics_and_plot(result_0_percent, result_05_percent, result_1_percent, result_2_percent, result_3_percent, result_4_percent):
    text = ['0% MC', '0.5% MC', '1% MC', '2% MC', '3% MC', '4% MC']
    E_FTM_all = []
    E_TMC_all = []
    E_TMO_all = []
    E_TME_all = []
    E_TMEP_all = []
    
    E_FTM_osmotic_all = []
    E_TMC_osmotic_all = []
    E_TMO_osmotic_all = []
    E_TME_osmotic_all = []
    E_TMEP_osmotic_all = []
    
    E_FTM_plus_osmotic_all = []
    E_TMC_plus_osmotic_all = []
    E_TMO_plus_osmotic_all = []
    E_TME_plus_osmotic_all = []
    E_TMEP_plus_osmotic_all = []
    
    E_FTM_osmotic_diff_all = []
    E_TMC_osmotic_diff_all = []
    E_TMO_osmotic_diff_all = []
    E_TME_osmotic_diff_all = []
    E_TMEP_osmotic_diff_all = []
    
    W_peak_FTM_all = []
    W_peak_TMC_all = []
    W_peak_TMO_all = []
    W_peak_TME_all = []
    W_peak_TMEP_all = []
    
    P_peak_FTM_all = []
    P_peak_TMC_all = []
    P_peak_TMO_all = []
    P_peak_TME_all = []
    P_peak_TMEP_all = []
    for i, result in enumerate([result_0_percent, result_05_percent, result_1_percent, result_2_percent, result_3_percent, result_4_percent]):
        E_FTM_all.append(result['FTM']['E'])
        E_TMC_all.append(result['TMC']['E'])
        E_TMO_all.append(result['TMO']['E'])
        E_TME_all.append(result['TME']['E'])
        E_TMEP_all.append(result['TMEP']['E'])
        
        E_FTM_osmotic_all.append(result['FTM']['E_osmotic'])
        E_TMC_osmotic_all.append(result['TMC']['E_osmotic'])
        E_TMO_osmotic_all.append(result['TMO']['E_osmotic'])
        E_TME_osmotic_all.append(result['TME']['E_osmotic'])
        E_TMEP_osmotic_all.append(result['TMEP']['E_osmotic'])
        
        E_FTM_plus_osmotic_all.append(np.array(result['FTM']['E_osmotic']) + np.array(result['FTM']['E']))
        E_TMC_plus_osmotic_all.append(np.array(result['TMC']['E_osmotic']) + np.array(result['TMC']['E']))
        E_TMO_plus_osmotic_all.append(np.array(result['TMO']['E_osmotic']) + np.array(result['TMO']['E']))
        E_TME_plus_osmotic_all.append(np.array(result['TME']['E_osmotic']) + np.array(result['TME']['E']))
        E_TMEP_plus_osmotic_all.append(np.array(result['TMEP']['E_osmotic']) + np.array(result['TMEP']['E']))
        
        E_FTM_osmotic_diff_all.append(np.array(result['FTM']['E_osmotic']) - np.array(result['FTM']['E']))
        E_TMC_osmotic_diff_all.append(np.array(result['TMC']['E_osmotic']) - np.array(result['TMC']['E']))
        E_TMO_osmotic_diff_all.append(np.array(result['TMO']['E_osmotic']) - np.array(result['TMO']['E']))
        E_TME_osmotic_diff_all.append(np.array(result['TME']['E_osmotic']) - np.array(result['TME']['E']))
        E_TMEP_osmotic_diff_all.append(np.array(result['TMEP']['E_osmotic']) - np.array(result['TMEP']['E']))

        W_peak_FTM_all.append(result['FTM']['W_peak'])
        W_peak_TMC_all.append(result['TMC']['W_peak'])
        W_peak_TMO_all.append(result['TMO']['W_peak'])
        W_peak_TME_all.append(result['TME']['W_peak'])
        W_peak_TMEP_all.append(result['TMEP']['W_peak'])

        P_peak_FTM_all.append(result['FTM']['P_peak'])
        P_peak_TMC_all.append(result['TMC']['P_peak'])
        P_peak_TMO_all.append(result['TMO']['P_peak'])
        P_peak_TME_all.append(result['TME']['P_peak'])
        P_peak_TMEP_all.append(result['TMEP']['P_peak'])
    
    stat_E = {'FTM': {'ANOVA': None,
                    'Kruskal': None}, 
            'TMC': {'ANOVA': None,
                    'Kruskal': None},
            'TME': {'ANOVA': None,
                    'Kruskal': None},
            'TMO': {'ANOVA': None,
                    'Kruskal': None},
            'TMEP': {'ANOVA': None,
                    'Kruskal': None}}
    stat_E_plus_osmotic = {'FTM': {'ANOVA': None,
                    'Kruskal': None}, 
            'TMC': {'ANOVA': None,
                    'Kruskal': None},
            'TME': {'ANOVA': None,
                    'Kruskal': None},
            'TMO': {'ANOVA': None,
                    'Kruskal': None},
            'TMEP': {'ANOVA': None,
                    'Kruskal': None}}
    stat_E_osmotic_diff = {'FTM': {'paired_t': None}, 
            'TMC': {'paired_t': None},
            'TME': {'paired_t': None},
            'TMO': {'paired_t': None},
            'TMEP': {'paired_t': None}}
    stat_P = {'FTM': {'ANOVA': None,
                    'Kruskal': None}, 
            'TMC': {'ANOVA': None,
                    'Kruskal': None},
            'TME': {'ANOVA': None,
                    'Kruskal': None},
            'TMO': {'ANOVA': None,
                    'Kruskal': None},
            'TMEP': {'ANOVA': None,
                    'Kruskal': None}}
    stat_W = {'FTM': {'ANOVA': None,
                    'Kruskal': None}, 
            'TMC': {'ANOVA': None,
                    'Kruskal': None},
            'TME': {'ANOVA': None,
                    'Kruskal': None},
            'TMO': {'ANOVA': None,
                    'Kruskal': None},
            'TMEP': {'ANOVA': None,
                    'Kruskal': None}}
    stat_E['FTM']['ANOVA'] = stats.f_oneway(*E_FTM_all)
    stat_E['TMC']['ANOVA'] = stats.f_oneway(*E_TMC_all)
    stat_E['TMO']['ANOVA'] = stats.f_oneway(*E_TMO_all)
    stat_E['TME']['ANOVA'] = stats.f_oneway(*E_TME_all)
    stat_E['TMEP']['ANOVA'] = stats.f_oneway(*E_TMEP_all)
    
    stat_E['FTM']['Kruskal'] = stats.kruskal(*E_FTM_all)
    stat_E['TMC']['Kruskal'] = stats.kruskal(*E_TMC_all)
    stat_E['TMO']['Kruskal'] = stats.kruskal(*E_TMO_all)
    stat_E['TME']['Kruskal'] = stats.kruskal(*E_TME_all)
    stat_E['TMEP']['Kruskal'] = stats.kruskal(*E_TMEP_all)
    
    stat_E_plus_osmotic['FTM']['ANOVA'] = stats.f_oneway(*list(E_FTM_plus_osmotic_all))
    stat_E_plus_osmotic['TMC']['ANOVA'] = stats.f_oneway(*list(E_TMC_plus_osmotic_all))
    stat_E_plus_osmotic['TMO']['ANOVA'] = stats.f_oneway(*list(E_TMO_plus_osmotic_all))
    stat_E_plus_osmotic['TME']['ANOVA'] = stats.f_oneway(*list(E_TME_plus_osmotic_all))
    stat_E_plus_osmotic['TMEP']['ANOVA'] = stats.f_oneway(*list(E_TMEP_plus_osmotic_all))
    
    stat_E_plus_osmotic['FTM']['Kruskal'] = stats.kruskal(*list(E_FTM_plus_osmotic_all))
    stat_E_plus_osmotic['TMC']['Kruskal'] = stats.kruskal(*list(E_TMC_plus_osmotic_all))
    stat_E_plus_osmotic['TMO']['Kruskal'] = stats.kruskal(*list(E_TMO_plus_osmotic_all))
    stat_E_plus_osmotic['TME']['Kruskal'] = stats.kruskal(*list(E_TME_plus_osmotic_all))
    stat_E_plus_osmotic['TMEP']['Kruskal'] = stats.kruskal(*list(E_TMEP_plus_osmotic_all))
    
    stat_E_osmotic_diff['FTM']['paired_t'] = stats.ttest_1samp(np.hstack(E_FTM_osmotic_diff_all), popmean = 0, alternative='less')
    stat_E_osmotic_diff['TMC']['paired_t'] = stats.ttest_1samp(np.hstack(E_TMC_osmotic_diff_all), popmean = 0, alternative='less')
    stat_E_osmotic_diff['TMO']['paired_t'] = stats.ttest_1samp(np.hstack(E_TMO_osmotic_diff_all), popmean = 0, alternative='less')
    stat_E_osmotic_diff['TME']['paired_t'] = stats.ttest_1samp(np.hstack(E_TME_osmotic_diff_all), popmean = 0, alternative='less')
    stat_E_osmotic_diff['TMEP']['paired_t'] = stats.ttest_1samp(np.hstack(E_TMEP_osmotic_diff_all), popmean = 0, alternative='less')
    
    stat_W['FTM']['ANOVA'] = stats.f_oneway(*W_peak_FTM_all)
    stat_W['TMC']['ANOVA'] = stats.f_oneway(*W_peak_TMC_all)
    stat_W['TMO']['ANOVA'] = stats.f_oneway(*W_peak_TMO_all)
    stat_W['TME']['ANOVA'] = stats.f_oneway(*W_peak_TME_all)
    stat_W['TMEP']['ANOVA'] = stats.f_oneway(*W_peak_TMEP_all)

    stat_W['FTM']['Kruskal'] = stats.kruskal(*W_peak_FTM_all)
    stat_W['TMC']['Kruskal'] = stats.kruskal(*W_peak_TMC_all)
    stat_W['TMO']['Kruskal'] = stats.kruskal(*W_peak_TMO_all)
    stat_W['TME']['Kruskal'] = stats.kruskal(*W_peak_TME_all)
    stat_W['TMEP']['Kruskal'] = stats.kruskal(*W_peak_TMEP_all)
    
    stat_P['FTM']['ANOVA'] = stats.f_oneway(*P_peak_FTM_all)
    stat_P['TMC']['ANOVA'] = stats.f_oneway(*P_peak_TMC_all)
    stat_P['TMO']['ANOVA'] = stats.f_oneway(*P_peak_TMO_all)
    stat_P['TME']['ANOVA'] = stats.f_oneway(*P_peak_TME_all)
    stat_P['TMEP']['ANOVA'] = stats.f_oneway(*P_peak_TMEP_all)

    stat_P['FTM']['Kruskal'] = stats.kruskal(*P_peak_FTM_all)
    stat_P['TMC']['Kruskal'] = stats.kruskal(*P_peak_TMC_all)
    stat_P['TMO']['Kruskal'] = stats.kruskal(*P_peak_TMO_all)
    stat_P['TME']['Kruskal'] = stats.kruskal(*P_peak_TME_all)
    stat_P['TMEP']['Kruskal'] = stats.kruskal(*P_peak_TMEP_all)
    
    result_summarized = {'E_FTM_all': E_FTM_all, 
                        'E_TMC_all': E_TMC_all, 
                        'E_TMO_all': E_TMO_all, 
                        'E_TME_all': E_TME_all,
                        'E_TMEP_all': E_TMEP_all, 
                         'E_FTM_osmotic_all': E_FTM_osmotic_all,
                         'E_TMC_osmotic_all': E_TMC_osmotic_all,
                         'E_TMO_osmotic_all': E_TMO_osmotic_all,
                         'E_TME_osmotic_all': E_TME_osmotic_all,
                         'E_TMEP_osmotic_all': E_TMEP_osmotic_all,
                         'E_FTM_osmotic_diff_all': E_FTM_osmotic_diff_all,
                         'E_TMC_osmotic_diff_all': E_TMC_osmotic_diff_all,
                         'E_TMO_osmotic_diff_all': E_TMO_osmotic_diff_all,
                         'E_TME_osmotic_diff_all': E_TME_osmotic_diff_all,
                         'E_TMEP_osmotic_diff_all': E_TMEP_osmotic_diff_all,
                         'E_FTM_plus_osmotic_all': E_FTM_plus_osmotic_all,
                         'E_TMC_plus_osmotic_all': E_TMC_plus_osmotic_all,
                         'E_TMO_plus_osmotic_all': E_TMO_plus_osmotic_all,
                         'E_TME_plus_osmotic_all': E_TME_plus_osmotic_all,
                         'E_TMEP_plus_osmotic_all': E_TMEP_plus_osmotic_all,
                        'W_peak_FTM_all': W_peak_FTM_all,
                        'W_peak_TMC_all': W_peak_TMC_all,
                        'W_peak_TMO_all': W_peak_TMO_all,
                        'W_peak_TME_all': W_peak_TME_all,
                        'W_peak_TMEP_all': W_peak_TMEP_all, 
                        'P_peak_FTM_all': P_peak_FTM_all,
                        'P_peak_TMC_all': P_peak_TMC_all,
                        'P_peak_TMO_all': P_peak_TMO_all,
                        'P_peak_TME_all': P_peak_TME_all,
                        'P_peak_TMEP_all': P_peak_TMEP_all, 
                        'stat_E': stat_E, 'stat_W': stat_W, 'stat_P': stat_P, 'stat_E_osmotic_diff': stat_E_osmotic_diff, 'stat_E_plus_osmotic': stat_E_plus_osmotic}
    return result_summarized

def L_tot_calc(L, x0, k):
    return L*(1 - 1/(1 + np.exp(k*x0)))

def nucleus_speed_varPT(x_nuc, t, popt_L, popt_x0, popt_k, L_T, d, R, k, mu, H, lambd):
    # polar tube is elongated after ejection.
    if (1+lambd)*sigmoid(t, popt_L, popt_x0, popt_k)*1E-6 - x_nuc - lambd*L_T < 0:
        dxdt = 0
    else:
        dxdt = sigmoid_velocity(t, popt_L, popt_x0, popt_k)*1E-6 + (d/R)**3*k*(
        (1+lambd)*sigmoid(t, popt_L, popt_x0, popt_k)*1E-6 - x_nuc - lambd*L_T)/lambd/(L_T - sigmoid(t, popt_L, popt_x0, popt_k)*1E-6)/(6*np.pi*mu*H)
    return dxdt
