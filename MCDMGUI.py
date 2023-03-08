import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from tabulate import tabulate
from pymcdm.methods import TOPSIS
from pymcdm.methods import promethee
from pymcdm.methods import VIKOR
from pymcdm.normalizations import minmax_normalization, max_normalization, linear_normalization,vector_normalization
from pymcdm.helpers import rrankdata,rankdata
from pyDecision.algorithm import ahp_method


    
def MCDM_AHP(dataAHP):
    from ahpy import ahpy
    #Compare X with Y
    weights_RES = {(row["X"], row["Compared with Y"]): row["Weights"] for _, row in dataAHP.iterrows()}
    #The AHP method is applied
    criterias= ahpy.Compare(name='AHP', comparisons=weights_RES, precision=3, random_index='saaty')
    d=criterias.target_weights
    d=criterias.target_weights
    df = pd.DataFrame(list(d.items()), columns=['Criteria', 'Weight'])
    df_np=df.to_numpy()
    CR=criterias.consistency_ratio
    return df_np,CR

def MCDM_TOPSIS(data):
    evaluation_matrix=(data.iloc[7:,1:]).to_numpy(dtype=float)
    types=(data.iloc[0,1:]).to_numpy(dtype=int)
    weights = (data.iloc[1,1:]).to_numpy(dtype=float)
    criteria = data.iloc[2,1:]
    alternatives = data.iloc[7:,0]
    
    # step 2 normalization
    if np.shape(evaluation_matrix)[0]<=5:
        body = TOPSIS(normalization_function=minmax_normalization)
        #print('Minmax normalization has been used.\n')
    else:
        body = TOPSIS(normalization_function=max_normalization)
        #print('Max normalization has been used.\n')
    # step 3 ranking
    
    # Determine preferences and ranking for alternatives
    #print('\nTOPSIS\n')

    pref=body(evaluation_matrix,weights,types)
    ranking = rrankdata(pref)
    
    # all values
    res_data = np.vstack((alternatives,np.round(pref,3),ranking))
    res_data_t = np.transpose(res_data)

    
    #add titles
    titles = np.array(['Alternative','Preference','Ranking'])
    res_data_table = np.vstack((titles,res_data_t[res_data_t[:,2].argsort()]))
    

    return res_data_table


def MCDM_PROMETHEEII(data):

    # Information
# =============================================================================
#     print('The PROMETHEE_II method needs:\n')
#     print('Evaluation matrix -  Alternative x Criteria matrix containing evaluation values.')
#     print('Weights - Array containing the weight of each criteria (sum must be 1).\n')
#     print('Types - Array containing True if criteria value is preferred when increasing or False if decreasing value is preferred.\n')
# =============================================================================
     # step 1 upload data 
    evaluation_matrix=(data.iloc[7:,1:]).to_numpy(dtype=float)
    types=(data.iloc[0,1:]).to_numpy(dtype=int)
    weights = (data.iloc[1,1:]).to_numpy(dtype=float)
    criteria = data.iloc[2,1:]
    Q = data.iloc[4,1:].to_numpy(dtype=float)
    P = data.iloc[5,1:].to_numpy(dtype=float)
    
    if np.all(np.isnan(P))==True:
        P=None
    else:
        pass
    
    if np.all(np.isnan(Q))==True:
        Q=None
    else:
        pass
    
    alternatives = data.iloc[7:,0]
    
    
    body = promethee.PROMETHEE_II(preference_function='usual')
    pref = body(evaluation_matrix,weights,types,p=P,q=Q)
    ranking = rrankdata(pref)
    
    # all values
    res_data = np.vstack((alternatives,np.round(pref,3),ranking))
    res_data_t = np.transpose(res_data)

    
    #add titles
    titles = np.array(['Alternative','Preference','Ranking'])
    res_data_table = np.vstack((titles,res_data_t[res_data_t[:,2].argsort()]))
    
    return res_data_table

def MCDM_VIKOR(data):
    # Information
# =============================================================================
#    # print('The VIKOR method needs:\n')
#     print('Evaluation matrix -  Alternative x Criteria matrix containing evaluation values.')
#     print('Weights - Array containing the weight of each criteria (sum must be 1).\n')
#     print('Types - Array containing True if criteria value is preferred when increasing or False if decreasing value is preferred.\n')
#     
# =============================================================================
    # step 1 upload data
    
    evaluation_matrix=(data.iloc[7:,1:]).to_numpy(dtype=float)
    types=(data.iloc[0,1:]).to_numpy(dtype=int)
    weights = (data.iloc[1,1:]).to_numpy(dtype=float)
    criteria = data.iloc[2,1:]
    alternatives = data.iloc[7:,0]
    

    # step 2 normalization
    if np.shape(evaluation_matrix)[0]<=5:
        body = VIKOR(normalization_function=linear_normalization)
        #print('Minmax normalization has been used.\n')
    else:
        body = VIKOR(normalization_function=max_normalization)
       # print('Max normalization has been used.\n')

    
    # Determine preferences and ranking for alternatives
   # print('\nVIKOR\n')
    
    pref=body(evaluation_matrix,weights,types)
    ranking = rankdata(pref)
    
    # all values
    res_data = np.vstack((alternatives,np.round(pref,3),ranking))
    res_data_t = np.transpose(res_data)

    
    #add titles
    titles = np.array(['Alternative','Preference','Ranking'])
    res_data_table = np.vstack((titles,res_data_t[res_data_t[:,2].argsort()]))


    return res_data_table





def run_file():
    filepath = filedialog.askopenfilename()
    data = pd.read_csv(filepath,header=None)
    dataAHP = pd.read_csv(filepath)
    mcdm = var.get()
   # print(mcdm)
    if mcdm == "AHP":
       results, CR = MCDM_AHP(dataAHP)
       title = tk.Label(root, text="\nMethod selected: AHP")
       title.pack()
       label = tk.Label(root, text =tabulate(results,headers=['Criteria','Weight'],tablefmt="pretty"))
       label.pack()
       Consistency_ratio = tk.Label(root,text="Consistency ratio: "+str(CR)+"\nA ratio <0.1 is considered acceptable" )
       Consistency_ratio.pack()
    elif mcdm == "TOPSIS":
        results = MCDM_TOPSIS(data)
        title = tk.Label(root, text="\nMethod selected: TOPSIS")
        title.pack()
        label = tk.Label(root, text =tabulate(results,headers='firstrow',tablefmt="pretty"))
        label.pack()

    elif mcdm == "PROMETHEE II":
        title = tk.Label(root, text="\nMethod selected: PROMETHEE II")
        title.pack()
        results = MCDM_PROMETHEEII(data)
        label = tk.Label(root, text =tabulate(results,headers='firstrow',tablefmt="pretty"))
        label.pack()
    elif mcdm == "VIKOR":
         results = MCDM_VIKOR(data)
         title = tk.Label(root, text="\nMethod selected: VIKOR")
         title.pack()
         label = tk.Label(root, text =tabulate(results,headers='firstrow',tablefmt="pretty"))
         label.pack()
    else:
        label = tk.Label(root, text = "Invalid Selection")
        label.pack()

root = tk.Tk()
root.title("MCDM GUI")

var = tk.StringVar(value="AHP")

AHP_button = tk.Radiobutton(root, text="AHP", variable=var, value="AHP")
AHP_button.pack()

topsis_button = tk.Radiobutton(root, text="TOPSIS", variable=var, value="TOPSIS")
topsis_button.pack()


promethee_button = tk.Radiobutton(root, text="PROMETHEE II", variable=var, value="PROMETHEE II")
promethee_button.pack()

vikor_button = tk.Radiobutton(root, text="VIKOR", variable=var, value="VIKOR")
vikor_button.pack()

run_button = tk.Button(root, text="Run", command=run_file)
run_button.pack()

exit_button = tk.Button(root, text="Exit", command=root.destroy)
exit_button.pack()

root.mainloop()