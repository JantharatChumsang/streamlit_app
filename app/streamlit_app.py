from email.policy import default
from tkinter import CENTER
import streamlit as st 
from streamlit_option_menu import option_menu
import codecs
import pickle
import joblib
from rdkit.Chem import AllChem
from rdkit import Chem
import requests
from streamlit_lottie import st_lottie
import json
import pandas as pd
from rdkit.Chem import Descriptors, Lipinski
import numpy as np
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import sklearn
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from rdkit.Avalon import pyAvalonTools
import os
from os import path
import zipfile
import glob
import random
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps, IPythonConsole
from chembl_webresource_client.new_client import new_client
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')



st.set_page_config(layout="wide")

#### tab bar ####
selected = option_menu(
    menu_title=None, 
    options=["Home", "About us", "Predict new smiles molecule","Check your smiles molecule"], 
    icons=["house","book","check2-all","search"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal", #แนวนอน
    styles={
        "container": {"padding": "0!important", "background-color": "#CAF1E9"},
        "nav-link": {"font-size": "19px", "text-align": "center", "margin":"8px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#B5E6E5"},
    }
)

#### sticker image ####
def load_lottiefile(filepath: str):
    with open (filepath,"r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#### import html ####
import streamlit.components.v1 as stc 
def st_webpage(page_html,width=1370,height=2000):
    page_file = codecs.open(page_html,'r')
    page =page_file.read()
    stc.html(page,width=width, height=height , scrolling = False)

#### selected tab bar ####
if selected =="Home":
    st.title(f"Welcome to Developing web applications for Breast Cancer Novel Drug Discovery Using the ChemBL Database and Deep Learning approach.")
    thai_title = '<p style="ont-family:Courier; color:#3D0E04; font-size: 18px;"> การพัฒนาเว็บแอพพลิเคชั่นสำหรับการค้นคว้ายาใหม่สำหรับมะเร็งเต้านมด้วยฐานข้อมูล ChemBL และแนวทางการเรียนรู้เชิงลึก</p>'
    st.markdown(thai_title, unsafe_allow_html=True)
    
    # ---- LOAD ASSETS ----
    st.write("##")
    lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_nw19osms.json")
    st_lottie(lottie_coding, height=350, key="coding")
    # index_title = '<p style="ont-family:Courier; color:Black; font-size: 20px;">This is the introduction of a drug molecule from an old smiles molecule into a new one from the eight target protein targets mTOR, HER2, aromatase, CDK4/6, Trop-2, Estrogen Receptor, PI3K and Akt of breast cancer. To researchers or individuals who wish to discover drugs or produce drugs in the drug discovery process to explore the possibilities of molecules before studying further research into the production of future drugs.</p>'
    # st.markdown(index_title, unsafe_allow_html=True)
    st.info('This is the introduction of a drug molecule from an old smiles molecule into a new one from the eight target protein targets mTOR, HER2, aromatase, CDK4/6, Trop-2, Estrogen Receptor, PI3K and Akt of breast cancer. To researchers or individuals who wish to discover drugs or produce drugs in the drug discovery process to explore the possibilities of molecules before studying further research into the production of future drugs.')
   

if selected =="About us":
    with st.container():
        st.title("About us 👥")
        Welcome_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; ">  Welcome to design drug discovery of breast cancer from pre-clinical stage and check of new smiles molecules.</p>'
        st.markdown(Welcome_title, unsafe_allow_html=True)
        st.write("[Database >](https://www.ebi.ac.uk/chembl/)")
            
    with st.container():
        st.write("---")
        st.header("Goal of the projects")
        st.write("""" The goal of this project is to introduce non-toxic drug molecules at the pre-clinical stage before the results of the study can be used to produce or create future drugs. " """)
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("##")
            Ideal_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; "> 🧬 Ideal.</p>'
            st.markdown(Ideal_title, unsafe_allow_html=True)
            st.info(
                """ 
                - It is a process of drug discovery that can Commonly discovered by predicting protein target molecules on web applications
                  And the drug molecules obtained from the prediction can be further studied before producing or developing drugs in the future.
             """)
            st.write("##")
            Reality_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size:20px; "> 🧬 Reality.</p>'
            st.markdown(Reality_title, unsafe_allow_html=True)
            st.info(
                """ 
                - In reality, the field of medicine is more complex than we think, starting with observing, experimenting and researching the properties of the natural surroundings. 
                  development of medicinal substances with the synthesis of chemical compounds or compounds that imitate important substances in nature, which the discovery and manufacture of each drug knowledge 
                  required in many disciplines Important substances that have medicinal properties and are available for sale. must be extracted synthesis or compound analysis number of more than ten thousand species 
                  To be selected to study the potency and toxicity of the drug in vitro.
            """)

            st.write("##")
            Consequences_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; "> 🧬 Consequences.</p>'
            st.markdown(Consequences_title, unsafe_allow_html=True)
            st.info(
                """ 
                - To discover and produce the desired drug If there is no application or technology to help at all It takes an average period of up to 15 years and costs a minimum of approximately $800 million. 
                  Therefore, each discovery of a drug takes a long time and a large budget. The group therefore chose to discover new drugs. together with a machine learning model to help mitigate this problem for future drug development circles.
             """)
            st.write("##")
        with right_column:
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            lottie2_coding = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_pk5mpw6j.json")
            st_lottie(lottie2_coding, height=400,  key="coding")  


if selected =="Predict new smiles molecule":
    Welcome_title = '<p style="font-family:sans-serif; color:#06BBCC; font-size: 20px; ">  Welcome to design drug discovery of breast cancer from pre-clinical stage and check of new smiles molecules.</p>'
    st.markdown(Welcome_title, unsafe_allow_html=True)


if selected =="Check your smiles molecule":
    st.title(f"Check your smiles molecule")
    st.write("The Lipinski's Rule stated the following: Molecular weight < 500 Dalton, Octanol-water partition coefficient (LogP) < 5, Hydrogen bond donors < 5, Hydrogen bond acceptors < 10 ")
    canonical_smiles = st.text_input("Enter your smiles molecules string")

    if st.button("Predict"):
        try:
            if canonical_smiles=="" :
                st.write(f"Don't have smiles molecules")
            
            else:
                # model1 = joblib.load('pIC50_predictor.joblib')
                # model2 = joblib.load('pIC50_predictor.joblib') 
                model3 = joblib.load('pIC50_predictor1.joblib')
                model4 = joblib.load('active-inactive_predictor3.joblib')
                model5 = joblib.load('BalancedRandomForestClassifier_model6.joblib')

                def draw_compound(canonical_smiles):
                    pic = Chem.MolFromSmiles(canonical_smiles)
                    weight = Descriptors.MolWt(pic)
                    return Draw.MolToImage(pic)
                picim = draw_compound(canonical_smiles)

                col1, col2 = st.columns(2)
                col1.write('')
                col1.write("""<style>.font-family {font-size:15px !important;}</style>""", unsafe_allow_html=True)
                col1.write('<p class="font-family">This is your smile molecule image</p>', unsafe_allow_html=True)
                col1.image(picim)
                
                

                def analyze_compound(canonical_smiles):
                    m = Chem.MolFromSmiles(canonical_smiles)
                    col2.success("The Lipinski's Rule stated the following: Molecular weight < 500 Dalton, Octanol-water partition coefficient (LogP) < 5, Hydrogen bond donors < 5, Hydrogen bond acceptors < 10 ")
                    col2.write('<p class="font-family">Molecule Weight:</p>', unsafe_allow_html=True)
                    col2.code( Descriptors.MolWt(m), "g/mol")
                    col2.write('<p class="font-family">LogP: </p>', unsafe_allow_html=True)
                    col2.code(Descriptors.MolLogP(m))
                    col2.write('<p class="font-family">Hydrogen bond donors: </p>', unsafe_allow_html=True)
                    col2.code(Lipinski.NumHDonors(m))
                    col2.write('<p class="font-family">Hydrogen bond acceptors:</p>', unsafe_allow_html=True)
                    col2.code(Lipinski.NumHAcceptors(m))

                    if Descriptors.MolWt(m) <= np.array(500): 
                        if Descriptors.MolLogP(m) <= np.array(5):
                            if Lipinski.NumHDonors(m) <= np.array(5):
                                if Lipinski.NumHAcceptors(m) <= np.array(10):
                                    str = "your smile is well ✔️"
                                    return str
                                else:
                                    str = "Warning!! your smiles molecule don't pass Lipinski's Rule ❌"
                                    return str
                            else:
                                str = "Warning!! your smiles molecule don't pass Lipinski's Rule ❌"
                                return str
                        else:
                            str = "Warning!! your smiles molecule don't pass Lipinski's Rule ❌"
                            return str
                    else:
                        str = "Warning!! your smiles molecule don't pass Lipinski's Rule ❌"
                        return str
                # analyze_compound(canonical_smiles)
                # st.write(analyze_compound(canonical_smiles))
                col2.warning(analyze_compound(canonical_smiles))
            

                def prediction_pIC50(canonical_smiles):
                    test_morgan_fps = []
                    mol = Chem.MolFromSmiles(canonical_smiles) 
                    info = {}
                    temp = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048,bitInfo=info)
                    test_morgan_fps.append(temp)
                    prediction = model3.predict(test_morgan_fps)
                    return prediction


                def get_h_bond_donors(mol):
                    idx = 0
                    donors = 0
                    while idx < len(mol)-1:
                        if mol[idx].lower() == "o" or mol[idx].lower() == "n":
                            if mol[idx+1].lower() == "h":
                                donors+=1
                        idx+=1
                    return donors
                def get_h_bond_acceptors(mol):
                    acceptors = 0
                    for i in mol:
                        if i.lower() == "n" or i.lower() == "o":
                            acceptors+=1
                    return acceptors

                m = Chem.MolFromSmiles(canonical_smiles)
                MW = Descriptors.MolWt(m)
                NA = m.GetNumAtoms()
                LP =  Descriptors.MolLogP(m)
                SA =  Descriptors.TPSA(m)
                mdataf = {'Molecule Weight': MW , 'ALogP': LP , 'HBD' : NA , 'HBA': SA}
                dfm  = pd.DataFrame([mdataf])
                my_array = np.array(dfm)


                # prediction1 = model3.predict(test_morgan_fps)
                predict_pIC50 = prediction_pIC50(canonical_smiles)
                prediction3 = ' '.join(map(str, predict_pIC50))
                prediction4 = model4.predict(my_array)
                prediction4_2 = ' '.join(map(str, prediction4))
                prediction5 = model5.predict(my_array)
                prediction5_2 = ' '.join(map(str, prediction5))

                # st.text(f"This is predict generate new string smiles molecules : {prediction1}")
                
                # st.write(f"This is predict pIC50: {prediction3}")
                # pIC50 = st.write("This is predict pIC50:", {prediction3} )
                # actin = st.write(f"This is predict active/inactive:", {prediction4_2})
                # appnon = st.write(f"This is predict approve/non-approve:", {prediction5_2})

                with open('style.css') as f:
                    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


                col1, col2, col3 = st.columns(3)
                col1.write("""<style>.font-family {font-size:15px !important;}</style>""", unsafe_allow_html=True)
                col1.write('<p class="font-family">This is predict pIC50 👇</p>', unsafe_allow_html=True)
                col1.code(prediction3)
                
                col2.write("""<style>.font-family {font-size:15px !important;}</style>""", unsafe_allow_html=True)
                col2.write('<p class="font-family">This is predict active/inactive 👇</p>', unsafe_allow_html=True)
                col2.code(prediction4_2)

                col3.write("""<style>.font-family {font-size:15px !important;}</style>""", unsafe_allow_html=True)
                col3.write('<p class="font-family">This is predict approve/non-approve 👇</p>', unsafe_allow_html=True)
                col3.code(prediction5_2)
        except:
            st.error(f"Your smile does not meet the principles of the Lipinski Rules!! ❌")
        
        
       

    

