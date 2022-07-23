from email.policy import default
import streamlit as st 
from streamlit_option_menu import option_menu
import codecs
import pickle
import joblib
from rdkit.Chem import AllChem
from rdkit import Chem
import requests
from streamlit_lottie import st_lottie


st.set_page_config(layout="wide")

#### tab bar ####
selected = option_menu(
    menu_title=None, 
    options=["Home", "About us", "How to use","Predict new smiles"], 
    icons=["house","book","question-circle","search"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal", #แนวนอนนน
    styles={
        "container": {"padding": "0!important", "background-color": "#CAF1E9"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#B5E6E5"},
    }
)



#### sticker image ####
def load_lottieurl(url):
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
    st.title(f"Welcome to Design drug discovery of Breast cancer from Pre-clinical stage and check of new smiles molecules.")
    # ---- LOAD ASSETS ----
    lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_nw19osms.json")
    st_lottie(lottie_coding, height=350, key="coding")
    index_title = '<p style="ont-family:Courier; color:Black; font-size: 20px;">This is the introduction of drug molecules from the protein targets of the disease that require drug discovery to explore possibilities before further study in drug productio.</p>'
    st.markdown(index_title, unsafe_allow_html=True)

if selected =="About us":
    st_webpage('about.html')
if selected =="How to use":
    st_webpage('How_to_use.html')
if selected =="Predict new smiles":
    st.title(f"Predict new smiles")
    canonical_smiles = st.text_input("Enter canonical_smiles")
    

    if st.button("Predict"):
    
        model2 = joblib.load('pIC50_predictor.joblib') 
        test_morgan_fps = []
        mol = Chem.MolFromSmiles(canonical_smiles) 
        info = {}
        temp = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048,bitInfo=info)
        test_morgan_fps.append(temp)
        prediction = model2.predict(test_morgan_fps)

        st.text(f"This is predict : {prediction}")

    
   


