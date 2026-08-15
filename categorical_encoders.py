# encoders for tmb, ages, ecog, gender, smoking status, DNA sequencing, treatment line, msi
# initial dx stage, clinical trial flag, ethnicity, RNA sequencing, has image, histology,
# primary tumor site

import pandas as pd
import numpy as np

all_vars = ["Race", "DNA Sequencing", "ECOG Score", "Age at Biopsy", "Multiple Primary Cancers", "Ethnicity",
"Specimen Sites", "Num Prior Lines", "Tumor Lineage", "Primary Tumor Site", "RNA Sequencing",
"Initial Dx Smoking Status", "Histology", "Initial Dx Stage", "TMB Value", "LOH Percentage",
"Mutation Count", "Gender", "Age at Sequencing", "patient_is_stage_4", "Age At Diagnosis", "MSI Result", "os_dx_status"]

def nothing(n):
    return n

def yesorno(b):
    if (b == "yes") | (b == "Yes"):
        return 1
    elif (b == "no") | (b == "No"):
        return 0
    else:
        raise Exception()

def age(s):
    if (s == ">89"):
        return 89.0
    else:
        return float(s)
   
def fill_median(col):
    median_tmb = col.median()
    col = col.fillna(median_tmb)
    return col
   
def clip_numbers(x):
    if isinstance(x, str) and x.startswith('>'):
        return float(x[1:])
    return float(x)

def ecog(score):
    if score == 'Grade 5':
        return 5
    elif score == 'Grade 4':
        return 4
    elif score == 'Grade 3':
        return 3
    elif score == 'Grade 2':
        return 2
    elif score == 'Grade 1':
        return 1
    else:
        return 1
   
def gender(g):
    if g == 'Male':
        return 0
    elif g == 'Female':
        return 1
    else:
        raise Exception()

def smoking(status):
    if status == 'non-smoker':
        return 0
    elif status == 'former smoker':
        return 1
    else:
        return 2
   
def dna_sequencing(s):
    if s == "NGSQ3":
        return 1
    elif s == "Exome":
        return 0
    else:
        return 0.5
   
def treatment_line(l):
    if l == "l1":
        return 1
    elif l == "l2":
        return 2
    elif l == "l3":
        return 3
    else:
        raise Exception()
   
def msi_result(m):
    if m == "Stable":
        return 0
    elif m == "High":
        return 1
    else:
        return 0.5
   
def initial_dx_stage(s):
    if s == "Stage 0":
        return 0
    elif s == "Stage 1":
        return 1
    elif ((s == "Stage 2A") | (s == "Stage 2B") | (s == "Stage 2C") | (s == "Stage 2")):
        return 2
    elif ((s == "Stage 3A") | (s == "Stage 3B") | (s == "Stage 3C") | (s == "Stage 3")):
        return 3
    elif ((s == "Stage 4A") | (s == "Stage 4B") | (s == "Stage 4C") | (s == "Stage 4")):
        return 4
    else:
        return 2
   
def clinical_trial_flag(f):
    if f == "NOT On Clinical Trial":
        return 0
    elif f == "On Clinical Trial":
        return 1
    else:
        raise Exception()
   
def ethnicity(e):
    if e == "Hispanic or Latino":
        return 1
    elif e == "Not Hispanic or Latino":
        return 0
    else:
        return 0.5
   
def rna_sequencing(r):
    if r == "Transcriptome":
        return 0
    elif r == "Hybrid":
        return 1
    else:
        raise Exception()
   
def has_image(y):
    if y == "Yes":
        return 1
    elif y == "No":
        return 0
    else:
        raise Exception()
   
def histology(h):
    h = h.lower()

    if "signet ring" in h:
        return "Signet ring cell carcinoma"
    elif "mucinous" in h:
        return "Mucinous adenocarcinoma"
    elif "adenocarcinoma" in h:
        return "Adenocarcinoma"
    elif "carcinoma" in h:
        return "Carcinoma NOS"
    else:
        return "Other histology"

def primary_tumor_site(site):
    s = site.lower()

    if any(x in s for x in ["cecum", "ascending", "ileocecal", "transverse"]):
        return "Right colon"
    elif any(x in s for x in ["descending", "sigmoid"]):
        return "Left colon"
    elif "rectosigmoid" in s:
        return "Rectosigmoid"
    elif "rectum" in s:
        return "Rectum"
    elif "colon, nos" in s:
        return "Colon NOS"
    elif "overlapping" in s:
        return "Overlapping / Other"
    else:
        return "Other primary tumor site"

def cms(t):
    if t == "CMS1":
        return 1
    elif t == "CMS2":
        return 2
    elif t == "CMS3":
        return 3
    elif t == "CMS4":
        return 4
    else:
        return 2.5

def specimen_site(site):
    if site is None or (isinstance(site, float) and np.isnan(site)):
        return "unknown"
   
    s = site.lower()

    # keyword dictionary
    categories = {
        "colon_rectum": [
            "colon", "rectum", "rectosigmoid", "cecum", "appendix",
            "ileocecal", "anus", "anorectum"
        ],
        "lung": [
            "lung", "bronchus", "bronchial", "pleura"
        ],
        "liver": [
            "liver", "hepatic", "porta hepatis"
        ],
        "lymph_node": [
            "lymph node", "lymph nodes"
        ],
        "peritoneal": [
            "peritoneum", "omentum", "mesentery", "mesocolon",
            "retroperitoneum", "retroperitoneal", "peritoneal cavity"
        ],
        "brain_cns": [
            "brain", "cerebellum", "frontal lobe", "parietal lobe",
            "occipital lobe", "spinal cord", "posterior cranial fossa"
        ],
        "bone": [
            "bone", "femur", "vertebra", "rib", "sternum", "clavicle",
            "sacrum", "ilium", "maxilla"
        ],
        "female_reproductive": [
            "ovary", "uterus", "cervix", "endocervix", "fallopian",
            "vagina", "tubo-ovarian", "adnexa", "endometrium",
            "corpus uteri"
        ],
        "urinary": [
            "bladder", "ureter", "urethra"
        ],
        "skin_soft_tissue": [
            "skin", "subcutaneous", "soft tissue", "buttock",
            "gluteal", "perineum", "abdominal wall", "chest wall"
        ],
        "other_gi": [
            "stomach", "duodenum", "jejunum", "ileum",
            "small bowel", "small intestine", "pancreas",
            "gallbladder", "bile duct", "gastroesophageal junction"
        ],
    }

    for category, keywords in categories.items():
        if any(k in s for k in keywords):
            return category

    return "other_specimen_site"

def initial_dx_code_names(text):
    text = text.lower()
    regions = set()
   
    if any(x in text for x in ["cecum", "ascending colon", "hepatic flexure"]):
        regions.add("right_colon")
       
    if any(x in text for x in [
        "transverse colon",
        "splenic flexure",
        "descending colon",
        "sigmoid colon"
    ]):
        regions.add("left_colon")
       
    if any(x in text for x in ["rectosigmoid junction", "rectum"]):
        regions.add("rectum")
       
    if "appendix" in text:
        regions.add("appendix")
       
    return list(regions)
 
def treatment(t, t0, t1):
    if t == t0:
        return 0
    elif t == t1:
        return 1
    else:
        raise Exception()
   
def event_status(s):
    if s == "1:DECEASED" or s == 1:
        return 1
    elif s == "0:LIVING" or s == 0:
        return 0
    else:
        raise Exception()
   



func_dict = {"Race": nothing,
        "DNA Sequencing": dna_sequencing,
        "ECOG Score": ecog,
        "Age at Biopsy": age,
        "Multiple Primary Cancers": yesorno,
        "Ethnicity": ethnicity,
        "CD8A TPM": fill_median,
        "Specimen Sites": specimen_site,
        "Num Prior Lines": nothing,
        "Tumor Lineage": nothing, # stays categorical
        "Primary Tumor Site": primary_tumor_site,
        "RNA Sequencing": rna_sequencing,
        "Initial Dx Smoking Status": smoking,
        "Histology": histology,
        "Initial Dx Stage": initial_dx_stage,
        "TMB Value": fill_median, # tmb
        "LOH Percentage": fill_median,
        "Mutation Count": fill_median,
        "initial_dx_code_names": nothing,
        "Gender": gender,
        "Age at Sequencing": age,
        "patient_is_stage_4": yesorno,
        "CMS Subtype": cms,
        "Age At Diagnosis": age,
        "MSI Result": msi_result,
        "Call": cms,
        "os_dx_status": event_status}
   

# use these encoders if the category is there
def encode(df, t1_name, t2_name):
    cat_cols = []
    for col in func_dict.keys():
        if col in df.columns:
            func = func_dict[col]
            if func != fill_median:
                df[col] = df[col].apply(func)
            else:
                df[col] = fill_median(df[col])
            if col in ["initial_dx_code_names", "Primary Tumor Site", "Histology", "Specimen Sites", "Tumor Lineage", "Race"]:
                cat_cols.append(col)
    # also encode treatment
    if (t1_name != None):
        df["Treatment"] = df["Treatment"].apply(treatment, args=(t1_name, t2_name))
   
    return pd.get_dummies(df, columns=cat_cols)
