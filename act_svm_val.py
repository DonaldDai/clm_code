import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from joblib import dump, load

def smiles_to_fp(smiles, n_bits=2048):
    # 将SMILES转换为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits, ), dtype=int)
    # 计算分子的指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def predict_smiles(smiles, model_path='./drd2_act_svm_model.joblib'):
    clf = load(model_path)
    fps = np.array(list(map(smiles_to_fp, smiles)))
    prediction = clf.predict(fps)
    probability = clf.predict_proba(fps)
    return prediction, probability

# 使用一个新的SMILES字符串进行预测
smiles_example = ['COc1ccc(CN2CCc3cc4nc(N)sc4cc3CC2)cc1', 'CCCN1CCC(c2cn(C)c3cc(F)ccc23)CC1']  # 苯环
prediction, probability = predict_smiles(np.array(smiles_example))
print(f"Prediction: {prediction}, Probability: {probability}")