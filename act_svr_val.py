from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import load
import pandas as pd

def smiles_to_features(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

def evaluate_model(model, X_test, y_test, r=0.05):
    y_pred = model.predict(X_test)
    percent = np.mean(np.abs((y_pred - y_test) / y_test) <= r) * 100
    print(f"Percentage of predictions within {r*100}% of actual values: {percent}%")
    return percent

# 使用一个新的SMILES字符串进行预测
# 假设数据存储在CSV文件中
data_path = '/work/09735/yichao/ls6/zhilian/clm_code/sample_data/drd2_ki_test.csv'
data = pd.read_csv(data_path)

# 假设数据列名为 'cpd2SMILES' 和 'cpd2Value'
features = np.array([smiles_to_features(sm) for sm in data['cpd2SMILES']])
targets = data['cpd2Value'].values
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=678)
model = load('./drd2_act_svr_model.joblib')
evaluate_model(model, X_test, y_test, r=0.05)