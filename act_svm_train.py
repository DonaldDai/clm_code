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

def load_data(filename):
    df = pd.read_csv(filename)
    # 转换SMILES列到指纹
    df['features'] = df['cpd2SMILES'].apply(smiles_to_fp)
    # 分割特征和标签
    X = np.array(list(df['features']))
    y = df['cpd2_act'].values
    return X, y
  
def train_and_evaluate(X, y):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=678)
    
    # 创建一个包含预处理的SVM分类器
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    dump(clf, './drd2_act_svm_model.joblib')
    return clf

# 主执行流
filename = '/work/09735/yichao/ls6/zhilian/clm_code/sample_data/drd2_ki_test.csv'  # 假设你的文件名为data.csv
X, y = load_data(filename)
model = train_and_evaluate(X, y)