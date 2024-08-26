#%%
import os,sys
import pandas as pd, numpy as np
from pathlib import Path
import argparse
from mmpdblib.fragment_io import read_fragment_records
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvFile", help="the name of csv file", required=True, default='')
    parser.add_argument("--smilesCol", nargs='+', help="the column name of SMILES", required=False, default='SMILES')
    parser.add_argument("--titleCol", nargs='+', help="the column name of SMILES", required=False, default='title')
    parser.add_argument("--maxRatio", type=float, help="the column name of SMILES", required=False, default=0.5)
    args = parser.parse_args()
    return args

args = get_parser()

class Index_Dummy:
    ''' add index to the dummy atoms  '''
    def __init__(self, df=''):
        self.df = df
   
    def index_constant(self,constSmi,attachmentOrder):
        count=-1
        newConstSmi=""
        for idx,ichar in enumerate(constSmi):
            if ichar=='*':
                count+=1
                ichar=f"[*:{int(attachmentOrder[count])+1}]"  
            newConstSmi+=ichar
        return newConstSmi  

    def index_var(self,varSmi):
        count=0
        newVarSmi=''
        for idx,ichar in enumerate(varSmi):
            if ichar=='*':
                count+=1
                ichar=f"[*:{count}]"
            newVarSmi+=ichar
        return newVarSmi  


    def list2str(self,strlist):
        newStr=''
        for istr in strlist:
            newStr+=istr
        return newStr

    def add_index(self):
        for idx,irow in self.df.iterrows():
            varSmi=irow['fromVarSMILES']
            constSmi=irow['constantSMILES']
            attachmentOrder=irow['attachmentOrder']
            self.df.loc[idx,'fromVarSMILES']=self.index_var(varSmi)
            self.df.loc[idx,'constantSMILES']=self.index_constant(constSmi,attachmentOrder)
        return self.df
    
def count_heavy_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    heavy_count=len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
    return heavy_count

'''  Fragmentize the molecules  '''
inputFile=args.csvFile
inputFile_path=Path(inputFile)
inputFile_stem=inputFile_path.stem

dfInput=pd.read_csv(inputFile_path)
dfInput['SMILES']=dfInput[args.smilesCol]
if args.titleCol not in dfInput.columns:
    dfInput['title']=[f"ID-{i}" for i in range(len(dfInput))]
else:
    dfInput['title']=dfInput[args.titleCol]

os.chdir(inputFile_path.parent)
dfInput[['SMILES', 'title']].to_csv(f"{inputFile_stem}.smi", sep='\t', header=None, index=None)


os.system(f"mmpdb fragment {inputFile_stem}.smi -o {inputFile_stem}.fragments")

fragmentReader=read_fragment_records(f"{inputFile_stem}.fragments")
fragList=[]
for recno, record in enumerate(fragmentReader, 1):
    print(record.id, record.normalized_smiles)
    ''' frag.num_cuts, frag.enumeration_label,
                        frag.variable_num_heavies, 
                        frag.variable_symmetry_class, 
                        frag.variable_smiles,
                        frag.attachment_order, 
                        frag.constant_num_heavies, 
                        frag.constant_symmetry_class,
                        frag.constant_smiles, 
                        frag.constant_with_H_smiles
    '''
    for frag in record.fragments:
        if count_heavy_atoms(frag.variable_smiles) < count_heavy_atoms(record.normalized_smiles)*args.maxRatio:
            fragList.append([frag.variable_smiles,frag.constant_smiles,record.id, record.normalized_smiles,frag.attachment_order])

dfFrag=pd.DataFrame(fragList, columns=['fromVarSMILES','constantSMILES','id','normalizedSMILES',"attachmentOrder"])
index_dummy=Index_Dummy(dfFrag)
dfFrag=index_dummy.add_index()
fragmentCsv=f"{inputFile_stem}-frag.csv"
dfFrag.to_csv(fragmentCsv, index=None)

# os.system(f"python /shared/data/jay.zhang/Codes/Scripts/csv2Excel.py --csvFile {fragmentCsv} --smilesCol fromVarSMILES constantSMILES  normalizedSMILES")




#%%
