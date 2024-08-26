
prepPath=/public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/preprocess.py

## MedChem AMPK
python $prepPath -i  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/AIXB-3_AMPK/AIXB-3_AMPK_MMP.csv  -d 1 -r 0.8
## MedChem JAK
python $prepPath -i  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/AIXB-3_JAK1/AIXB-3_JAK1_MMP.csv  -d 1 -r 0.8
# JAK + MedChem
python $prepPath -i  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/JAK1_addAIXB/JAK1_addAIXB_MMP.csv   -d 1 -r 0.8
## JAK
python $prepPath -i  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/JAK1_cano/JAK1_cano_MMP.csv  -d 1 -r 0.8
## PRKAB1 + MedChem
python $prepPath -i  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/PRKAB1_addAIXB/PRKAB1_addAIXB_MMP.csv  -d 1 -r 0.8
## PRKAB1 
python $prepPath -i  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/PRKAB1_cano/PRKAB1_cano_MMP.csv   -d 1 -r 0.8
