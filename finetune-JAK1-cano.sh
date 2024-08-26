pretrainPath="/public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/PretrainWork/pretrain_chembl32_2"
vocabPath="/public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/PretrainWork/ChEMBL32_Data"
 
 ## dataset AIXB-3_JAK1
 python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/JAK1_cano  --save-directory FinetunedModels/finetune-JAK1_cano --starting-epoch 36 --pretrain-path  ${pretrainPath}  --vocab-path ${vocabPath}