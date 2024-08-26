 
pretrainPath="/public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/PretrainWork/pretrain_chembl32_2"
vocabPath="/public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/PretrainWork/ChEMBL32_Data"

 
 ## dataset AIXB-3_AMPK
#  python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/AIXB-3_AMPK  --save-directory FinetunedModels/finetune-AIXB3-AMPK --starting-epoch 36 --pretrain-path  ${pretrainPath}  --vocab-path ${vocabPath} 

 ## dataset AIXB-3_JAK1
 python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/AIXB-3_JAK1  --save-directory FinetunedModels/finetune-AIXB3-JAK1 --starting-epoch 36 --pretrain-path  ${pretrainPath}  --vocab-path ${vocabPath}

#  ## dataset JAK1_addAIXB
#   python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/FinetuningData/AIXB-3/JAK1_addAIXB  --save-directory FinetunedModels/finetune-JAK1_addAIXB --starting-epoch 43 --pretrain-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/Pretrain/train  --vocab-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/TrainData_ChEMBL17

#  ## dataset JAK1_cano
#    python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/FinetuningData/AIXB-3/JAK1_cano  --save-directory FinetunedModels/finetune-JAK1_cano --starting-epoch 43 --pretrain-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/Pretrain/train  --vocab-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/TrainData_ChEMBL17

 ## dataset PRKAB1_addAIXB
   python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/PRKAB1_addAIXB  --save-directory FinetunedModels/PRKAB1_addAIXB --starting-epoch 36 --pretrain-path  ${pretrainPath}  --vocab-path ${vocabPath}

## dataset PRKAB1_cano
   python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v2/FinetuningData/AIXB-3/PRKAB1_cano  --save-directory FinetunedModels/PRKAB1_cano --starting-epoch 36 --pretrain-path  ${pretrainPath}  --vocab-path ${vocabPath}

