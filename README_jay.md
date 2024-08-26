# data preprocess
`python preprocess.py -i /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/FinetuningData/AIXB-3/AIXB-3_AMPK/AIXB-3_AMPK_MMP.csv -d 1`

```shell
python preprocess.py -d 1 -i /home/yichao/zhilian/GenAICode/Data/MMPFinised/BindingDB_All_202407_5k/BindingDB_All_202407_5k_MMP.csv
```

# pretrian
`python train.py --model-choice transformer  --data-path  PretrainWork/ChEMBL32_Data   --save-directory PretrainWork/pretrain_chembl32`

```shell
python train.py --model-choice transformer  --data-path  /home/yichao/zhilian/GenAICode/Data/MMPFinised/BindingDB_All_202407_5k  --save-directory /home/yichao/zhilian/GenAICode/CLModel_v2/pretrain_temp
```

# fine tuning
`python train.py --model-choice transformer  --data-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/FinetuningData/AIXB-3/AIXB-3_JAK1  --save-directory FinetunedModels/finetune-AIXB3-JAK1 --starting-epoch 43 --pretrain-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/Pretrain/train  --vocab-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/TrainData_ChEMBL17`

# generate
` python generate.py --model-choice transformer$$  --data-path  /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/test_data   --test-file-name  test  --model-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/pretrain_test/checkpoint  --epoch 40    --vocab-path /public/home/zhangjie/Projects/MMP/pot_clm/CLModel_v1/TrainData_ChEMBL17  --save-directory FinetunedModels/finetune-AIXB3-JAK1 `

```shell
python generate.py --model-choice transformer  --data-path  /home/yichao/zhilian/GenAICode/Data/MMPFinised/BindingDB_All_202407_5k   --test-file-name  test  --model-path /home/yichao/zhilian/GenAICode/CLModel_v2/pretrain/checkpoint  --epoch 200    --vocab-path /home/yichao/zhilian/GenAICode/Data/MMPFinised/BindingDB_All_202407_5k  --save-directory /home/yichao/zhilian/GenAICode/CLModel_v2/generate_temp
```