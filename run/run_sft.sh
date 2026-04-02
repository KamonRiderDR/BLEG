dataset="HCP"                   
dataset_name="HCP"               
device=1
date=`date +%F`
cur_root=`pwd`
root=$(dirname "${cur_root}")
batch_size=16 
epochs=300
epochs_pretrain=60
training_type="sft" 
dropout=0
lr=0.0001 
weight_decay=0.01 
patience=100
python_file="Trainer_sft.py"

exp_type="k_way"
few_shot_hyper=0.7
times=10

nohup \
    python -u ../${python_file} \
    --dataset ${dataset} \
    --device ${device} \
    --root ${root} \
    --path ${root}/data/${dataset_name} \
    --result_path ${root}/logs/results/${dataset}_${device}_${training_type}_${exp_type}.log \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --epochs_pretrain ${epochs_pretrain} \
    --patience ${patience} \
    --training_type ${training_type} \
    --dropout ${dropout} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
    --few_shot_hyper ${few_shot_hyper} \
    --exp_type ${exp_type} \
    --times ${times} \
    > ${root}/logs/out/${date}_${dataset}_${training_type}_${exp_type}_${device}.log  2>&1 &