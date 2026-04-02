dataset="pretrain"                    
dataset_name=""                       
device=2
date=`date +%F`
cur_root=`pwd`
root=$(dirname "${cur_root}")
batch_size=8 
epochs=300
epochs_pretrain=60
training_type="instruct" 
dropout=0
lr=0.0001 
weight_decay=0.01
patience=100
python_file="Trainer_instruct.py"



nohup \
    python -u ../${python_file} \
    --dataset ${dataset} \
    --device ${device} \
    --root ${root} \
    --path ${root}/data/${dataset_name} \
    --result_path ${root}/logs/results/${dataset}_${device}_${training_type}.log \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --epochs_pretrain ${epochs_pretrain} \
    --patience ${patience} \
    --training_type ${training_type} \
    --dropout ${dropout} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
    > ${root}/logs/out/${date}_${dataset}_${training_type}_${device}.log  2>&1 &