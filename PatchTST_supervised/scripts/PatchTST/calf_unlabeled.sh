if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=PatchTST

root_path_name=./dataset/
data_path_name=calf_unlabeled.csv
model_id_name=calf_unlabeled
data_name=custom

random_seed=2026

for seq_len in 100 200
do
  for pred_len in 100 200 300
  do
    python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 4 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 6 \
    --stride 3 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 10 \
    --lradj 'TST' \
    --pct_start 0.2 \
    --itr 1 --batch_size 128 --learning_rate 0.0001 > ./logs/LongForecasting/$model_id_name'_'$seq_len'_'$pred_len.log
  done
done