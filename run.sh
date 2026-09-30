# python src/distance_tuning/calculate_val_val_distances.py

# HAND="right"
# for DISTANCE in "L2" "Proc-weighted" "Quat"; do
# #   echo "=============================================="
# #   echo "🖐 Hand: $HAND | Base: $DISTANCE" 
# #   echo "Output: $OUTPUT_DIR"
# #   echo "=============================================="

#   # Step 1: generate distances + triplets
#   python src/pretraining/1_distance_to_triplets.py \
#       --metric "$DISTANCE" \
#       --hand $HAND \

#   # Step 2: train encoder
#   python src/pretraining/2_distance_encoder.py \
#     --path-extension "${DISTANCE}_r"

#   # python src/pretraining/compare_distances.py \
#   #   --path-extension "${DISTANCE}_r"


#   # Step 3: evaluate DTW + kNN for single hand
#   # python src/pretraining/3_evaluate_encoder_dtw_knn.py \
#   #   --path-extension "${DISTANCE}_r"
# done



#### ==================================================================================

common="--epochs 50 --batch_size 32 --lr 3e-4 --wd 1e-4 --heads 8 --layers 4 --model_dim 256 --use_posenc"

# 1. Baselines
# Run without hand encoder:

# python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split  --no_hand_encoder --seed 1 $common


# Run with hand encoder start from scratch:

# python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split --seed 1 $common


# Run with hand encoder start from scratch and freeze weights:
# python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split --seed 1 --freeze_encoder $common



# 2. Run augmenting with embeddings as a preprocessing step

# echo "$EMB_METHOD"
# python src/training/train.py --data_dir data/"$EMB_METHOD"_aug --epochs 50 --batch_size 32 --lr 3e-4 --wd 1e-4 --seed 1 --heads 8 --layers 4 --model_dim 256 --use_posenc


# 3. Run augmenting with embeddings inside network, should have similar results with 2.

# echo "$EMB_METHOD"
# python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split --encoder_ckpt ./outputs/"$EMB_METHOD"_triplets/dist_encoder.pt --freeze_encoder --seed 1 $common


# 4. Run augmenting with embeddings inside network and allow weights to be trained.

# echo "$EMB_METHOD"
# python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split --epochs 50 --batch_size 32 --lr 3e-4 --wd 1e-4 --seed 1337 --heads 8 --layers 4 --model_dim 256 --use_posenc --encoder_ckpt ./outputs/"$EMB_METHOD"_triplets/dist_encoder.pt 



DATASET="WLASL_npy_dataset_100_split"
# python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split  --no_hand_encoder --seed 4 $common
python src/training/train.py --data_dir data/WLASL_npy_dataset_100_split --seed 5 $common


# 5. Test many distances (multiple encoders)
# Quat_r
# for seed in 1 2 3 4 5; do
#   python src/training/train.py \
#     --data_dir data/$DATASET \
#     $common \
#     --seed $seed \
#     --encoder_ckpt \
#       ./outputs/$DATASET/Quat_r/dist_encoder.pt
# done

# ## Proc-weighted_r
# for seed in 1 2 3 4 5; do
#   python src/training/train.py \
#     --data_dir data/$DATASET \
#     $common \
#     --seed $seed \
#     --encoder_ckpt \
#       ./outputs/$DATASET/Proc-weighted_r/dist_encoder.pt
# done

# ## L2_r
# for seed in 1 2 3 4 5; do
#   python src/training/train.py \
#     --data_dir data/$DATASET \
#     $common \
#     --seed $seed \
#     --encoder_ckpt \
#       ./outputs/$DATASET/L2_r/dist_encoder.pt
# done


# ## L2 + Proc-weighted + Quat (concat)
# for seed in 1 2 3 4 5; do
#   python src/training/train.py \
#     --data_dir data/$DATASET \
#     $common \
#     --seed $seed \
#     --encoder_ckpt \
#       ./outputs/$DATASET/Quat_r/dist_encoder.pt \
#       ./outputs/$DATASET/Proc-weighted_r/dist_encoder.pt \
#       ./outputs/$DATASET/L2_r/dist_encoder.pt \
#       --encoder_fusion concat
# done


# ## L2 + Proc-weighted + Quat (weighted)
# for seed in 1 2 3 4 5; do
#   python src/training/train.py \
#     --data_dir data/$DATASET \
#     $common \
#     --seed $seed \
#     --encoder_ckpt \
#       ./outputs/$DATASET/Quat_r/dist_encoder.pt \
#       ./outputs/$DATASET/Proc-weighted_r/dist_encoder.pt \
#       ./outputs/$DATASET/L2_r/dist_encoder.pt \
#       --encoder_fusion weighted
# done
