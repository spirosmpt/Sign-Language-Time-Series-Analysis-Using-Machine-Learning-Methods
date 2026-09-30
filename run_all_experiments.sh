#!/bin/bash
set -e

DATA_PATH="data/WLASL_npy_dataset_100_split"
NUM_TRIPLETS=200000
EMB_DIM=64
SUBSAMPLING=0.1
EPOCHS=200

BASE_DISTANCES=("Quat" "L2" "Proc")
HANDS=("right" "left")

# Helper: determine weights
function get_weights() {
    local base=$1
    local with_wtr=$2
    if [ "$with_wtr" == "yes" ]; then
        if [ "$base" == "L2" ]; then
            echo "0.9 0.1"
        else
            echo "0.7 0.3"
        fi
    else
        echo "1.0"
    fi
}

# =====================================================
# 1️⃣ TRAIN & EVALUATE EACH HAND SEPARATELY
# =====================================================
for HAND in "${HANDS[@]}"; do
  for BASE in "${BASE_DISTANCES[@]}"; do
    for WITH_WTR in "no" "yes"; do

      if [ "$WITH_WTR" == "yes" ]; then
        DISTANCES=("$BASE" "wtr")
        WEIGHTS=($(get_weights "$BASE" "$WITH_WTR"))
        SUFFIX="${BASE,,}_wtr"
      else
        DISTANCES=("$BASE")
        WEIGHTS=($(get_weights "$BASE" "$WITH_WTR"))
        SUFFIX="${BASE,,}"
      fi

      OUTPUT_DIR="outputs/${HAND}_${SUFFIX}_triplets"

      echo "=============================================="
      echo "🖐 Hand: $HAND | Base: $BASE | WTR: $WITH_WTR"
      echo "Output: $OUTPUT_DIR"
      echo "=============================================="

      # Step 1: generate distances + triplets
      python src/pretraining/1_distance_to_triplets.py \
        --distances "${DISTANCES[@]}" \
        --weights "${WEIGHTS[@]}" \
        --subsampling-rate $SUBSAMPLING \
        --hand $HAND \
        --num-triplets $NUM_TRIPLETS \
        --output $OUTPUT_DIR \
        --pos-k 30 \
        --neg-start 50 \
        --neg-end 100


      # Step 2: train encoder
      python src/pretraining/2_distance_encoder.py \
        --X-path $OUTPUT_DIR/X_train.npy \
        --triplets-path $OUTPUT_DIR/triplets.npy \
        --triplets-test-path $OUTPUT_DIR/triplets_test.npy \
        --emb-dim $EMB_DIM \
        --epochs $EPOCHS \
        --save-path $OUTPUT_DIR/dist_encoder.pt \
        --lr 1e-3

      # Step 3: evaluate DTW + kNN for single hand
      python src/pretraining/3_evaluate_encoder_dtw_knn.py \
        --encoder-path $OUTPUT_DIR/dist_encoder.pt \
        --data-path $DATA_PATH \
        --hand $HAND \
        --emb-dim $EMB_DIM

    done
  done
done


# =====================================================
# 2️⃣ EVALUATE BOTH HANDS FUSED
# =====================================================
echo "=============================================="
echo "👐 Evaluating both hands fused (for all configs)"
echo "=============================================="

for BASE in "${BASE_DISTANCES[@]}"; do
  for WITH_WTR in "no" "yes"; do

    if [ "$WITH_WTR" == "yes" ]; then
      SUFFIX="${BASE,,}_wtr"
    else
      SUFFIX="${BASE,,}"
    fi

    RIGHT_PATH="outputs/right_${SUFFIX}_triplets/dist_encoder.pt"
    LEFT_PATH="outputs/left_${SUFFIX}_triplets/dist_encoder.pt"

    echo "🤝 Fusion: Base=$BASE | WTR=$WITH_WTR"
    echo "Right: $RIGHT_PATH"
    echo "Left:  $LEFT_PATH"

    python src/pretraining/3_evaluate_encoder_dtw_knn.py \
      --encoder-paths "$RIGHT_PATH" "$LEFT_PATH" \
      --hands right left \
      --emb-dims $EMB_DIM $EMB_DIM \
      --weights 0.5 0.5 \
      --data-path $DATA_PATH
  done
done
