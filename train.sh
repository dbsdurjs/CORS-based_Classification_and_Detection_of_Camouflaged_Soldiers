export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export BASE_INSTANCE_DIR="./data_dir"
export CLASS_DIR="./class_dir"
export BASE_OUTPUT_DIR="./saved_results"

for i in {1..10}
do
  INSTANCE_DIR="$BASE_INSTANCE_DIR/train${i}"
  OUTPUT_DIR="$BASE_OUTPUT_DIR/result${i}"

  echo "Processing $INSTANCE_DIR and saving to $OUTPUT_DIR"

  accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir=$CLASS_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks camouflaged soldier" \
    --class_prompt="a photo of camouflaged soldier" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=256 \
    --max_train_steps=1000 \
    --push_to_hub \
    --use_8bit_adam \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    #--set_grads_to_none
done