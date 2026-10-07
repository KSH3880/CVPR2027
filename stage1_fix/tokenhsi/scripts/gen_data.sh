#!/bin/bash

cfg_path="./tokenhsi/data/dataset_cfg.yaml"

dataset="dataset_amass_loco"
printf "\n\n"
echo "Processing $dataset"
printf "\n\n"
python ./tokenhsi/data/$dataset/preprocess.py --dataset_cfg $cfg_path
python ./tokenhsi/data/$dataset/generate_motion.py
printf "\n\n"
echo "Done!"

dataset="dataset_amass_loco_stairs"
printf "\n\n"
echo "Processing $dataset"
printf "\n\n"
python ./tokenhsi/data/$dataset/preprocess.py --dataset_cfg $cfg_path
python ./tokenhsi/data/$dataset/generate_motion.py
printf "\n\n"
echo "Done!"

dataset="dataset_amass_climb"
printf "\n\n"
echo "Processing $dataset"
printf "\n\n"
python ./tokenhsi/data/$dataset/preprocess.py --dataset_cfg $cfg_path
python ./tokenhsi/data/$dataset/generate_motion.py
python ./tokenhsi/data/$dataset/generate_object.py
printf "\n\n"
echo "Done!"

dataset="dataset_sit"
printf "\n\n"
echo "Processing $dataset"
printf "\n\n"
python ./tokenhsi/data/$dataset/preprocess_samp.py --dataset_cfg $cfg_path
python ./tokenhsi/data/$dataset/generate_motion.py
python ./tokenhsi/data/$dataset/generate_object.py
printf "\n\n"
echo "Done!"

dataset="dataset_carry"
printf "\n\n"
echo "Processing $dataset"
printf "\n\n"
python ./tokenhsi/data/$dataset/preprocess_amass.py --dataset_cfg $cfg_path
python ./tokenhsi/data/$dataset/preprocess_omomo.py --dataset_cfg $cfg_path
python ./tokenhsi/data/$dataset/generate_motion.py
python ./tokenhsi/data/$dataset/generate_object.py
printf "\n\n"
echo "Done!"
