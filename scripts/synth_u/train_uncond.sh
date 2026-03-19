#CUDA_VISIBLE_DEVICES=1 python run.py \
#    --cond_modal text \
#    --training_stage pretrain \
#    --save_folder ./verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv \
#    --model_diff_config_path configs/synth_u/diff/model_text2ts_dep.yaml \
#    --model_cond_config_path configs/synth_u/cond/text_msmdiffmv.yaml \
#    --train_config_path configs/synth_u/train.yaml \
#    --evaluate_config_path configs/synth_u/evaluate.yaml \
#    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u \
#    --clip_folder "" \
#    --multipatch_num 3 \
#    --L_patch_len 2 \
#    --base_patch 4 \
#    --epochs 2500 \
#    --batch_size 512 \
#    --clip_cache_path "" \
#    --only_evaluate True \
#    --samples_name "samples.pt"

CUDA_VISIBLE_DEVICES=3 python run_debug.py \
    --cond_modal text \
    --training_stage pretrain \
    --save_folder ./verbalts_orig_save/uncond_synth_m/text2ts_msmdiffmv \
    --model_diff_config_path configs/synth_m_debug/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth_m_debug/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth_m_debug/train.yaml \
    --evaluate_config_path configs/synth_m_debug/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/synthetic_m \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --samples_name "samples.pt" \
    --only_evaluate True

  CUDA_VISIBLE_DEVICES=3 python run_debug.py \
    --cond_modal text \
    --training_stage pretrain \
    --save_folder ./verbalts_orig_save/uncond_istanbul_traffic/text2ts_msmdiffmv \
    --model_diff_config_path configs/istanbul_debug/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/istanbul_debug/cond/text_msmdiffmv.yaml \
    --train_config_path configs/istanbul_debug/train.yaml \
    --evaluate_config_path configs/istanbul_debug/evaluate.yaml \
    --data_folder /playpen/haochenz/LitsDatasets/128_len_ts/istanbul_traffic \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 2 \
    --base_patch 4 \
    --epochs 2500 \
    --batch_size 512 \
    --clip_cache_path "" \
    --samples_name "samples.pt" \
    --only_evaluate True
