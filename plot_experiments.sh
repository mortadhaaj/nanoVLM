#!/bin/bash\

#################
# Global Image Token or not
#################
python plot_eval_results.py \
    Token:'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-091343' \
    No-Token:'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-131841' \
    'Token&Resize':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-144934' \
    --output global_image_token

#################
# Experiments 4.1 (Against Baselines)
# TODO: Rerun Cauldron, Cambrian and LLaVa
#################
python plot_eval_results.py \
    Cauldron:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_3489samples_bs512_40000_lr5e-05-0.00512_0811-092351' \
    Cambrian:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_13814samples_bs512_40000_lr5e-05-0.00512_0811-101603' \
    LLaVa:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_7726samples_bs512_40000_lr5e-05-0.00512_0811-130750' \
    FineVision:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-153841' \
    --output against_baselines \
    --tasks 'mmmu_val_mmmu_acc' 'mmstar_average' 'ocrbench_ocrbench_accuracy' 'average' 'textvqa_val_exact_match'

#################
# Experiments 4.b (Internal Deduplication)
# TODO: Run additional Benchmarks
#################
python plot_eval_results.py \
    IntDedup:'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_36851samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-132458' \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results_andi/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0814-091343' \
    --output internal_deduplication \
    --steps 300 1500 2700 3900 5100 6300 7500 8700 9900 11100 12300 13500 14700 15900 17100 18300 19500 20700 21900 23100 24300 25500 26700 27900 29100 30300 31500 32700 33900 35100 36300 37500 38700 39900

#################
# Experiments 4.d) i) (Individual ratings)
#################

# Plot Baseline
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-153841' \
    --output baseline \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

# Plot Relevance Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-153841' \
    '>=2':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-165157' \
    '>=3':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-172025' \
    '>=4':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-173121' \
    '>=5':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-174041' \
    --output relevance_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

# Plot Image Correspondence Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-153841' \
    '>=2':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-205752' \
    '>=3':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-210619' \
    '>=4':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-105432' \
    '>=5':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-145130' \
    --output image_correspondence_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

# Plot Visual Dependency Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-153841' \
    '>=2':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-130314' \
    '>=3':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-150042' \
    '>=4':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20000_lr_vision_5e-05-language_5e-05-0.00512_0820-165133' \
    '>=5':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-095710' \
    --output visual_dependency_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

# Plot Formatting Filters
python plot_eval_results.py \
    Baseline:'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0819-153841' \
    '>=2':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-100810' \
    '>=3':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-103222' \
    '>=4':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-131717' \
    '>=5':'/fsx/luis_wiedmann/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_1536_mp4_SmolLM2-360M-Instruct_32xGPU_48206samples_bs512_20100_lr_vision_5e-05-language_5e-05-0.00512_0821-115740' \
    --output formatting_filters \
    --steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000

#################
# Experiments 4.d) ii) (All ratings)
# TODO: Rerun with proper setup
#################
python plot_eval_results.py \
    'All_Samples':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0812-110026' \
    '>=2':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-063155' \
    '>=3':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0812-004500' \
    '>=4':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0812-033512' \
    '>=5':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0815-082051' \
    --output all_ratings
    
    #'>=4cont':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-081736' \

#################
# Experiments 4.e) (Multiple Stages)
# TODO: Rerun with proper setup
#################
python plot_eval_results.py \
    '1_on_top_1':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-081736' \
    '3_on_top_1':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-101418' \
    '5_on_top_1':'/fsx/andi/nanoVLM/eval_results/nanoVLM_siglip2-base-patch16-512_2048_mp4_SmolLM2-360M-Instruct_32xGPU_39902samples_bs512_40000_lr_vision_5e-05-language_5e-05-0.00512_0813-125149' \
    --output multi_stage
