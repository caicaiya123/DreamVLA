#!/bin/bash

# --- 1. 环境设置 ---
source examples/libero_plus/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/inspire/ssd/project/wuliqifa/zhousizhuo-240107010006/moe-vla/LIBERO-plus

# --- 2. 任务定义 ---
TASKS=(
    # "Objects Layout|layout"
    # "Light Conditions|light"
    # "Background Textures|texture"
    "Camera Viewpoints|camera"
    "Robot Initial States|state"
    "Language Instructions|language"
    # "Sensor Noise|noise"
)

# --- 3. 配置参数 ---
GPUS=(1 0)
BASE_PORT=8041
NUM_GPUS=${#GPUS[@]}

# --- 【关键修改：使用带时间戳的日志文件夹】 ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="tmp/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "开始并行测试，共 ${#TASKS[@]} 个任务，分配在 ${NUM_GPUS} 张显卡上..."
echo "日志将保存至: $LOG_DIR"

# --- 4. 循环执行 ---
counter=0
for task_str in "${TASKS[@]}"; do
    category="${task_str%|*}"
    keyword="${task_str#*|}"

    gpu_index=$((counter % NUM_GPUS))
    gpu_id=${GPUS[$gpu_index]}
    port=$((BASE_PORT))

    out_path="data/pi0_libero_object_no_control_cst_lr_libero_spatial_${keyword}/"
    json_path="${out_path}results.json"

    echo "------------------------------------------------"
    echo "任务 [$counter]: $category (GPU: $gpu_id, Port: $port)"
    
    mkdir -p "$out_path"

    # 使用新的 LOG_DIR 存放日志
    CUDA_VISIBLE_DEVICES=$gpu_id python examples/libero_plus/main.py \
      --args.task-suite-name libero_spatial \
      --args.category "$category" \
      --args.video_out_path "$out_path" \
      --args.results_json_path "$json_path" \
      --args.num_trials_per_task 1 \
      --args.port $port \
      > "${LOG_DIR}/${keyword}.log" 2>&1 &

    ((counter++))
    sleep 1
done

echo "------------------------------------------------"
echo "所有任务已启动。查看进度请执行:"
echo "tail -f ${LOG_DIR}/*.log"
echo "------------------------------------------------"

wait
echo "所有测试完成！结果已存入各 data/ 子目录。"