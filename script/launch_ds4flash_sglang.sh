pkill -9 python | pkill -9 sglang
sleep 3
pkill -9 python | pkill -9 sglang
sleep 2
rm -rf kernel_meta
rm -rf extra-info
rm -rf /root/ascend/

# cpu high performance
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
export SGLANG_SET_CPU_AFFINITY=1

# unset proxy
unset http_proxy
unset https_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

# CANN env var
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
export TASK_QUEUE_ENABLE=1
# unset sync kernel
unset ASCEND_LAUNCH_BLOCKING

# torch stream upper limit setting
export STREAMS_PER_DEVICE=32
# torch optimize memory fragmentation
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 设置SGLANG代码路径
export PYTHONPATH=${PWD}/python:$PYTHONPATH
MODEL_PATH=/workspace/models/DeepSeek-V4-Flash-W8A8/

### HCCL env
IFNAMES=eth0  # get name according to ifconfig
export HCCL_SOCKET_IFNAME=$IFNAMES
export GLOO_SOCKET_IFNAME=$IFNAMES
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=2048

# Fused Kernels
# Compressor
export USE_FUSED_COMPRESSOR=1
# Lightning Indexer
export LI_KV_DTYPE_INT8=1
# SparseAttentionSharedKV
export USE_PA_DECODE=1
export USE_PA_PREFILL=1
#HC Pre / HC Post
export USE_FUSED_HC_POST_ASCENDC=1
export USE_FUSED_HC_PRE_ASCENDC=1
# MOE Gating TopK
export USE_NPU_MOE_GATING_TOP_K=1
# Transpose BatchMatmul
export USE_FUSED_TRANSPOSE_BATCHMATMUL=1
# Inplace Partial Rotary Mul
export USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=1

# deepep vars
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=24
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export ASCEND_USE_FIA=1
# transfer dsv4 weight suffix, will be removed in a future version
export IS_DEEPSEEK_V4=1

QUANT_MODE=compressed-tensors  # [compressed-tensors, modelslim]

# MTP
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 8 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --watchdog-timeout 18000 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.75 \
    --cuda-graph-bs 1  \
    --disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 65535 --context-length 65536 \
    --max-running-requests 8 \
    --dtype bfloat16 \
    --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
    --quantization ${QUANT_MODE} --disable-shared-experts-fusion \
    --skip-server-warmup \
    --moe-a2a-backend deepep --deepep-mode auto \
    --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3

