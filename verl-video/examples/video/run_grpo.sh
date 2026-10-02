#!/usr/bin/env bash
# GRPO/RDR-GRPO recipe. Run on a shared-filesystem Ray cluster.
set -euo pipefail
PUBLIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="${PUBLIC_ROOT}/verl-video"
cd "${REPO_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to your SFT checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the generated train.parquet}"
: "${TEST_FILE:?Set TEST_FILE to held-out validation parquet}"
: "${TOOL_CONFIG:?Run scripts/prepare_tool_config.py and set TOOL_CONFIG}"
: "${JUDGER_MODEL:?Set JUDGER_MODEL}"
: "${JUDGER_API_KEY:?Set JUDGER_API_KEY}"
: "${JUDGER_BASE_URL:?Set JUDGER_BASE_URL}"
export VSS_VIDEO_AGENT_PATH="${PUBLIC_ROOT}"
export PYTHONPATH="${PUBLIC_ROOT}:${REPO_ROOT}:${REPO_ROOT}/video_search_sim${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
INFER_BACKEND=vllm
PROJECT_NAME=VideoResearchAgent
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rdr_grpo}"
NDEVICES_PER_NODE="${NDEVICES_PER_NODE:-8}"
NNODES="${NNODES:-2}"
TP="${TP:-2}"
PP="${PP:-1}"
CP="${CP:-4}"
EP=1
ETP=1
GEN_TP="${GEN_TP:-2}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.65}"
ALL_OFFLOAD=True
USE_DYNAMIC_BSZ=True
CKPTS_DIR="${PUBLIC_ROOT}/data/checkpoints/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${PUBLIC_ROOT}/data/rollouts/${EXPERIMENT_NAME}"
VALIDATION_DATA_DIR="${PUBLIC_ROOT}/data/validation/${EXPERIMENT_NAME}"
REWARD_PATH="${REPO_ROOT}/verl/utils/reward_score/video_research_judge.py"
REWARD_NAME=compute_score
max_prompt_length=8192
max_response_length=32768
val_max_response_length=65536
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length + CP) / CP))
log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=True
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=64
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.return_raw_chat=True
    data.image_key=images
    data.video_key=videos
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=2e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=2
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
    actor_rollout_ref.ref.megatron.use_remove_padding=True
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}"
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle
    actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=32
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.response_length=${val_max_response_length}

)

REWARD=(
    custom_reward_function.path="${REWARD_PATH}"
    custom_reward_function.name="${REWARD_NAME}"
)

TRAINER=(
    trainer.critic_warmup=0
    "trainer.logger=['console']"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=True
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.val_before_train=True
    trainer.save_freq=5
    trainer.test_freq=5
    trainer.total_epochs=5
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
)

EXTRA=(
    model_engine=megatron
)


python "${PUBLIC_ROOT}/scripts/submit_training.py" -- \
  python -m verl.trainer.main_ppo \
  "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" \
  "${ROLLOUT[@]}" "${REWARD[@]}" "${TRAINER[@]}" "${EXTRA[@]}" "$@"
