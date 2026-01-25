from asyncio import get_event_loop
import asyncio
from typing import Dict
import torch
import logging
import os
import copy
from omegaconf import OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from verl.protocol import DataProto
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler.performance import reduce_timing, simple_timer, topk_reduce_ratio_min_max
from verl.utils.profiler.profile import DistProfiler
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker, get_sharding_strategy, get_vl_model_vision_tower
from verl.utils.fs import copy_to_local
from verl.workers.config.optimizer import build_optimizer
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register, Dispatch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.model import convert_weight_keys, print_model_size, get_generation_config, update_model_config, load_valuehead_model
from verl.utils.flops_counter import FlopsCounter
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.fsdp_utils import collect_lora_params, collect_task_lora_params, fsdp_version, get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager, apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu, offload_fsdp_optimizer, replace_lora_wrapper
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.fsdp_utils import CPUOffloadPolicy, MixedPrecisionPolicy
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForImageTextToText
from verl.models.transformers.monkey_patch import apply_monkey_patch
import warnings
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.workers.rollout.vllm_rollout.multilora_vllm_rollout import MultiLoraVLLMRollout

logger = logging.getLogger(__file__)

ADAPTER_NAME_PREFIX = "lora_"

class MultiTaskActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config, tasks_config: dict, role, **kwargs):
        # Initialize with the base config
        super().__init__(config, role, **kwargs)
        self._is_lora = True
        self.tasks_config = tasks_config
        self.active_task_id = None
        self.task_optimizers = {}
        self.task_schedulers = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )
            
    
    def _build_multitask_model_optimizer(self, model_path, fsdp_config, optim_config, override_model_config, **kwargs):
        # Adapted from ActorRolloutRefWorker._build_model_optimizer
        role = kwargs.get("role", "actor")
        trust_remote_code = kwargs.get("trust_remote_code", False)
        use_liger = kwargs.get("use_liger", False)
        use_remove_padding = kwargs.get("use_remove_padding", False)
        use_fused_kernels = kwargs.get("use_fused_kernels", False)
        enable_gradient_checkpointing = kwargs.get("enable_gradient_checkpointing", False)
        enable_activation_offload = kwargs.get("enable_activation_offload", False)

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        
        self.tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(model_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        
        # ... (skipping some model config patches for brevity, assume they are handled or add if critical) ...
        # Copied critical patches:
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        self.generation_config = get_generation_config(model_path, trust_remote_code=trust_remote_code)
        
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Determine model class
            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                actor_module_class = AutoModelForCausalLM
            else:
                actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )
            
            actor_module.to(torch_dtype)
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Multi-task LoRA loading
        assert self._is_lora
        print("Applying Multi-Task LoRA to actor module")
        actor_module.enable_input_require_grads()
        
        first_task = True
        first_task_id = None
        for task_id, task_cfg in self.tasks_config.items():
            if self.rank == 0:
                print(f"Loading adapter for task: {task_id}")
            
            # Get LoRA config for this task
            # Assuming task_cfg structure matches the expected structure
            # We need to access the model config within task_cfg
            model_cfg = task_cfg.actor_rollout_ref.model
            lora_rank = model_cfg.get("lora_rank", 8)
            lora_alpha = model_cfg.get("lora_alpha", 32)
            target_modules = model_cfg.get("target_modules", "all-linear")
            lora_adapter_path = model_cfg.get("lora_adapter_path", None)
            
            if first_task:
                # For the first task, we use from_pretrained or get_peft_model
                if lora_adapter_path:
                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_cfg.get("use_shm", False))
                    actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, adapter_name=task_id, is_trainable=True)
                else:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=convert_to_regular_types(target_modules),
                        bias="none"
                    )
                    actor_module = get_peft_model(actor_module, lora_config, adapter_name=task_id)
                first_task = False
                first_task_id = task_id
            else:
                # For subsequent tasks, we add adapter
                if lora_adapter_path:
                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_cfg.get("use_shm", False))
                    actor_module.load_adapter(local_adapter_path, adapter_name=task_id, is_trainable=True)
                else:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=convert_to_regular_types(target_modules),
                        bias="none"
                    )
                    actor_module.add_adapter(task_id, lora_config)

        # Ensure a default adapter entry exists for downstream PEFT utils (e.g., layered_summon)
        # if first_task_id is not None and hasattr(actor_module, "peft_config"):
        #     peft_cfg = actor_module.peft_config
        #     if "default" not in peft_cfg:
        #         peft_cfg["default"] = peft_cfg[first_task_id]
        print("peft config:", actor_module.peft_config)

        # FSDP Wrapping
        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        # Ensure use_orig_params is True for LoRA
        if self._is_lora:
            self.use_orig_params = True

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.get("reshard_after_forward", False)
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)
        
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        
        # Assuming FSDP1 for simplicity as in base class
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=self.use_orig_params,
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
        )

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, self.config.actor.strategy, enable_gradient_checkpointing)

        # Create Optimizers for each task
        if role == "actor" and optim_config is not None:
            for task_id in self.tasks_config.keys():
                # Filter parameters for this task
                # With use_orig_params=True, we can filter by name
                # LoRA params usually have "lora" in name, and "adapter_name" (task_id) if we use different names
                # PEFT names parameters like "base_model.model.layers.0.self_attn.q_proj.lora_A.task_id"
                task_params = []
                actor_module_fsdp.set_adapter(task_id)

                task_params = [
                    p for p in actor_module_fsdp.parameters()
                    if p.requires_grad and p.numel() > 0
                ]
                
                if len(task_params) == 0:
                    if self.rank == 0:
                        logger.warning(f"Building actor: No parameters found for task {task_id}")
                
                # Create optimizer
                # We use the task-specific optim config if available, otherwise base optim config
                if "actor" in self.tasks_config[task_id] and "optim" in self.tasks_config[task_id].actor:
                    task_optim_config = self.tasks_config[task_id].actor.optim
                else:
                    task_optim_config = optim_config
                optimizer = build_optimizer(task_params, task_optim_config)
                
                # Create scheduler
                total_steps = task_optim_config.get("total_training_steps", 0)
                num_warmup_steps = int(task_optim_config.get("lr_warmup_steps", -1))
                lr_scheduler_type = task_optim_config.get("lr_scheduler_type", "constant")
                if num_warmup_steps < 0:
                    num_warmup_steps_ratio = task_optim_config.get("lr_warmup_steps_ratio", 0.0)
                    num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

                if lr_scheduler_type == "constant":
                    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
                elif lr_scheduler_type == "cosine":
                     scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=total_steps,
                        min_lr_ratio=task_optim_config.get("min_lr_ratio", 0.0),
                        num_cycles=task_optim_config.get("num_cycles", 0.5),
                    )
                else:
                    raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

                self.task_optimizers[task_id] = optimizer
                self.task_schedulers[task_id] = scheduler
            
            # Set initial optimizer/scheduler (will be updated by switch_task)
            # We pick the first one just to populate the fields
            first_task_id = list(self.tasks_config.keys())[0]
            actor_optimizer = self.task_optimizers[first_task_id]
            actor_lr_scheduler = self.task_schedulers[first_task_id]
            self.active_task_id = first_task_id
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        device_name = get_device_name()
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )

        self.rollout_device_mesh = rollout_device_mesh

        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = MultiLoraVLLMRollout(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh, tasks_config=self.tasks_config
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        # Note: sync mode is deprecated and rejected in RolloutConfig.__post_init__

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for qwen2.5-vl: when using flash_attention_3, set vision tower to use flash_attention_2
        # because the vision tower does not support flash_attention_3
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # add lora adapters to actor_module
        print("Applying LoRA to actor module")
        actor_module.enable_input_require_grads()
        first_task = True
        for task_id, task_cfg in self.tasks_config.items():
            if self.rank == 0:
                print(f"Loading adapter for task: {task_id}")
            
            # Get LoRA config for this task
            # Assuming task_cfg structure matches the expected structure
            # We need to access the model config within task_cfg
            model_cfg = task_cfg.actor_rollout_ref.model
            lora_rank = model_cfg.get("lora_rank", 8)
            lora_alpha = model_cfg.get("lora_alpha", 32)
            target_modules = model_cfg.get("target_modules", "all-linear")
            lora_adapter_path = model_cfg.get("lora_adapter_path", None)

            adapter_name = ADAPTER_NAME_PREFIX + str(task_id)
            
            if first_task:
                # For the first task, we use from_pretrained or get_peft_model
                if lora_adapter_path:
                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_cfg.get("use_shm", False))
                    actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, adapter_name=adapter_name, is_trainable=True)
                else:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=convert_to_regular_types(target_modules),
                        bias="none"
                    )
                    actor_module = get_peft_model(actor_module, lora_config, adapter_name=adapter_name)
                first_task = False
            else:
                # For subsequent tasks, we add adapter
                if lora_adapter_path:
                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_cfg.get("use_shm", False))
                    actor_module.load_adapter(local_adapter_path, adapter_name=adapter_name, is_trainable=True)
                else:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=convert_to_regular_types(target_modules),
                        bias="none"
                    )
                    actor_module.add_adapter(adapter_name, lora_config)

        print("peft config:", actor_module.peft_config)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.reshard_after_forward
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # Create Optimizers for each task
        if role == "actor" and optim_config is not None:
            for task_id in self.tasks_config.keys():
                actor_module_fsdp.set_adapter(ADAPTER_NAME_PREFIX + str(task_id))

                print("building optimizer for task:", task_id)
                # for n, p in actor_module_fsdp.named_parameters():
                #     print(n, p.requires_grad, p.grad, p.numel())

                # task_params = [
                #     p for p in actor_module_fsdp.parameters()
                #     if p.requires_grad and p.numel() > 0
                # ]
                
                # Create optimizer
                # We use the task-specific optim config if available, otherwise base optim config
                if "actor" in self.tasks_config[task_id] and "optim" in self.tasks_config[task_id].actor:
                    task_optim_config = self.tasks_config[task_id].actor.optim
                else:
                    task_optim_config = optim_config
                optimizer = build_optimizer(actor_module_fsdp.parameters(), task_optim_config)
                
                # Create scheduler
                total_steps = task_optim_config.get("total_training_steps", 0)
                num_warmup_steps = int(task_optim_config.get("lr_warmup_steps", -1))
                lr_scheduler_type = task_optim_config.get("lr_scheduler_type", "constant")
                if num_warmup_steps < 0:
                    num_warmup_steps_ratio = task_optim_config.get("lr_warmup_steps_ratio", 0.0)
                    num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

                if lr_scheduler_type == "constant":
                    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
                elif lr_scheduler_type == "cosine":
                     scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=total_steps,
                        min_lr_ratio=task_optim_config.get("min_lr_ratio", 0.0),
                        num_cycles=task_optim_config.get("num_cycles", 0.5),
                    )
                else:
                    raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

                self.task_optimizers[task_id] = optimizer
                self.task_schedulers[task_id] = scheduler
            
            # Set initial optimizer/scheduler (will be updated by switch_task)
            # We pick the first one just to populate the fields
            first_task_id = list(self.tasks_config.keys())[0]
            actor_optimizer = self.task_optimizers[first_task_id]
            actor_lr_scheduler = self.task_schedulers[first_task_id]
            # self.active_task_id = first_task_id
        else:
            actor_optimizer = None
            actor_lr_scheduler = None     

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config


    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="multi_lora_rollout_generate")
    def generate_sequences_multi_lora(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        print("multi-lora worker generate.")
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences_multi_lora", timing_generate):
            output = self.rollout.generate_sequences_multi_lora(prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences_multi_lora"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output
    
    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        peft_config = peft_model.peft_config

        # collect all the adapters in peft model
        adapter_params = {}
        for adapter_name in peft_config.keys():
            lora_config = peft_config[adapter_name]
            params = collect_task_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
                adapter_name=adapter_name
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, lora_config): v for k, v in params.items()}

            params = convert_weight_keys(
                params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

            adapter_params[adapter_name] = params

            print(f"adapter {adapter_name} params: {len(params)}")

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            # base_model_params = {replace_lora_wrapper(k, lora_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )
            print("base model params:", len(base_model_params))

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # vllm update base model params
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_base_weights(per_tensor_base_params)
            del base_model_params, per_tensor_base_params

        # vllm update lora params
        for adapter_name in adapter_params.keys():
            params = adapter_params[adapter_name]
            if peft_config is not None and self.base_sync_done:
                per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
            else:
                device = get_device_id()  # used when fsdp2 set cpu_offload_policy
                per_tensor_param = (
                    (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                    for name, param in params.items()
                )
            
            lora_config = peft_config[adapter_name]
            task_id = int(adapter_name[len(ADAPTER_NAME_PREFIX):])
            await self.rollout.update_lora_weights(per_tensor_param, task_id=task_id, peft_config=lora_config, base_sync_done=self.base_sync_done)
            del params, per_tensor_param

        del adapter_params
        log_gpu_memory_usage("After update_weights", logger=logger)
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)



class MultiTaskCriticWorker(CriticWorker):
    def __init__(self, config, tasks_config, **kwargs):
        super().__init__(config, **kwargs)
        self.tasks_config = tasks_config
        self.active_task_id = None
        self.task_optimizers = {}
        self.task_schedulers = {}

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # def init_model(self):
    #     import_external_libs(self.config.model.get("external_lib", None))
    #     from verl.workers.critic import DataParallelPPOCritic

    #     # We need a custom build method for multi-task critic
    #     self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_multitask_critic_model_optimizer(
    #         self.config
    #     )

    #     if self._is_offload_param:
    #         # offload_fsdp_model_to_cpu(self.critic_module)
    #         pass 
    #     if self._is_offload_optimizer:
    #         # offload_fsdp_optimizer(optimizer=self.critic_optimizer)
    #         pass

    #     self.critic = DataParallelPPOCritic(
    #         config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
    #     )
        
    #     # Flops counter and checkpoint manager might need adjustment or just use current task's
    #     # For simplicity, we initialize them with the base config/model
    #     self.flops_counter = FlopsCounter(self.critic_model_config)
    #     self.checkpoint_manager = None # TODO: Custom checkpoint manager for multi-task?

    def _build_multitask_critic_model_optimizer(self, config):
        # Adapted from CriticWorker._build_critic_model_optimizer
        # Similar logic to MultiTaskActorRolloutRefWorker._build_multitask_model_optimizer
        # Load base model, add adapters, wrap FSDP, create optimizers
        
        # ... (Loading base model logic) ...
        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        
        # ... (Model config and initialization) ...
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(local_path, attn_implementation=attn_implementation, trust_remote_code=config.model.get("trust_remote_code", False))
        critic_model_config.num_labels = 1
        
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = load_valuehead_model(
                local_path,
                PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "fp32")),
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )
            
            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )
            critic_module.to(PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "fp32")))
            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Multi-task LoRA
        if self._is_lora:
            print("Applying Multi-Task LoRA to critic module")
            critic_module.enable_input_require_grads()
            first_task = True
            first_task_id = None
            for task_id, task_cfg in self.tasks_config.items():
                if self.rank == 0:
                    print(f"Loading critic adapter for task: {task_id}")
                
                model_cfg = task_cfg.critic.model
                lora_rank = model_cfg.get("lora_rank", 8)
                lora_alpha = model_cfg.get("lora_alpha", 32)
                target_modules = model_cfg.get("target_modules", "all-linear")
                lora_adapter_path = model_cfg.get("lora_adapter_path", None)
                
                if first_task:
                    if lora_adapter_path:
                         local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_cfg.get("use_shm", False))
                         critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, adapter_name=task_id, is_trainable=True)
                    else:
                        lora_config = LoraConfig(
                            task_type=TaskType.TOKEN_CLS,
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            target_modules=convert_to_regular_types(target_modules),
                            bias="none"
                        )
                        critic_module = get_peft_model(critic_module, lora_config, adapter_name=task_id)
                    first_task = False
                    first_task_id = task_id
                else:
                    if lora_adapter_path:
                        local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_cfg.get("use_shm", False))
                        critic_module.load_adapter(local_adapter_path, adapter_name=task_id, is_trainable=True)
                    else:
                        lora_config = LoraConfig(
                            task_type=TaskType.TOKEN_CLS,
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            target_modules=convert_to_regular_types(target_modules),
                            bias="none"
                        )
                        critic_module.add_adapter(task_id, lora_config)

            if first_task_id is not None and hasattr(critic_module, "peft_config"):
                peft_cfg = critic_module.peft_config
                if "default" not in peft_cfg:
                    peft_cfg["default"] = peft_cfg[first_task_id]

        self.critic_model_config = critic_model_config

        # FSDP Wrapping
        self.use_orig_params = True if self._is_lora else config.model.fsdp_config.get("use_orig_params", False)
        
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )
        
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        
        critic_module = FSDP(
            critic_module,
            param_init_fn=init_fn,
            use_orig_params=self.use_orig_params,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
            sync_module_states=True,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        # Create Optimizers
        if not self._is_lora:
            # No task-specific adapters: share one optimizer/scheduler across tasks.
            task_params = list(critic_module.parameters())
            base_task_cfg = next(iter(self.tasks_config.values()))
            if "critic" in base_task_cfg and "optim" in base_task_cfg.critic:
                task_optim_config = base_task_cfg.critic.optim
            else:
                task_optim_config = config.optim
            optimizer = build_optimizer(task_params, task_optim_config)

            total_steps = task_optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(task_optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = task_optim_config.get("lr_scheduler_type", "constant")
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = task_optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if lr_scheduler_type == "constant":
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
            elif lr_scheduler_type == "cosine":
                 scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=task_optim_config.get("min_lr_ratio", 0.0),
                    num_cycles=task_optim_config.get("num_cycles", 0.5),
                )

            for task_id in self.tasks_config.keys():
                self.task_optimizers[task_id] = optimizer
                self.task_schedulers[task_id] = scheduler
        else:
            for task_id in self.tasks_config.keys():
                task_params = []
                for n, p in critic_module.named_parameters():
                    if task_id in n:
                        task_params.append(p)
                
                if len(task_params) == 0:
                    if self.rank == 0:
                        logger.warning(f"Building Critic: No parameters found for task {task_id}")

                if "critic" in self.tasks_config[task_id] and "optim" in self.tasks_config[task_id].critic:
                    task_optim_config = self.tasks_config[task_id].critic.optim
                else:
                    task_optim_config = config.optim
                optimizer = build_optimizer(task_params, task_optim_config)
                
                total_steps = task_optim_config.get("total_training_steps", 0)
                num_warmup_steps = int(task_optim_config.get("lr_warmup_steps", -1))
                lr_scheduler_type = task_optim_config.get("lr_scheduler_type", "constant")
                if num_warmup_steps < 0:
                    num_warmup_steps_ratio = task_optim_config.get("lr_warmup_steps_ratio", 0.0)
                    num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

                if lr_scheduler_type == "constant":
                    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
                elif lr_scheduler_type == "cosine":
                     scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=total_steps,
                        min_lr_ratio=task_optim_config.get("min_lr_ratio", 0.0),
                        num_cycles=task_optim_config.get("num_cycles", 0.5),
                    )
                
                self.task_optimizers[task_id] = optimizer
                self.task_schedulers[task_id] = scheduler

        first_task_id = list(self.tasks_config.keys())[0]
        critic_optimizer = self.task_optimizers[first_task_id]
        critic_lr_scheduler = self.task_schedulers[first_task_id]
        self.active_task_id = first_task_id

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_task(self, task_id):
        if task_id == self.active_task_id:
            return
        
        if self.rank == 0:
            print(f"Switching Critic worker to task: {task_id}")
            
        model = self.critic_module
        if hasattr(model, "_fsdp_wrapped_module") and hasattr(model._fsdp_wrapped_module, "set_adapter"):
             model._fsdp_wrapped_module.set_adapter(task_id)
        
        if task_id in self.task_optimizers:
            self.critic_optimizer = self.task_optimizers[task_id]
            self.critic_lr_scheduler = self.task_schedulers[task_id]
            if hasattr(self, "critic") and hasattr(self.critic, "critic_optimizer"):
                self.critic.critic_optimizer = self.critic_optimizer

        self.active_task_id = task_id
