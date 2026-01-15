import torch
import logging
import os
import copy
from omegaconf import OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker, get_sharding_strategy
from verl.utils.fs import copy_to_local
from verl.workers.config.optimizer import build_optimizer
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from verl.single_controller.base.decorator import register, Dispatch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.model import print_model_size, get_generation_config, update_model_config, load_valuehead_model
from verl.utils.flops_counter import FlopsCounter
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager, apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.fsdp_utils import CPUOffloadPolicy, MixedPrecisionPolicy
from verl.utils.device import get_device_id
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForImageTextToText
from verl.models.transformers.monkey_patch import apply_monkey_patch
import warnings

logger = logging.getLogger(__file__)

class MultiTaskActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config, tasks_config, role, **kwargs):
        # Initialize with the base config
        super().__init__(config, role, **kwargs)
        self._is_lora = True
        self.tasks_config = tasks_config
        self.active_task_id = None
        self.task_optimizers = {}
        self.task_schedulers = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                # Use base optim config as default, but we will create per-task optimizers
                optim_config = self.config.actor.optim
                fsdp_config = OmegaConf.to_object(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            
            # Use our custom build method that handles multiple adapters
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_multitask_model_optimizer(
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

            # get the original unwrapped module if FSDP1
            # Note: with FSDP, accessing the unwrapped module might be tricky if sharded
            if hasattr(self.actor_module_fsdp, "_fsdp_wrapped_module"):
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        if self._is_actor:
            from verl.workers.actor import DataParallelPPOActor
            actor_cfg = self.config.actor
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            # Reference model usually doesn't need multi-task LoRA switching for PPO if we use the same ref logic
            # But if tasks have different ref models, that's complex. 
            # Assuming shared ref model (base model) or ref is just the base model frozen.
            # If ref is also LoRA, we might need multi-task ref.
            # For now, let's assume ref is single task or handled by base class for simplicity,
            # or if needed, we can extend it.
            # The base class init_model handles ref model loading. We can call it or copy logic.
            # Copying logic for safety.
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            # Ref uses standard build method (single task or no LoRA)
            # If ref needs LoRA, we might need to handle it.
            # For simplicity, assuming Ref is just base model for now.
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=OmegaConf.to_object(self.config.ref.fsdp_config),
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
                self.config.ref.checkpoint = None

            
    
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_task(self, task_id):
        if task_id == self.active_task_id:
            return
        
        if self.rank == 0:
            print(f"Switching Actor/Rollout worker to task: {task_id}")

        # Switch LoRA adapter
        # We need to access the underlying PeftModel. 
        # FSDP wraps it.
        model = self.actor_module_fsdp
        
        # Traverse to find set_adapter
        # Note: FSDP module itself doesn't have set_adapter unless we expose it.
        # But we can access the underlying module via _fsdp_wrapped_module if it's FSDP1
        # For FSDP2, it might be different.
        
        if hasattr(model, "_fsdp_wrapped_module") and hasattr(model._fsdp_wrapped_module, "set_adapter"):
             model._fsdp_wrapped_module.set_adapter(task_id)
        else:
            # Fallback recursion or check
            # Sometimes FSDP wraps layers, so we might need to call set_adapter on all modules?
            # Peft usually injects set_adapter into the model.
            # If we call set_adapter on the root model (inside FSDP), it propagates.
            # But FSDP hides attributes.
            # We can try to use a recursive function to find set_adapter
            def apply_set_adapter(module, adapter_name):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                # Recurse for children if needed, but usually set_adapter on root is enough if PEFT is root
                for child in module.children():
                    apply_set_adapter(child, adapter_name)
            
            # apply_set_adapter(model, task_id) 
            # This might be slow.
            # Better: access the base module.
            # If use_orig_params=True, maybe we can just call it?
            # No, FSDP object doesn't forward unknown methods by default.
            
            # Try getting the inner module
            if hasattr(model, "module"):
                model.module.set_adapter(task_id)
            else:
                 # If we can't find it easily, use a known path if possible
                 pass

        # Switch Optimizer and Scheduler
        if task_id in self.task_optimizers:
            self.actor_optimizer = self.task_optimizers[task_id]
            self.actor_lr_scheduler = self.task_schedulers[task_id]
            # We also need to update the actor.actor_optimizer if it exists
            if hasattr(self, "actor") and hasattr(self.actor, "actor_optimizer"):
                self.actor.actor_optimizer = self.actor_optimizer

        self.active_task_id = task_id


class MultiTaskCriticWorker(CriticWorker):
    def __init__(self, config, tasks_config, **kwargs):
        super().__init__(config, **kwargs)
        self.tasks_config = tasks_config
        self.active_task_id = None
        self.task_optimizers = {}
        self.task_schedulers = {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))
        from verl.workers.critic import DataParallelPPOCritic

        # We need a custom build method for multi-task critic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_multitask_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            # offload_fsdp_model_to_cpu(self.critic_module)
            pass 
        if self._is_offload_optimizer:
            # offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            pass

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )
        
        # Flops counter and checkpoint manager might need adjustment or just use current task's
        # For simplicity, we initialize them with the base config/model
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = None # TODO: Custom checkpoint manager for multi-task?

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
