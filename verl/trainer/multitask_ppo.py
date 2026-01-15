# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-task parallel PPO training interface.
This module supports running multiple PPO training tasks in parallel using Ray,
with support for shared base models and task-specific LoRA adapters.

Usage:
    Single task (backward compatible):
        Use standard config structure as in main_ppo.py
    
    Multi-task:
        Configure with a 'tasks' list in your config:
        ```yaml
        tasks:
          - task_id: "task_1"
            data:
              train_files: ["path/to/task1/train.parquet"]
            actor_rollout_ref:
              model:
                lora_adapter_path: "path/to/task1/adapter"
          - task_id: "task_2"
            data:
              train_files: ["path/to/task2/train.parquet"]
            actor_rollout_ref:
              model:
                lora_adapter_path: "path/to/task2/adapter"
        ```
"""

import os
import socket
from typing import List, Optional, Union

import hydra
import ray
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.multitask_ray_trainer import MultiTaskRayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.workers.multitask_fsdp_workers import MultiTaskActorRolloutRefWorker, MultiTaskCriticWorker
from verl.utils.device import auto_set_ascend_device_name, is_cuda_available
from verl.utils import hf_tokenizer, hf_processor
from verl.trainer.ppo.ray_trainer import Role

@hydra.main(config_path="config", config_name="multitask_ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management."""
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_ascend_device_name(config)

    if hasattr(config, "tasks") and config.tasks is not None:
        # Multi-task mode
        run_multitask_ppo(config)
    else:
        # Single task mode: backward compatibility
        from verl.trainer.main_ppo import run_ppo
        run_ppo(config)


def run_multitask_ppo(config):
    """Initialize Ray cluster and run distributed Multi-task PPO training process. \
    Similar to run_ppo in main_ppo.py. 

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters. Specially, 'tasks' is a 
                list of task-specific configs for each task.
    """
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(MultiLoraTaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class MultiLoraTaskRunner(TaskRunner):
    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker. Currently, only choose MultiTaskActorRolloutRefWorker"""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.multitask_fsdp_workers import MultiTaskActorRolloutRefWorker

            actor_rollout_cls = MultiTaskActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping.   """
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.multitask_fsdp_workers import MultiTaskCriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(MultiTaskCriticWorker)
        self.mapping[Role.Critic] = "global_pool"


    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"Multi-lora TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # process task-specific configs, and pass them to trainer
        tasks_list = config.tasks
        if not isinstance(tasks_list, (list, ListConfig)):
            raise ValueError("config.tasks must be a list of task configurations")
        if len(tasks_list) == 0:
            raise ValueError(
                "config.tasks is empty. Provide at least one task via CLI (e.g., tasks.0.task_id=foo) or a config overlay."
            )
    
        num_tasks = len(tasks_list)
        print(f"[MultiTask] Starting {num_tasks} PPO training tasks with shared workers")

        # Build a global/base config by stripping tasks out of the original config.
        # This ensures task-local overrides (e.g., LoRA rank) do not leak into the shared setup.
        base_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        if "tasks" in base_config:
            del base_config["tasks"]

        # Create a dictionary of task configs
        tasks_config = {}
        for i, task_config in enumerate(tasks_list):
            merged_config = OmegaConf.merge(base_config, task_config)
            task_id = task_config.get("task_id", f"task_{i}")
            merged_config.task_id = task_id
            tasks_config[task_id] = merged_config

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        # different tasks can have different reward managers
        reward_fn_map = {}
        val_reward_fn_map = {}
        for task_id, task_cfg in tasks_config.items():
            reward_fn_map[task_id] = load_reward_manager(
                task_cfg, tokenizer, num_examine=0, **task_cfg.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn_map[task_id] = load_reward_manager(
                task_cfg, tokenizer, num_examine=1, **task_cfg.reward_model.get("reward_kwargs", {})
            )
        # default to first task for backward compatibility
        reward_fn = reward_fn_map[list(tasks_config.keys())[0]]
        val_reward_fn = val_reward_fn_map[list(tasks_config.keys())[0]]

        resource_pool_manager = self.init_resource_pool_mgr(config)

        trainer = MultiTaskRayPPOTrainer(
            config=base_config,
            tasks_config=tasks_config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            reward_fn_map=reward_fn_map,
            val_reward_fn_map=val_reward_fn_map
        )
        
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()

if __name__ == "__main__":
    main()
