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
Multi-task PPO Trainer with Ray-based single controller.
"""

import os
import uuid
from contextlib import contextmanager
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.utils import Role
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking
from verl.utils.import_utils import load_class_from_fqn


class MultiTaskRayPPOTrainer(RayPPOTrainer):
    """
    Multi-task Distributed PPO trainer using Ray.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, ray.remote],
        resource_pool_manager: ResourcePoolManager,
        tasks_config: dict,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        reward_fn_map: Optional[dict] = None,
        val_reward_fn_map: Optional[dict] = None,
        collate_fn=None,
    ):
        # Store tasks config
        self.tasks_config = tasks_config
        self.active_task_id = None
        self.async_rollout_mode = None
        self.async_rollout_manager = None
        self.reward_fn_map = reward_fn_map or {}
        self.val_reward_fn_map = val_reward_fn_map or {}
        
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            collate_fn=collate_fn,
        )

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders for each task.
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        if collate_fn is None:
            collate_fn = default_collate_fn

        self.train_dataloaders = {}
        self.val_dataloaders = {}
        self.train_samplers = {}
        self.task_total_steps: dict[str, int] = {}
        
        total_training_steps = 0

        for task_id, task_cfg in self.tasks_config.items():
            print(f"Creating dataloaders for task: {task_id}")
            
            # Create datasets
            task_train_dataset = create_rl_dataset(
                task_cfg.data.train_files,
                task_cfg.data,
                self.tokenizer,
                self.processor,
                max_samples=task_cfg.data.get("train_max_samples", -1),
            )
            
            task_val_dataset = create_rl_dataset(
                task_cfg.data.val_files,
                task_cfg.data,
                self.tokenizer,
                self.processor,
                max_samples=task_cfg.data.get("val_max_samples", -1),
            )
            
            # Create sampler
            task_train_sampler = create_rl_sampler(task_cfg.data, task_train_dataset)
            self.train_samplers[task_id] = task_train_sampler
            
            # Create dataloaders
            num_workers = task_cfg.data["dataloader_num_workers"]
            
            self.train_dataloaders[task_id] = StatefulDataLoader(
                dataset=task_train_dataset,
                batch_size=task_cfg.data.get("gen_batch_size", task_cfg.data.train_batch_size),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=task_train_sampler,
            )
            
            val_batch_size = task_cfg.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(task_val_dataset)
                
            self.val_dataloaders[task_id] = StatefulDataLoader(
                dataset=task_val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=task_cfg.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            
            task_steps = len(self.train_dataloaders[task_id]) * self.config.trainer.total_epochs
            total_training_steps += task_steps
            self.task_total_steps[task_id] = task_steps
            print(f"Task {task_id}: Train size={len(self.train_dataloaders[task_id])}, Val size={len(self.val_dataloaders[task_id])}")

            # inject per-task total_training_steps for schedulers
            try:
                with open_dict(task_cfg):
                    if OmegaConf.select(task_cfg, "actor_rollout_ref.actor.optim"):
                        task_cfg.actor_rollout_ref.actor.optim.total_training_steps = task_steps
                    if OmegaConf.select(task_cfg, "critic.optim"):
                        task_cfg.critic.optim.total_training_steps = task_steps
            except Exception as e:
                print(f"Warning: Could not set total_training_steps for task {task_id}. Error: {e}")

        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = total_training_steps
            
        print(f"Total training steps across all tasks: {self.total_training_steps}")

        # Set total training steps in config for schedulers
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Error: {e}")

    # def init_workers(self):
    #     """Initialize distributed training workers with multi-task config."""
    #     self.resource_pool_manager.create_resource_pool()
    #     self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    #     # Create actor and rollout
    #     actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
    #     if self.hybrid_engine:
    #         resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
    #         actor_rollout_cls = RayClassWithInitArgs(
    #             cls=self.role_worker_mapping[actor_role],
    #             config=self.config.actor_rollout_ref,
    #             role=str(actor_role),
    #             tasks_config=self.tasks_config
    #         )
    #         self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
    #     else:
    #         raise NotImplementedError

    #     # Create critic
    #     if self.use_critic:
    #         resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
    #         from verl.workers.config import CriticConfig
    #         critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
            
    #         critic_cls = RayClassWithInitArgs(
    #             cls=self.role_worker_mapping[Role.Critic], 
    #             config=critic_cfg,
    #             tasks_config=self.tasks_config
    #         )
    #         self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    #     # Initialize workers
    #     self._init_workers_impl()

    # def _init_workers_impl(self):
    #     """Implementation of worker initialization (shared logic)."""
    #     # This is copied from RayPPOTrainer because it's not easily overridable without duplication
    #     # But wait, RayPPOTrainer.init_workers calls create_colocated_worker_cls
    #     # We constructed resource_pool_to_cls, now we need to actually instantiate them.
        
    #     # We can just call the super method if we populate resource_pool_to_cls correctly.
    #     # But RayPPOTrainer.init_workers does both: populates resource_pool_to_cls AND instantiates.
    #     # So I have to duplicate the instantiation logic here or refactor base class.
    #     # Since I cannot modify base class easily, I will duplicate the instantiation part.
        
    #     from verl.single_controller.ray.base import create_colocated_worker_cls
        
    #     self.workers = {}
    #     for resource_pool, class_dict in self.resource_pool_to_cls.items():
    #         worker_cls = create_colocated_worker_cls(class_dict=class_dict)
    #         wg = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_cls)
    #         spawn_wg = wg.spawn(prefix_set=class_dict.keys())
    #         for role, worker in spawn_wg.items():
    #             self.workers[Role.from_string(role)] = worker
        
    #     self.actor_rollout_wg = self.workers[Role.ActorRolloutRef] if Role.ActorRolloutRef in self.workers else self.workers[Role.ActorRollout]
    #     self.actor_rollout_wg.init_model()
        
    #     if self.use_critic:
    #         self.critic_wg = self.workers[Role.Critic]
    #         self.critic_wg.init_model()
    #     # Reference policy handling: reuse actor worker when ref is in actor (LoRA) or no separate ref worker
    #     if self.use_reference_policy:
    #         if hasattr(self, "ref_in_actor") and self.ref_in_actor:
    #             self.ref_policy_wg = self.actor_rollout_wg
    #         elif Role.RefPolicy in self.workers:
    #             self.ref_policy_wg = self.workers[Role.RefPolicy]
    #             self.ref_policy_wg.init_model()

    #     # Configure async rollout manager (mirrors base trainer)
    #     self.async_rollout_mode = True
    #     manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
    #     if manager_class_fqn:
    #         AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
    #     else:
    #         from verl.experimental.agent_loop import AgentLoopManager

    #     if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
    #         rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
    #     else:
    #         rm_resource_pool = None

    #     self.async_rollout_manager = AgentLoopManager(
    #         config=self.config,
    #         worker_group=self.actor_rollout_wg,
    #         rm_resource_pool=rm_resource_pool,
    #     )

    def switch_task(self, task_id):
        """Switch all workers to the specified task."""
        if self.active_task_id == task_id:
            return
            
        print(f"Switching trainer to task: {task_id}")
        self.actor_rollout_wg.switch_task(task_id)
        if self.use_critic:
            self.critic_wg.switch_task(task_id)
        self.active_task_id = task_id

    def fit(self):
        """
        Multi-task Training Loop.
        """
        print("Starting multi-task training loop")
        
        self.global_steps = 0
        
        # Create iterators for all dataloaders
        train_iterators = {task_id: iter(dl) for task_id, dl in self.train_dataloaders.items()}
        
        # Round-robin or weighted sampling
        task_ids = list(self.tasks_config.keys())
        num_tasks = len(task_ids)
        
        # Main training loop
        # We run for total_training_steps, cycling through tasks
        # Note: This is a simplified scheduling strategy (Round Robin)
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        while self.global_steps < self.total_training_steps:
            current_task_id = task_ids[self.global_steps % num_tasks]
            self.switch_task(current_task_id)
            try:
                batch = next(train_iterators[current_task_id])
            except StopIteration:
                train_iterators[current_task_id] = iter(self.train_dataloaders[current_task_id])
                batch = next(train_iterators[current_task_id])
            metrics = self._fit_batch(batch, task_id=current_task_id)
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/task_id": current_task_id,
                }
            )
            logger.log(data=metrics, step=self.global_steps)
            self.global_steps += 1
            if self.global_steps % self.config.trainer.val_check_interval == 0:
                self._validate_multitask()
            if self.global_steps % self.config.trainer.save_checkpoint_steps == 0:
                self._save_checkpoint_multitask()

    def _fit_batch(self, batch, task_id):
        """Run a single PPO step for the given batch and task."""
        # This mirrors the logic inside RayPPOTrainer.fit loop
        timing_raw = {}

        # 1. Prepare batch
        batch = DataProto.from_single_dict(batch)
        batch.meta_info["task_id"] = task_id  # Pass task_id in meta info if needed
        task_cfg = self.tasks_config[task_id]
        rollout_cfg = task_cfg.actor_rollout_ref.rollout
        batch.meta_info["temperature"] = rollout_cfg.temperature
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        
        # 2. Generation (Rollout)
        # Note: actor_rollout_wg is already switched to task_id
        
        # Pad batch
        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else rollout_cfg.get("agent", {}).get("num_workers", self.actor_rollout_wg.world_size)
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        with marked_timer("gen", timing_raw):
            if not self.async_rollout_mode:
                gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            else:
                gen_batch_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
        gen_batch = unpad_dataproto(gen_batch_padded, pad_size=pad_size)
        
        # 3. Compute Reward
        # Use task-specific reward function if available? 
        # For now, assume reward_fn handles it or it's generic.
        # Ideally, we should have task-specific reward functions.
        # But RayPPOTrainer init takes a single reward_fn.
        # We might need to wrap it.
        
        # Expand batch with generated responses
        batch = batch.union(gen_batch)
        
        # Compute rewards
        reward_fn = self.reward_fn_map.get(task_id, self.reward_fn)
        reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(batch, reward_fn=reward_fn)
        batch.batch["token_level_rewards"] = reward_tensor

        # Ensure response_mask exists before old_log_prob/updates
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        # Recompute old log prob (anchor) for actor update
        old_log_prob, _ = self._compute_old_log_prob(batch)
        batch = batch.union(old_log_prob)
        # Reference log prob if enabled
        if self.use_reference_policy:
            ref_log_prob = self._compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)
        
        # 4. Compute Advantages (Critic)
        if self.use_critic:
            # Compute values
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
            values_padded = self.critic_wg.compute_values(batch_padded)
            values = unpad_dataproto(values_padded, pad_size=pad_size)
            batch = batch.union(values)
            
            # Compute advantages
            batch = compute_advantage(
                batch,
                adv_estimator=task_cfg.algorithm.adv_estimator,
                gamma=task_cfg.algorithm.gamma,
                lam=task_cfg.algorithm.lam,
                num_repeat=rollout_cfg.n,
                config=task_cfg.algorithm,
            )
        else:
            # GRPO or other methods without critic
            batch = compute_advantage(
                batch,
                adv_estimator=task_cfg.algorithm.adv_estimator,
                gamma=task_cfg.algorithm.gamma,
                lam=task_cfg.algorithm.lam,
                num_repeat=rollout_cfg.n,
                config=task_cfg.algorithm,
            )

        # 5. Update (Train)
        # Update Actor
        actor_metrics = self.actor_rollout_wg.update_actor(batch)
        
        # Update Critic
        critic_metrics = {}
        if self.use_critic:
            critic_metrics = self.critic_wg.update_critic(batch)
            
        # 6. Logging
        metrics = {**actor_metrics, **critic_metrics}
        # Add task prefix to metrics
        metrics = {f"{task_id}/{k}": v for k, v in metrics.items()}
        
        # Log reward metrics
        reward_metrics = reduce_metrics(reward_extra_infos_dict)
        reward_metrics = {f"{task_id}/reward/{k}": v for k, v in reward_metrics.items()}
        metrics.update(reward_metrics)
        
        return metrics

    def _validate_multitask(self):
        """Run validation for all tasks."""
        print("Running multi-task validation")
        for task_id in self.tasks_config.keys():
            self.switch_task(task_id)
            # Use self.val_dataloaders[task_id]
            # ... implementation similar to _validate but iterating over task's val loader ...
            # For brevity, reusing _validate logic adapted for task
            self._validate_task(task_id)

    def _validate_task(self, task_id):
        # ... logic from RayPPOTrainer._validate ...
        # But using self.val_dataloaders[task_id]
        pass # To be implemented if needed, or just skip for now as proof of concept

    def _save_checkpoint_multitask(self):
        # Save checkpoints for all tasks (or just shared weights + adapters)
        pass
