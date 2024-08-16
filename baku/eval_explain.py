#!/usr/bin/env python3

import warnings
import os
import cv2

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
from store_obs import DictionarySaver

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )

        self.env, self.task_descriptions, self.task = hydra.utils.call(self.cfg.suite.task_make_fn)
        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )
        self.agent_name = str(self.cfg.agent)
        self.suite_name = self.cfg.suite.name
        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        
        self.render_image = None

        #store actions, obs
        self.store_data = DictionarySaver(self.work_dir)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval_x(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        data_list=[]
        attn_maps_across_layer = []
        observations = []
        actions=[]

        # TODO: generate explainability data from different short-horizon Task!
        # --> 10 short horizon task

        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []
            #data = {}
            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)


                # datastorage per task / episode / run
                data = {
                    "prompt": None,
                    "action": [],
                    "pixels": [],
                    "pixels_egocentric": [],
                    "proprioceptive": [],
                    "task_emb": None,
                    "goal_achieved": [],
                    "last_layer_atten_maps":[],
                    "atten_maps": [],
                    "grad_atten": [],
                    "grad_atten_ft":[],
                    "generic_atten_wrt_action_gradient":[],
                    "action_gradient_weighted_tokens":[],
                    "features_gradient_weighted_tokens":[],
                    "gradients_wrt_action_features":[],
                    "gradients_wrt_action":[],
                    "environment": None,
                    "agent":[],
                    "suite":[],
                    "task": [],
                }
                data["environment"] = self.task
                data["task"] = self.task_descriptions[0]
                data["agent"] = self.agent_name
                data["suite"] = self.suite_name

                # loop of episode!
                # we want to explain what is going on here!
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with utils.eval_mode(self.agent): #torch.no_grad was also here, Modified it to calculate gradients
                        action, grad_attn_map, grad = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward


                    # store data per timestep

                    # TODO append data to list in data storage


                    #Observations
                    obs = time_step.observation
                    # language
                    if data["prompt"] is None:
                        data["prompt"] = prompt
                        data["task_emb"] = obs["task_emb"]

                    # import pdb; pdb.set_trace()
                    data["action"].append(action)
                    #data["features"].append(obs["features"])
                    data["pixels"].append(obs["pixels"])
                    data["pixels_egocentric"].append(obs["pixels_egocentric"])
                    data["proprioceptive"].append(obs["proprioceptive"])
                    data["goal_achieved"].append(obs["goal_achieved"])
                    
                    step += 1
                    print(step)

                    # EXTRACT DATA FROM MODEL AND ENV 

                    # store attention maps and gradients 
                    # TODO get gradients from action output

                    grad_attn_map_ft = self.agent.actor._gradient_attn_maps #Gradient Weighted Maps wrt Features
                    gradients_wrt_action_features = self.agent.actor._gradients_wrt_ft_token

                    data["gradients_wrt_action_features"].append(gradients_wrt_action_features)
                    data["gradients_wrt_action"].append(grad)

                    attn_map = self.agent.actor.attn_maps
                    data["atten_maps"].append(attn_map)

                    attn_map = torch.tensor(attn_map[-1][-1]).mean(dim=0)
                    attn_map = np.array(attn_map)
                  
                    # TODO just store gradients is fine!

                    data["last_layer_atten_maps"].append(attn_map)#Code changed here to save only last layer maps taken mean across heads

                    # make atten_maps to array and store it
                    data["grad_atten"].append(grad_attn_map) #Gradient Weighted Maps wrt action token
                    data["grad_atten_ft"].append(grad_attn_map_ft) #Gradient Weighted Maps wrt action feature token
            
                    if self.cfg.suite.name == "calvin" and time_step.reward == 1:
                        self.agent.buffer_reset()

                episode += 1
                success.append(time_step.observation["goal_achieved"])
                break 
            
            self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))
            break 

        print("finished")
        # TODO: store Explainability data and Obervations as pickle files

        for i in range(len(data["grad_atten"])):
            step_atten = data["grad_atten"][i]
            R = torch.eye(5, 5).cuda()
            R = R.unsqueeze(0).expand(1, 5, 5)
            # check attention maps here!!
            for j, blk in enumerate(step_atten):
                cam = blk.detach()
                cam = torch.abs(cam)
                cam = cam.clamp(min=0).mean(dim=1)
                R = R + torch.bmm(cam, R)

                R[0]= R[0]- torch.eye(5,5).cuda()
                sum = R[0].sum(dim=1)
                for i in range(5):
                  if sum[i] == 0:
                       sum[i] = 1
                  R[0][i] = R[0][i]/sum[i]
                R[0] =  R[0] + torch.eye(5,5).cuda()

            attention_relevance = R[0]/2  # batch dim
          
            attention_relevance = attention_relevance.cpu().numpy()
            data["generic_atten_wrt_action_gradient"].append(attention_relevance)  
            data["action_gradient_weighted_tokens"].append(attention_relevance[-1, :]) 

        for i in range(len(data["grad_atten_ft"])):
            step_atten = data["grad_atten_ft"][i]
            R = torch.eye(5, 5).cuda()
            R = R.unsqueeze(0).expand(1, 5, 5)
            # check attention maps here!!
            for j, blk in enumerate(step_atten):
                cam = blk.detach()
                cam = torch.abs(cam)
                cam = cam.clamp(min=0).mean(dim=1)
                R = R + torch.bmm(cam, R)

            attention_relevance = R[0]  # batch dim
            attention_relevance = attention_relevance.cpu().numpy()    
            data["features_gradient_weighted_tokens"].append(attention_relevance[-1, :])  

        max_values = [t.max() for t in data["features_gradient_weighted_tokens"]]
        overall_max_value = max(max_values)

        for i in range(len(data["features_gradient_weighted_tokens"])):
            data["features_gradient_weighted_tokens"][i] = data["features_gradient_weighted_tokens"][i]/overall_max_value

        # generate pickle here
        self.store_data.save_as_pkl(filename= f"data_{self.task}.pkl", dictionary=data)
        self.store_data.save_as_txt(filename= f"data_{self.task}.txt", dictionary=data)

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    from eval_explain import WorkspaceIL as W

    root_dir = Path.cwd()
    workspace = W(cfg)

    # Load weights
    snapshots = {}
    # bc
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    # TODO: check what weights are getting loaded
    workspace.load_snapshot(snapshots)
    workspace.eval_x()
    # workspace.eval()


if __name__ == "__main__":
    main()
