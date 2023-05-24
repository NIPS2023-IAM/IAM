import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import math as m
import wandb 
import torch.nn.functional as F

class NQLearner_act_rew:
    def __init__(self, mac, action_models, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
            
        #! action_model
        self.action_models = action_models #! add action_models here
        self.am_optims = [Adam(action_model.parameters(), lr=args.am_lr) for action_model in self.action_models] 
        self.am_belta_cnt = 0
        self.am_belta = args.am_belta
        self.act_min_norm = args.act_min_norm
        self.act_min_var_norm = args.act_min_var_norm
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, env_feat_shape: dict, args, per_weight=None):        
        if args.am_belta_decay and ((t_env // args.am_decay_T) > self.am_belta_cnt):
            self.am_belta = self.am_belta * 0.9
            self.am_belta_cnt += 1
        
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals
        
        #! calculate intrinsic reward
        intr_rews = self.add_intr_rew(batch, env_feat_shape, args, mac_out.detach()).to('cuda')  # shape: (batch_size, max_seq_length-1, num_agents)
        
        #! normalize intrinsic reward by substract min
        if self.act_min_norm:
            intr_rews -= intr_rews.min(-1,keepdim=True).values
        elif self.act_min_var_norm:
            intr_rews = 5e-4 * (intr_rews - intr_rews.min(-1,keepdim=True).values)/(intr_rews.var(-1, keepdim=True) + 1e-7)
        
        
        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])
        
        
        #! add vdn_intr_rew, qmix_intr_rew learning here
        #* VDN_intr_rew
        if self.args.mixer == "vdn":
            # Mixer
            chosen_action_qvals_mixer = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            
            
            # Optimise
            self.optimiser.zero_grad()
            grad_norm_sum = 0
            intr_rew_epi_ls=[]
            for i in range(self.args.n_agents):
                
                if getattr(self.args, 'q_lambda', False):
                    qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                    qvals = self.target_mixer(qvals, batch["state"])

                    targets = build_q_lambda_targets(rewards + self.am_belta * intr_rews[:,:, i].unsqueeze(-1).detach(), terminated, mask, target_max_qvals, qvals,
                                        self.args.gamma, self.args.td_lambda)
                else:
                    targets = build_td_lambda_targets(rewards + self.am_belta * intr_rews[:,:, i].unsqueeze(-1).detach(), terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
                
                td_error = (chosen_action_qvals_mixer.detach() - targets.detach())
                td_error2 = 0.5 * (td_error - chosen_action_qvals[:,:,i].unsqueeze(-1).detach() + chosen_action_qvals[:,:,i].unsqueeze(-1)).pow(2)
                mask = mask.expand_as(td_error2)
                masked_td_error = td_error2 * mask
                loss = L_td = masked_td_error.sum() / mask.sum()

                loss.backward(retain_graph=True)
                grad_norm_sum += th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
                
                intr_rew_epi_ls.append(self.am_belta*intr_rews[0,:, i].mean().item())
                
            self.optimiser.step()
                
    
        #* Qmix_intr_rew
        if self.args.mixer == "qmix":
             # Mixer
            chosen_action_qvals.retain_grad()
            chosen_action_qvals_mixer = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            chosen_action_qvals_mixer.sum().backward(retain_graph=True)
            Mixer_grad = chosen_action_qvals.grad
            self.optimiser.zero_grad()
            # Qi params
            intr_rew_epi_ls=[]
            for i in range(self.args.n_agents):
                if getattr(self.args, 'q_lambda', False):
                    qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                    qvals = self.target_mixer(qvals, batch["state"])

                    targets = build_q_lambda_targets(rewards + self.am_belta * intr_rews[:,:, i].unsqueeze(-1).detach(), terminated, mask, target_max_qvals, qvals,
                                        self.args.gamma, self.args.td_lambda)
                else:
                    targets = build_td_lambda_targets(rewards + self.am_belta * intr_rews[:,:, i].unsqueeze(-1).detach(), terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
                
                td_error = (chosen_action_qvals_mixer.detach() - targets.detach())
                td_error2 = td_error.squeeze(-1) * Mixer_grad[:,:,i] * chosen_action_qvals[:,:,i]
                masked_td_error = td_error2 * mask.squeeze(-1)
                loss = L_td = masked_td_error.sum() / ((mask.squeeze(-1)).sum())
                loss.backward(retain_graph=True)
                
                intr_rew_epi_ls.append(self.am_belta*intr_rews[0,:, i].mean().item())
            # hpyer params
            chosen_action_qvals_clone = chosen_action_qvals.clone().detach()
            chosen_action_qvals_mixer_clone = self.mixer(chosen_action_qvals_clone, batch["state"][:, :-1])
            hyper_loss = td_error  * chosen_action_qvals_mixer_clone
            hyper_loss = hyper_loss * mask
            (hyper_loss.sum()/(mask.sum())).backward(retain_graph=True)
            
            grad_norm_sum = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()
            pass
            
        
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        #! Action Models Training here
        # ground truth: mac_out[:, :-1]: (batch_size, max_seq_length-1, num_agents, num_actions)
        mac_out_clone = mac_out[:, :-1].clone().detach()
        am_loss_mean = 0
        am_grad_norm_sum = 0
        for i in range(self.args.n_agents):
            action_model = self.action_models[i]
            i_obs = batch['obs'][:,:-1,i].unsqueeze(2)
        
            pred_Qs_i = []
            for t in range(batch.max_seq_length-1):
                pred_Q_i = action_model.forward(i_obs, batch, t=t,i=i) #shape: (batch_size, 1, num_actions)
                pred_Qs_i.append(pred_Q_i)
            pred_Qs_i = th.stack(pred_Qs_i, dim=1).squeeze(2)  # Concat over time shape: (batch_size, seq_length - 1, num_actions)
            
            # loss
            am_loss = (0.5*(mac_out_clone[:,:,i] - pred_Qs_i).pow(2)).mean(-1).unsqueeze(-1) * mask
            am_loss = am_loss.sum() / mask.sum()
            
            self.am_optims[i].zero_grad()
            am_loss.backward()
            
            am_grad_norm_sum += th.nn.utils.clip_grad_norm_(action_model.parameters(), self.args.grad_norm_clip)
            self.am_optims[i].step()  
            
            am_loss_mean += am_loss.item()  
        #! end


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm_sum/self.args.n_agents, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

            #* Logging [intrinsic reward] here: for debug
            act_rew_episode_mean = sum(intr_rew_epi_ls) / len(intr_rew_epi_ls)
            
            if self.args.wandb_enabled:
                wandb.log({
                        "act_rew_episode_mean": act_rew_episode_mean,
                        "am_loss": am_loss_mean/self.args.n_agents,
                        "am_grad_norm": am_grad_norm_sum/self.args.n_agents,
                        })
            
        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info
    
    
    #! Add a method to add intrinsic rewards to the buffer    返回numpy: intr_rew
    def add_intr_rew(self, sampled_batch, env_feat_size, args, mac_out):
        '''
        1. Calculate intrinsic rewards 
        intrinsic_rewards is from alignment.
        '''
        num_agents = sampled_batch['obs'].shape[2]
        batch_size = sampled_batch['obs'].shape[0]
        intr_rew = th.zeros(sampled_batch['actions'].shape[0],sampled_batch['actions'].shape[1]-1, sampled_batch['actions'].shape[2]) # shape: (batch_size, max_seq_len, num_agents)
        j_list = list(range(num_agents))
        
        for i in range(num_agents):
            i_obs = sampled_batch['obs'][:,:,i].clone().detach()
            obs_j_list = self.imagine_agent_j_obs(i_obs, i, j_list, env_feat_size)
            # print("end of imagine_agent_j_obs")
            obs_n = th.cat(obs_j_list, 2)[:,:-1] # shape: (batch_size, max_seq_len, num_agents, obs_size)
            obs_size = th.cat(obs_j_list, 2).shape[-1]
            # act_n = sampled_batch['actions'][:,:,i,:].unsqueeze(2).repeat(1,1,num_agents,1) # shape: (batch_size, max_seq_len, num_ag            
            # inputs = th.cat([obs_n, act_n], 3).view(-1, self.world_models[i].state_shape+1).to(self.world_models[i].device) # shape: (batch_size*max_seq_len*num_agents, obs_size+1)
            
            
            # get the prediction losses
            action_model = self.action_models[i]
            

            pred_Qs_n_i = []
            action_model.agent.cuda()
            action_model.init_hidden(sampled_batch.batch_size)
            for t in range(sampled_batch.max_seq_length-1):
                pred_Q_n_i = action_model.forward(obs_n, sampled_batch, t=t ,i=i)
                pred_Qs_n_i.append(pred_Q_n_i)
            pred_Qs_n_i = th.stack(pred_Qs_n_i, dim=1)  # Concat over time
           
            # MSE           
            pred_losses = th.sqrt((mac_out[:,:-1,i].unsqueeze(2).repeat(1,1,num_agents,1) - pred_Qs_n_i).pow(2).sum(-1)) # shape: (batch_size, max_seq_len, num_agents)
            
            # zero out the losses for invisible agents
            obs_mask = ~th.all(obs_n == 0, -1).to('cuda')
            pred_losses *= obs_mask
            
            # calculate nonzero_obs_count
            ally_num, ally_feat_size = env_feat_size['ally_feat_shape'][0], env_feat_size['ally_feat_shape'][1]
            own_feat_size = env_feat_size['unit_feat_shape']
            nonzero_obs_count = i_obs[:,:-1,-own_feat_size-ally_num*ally_feat_size:-own_feat_size].view(batch_size, -1, ally_num,ally_feat_size)# N(i), 表示i的可见的ally的数量(包括自己)
            nonzero_obs_count = th.sum(nonzero_obs_count[:,:,:,0], -1) + 1 # N(i) + 1
            
            # calculate intrinsic rewards
            intr_rew[:,:,i] = (-pred_losses.sum(2)/ nonzero_obs_count).detach()
        
        # if args.intr_rew_sqrt_norm:
        #     intr_rew = 1 / m.sqrt((i_obs.shape[-1])) * intr_rew #! 这个归一化可能有点太大了，但是先这样吧，后面再改，先看看效果，后面实在不行改成sqrt(i_obs.shape[-1])，或者1
        # elif args.intr_rew_sqrt_mean_norm:
        #     intr_rew = 1 / m.sqrt((i_obs.shape[-1])) * intr_rew
        #     intr_rew_mean = np.repeat(np.expand_dims(intr_rew.mean(-1),-1),intr_rew.shape[-1], axis=-1)
        #     intr_rew = intr_rew - intr_rew_mean
        # elif args.intr_rew_sqrt_min_norm:
        #     intr_rew = 1 / m.sqrt((i_obs.shape[-1])) * intr_rew
        #     intr_rew_min = np.repeat(np.expand_dims(intr_rew.min(-1),-1),intr_rew.shape[-1], axis=-1)
        #     intr_rew = intr_rew - intr_rew_min
        # elif args.intr_rew_mean_norm:
        #     intr_rew = 1 / (i_obs.shape[-1]) * intr_rew
        #     intr_rew_mean = np.repeat(np.expand_dims(intr_rew.mean(-1),-1),intr_rew.shape[-1], axis=-1)
        #     intr_rew = intr_rew - intr_rew_mean
        # elif args.intr_rew_min_norm:
        #     intr_rew = 1 / (i_obs.shape[-1]) * intr_rew
        #     intr_rew_min = np.repeat(np.expand_dims(intr_rew.min(-1),-1),intr_rew.shape[-1], axis=-1)
        #     intr_rew = intr_rew - intr_rew_min    
        # else:
        #     intr_rew = 1 / (i_obs.shape[-1]) * intr_rew
        # print("end of calculate intrinsic rewards.")
        
        return intr_rew # shape: (batch_size, max_seq_len, num_agents)
    

    # ! Add a method to add agent_j_obs to the buffer
    def imagine_agent_j_obs(self, sampled_batch_obs_i, i, j_list, env_feat_shape, sight_range=1, shooting_range=2/3):
        '''
        以i为圆心, 计算i观测内所有ally的想象的obs
        sampled_batch_obs_i: (batch_size, max_seq_len, obs_size)
        TODO: return obs_j_img: list:{(batch_size, max_seq_len, obs_size),...(batch_size, max_seq_len, obs_size)}
        ''' 
        i_obs = sampled_batch_obs_i.clone().detach()
        # size
        move_feat_size = env_feat_shape['agent_movement_feat_shape']
        enemy_num, enemy_feat_size = env_feat_shape['enemy_feat_shape'][0], env_feat_shape['enemy_feat_shape'][1]
        ally_num, ally_feat_size = env_feat_shape['ally_feat_shape'][0], env_feat_shape['ally_feat_shape'][1]
        own_feat_size = env_feat_shape['unit_feat_shape']
        batch_size = i_obs.shape[0]
        max_len = i_obs.shape[1]
        
        # obs_i
        i_move_feat = i_obs[:,:,:move_feat_size] # shape: (batch_size, max_seq_len, move_feat_size)
        i_enemy_feat = i_obs[:,:,move_feat_size:move_feat_size+enemy_num*enemy_feat_size].view(
                        batch_size,max_len,enemy_num,enemy_feat_size) # shape: (batch_size, max_seq_len, agt_n, enemy_feat_size)
        i_ally_feat = i_obs[:,:,move_feat_size+enemy_num*enemy_feat_size:
                      move_feat_size+enemy_num*enemy_feat_size+ally_num*ally_feat_size].view(
                       batch_size,max_len,ally_num,ally_feat_size) # shape: (batch_size, max_seq_len, agt_n, ally_feat_size)
        i_own_feat = i_obs[:,:,-own_feat_size:]
        
        # middle variables for j_enemy_feat
        i_enemy_mask = i_enemy_feat[:,:,:,0:1] # avialale to attack  # shape: (batch_size, max_seq_len, n_agt, feat_size)
        i_enemy_health = i_enemy_feat[:,:,:,1] # enemy health
        i_enemy_shield = i_enemy_feat[:,:,:,4] # enemy shield
        i_enemy_pos = i_enemy_feat[:,:,:,2:4] # enemy position
        
        # cal obs_j_list
        obs_j_list = []
        for j in j_list:
            if j == i:
                obs_j_list.append(i_obs.view(batch_size,max_len,1,-1))
            else:
                obs_j = i_obs.clone().detach()
                #* calculate obs_j: j_ally_feat, j_enemy_feat, j_own_feat
                #* j_ally_feat
                    # 1. cal i_feat in j's view :shape: (batch_size, max_seq_len, ally_feat_size)
                j_feat = i_ally_feat[:,:,j,:].clone().detach() if j < i else i_ally_feat[:,:,j-1,:].clone().detach() # shape: (batch_size, max_seq_len, ally_feat_size)
                i_feat_in_j = i_ally_feat[:,:,0,:].clone().detach() # shape: (batch_size, max_seq_len, ally_feat_size)
                i_feat_in_j[:,:,0:1] = 1 # visible 
                i_feat_in_j[:,:,1:2] = j_feat[:,:,1:2] # distance
                i_feat_in_j[:,:,2:4] = th.zeros(j_feat[:,:,2:4].shape) # relative x , relative y
                i_feat_in_j[:,:,-ally_feat_size+4:] = i_own_feat.clone().detach() # shield + unit_type
                    # 2. ally_feat: add i_feat in i_ally_feat
                device = 'cuda' if th.cuda.is_available() else 'cpu'
                ally_feat = th.zeros(batch_size,max_len,ally_num+1,ally_feat_size).to(device) # shape: (batch_size, max_seq_len, ally_num+1, ally_feat_size)
                for k in range(ally_num+1):
                    if k < i:
                        ally_feat[:,:,k,:] = i_ally_feat[:,:,k,:].clone().detach() 
                    elif k==i:
                        ally_feat[:,:,k,:] = i_feat_in_j.clone().detach()
                    else:
                        ally_feat[:,:,k,:] = i_ally_feat[:,:,k-1,:].clone().detach()
                    # 3. cal j_vis_mask, ji_ally_pos_delta, ji_ally_mask, 
                # middle variables for j_ally_feat
                j_vis_mask = j_feat[:,:,0]              # shape: (batch_size, max_seq_len, 1)
                j_pos = j_feat[:,:,2:4]                 # shape: (batch_size, max_seq_len, 2)
                ally_pos = ally_feat[:,:,:,2:4]         # ally position
                ji_ally_pos_delta = ally_pos - j_pos.unsqueeze(2).repeat(1,1,ally_num+1,1)     # agent j 相对于其他agent的position(包含agent i)
                ji_ally_mask = (th.sqrt(th.square(ji_ally_pos_delta).sum(-1)) <= sight_range).unsqueeze(-1) # shape: (batch_size, max_seq_len, ally_num+1, 1)
                ji_ally_mask = ji_ally_mask * (ally_feat[:,:,:,0:1] > 0)
                    # 4. cal ji_ally_feat
                ji_ally_dis = (th.sqrt(th.square(ji_ally_pos_delta).sum(-1))).unsqueeze(-1) * ji_ally_mask # shape: (batch_size, max_seq_len, ally_num+1, 1)
                ji_ally_pos = ji_ally_pos_delta * ji_ally_mask # shape: (batch_size, max_seq_len, ally_num, 2)
                ji_ally_shield = (ally_feat[:,:,:,4].unsqueeze(-1)) * ji_ally_mask # shape: (batch_size, max_seq_len, ally_num+1, 1)
                ji_ally_feat = th.cat((ji_ally_mask, ji_ally_dis, ji_ally_pos, ji_ally_shield), -1) # shape: (batch_size, max_seq_len, ally_num, 5)
                    # 5. cal j_ally_feat: remove j_feat from ji_ally_feat
                j_ally_feat = th.zeros(batch_size,max_len,ally_num,ally_feat_size).to(device) # shape: (batch_size, max_seq_len, ally_num, ally_feat_size)
                for k in range(ally_num+1):
                    if k < j:
                        j_ally_feat[:,:,k,:5] = ji_ally_feat[:,:,k,:5]
                        j_ally_feat[:,:,k,-ally_feat_size+4:] = ally_feat[:,:,k,-ally_feat_size+4:]
                    elif k > j:
                        j_ally_feat[:,:,k-1,:5] = ji_ally_feat[:,:,k,:5]
                        j_ally_feat[:,:,k-1,-ally_feat_size+4:] = ally_feat[:,:,k,-ally_feat_size+4:]
                j_ally_feat = j_ally_feat.view(batch_size,max_len,-1)
                                        
                #* j_enemy feat 
                enemy_pos_delta = i_enemy_pos - j_pos.unsqueeze(2).repeat(1,1,enemy_num,1) 
                j_enemy_vis_mask = (th.sqrt(th.square(enemy_pos_delta).sum(-1)) <= sight_range).unsqueeze(-1) * (i_enemy_feat[:,:,:,1:2] > 0)
                j_enemy_attack_mask = (th.sqrt(th.square(enemy_pos_delta).sum(-1)) <= shooting_range).unsqueeze(-1) * (i_enemy_feat[:,:,:,1:2] > 0)
                # j_enemy_mask = (th.sqrt(th.square(enemy_pos_delta).sum(-1)) <= shooting_range).unsqueeze(-1) * (i_enemy_feat[:,:,:,0:1] > 0) # shape: (batch_size, max_seq_len, enemy_num, 1)
                j_enemy_pos = enemy_pos_delta * j_enemy_vis_mask # shape: (batch_size, max_seq_len, enemy_num, 2)
                j_enemy_health = (i_enemy_health.unsqueeze(-1)) * j_enemy_vis_mask # shape: (batch_size, max_seq_len, enemy_num, 1)
                j_enemy_shield = (i_enemy_shield.unsqueeze(-1)) * j_enemy_vis_mask # shape: (batch_size, max_seq_len, enemy_num, 1)
                if enemy_feat_size > 5:
                    j_enemy_unit_type = (i_enemy_feat[:,:,:,-enemy_feat_size+5:]) * j_enemy_vis_mask # shape: (batch_size, max_seq_len, enemy_num, 1)
                    j_enemy_feat = th.cat((j_enemy_attack_mask, j_enemy_health, j_enemy_pos, j_enemy_shield,j_enemy_unit_type), -1) # shape: (batch_size, max_seq_len, enemy_num, 5)
                else:
                    j_enemy_feat = th.cat((j_enemy_attack_mask, j_enemy_health, j_enemy_pos, j_enemy_shield), -1) # shape: (batch_size, max_seq_len, enemy_num, 5)
                j_enemy_feat = j_enemy_feat.view(batch_size,max_len,-1) # shape: (batch_size, max_seq_len, enemy_num*5
                       
                #* obs_j[:,:,:move_feat_shape] = 1
                obs_j[:,:,move_feat_size:move_feat_size+enemy_num*enemy_feat_size] = j_enemy_feat
                obs_j[:,:,-ally_num*ally_feat_size-own_feat_size:-own_feat_size] = j_ally_feat
                obs_j[:,:,-own_feat_size] = j_feat[:,:,4]
                if -own_feat_size+1 < 0:
                    obs_j[:,:,-own_feat_size+1:] = j_feat[:,:,-own_feat_size+1:]
                final_obs_j = (j_vis_mask.unsqueeze(2).repeat(1,1,obs_j.shape[-1]) * obs_j).view(batch_size,max_len,1,-1)
                obs_j_list.append(final_obs_j)
        
        return obs_j_list

    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
