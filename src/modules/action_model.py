from controllers import REGISTRY as mac_REGISTRY
from controllers.n_controller import NMAC
import torch as th


class Action_Model(NMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        
        
    def forward(self, obs_n, ep_batch,  t, i, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        num_agents = obs_n.shape[2]
        agent_inputs = self._build_inputs(obs_n, ep_batch, t, i)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states.repeat(1,num_agents,1) )
        self.hidden_states = self.hidden_states[:,:1,:]
        return agent_outs

    
    def _build_inputs(self, obs_n, batch, t, i ):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        num_agents = batch['obs'].shape[2]
        bs = batch.batch_size
        inputs = []
        inputs.append(obs_n[:, t])  # b1av
        num_agents = obs_n.shape[2]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t, i].unsqueeze(1).repeat(1,num_agents,1)))
            else:
                inputs.append(batch["actions_onehot"][:, t-1, i].unsqueeze(1).repeat(1,num_agents,1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)[:,i,:].unsqueeze(1).repeat(1,num_agents,1))

        inputs = th.cat([x.reshape(bs, num_agents, -1) for x in inputs], dim=-1)
            
        return inputs # [batch_size, n_agents, input_shape]
    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, 1, -1)  # bav