from .nq_learner import NQLearner
from .nq_learner_act_rew import NQLearner_act_rew

REGISTRY = {}
#! add intr reward in iql
REGISTRY["nq_learner_act_rew"] = NQLearner_act_rew
