# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# jax version
# __init__.py for the Reinforcement Learning Framework
__version__ = "0.0.1"
# Agentic behavior and cognition modules
from .agentic_behavior import AgenticBehavior
from .cognition import CognitionAgent

# RL environment and frameworks
from .environment import Environment
from .rl_environments import GymEnvironment

# Advanced RL algorithms
from .advanced_rl_algorithms import SACAgent, TD3Agent
from .curiosity_driven_exploration import CuriosityDrivenAgent
from .deep_rl_algorithms import DQNAgent, PPOAgent
from .imitation_learning import BehavioralCloning, DAgger
from .meta_learning import MAML, Reptile
from .model_based_rl import MBPOAgent, MCTSAgent
from .model_free_rl import QLearningAgent, SARSAgent
from .multi_agent_rl import MultiAgentRL
from .reinforcement_learning_advancements import QNetwork, AdvancedRLAgent
from .self_curing_rl import SelfCuringRL

# RL algorithms and modules
from .rl_algorithms import PolicyGradient, QLearning, ActorCritic
from .rl_module import (
    PPOBuffer,
    discount_cumsum,
    Actor,
    Critic,
    RLAgent,
    select_action,
    update_ppo,
    get_minibatches,
    train_rl_agent
)

# Newly added advanced RL components
from .hierarchical_rl import HierarchicalRL
from .offline_rl import OfflineRL
from .rl_withcasual_reasoning import RLWithCasualReasoning
from .q_rl import QRLAgent

__all__ = [
    # Agentic behavior and cognition modules
    'AgenticBehavior',
    'CognitionAgent',
    
    # RL environment and frameworks
    'Environment',
    'GymEnvironment',
    
    # Advanced RL algorithms
    'SACAgent',
    'TD3Agent',
    'CuriosityDrivenAgent',
    'DQNAgent',
    'PPOAgent',
    'BehavioralCloning',
    'DAgger',
    'MAML',
    'Reptile',
    'MBPOAgent',
    'MCTSAgent',
    'QLearningAgent',
    'SARSAgent',
    'MultiAgentRL',
    'QNetwork',
    'AdvancedRLAgent',
    'SelfCuringRL',
    
    # RL algorithms and modules
    'PolicyGradient',
    'QLearning',
    'ActorCritic',
    'PPOBuffer',
    'discount_cumsum',
    'Actor',
    'Critic',
    'RLAgent',
    'select_action',
    'update_ppo',
    'get_minibatches',
    'train_rl_agent',
    
    # Newly added advanced RL components
    'HierarchicalRL',
    'OfflineRL',
    'RLWithCasualReasoning',
    'QRLAgent',
]

