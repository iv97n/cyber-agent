# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deep Recurrent Q-learning (DRQN) agent"""

from numpy import ndarray
from cyberbattle._env import cyberbattle_env
import numpy as np
from typing import List, NamedTuple, Optional, Tuple, Union
import random

# deep learning packages
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

from cyberbattle.agents.baseline.learner import Learner
from cyberbattle.agents.baseline.agent_wrapper import EnvironmentBounds
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.agent_randomcredlookup import CredentialCacheExploiter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_argmax(array):
    """Just like `argmax` but if there are multiple elements with the max
    return a random index to break ties instead of returning the first one."""
    max_value = np.max(array)
    max_index = np.where(array == max_value)[0]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    return max_value, max_index


class CyberBattleStateActionModel:
    """Define an abstraction of the state and action space
    for a CyberBattle environment, to be used to train a Q-function.
    """

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(
            ep,
            [
                w.Feature_discovered_notowned_node_count(ep, None)
            ],
        )

        self.node_specific_features = w.ConcatFeatures(
            ep,
            [
                w.Feature_success_actions_at_node(ep),
                w.Feature_failed_actions_at_node(ep),
                w.Feature_active_node_properties(ep),
                w.Feature_active_node_age(ep),
            ],
        )

        self.state_space = w.ConcatFeatures(
            ep,
            self.global_features.feature_selection + self.node_specific_features.feature_selection,
        )

        self.action_space = w.AbstractAction(ep)

    def get_state_astensor(self, state: w.StateAugmentation):
        state_vector = self.state_space.get(state, node=None)
        state_vector_float = np.array(state_vector, dtype=np.float32)
        # Return discrete vector, batching will be handled later
        return torch.from_numpy(state_vector_float)

    def implement_action(
        self,
        wrapped_env: w.AgentWrapper,
        actor_features: ndarray,
        abstract_action: np.int32,
    ) -> Tuple[str, Optional[cyberbattle_env.Action], Optional[int]]:
        """Specialize an abstract model action into a CyberBattle gym action."""

        observation = wrapped_env.state.observation

        potential_source_nodes = [from_node for from_node in w.owned_nodes(observation) if np.all(actor_features == self.node_specific_features.get(wrapped_env.state, from_node))]

        if len(potential_source_nodes) > 0:
            source_node = np.random.choice(potential_source_nodes)

            gym_action = self.action_space.specialize_to_gymaction(source_node, observation, np.int32(abstract_action))

            if not gym_action:
                return "exploit[undefined]->explore", None, None

            elif wrapped_env.env.is_action_valid(gym_action, observation["action_mask"]):
                return "exploit", gym_action, source_node
            else:
                return "exploit[invalid]->explore", None, None
        else:
            return "exploit[no_actor]->explore", None, None


class Transition(NamedTuple):
    """One taken transition"""
    state: Tensor
    action: Tensor
    next_state: Optional[Tensor]
    reward: Tensor


class EpisodicReplayMemory(object):
    """Replay memory that stores full episodes"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.current_episode = []

    def push(self, state, action, next_state, reward):
        """Saves a transition to the current episode buffer."""
        self.current_episode.append(Transition(state, action, next_state, reward))

    def flush_episode(self):
        """Commits the current episode to the replay memory"""
        if len(self.current_episode) == 0:
            return

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = self.current_episode
        self.position = (self.position + 1) % self.capacity
        self.current_episode = []

    def sample(self, batch_size):
        """Samples a batch of episodes"""
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    """The Deep Recurrent Neural Network"""

    def __init__(self, ep: EnvironmentBounds):
        super(DRQN, self).__init__()

        model = CyberBattleStateActionModel(ep)
        linear_input_size = len(model.state_space.dim_sizes)
        output_size = model.action_space.flat_size()
        self.hidden_size = 128
        self.num_layers = 1 # Keep it simple for now

        self.gru = nn.GRU(linear_input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.head = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch, 1, input)
        
        out, new_hidden = self.gru(x, hidden)
        
        # out shape: (batch, seq, hidden)
        q_values = self.head(out)
        
        return q_values, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class ChosenActionMetadata(NamedTuple):
    abstract_action: np.int32
    actor_node: int
    actor_features: ndarray
    actor_state: ndarray
    hidden_state: Optional[Tensor]

    def __repr__(self) -> str:
        return f"[abstract_action={self.abstract_action}, actor={self.actor_node}]"


class DeepRecurrentQLearnerPolicy(Learner):
    """Deep Recurrent Q-Learning (DRQN)"""

    def __init__(
        self,
        ep: EnvironmentBounds,
        gamma: float,
        replay_memory_size: int,
        target_update: int,
        batch_size: int,
        learning_rate: float,
    ):
        self.stateaction_model = CyberBattleStateActionModel(ep)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.policy_net = DRQN(ep).to(device)
        self.target_net = DRQN(ep).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = target_update

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.memory = EpisodicReplayMemory(replay_memory_size)
        self.credcache_policy = CredentialCacheExploiter()
        
        # Current inference hidden state (h_t)
        self.inference_hidden = None

    def new_episode(self):
        """Reset the hidden state at start of episode"""
        self.inference_hidden = None
        self.memory.current_episode = []

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # 1. Sample episodes
        episodes = self.memory.sample(self.batch_size)
        
        # 2. Prepare batches
        # We need to pad episodes to valid length
        # batch structure:
        # [ [t0, t1, t2], [t0, t1], ... ]
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for ep in episodes:
            states = torch.stack([t.state for t in ep])
            actions = torch.stack([t.action for t in ep])
            rewards = torch.stack([t.reward for t in ep])
            
            # Next states: shift by 1. For the last step, if next_state is None, it's terminal
            # However, Transition stores next_state explicitly.
            # We need to handle the None (terminal) manually for padding.
            # Let's verify how we pushed them.
            
            # Helper to handle None in next_state batching
            next_states_list = []
            dones_list = []
            for t in ep:
                if t.next_state is None:
                    next_states_list.append(torch.zeros_like(t.state)) # Dummy zero state
                    dones_list.append(1.0)
                else:
                    next_states_list.append(t.next_state)
                    dones_list.append(0.0)
            
            next_states = torch.stack(next_states_list)
            dones = torch.tensor(dones_list, device=device, dtype=torch.float)
            
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)

        # Pad sequences
        # padded_states shape: (batch, max_seq, input_dim)
        padded_states = pad_sequence(batch_states, batch_first=True)
        padded_actions = pad_sequence(batch_actions, batch_first=True)
        padded_rewards = pad_sequence(batch_rewards, batch_first=True)
        padded_next_states = pad_sequence(batch_next_states, batch_first=True)
        padded_dones = pad_sequence(batch_dones, batch_first=True)
        
        # Create mask for padding
        # We can use the lengths to create a boolean mask
        lengths = torch.tensor([len(ep) for ep in episodes], device=device)
        # (batch, max_seq)
        mask = torch.arange(padded_states.size(1), device=device)[None, :] < lengths[:, None]
        
        # 3. Calculate Q(s, a)
        # Initialize hidden state to zeros for training
        hidden = self.policy_net.init_hidden(len(episodes))
        
        # Run through policy net
        # q_values: (batch, seq, action_dim)
        q_values_all, _ = self.policy_net(padded_states, hidden)
        
        # Gather the specific actions taken
        # padded_actions is (batch, seq, 1)
        state_action_values = q_values_all.gather(2, padded_actions) # (batch, seq, 1)
        
        # 4. Calculate V(s')
        with torch.no_grad():
            next_hidden = self.target_net.init_hidden(len(episodes))
            next_q_values_all, _ = self.target_net(padded_next_states, next_hidden)
            next_state_values = next_q_values_all.max(2)[0].unsqueeze(2) # (batch, seq, 1)
            
            # Zero out values for terminal states using padded_dones
            # dones: 1.0 if terminal
            next_state_values = next_state_values * (1.0 - padded_dones.unsqueeze(2))
            
            expected_state_action_values = padded_rewards + (self.gamma * next_state_values)

        # 5. Compute Loss
        # We only care about the loss at valid (masked) steps
        # Mask shape: (batch, seq) -> unsqueeze to (batch, seq, 1)
        mask_expanded = mask.unsqueeze(2)
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        masked_loss = loss * mask_expanded.float()
        
        # Average over valid steps
        final_loss = masked_loss.sum() / mask_expanded.sum()

        self.optimizer.zero_grad()
        final_loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray) -> ndarray:
        return np.concatenate(
            (
                np.array(global_state, dtype=np.float32),
                np.array(actor_features, dtype=np.float32),
            )
        )

    def on_step(self, wrapped_env, observation, reward, done, truncated, info, action_metadata):
        # 1. Store transition
        # We need to reconstruct the next state vector if not done
        if done:
            next_actor_state_tensor = None
            self.memory.push(
                torch.as_tensor(action_metadata.actor_state, dtype=torch.float, device=device),
                torch.tensor([[action_metadata.abstract_action]], device=device, dtype=torch.long),
                None,
                torch.tensor([[reward]], device=device, dtype=torch.float)
            )
            self.memory.flush_episode()
            self.inference_hidden = None # Reset hidden
        else:
            # Re-calculate state vector for the chosen actor (the one we focused on)
            # note: in the real world we would use the OBSERVED next state of that actor.
            # But here `agent_dql` re-calculates it.
            
            # Current agent state (after step)
            agent_state = wrapped_env.state
            next_global_state = self.stateaction_model.global_features.get(agent_state, node=None)
            next_actor_features = self.stateaction_model.node_specific_features.get(agent_state, action_metadata.actor_node)
            next_actor_state = self.get_actor_state_vector(next_global_state, next_actor_features)
            
            next_actor_state_tensor = torch.as_tensor(next_actor_state, dtype=torch.float, device=device)
            
            self.memory.push(
                torch.as_tensor(action_metadata.actor_state, dtype=torch.float, device=device),
                torch.tensor([[action_metadata.abstract_action]], device=device, dtype=torch.long),
                next_actor_state_tensor,
                torch.tensor([[reward]], device=device, dtype=torch.float)
            )
            
            # 2. Update Inference Hidden State
            # We want to carry over the hidden state that resulted from processing THIS specific action
            # We calculated it in `lookup_dqn`, but we computed it for ALL candidates.
            # We need to pick the one corresponding to the chosen actor.
            
            # We stored `hidden_state` in metadata? No, that would be memory expensive if we stored all.
            # But since we just took one step, let's just use the one we computed.
            # Wait, `lookup_dqn` doesn't return the hidden states.
            # We should probably modify `explore`/`exploit` to capture it.
            
            # Simplification: Just re-run the forward pass for the chosen state to get the new hidden
            # This is cheap (single vector) and robust.
            
            with torch.no_grad():
                current_tensor = torch.as_tensor(action_metadata.actor_state, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                _, new_hidden = self.policy_net(current_tensor, self.inference_hidden)
                self.inference_hidden = new_hidden

            # 3. Train
            self.optimize_model()

    def end_of_episode(self, i_episode, t):
        if i_episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def lookup_dqn(self, states_to_consider: List[ndarray]) -> Tuple[List[np.int32], List[np.int32]]:
        with torch.no_grad():
            state_batch = torch.tensor(states_to_consider, dtype=torch.float, device=device)
            # shape: (batch_actors, input_dim) -> (batch, 1, input)
            state_batch = state_batch.unsqueeze(1)
            
            # Broadcast hidden state to match batch size
            batch_size = len(states_to_consider)
            if self.inference_hidden is None:
                hidden = self.policy_net.init_hidden(batch_size)
            else:
                # inference_hidden is (num_layers, 1, hidden)
                # We need (num_layers, batch, hidden)
                hidden = self.inference_hidden.expand(-1, batch_size, -1).contiguous()

            # Forward output: (batch, 1, output_dim)
            q_out, _ = self.policy_net(state_batch, hidden)
            
            # Max over actions
            dnn_output = q_out.squeeze(1).max(1)
            action_lookups = dnn_output[1].tolist()
            expectedq_lookups = dnn_output[0].tolist()

        return action_lookups, expectedq_lookups

    def metadata_from_gymaction(self, wrapped_env, gym_action):
        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)
        actor_node = cyberbattle_env.sourcenode_of_action(gym_action)
        actor_features = self.stateaction_model.node_specific_features.get(wrapped_env.state, actor_node)
        abstract_action = self.stateaction_model.action_space.abstract_from_gymaction(gym_action)
        return ChosenActionMetadata(
            abstract_action=abstract_action,
            actor_node=actor_node,
            actor_features=actor_features,
            actor_state=self.get_actor_state_vector(current_global_state, actor_features),
            hidden_state=None # We don't store it here to avoid duplication
        )

    def explore(self, wrapped_env: w.AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
        return "explore", gym_action, metadata

    def try_exploit_at_candidate_actor_states(self, wrapped_env, current_global_state, actor_features, abstract_action):
        actor_state = self.get_actor_state_vector(current_global_state, actor_features)

        action_style, gym_action, actor_node = self.stateaction_model.implement_action(wrapped_env, actor_features, abstract_action)

        if gym_action:
            return (
                action_style,
                gym_action,
                ChosenActionMetadata(
                    abstract_action=abstract_action,
                    actor_node=actor_node,
                    actor_features=actor_features,
                    actor_state=actor_state,
                    hidden_state=None
                ),
            )
        else:
            # For failed exploit, we should technically record it as a transition that resulted in same state
            # But the baseline code just skips it and returns "explore" instruction.
            # We'll just return the failure signal.
            return "exploit[undefined]->explore", None, None

    def exploit(self, wrapped_env, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)
        active_actors_features: List[ndarray] = [self.stateaction_model.node_specific_features.get(wrapped_env.state, from_node) for from_node in w.owned_nodes(observation)]
        unique_active_actors_features: List[ndarray] = list(np.unique(active_actors_features, axis=0))
        candidate_actor_state_vector: List[ndarray] = [self.get_actor_state_vector(current_global_state, node_features) for node_features in unique_active_actors_features]

        remaining_action_lookups, remaining_expectedq_lookups = self.lookup_dqn(candidate_actor_state_vector)
        remaining_candidate_indices = list(range(len(candidate_actor_state_vector)))

        while remaining_candidate_indices:
            _, remaining_candidate_index = random_argmax(remaining_expectedq_lookups)
            actor_index = remaining_candidate_indices[remaining_candidate_index]
            abstract_action = remaining_action_lookups[remaining_candidate_index]
            actor_features = unique_active_actors_features[actor_index]

            action_style, gym_action, metadata = self.try_exploit_at_candidate_actor_states(wrapped_env, current_global_state, actor_features, abstract_action)

            if gym_action:
                return action_style, gym_action, metadata

            remaining_candidate_indices.pop(remaining_candidate_index)
            remaining_expectedq_lookups.pop(remaining_candidate_index)
            remaining_action_lookups.pop(remaining_candidate_index)

        return "exploit[undefined]->explore", None, None

    def stateaction_as_string(self, action_metadata) -> str:
        return ""
