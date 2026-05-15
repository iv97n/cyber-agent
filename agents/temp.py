# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Function DeepQLearnerPolicy.optimize_model:
#   Copyright (c) 2017, Pytorch contributors
#   All rights reserved.
#   https://github.com/pytorch/tutorials/blob/master/LICENSE

"""Deep Q-learning agent applied to chain network (notebook)
This notebooks can be run directly from VSCode, to generate a
traditional Jupyter Notebook to open in your browser
 you can run the VSCode command `Export Currenty Python File As Jupyter Notebook`.

Requirements:
    Nvidia CUDA drivers for WSL2: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
    PyTorch
"""

# pylint: disable=invalid-name

# %% [markdown]
# # Chain network CyberBattle Gym played by a Deeo Q-learning agent

# %%
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
from torch.nn.utils.clip_grad import clip_grad_norm_

from cyberbattle.agents.baseline.learner import Learner
from cyberbattle.agents.baseline.agent_wrapper import EnvironmentBounds
import cyberbattle.agents.baseline.agent_wrapper as w
from cyberbattle.agents.baseline.agent_randomcredlookup import CredentialCacheExploiter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CyberBattleStateActionModel:
    """Define an abstraction of the state and action space
    for a CyberBattle environment, to be used to train a Q-function.
    """

    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        self.global_features = w.ConcatFeatures(
            ep,
            [
                # w.Feature_discovered_node_count(ep),
                # w.Feature_owned_node_count(ep),
                w.Feature_discovered_notowned_node_count(ep, None)
                # w.Feature_discovered_ports(ep),
                # w.Feature_discovered_ports_counts(ep),
                # w.Feature_discovered_ports_sliding(ep),
                # w.Feature_discovered_credential_count(ep),
                # w.Feature_discovered_nodeproperties_sliding(ep),
            ],
        )

        self.node_specific_features = w.ConcatFeatures(
            ep,
            [
                # w.Feature_actions_tried_at_node(ep),
                w.Feature_success_actions_at_node(ep),
                w.Feature_failed_actions_at_node(ep),
                w.Feature_active_node_properties(ep),
                w.Feature_active_node_age(ep),
                # w.Feature_active_node_id(ep)
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
        state_tensor = torch.from_numpy(state_vector_float).unsqueeze(0)
        return state_tensor

    def implement_action(
        self,
        wrapped_env: w.AgentWrapper,
        actor_features: ndarray,
        abstract_action: np.int32,
    ) -> Tuple[str, Optional[cyberbattle_env.Action], Optional[int]]:
        """Specialize an abstract model action into a CyberBattle gym action.

        actor_features -- the desired features of the actor to use (source CyberBattle node)
        abstract_action -- the desired type of attack (connect, local, remote).

        Returns a gym environment implementing the desired attack at a node with the desired embedding.
        """

        observation = wrapped_env.state.observation

        # Pick source node at random (owned and with the desired feature encoding)
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


# %%

# Deep Q-learning


class Transition(NamedTuple):
    """One taken transition and its outcome"""

    state: Union[Tuple[Tensor], List[Tensor]]
    action: Union[Tuple[Tensor], List[Tensor]]
    next_state: Union[Tuple[Tensor], List[Tensor]]
    reward: Union[Tuple[Tensor], List[Tensor]]


class ReplayMemory(object):
    """Transition replay memory
    Uses bootstrapped random updates to allow the training of DRQN
    """
    # handle the iterations when the replay memory is still not full?
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition. Ring buffer"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, unroll_iterations=10):
        """randomly select batch_size starting points"""
        last_possible_start = len(self.memory) - unroll_iterations

        random_starts = [random.randint(0, last_possible_start) for _ in range(batch_size)]
        batch = [self.memory[start:start + unroll_iterations] for start in random_starts]
        return batch

        # return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    """The Deep Neural Network used to estimate the Q function"""

    def __init__(self, ep: EnvironmentBounds):
        super(DRQN, self).__init__()

        model = CyberBattleStateActionModel(ep)
        linear_input_size = len(model.state_space.dim_sizes)
        output_size = model.action_space.flat_size()

        self.fc1 = nn.Linear(linear_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)
        self.head = nn.Linear(128, output_size)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # recurrent forward
        if hidden is None:
            h0 = torch.zeros(1, batch_size, 128, device=x.device)
            c0 = torch.zeros(1, batch_size, 128, device=x.device)
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch, seq_len, 128)

        # apply head to all timesteps
        q_values = self.head(lstm_out)

        return q_values, hidden


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


class ChosenActionMetadata(NamedTuple):
    """Additonal info about the action chosen by the DQN-induced policy"""

    abstract_action: np.int32
    actor_node: int
    actor_features: ndarray
    actor_state: ndarray

    def __repr__(self) -> str:
        return f"[abstract_action={self.abstract_action}, actor={self.actor_node}, state={self.actor_state}]"


class DeepRecurrentQLearnerPolicy(Learner):
    """Deep Q-Learning on CyberBattle environments

    Parameters
    ==========
    ep -- global parameters of the environment
    model -- define a state and action abstraction for the gym environment
    gamma -- Q discount factor
    replay_memory_size -- size of the replay memory
    batch_size    -- Deep Q-learning batch
    target_update -- Deep Q-learning replay frequency (in number of episodes)
    learning_rate -- the learning rate

    Parameters from DeepDoubleQ paper
        - learning_rate = 0.00025
        - linear epsilon decay
        - gamma = 0.99

    Pytorch code from tutorial at
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

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

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)  # type: ignore
        self.memory = ReplayMemory(replay_memory_size)

        self.credcache_policy = CredentialCacheExploiter()
        self.inference_hidden = None

    def new_episode(self):
        self.inference_hidden = None

    def parameters_as_string(self):
        return f"γ={self.gamma}, lr={self.learning_rate}, replaymemory={self.memory.capacity},\n" f"batch={self.batch_size}, target_update={self.target_update}"

    def all_parameters_as_string(self) -> str:
        model = self.stateaction_model
        return (
            f"{self.parameters_as_string()}\n"
            f"dimension={model.state_space.flat_size()}x{model.action_space.flat_size()}, "
            f"Q={[f.name() for f in model.state_space.feature_selection]} "
            f"-> 'abstract_action'"
        )

    def optimize_model(self, norm_clipping=False, unroll_iteratons=10):
        # Be more strict? require more sampels to start optimizing? e.g self.batch_size*unroll_iterations
        if len(self.memory) < self.batch_size or len(self.memory) < unroll_iteratons:
            return

        # Sample batch_size random sequences of transitions from the replay memory D
        sequence_transitions = self.memory.sample(self.batch_size)

        # Unpack the batch of episodes
        # sequence_transitions is a list of lists: [ [Transition1, Transition2...], [Transition1...] ]
        # We need to process this into tensors for the RNN
        
        # 1. Separate inputs and targets
        state_sequences = []
        action_sequences = []
        reward_sequences = []
        next_state_sequences = []
        non_final_masks = []
        
        for episode in sequence_transitions:
             # unzip the episode: [(s,a,r,s'), ...] -> ([s...], [a...], [r...], [s'...])
            batch = Transition(*zip(*episode))
            
            state_sequences.append(torch.cat(batch.state))
            action_sequences.append(torch.cat(batch.action))
            reward_sequences.append(torch.cat(batch.reward))
            
            # For next states, we need to handle None (terminal states)
            # If s' is None, we can just use a zero tensor or the current state, 
            # as it will be masked out anyway by the value calculation
            next_states_proc = []
            episode_mask = []
            for s in batch.next_state:
                if s is not None:
                    next_states_proc.append(s)
                    episode_mask.append(True)
                else:
                    # Append a dummy state (same size)
                    next_states_proc.append(torch.zeros_like(batch.state[0]))
                    episode_mask.append(False)
            
            next_state_sequences.append(torch.cat(next_states_proc))
            non_final_masks.append(torch.tensor(episode_mask, device=device, dtype=torch.bool))

        # 2. Pad sequences to the same length for batch processing
        # Use batch_first=True
        state_batch_padded = torch.nn.utils.rnn.pad_sequence(state_sequences, batch_first=True)
        action_batch_padded = torch.nn.utils.rnn.pad_sequence(action_sequences, batch_first=True, padding_value=0) # dummy action 0
        reward_batch_padded = torch.nn.utils.rnn.pad_sequence(reward_sequences, batch_first=True, padding_value=0.0)
        next_state_batch_padded = torch.nn.utils.rnn.pad_sequence(next_state_sequences, batch_first=True)
        non_final_mask_padded = torch.nn.utils.rnn.pad_sequence(non_final_masks, batch_first=True, padding_value=False)
        
        # Create a general mask for the padding (True where data exists, False where padding)
        # lengths = torch.tensor([len(s) for s in state_sequences], device=device)
        # We can deduce the mask from one of the padded tensors, e.g. non_final_mask wasn't good for this
        # Let's create an explicit mask for valid steps vs padding
        valid_steps_mask = torch.zeros_like(reward_batch_padded, dtype=torch.bool)
        for i, seq in enumerate(reward_sequences):
            valid_steps_mask[i, :len(seq)] = True

        # 3. Forward Pass (Policy Network)
        # Initialize hidden state to zeros for the start of episodes
        # policy_net.forward(x) will handle zero initialization if hidden is None
        # Q(s_t, h_{t-1})
        q_values, _ = self.policy_net(state_batch_padded) 
        
        # Select the Q-values for the actions that were actually taken
        # q_values shape: (batch, seq_len, n_actions)
        # action_batch shape: (batch, seq_len, 1) -> need (batch, seq_len, 1) for gather
        state_action_values = q_values.gather(2, action_batch_padded).squeeze(2)

        # 4. Compute Target Values (Target Network)
        # V(s_{t+1}) = max_a Q'(s_{t+1}, h_t, a)
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_batch_padded)
            # Max over actions (dim 2)
            next_state_max_q = next_q_values.max(2)[0]
            
            # Apply mask: items where next_state is None (terminal) should have value 0
            # Also padding should be 0 (though handled by loss mask later)
            next_state_values = next_state_max_q * non_final_mask_padded
            
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch_padded

        # 5. Compute Loss
        # We only want to compute loss on valid time steps, not on padding
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        
        # Apply the mask to zero out loss for padding
        masked_loss = loss * valid_steps_mask.float()
        
        # Average over the number of valid elements
        final_loss = masked_loss.sum() / valid_steps_mask.sum()

        # 6. Optimize
        self.optimizer.zero_grad()
        final_loss.backward()

        if norm_clipping:
            clip_grad_norm_(self.policy_net.parameters(), 1.0)
        else:
             for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()


    def get_actor_state_vector(self, global_state: ndarray, actor_features: ndarray) -> ndarray:
        return np.concatenate(
            (
                np.array(global_state, dtype=np.float32),
                np.array(actor_features, dtype=np.float32),
            )
        )

    def update_q_function(
        self,
        reward: float,
        actor_state: ndarray,
        abstract_action: np.int32,
        next_actor_state: Optional[ndarray],
    ):
        # store the transition in memory
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
        action_tensor = torch.tensor([[np.int_(abstract_action)]], device=device, dtype=torch.long)
        current_state_tensor = torch.as_tensor(actor_state, dtype=torch.float, device=device).unsqueeze(0)
        if next_actor_state is None:
            next_state_tensor = None
        else:
            next_state_tensor = torch.as_tensor(next_actor_state, dtype=torch.float, device=device).unsqueeze(0)
        self.memory.push(current_state_tensor, action_tensor, next_state_tensor, reward_tensor)

        # optimize the target network
        self.optimize_model()

    def on_step(
        self,
        wrapped_env: w.AgentWrapper,
        observation,
        reward: float,
        done: bool,
        truncated: bool,
        info,
        action_metadata,
    ):
        agent_state = wrapped_env.state
        if done:
            self.update_q_function(
                reward,
                actor_state=action_metadata.actor_state,
                abstract_action=action_metadata.abstract_action,
                next_actor_state=None,
            )
        else:
            # DRQN: Update inference hidden state by passing the current step
            with torch.no_grad():
                # Format state: (1, 1, input_size)
                state_tensor = torch.as_tensor(action_metadata.actor_state, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                _, self.inference_hidden = self.policy_net(state_tensor, self.inference_hidden)

            next_global_state = self.stateaction_model.global_features.get(agent_state, node=None)
            next_actor_features = self.stateaction_model.node_specific_features.get(agent_state, action_metadata.actor_node)
            next_actor_state = self.get_actor_state_vector(next_global_state, next_actor_features)

            self.update_q_function(
                reward,
                actor_state=action_metadata.actor_state,
                abstract_action=action_metadata.abstract_action,
                next_actor_state=next_actor_state,
            )

    def end_of_episode(self, i_episode, t):
        # Update the target network, copying all weights and biases in DQN
        if i_episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def lookup_dqn(self, states_to_consider: List[ndarray]) -> Tuple[List[np.int32], List[np.int32]]:
        """Given a set of possible current states return:
        - index, in the provided list, of the state that would yield the best possible outcome
        - the best action to take in such a state"""
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # action: np.int32 = self.policy_net(states_to_consider).max(1)[1].view(1, 1).item()

            state_batch = torch.tensor(states_to_consider).to(device)
            
            # DRQN expects (batch, seq_len, input_size)
            # We have (batch, input_size), so unsqueeze to (batch, 1, input_size)
            state_batch = state_batch.unsqueeze(1)
            
            # Handle hidden state
            if self.inference_hidden is None:
                # Initialize hidden state if it doesn't exist yet
                # hidden is (h0, c0), each is (1, batch, hidden_size) for batch_first=True
                # But here we have a batch of candidate states, which all share the SAME history.
                # So we want to broadcast the current single hidden state to this batch size.
                batch_size = state_batch.size(0)
                # Let the model initialize zeros for batch 1, then expand? 
                # Easier to manually create zeros matching the batch size
                h0 = torch.zeros(1, batch_size, 128, device=device)
                c0 = torch.zeros(1, batch_size, 128, device=device)
                hidden = (h0, c0)
            else:
                # self.inference_hidden is (1, 1, 128) - one batch, one history
                # We need to expand it to (1, batch_size, 128)
                batch_size = state_batch.size(0)
                h, c = self.inference_hidden
                h = h.expand(-1, batch_size, -1).contiguous()
                c = c.expand(-1, batch_size, -1).contiguous()
                hidden = (h, c)

            dnn_output, _ = self.policy_net(state_batch, hidden)
            # Remove sequence dimension: (batch, 1, output) -> (batch, output)
            dnn_output = dnn_output.squeeze(1)
            
            dnn_output = dnn_output.max(1)
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
        )

    def explore(self, wrapped_env: w.AgentWrapper) -> Tuple[str, cyberbattle_env.Action, object]:
        """Random exploration that avoids repeating actions previously taken in the same state"""
        # sample local and remote actions only (excludes connect action)
        gym_action = wrapped_env.env.sample_valid_action(kinds=[0, 1, 2])
        metadata = self.metadata_from_gymaction(wrapped_env, gym_action)
        return "explore", gym_action, metadata

    def try_exploit_at_candidate_actor_states(self, wrapped_env, current_global_state, actor_features, abstract_action):
        actor_state = self.get_actor_state_vector(current_global_state, actor_features)

        action_style, gym_action, actor_node = self.stateaction_model.implement_action(wrapped_env, actor_features, abstract_action)

        if gym_action:
            assert actor_node is not None, "actor_node should be set together with gym_action"

            return (
                action_style,
                gym_action,
                ChosenActionMetadata(
                    abstract_action=abstract_action,
                    actor_node=actor_node,
                    actor_features=actor_features,
                    actor_state=actor_state,
                ),
            )
        else:
            # learn the failed exploit attempt in the current state
            self.update_q_function(
                reward=0.0,
                actor_state=actor_state,
                next_actor_state=actor_state,
                abstract_action=abstract_action,
            )

            return "exploit[undefined]->explore", None, None

    def exploit(self, wrapped_env, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        # first, attempt to exploit the credential cache
        # using the crecache_policy
        # action_style, gym_action, _ = self.credcache_policy.exploit(wrapped_env, observation)
        # if gym_action:
        #     return action_style, gym_action, self.metadata_from_gymaction(wrapped_env, gym_action)

        # Otherwise on exploit learnt Q-function

        current_global_state = self.stateaction_model.global_features.get(wrapped_env.state, node=None)

        # Gather the features of all the current active actors (i.e. owned nodes)
        active_actors_features: List[ndarray] = [self.stateaction_model.node_specific_features.get(wrapped_env.state, from_node) for from_node in w.owned_nodes(observation)]

        unique_active_actors_features: List[ndarray] = list(np.unique(active_actors_features, axis=0))

        # array of actor state vector for every possible set of node features
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
