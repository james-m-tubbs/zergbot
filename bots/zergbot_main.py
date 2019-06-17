# based on tutorial: https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e
# github: https://github.com/skjb/pysc2-tutorial

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

import numpy as np
import pandas as pd

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_OVERLORD = actions.FUNCTIONS.Train_Overlord_quick.id
_BUILD_SPAWNING_POOL = actions.FUNCTIONS.Build_SpawningPool_screen.id
_TRAIN_ZERGLING = actions.FUNCTIONS.Train_Zergling_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_BLD_HATCHERY = 86
_BLD_LAIR = 100
_BLD_HIVE = 101
_BLD_EXTRACTOR = 88
_UNT_DRONE = 104
_UNT_OVERLORD = 106
_UNT_LARVA = 151

_BLD_SPAWNING_POOL = 89
_UNT_QUEEN = 126
_UNT_ZERGLING = 105

_BLD_ROACH_WARRENS = 97
_UNT_ROACH = 110

_BLD_HYDRALISK_DEN = 91
_UNT_HYDRALISK = 107

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

basic_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK
]

ACTION_SELECT_HATCHERY = 'selecthatchery'
ACTION_SELECT_LARVA = 'selectlarva'
ACTION_TRAIN_QUEEN = 'selectlarva'
ACTION_TRAIN_DRONE = 'traindrone'
ACTION_SELECT_DRONE = 'selectdrone'
ACTION_TRAIN_OVERLORD = 'trainoverlord'
ACTION_BUILD_EXTRACTOR = 'buildextractor'

hatchery_actions = [
    ACTION_SELECT_HATCHERY,
    ACTION_SELECT_LARVA,
    ACTION_TRAIN_QUEEN,
    ACTION_TRAIN_DRONE,
    ACTION_SELECT_DRONE,
    ACTION_TRAIN_OVERLORD,
    ACTION_BUILD_EXTRACTOR
]

ACTION_BUILD_SPAWNING_POOL = 'buildspawningpool'
ACTION_TRAIN_ZERGLING = 'trainzergling'

zergling_actions = [
    ACTION_BUILD_SPAWNING_POOL,
    ACTION_TRAIN_ZERGLING
]

smart_actions = basic_actions + hatchery_actions + zergling_actions

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    self.actions = actions  # a list
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon = e_greedy
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

  def choose_action(self, observation):
    self.check_state_exist(observation)

    if np.random.uniform() < self.epsilon:
      # choose best action
      state_action = self.q_table.ix[observation, :]

      # some actions have the same value
      state_action = state_action.reindex(np.random.permutation(state_action.index))

      action = state_action.idxmax()
    else:
      # choose random action
      action = np.random.choice(self.actions)

    return action

  def learn(self, s, a, r, s_):
    self.check_state_exist(s_)
    self.check_state_exist(s)

    q_predict = self.q_table.ix[s, a]
    q_target = r + self.gamma * self.q_table.ix[s_, :].max()

    # update
    self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

  def check_state_exist(self, state):
    if state not in self.q_table.index:
      # append new state to q table
      self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

class ZergAgent(base_agent.BaseAgent):
  def __init__(self):
    super(ZergAgent, self).__init__()

    self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

    self.previous_killed_unit_score = 0
    self.previous_killed_building_score = 0

    self.previous_action = None
    self.previous_state = None
    
    self.attack_coordinates = None

  def unit_type_is_selected(self, obs, unit_type):
    if (len(obs.observation.single_select) > 0 and
        obs.observation.single_select[0].unit_type == unit_type):
      return True
    
    if (len(obs.observation.multi_select) > 0 and
        obs.observation.multi_select[0].unit_type == unit_type):
      return True
    
    return False

  def get_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
  
  def can_do(self, obs, action):
    return action in obs.observation.available_actions

  def step(self, obs):
    super(ZergAgent, self).step(obs)

    #find the enemy base & set attack coordinates
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                            features.PlayerRelative.SELF).nonzero()
      xmean = player_x.mean()
      ymean = player_y.mean()

      if xmean <= 31 and ymean <= 31:
        self.attack_coordinates = (49, 49)
      else:
        self.attack_coordinates = (12, 16)

    ###############################
    ### Capture Game State ###
    ###############################

    # Get the current game "score" #
    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    ##### capture current game state #####
    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]

    # Building Counts #
    spawning_pool_count = len(self.get_units_by_type(obs, units.Zerg.SpawningPool))

    # Unit Counts
    overlord_count = len(self.get_units_by_type(obs, units.Zerg.Overlord))
    queen_count = len(self.get_units_by_type(obs, units.Zerg.Queen))
    zergling_count = len(self.get_units_by_type(obs, units.Zerg.Zergling))
    larva_count = len(self.get_units_by_type(obs, units.Zerg.Larva))

    # build the state object
    current_state = [
      zergling_count,
      queen_count,
      overlord_count,
      larva_count,
      spawning_pool_count,
      supply_limit,
      army_supply,
    ]

    if self.previous_action is not None:
      reward = 0

      if killed_unit_score > self.previous_killed_unit_score:
        reward += KILL_UNIT_REWARD

      if killed_building_score > self.previous_killed_building_score:
        reward += KILL_BUILDING_REWARD

      self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

    rl_action = self.qlearn.choose_action(str(current_state))
    smart_action = smart_actions[rl_action]

    self.previous_killed_unit_score = killed_unit_score
    self.previous_killed_building_score = killed_building_score
    self.previous_state = current_state
    self.previous_action = rl_action

    print(smart_action)

    ###############################
    ### Basic Actions ###
    ###############################
    if smart_action == ACTION_DO_NOTHING:
        return actions.FunctionCall(_NO_OP, [])

    ###############################
    ### Hatchery Actions ###
    ###############################

    elif smart_action == ACTION_SELECT_LARVA:
      larvae = self.get_units_by_type(obs, units.Zerg.Larva)
      if len(larvae) > 0:
        larva = random.choice(larvae)
        return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                  larva.y))
    elif smart_action == ACTION_SELECT_DRONE:
        drones = self.get_units_by_type(obs, units.Zerg.Drone)
        if len(drones) > 0:
            drone = random.choice(drones)
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
                                                                      drone.y))
    elif smart_action == ACTION_TRAIN_OVERLORD:
      free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
      if free_supply == 0:
        if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
          return actions.FUNCTIONS.Train_Overlord_quick("now")

    elif smart_action == ACTION_TRAIN_DRONE:
        drones = self.get_units_by_type(obs, units.Zerg.Drone)
        if len(drones) < 16:
          if self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id):
            return actions.FUNCTIONS.Train_Drone_quick("now")

    elif smart_action == ACTION_SELECT_HATCHERY:
        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
        hatchery = random.choice(hatcheries)
        return actions.FUNCTIONS.select_point("select_all_type", (hatchery.x, hatchery.y))

    elif smart_action == ACTION_TRAIN_QUEEN:
        if self.can_do(obs, actions.FUNCTIONS.Train_Queen_quick.id):
            return actions.FUNCTIONS.Train_Queen_quick("now")

    #######################
    ### Zerging Actions ###
    #######################

    elif smart_action == ACTION_BUILD_SPAWNING_POOL:
      if len(self.get_units_by_type(obs, units.Zerg.SpawningPool)) < 1:
        if self.unit_type_is_selected(obs, units.Zerg.Drone):
            if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                x = random.randint(0, 83)
                y = random.randint(0, 83)

                return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

    elif smart_action == ACTION_TRAIN_ZERGLING:
        if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
            return actions.FUNCTIONS.Train_Zergling_quick("now")

    return actions.FunctionCall(_NO_OP, [])

def main(unused_argv):
  agent = ZergAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.zerg),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64),
              use_feature_units=True),
          step_mul=16,
          game_steps_per_episode=0,
          visualize=True) as env:
          
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
  except KeyboardInterrupt:
    pass
  
if __name__ == "__main__":
  app.run(main)
