# based on tutorial: https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e
# github: https://github.com/skjb/pysc2-tutorial

import random
import math

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from absl import app

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
  ACTION_DO_NOTHING,
  ACTION_SELECT_SCV,
  ACTION_BUILD_SUPPLY_DEPOT,
  ACTION_BUILD_BARRACKS,
  ACTION_SELECT_BARRACKS,
  ACTION_BUILD_MARINE,
  ACTION_SELECT_ARMY,
  ACTION_ATTACK,
]

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
  first_atack = False

  def __init__(self):
    super(ZergAgent, self).__init__()
    
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

  def build_structure(self, build_function):
      x = random.randint(0, 83)
      y = random.randint(0, 83)
      return build_function("now", (x, y))

  def step(self, obs):
    super(ZergAgent, self).step(obs)
    
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                            features.PlayerRelative.SELF).nonzero()
      xmean = player_x.mean()
      ymean = player_y.mean()
      
      if xmean <= 31 and ymean <= 31:
        self.attack_coordinates = (49, 49)
      else:
        self.attack_coordinates = (12, 16)

      self.first_atack = False

    queens = self.get_units_by_type(obs, units.Zerg.Queen)
    if len(queens) > 0:
      queen = random.choice(queens)
      return actions.FUNCTIONS.select_point("select_all_type", (queen.x, queen.y))

    zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
    if len(zerglings) >= 20:
      if self.unit_type_is_selected(obs, units.Zerg.Zergling):
        if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
          self.first_atack = True
          return actions.FUNCTIONS.Attack_minimap("now",
                                                  self.attack_coordinates)

      if self.can_do(obs, actions.FUNCTIONS.select_army.id):
        return actions.FUNCTIONS.select_army("select")

    spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
    # build spawning pool branch
    if len(spawning_pools) == 0:
      if self.unit_type_is_selected(obs, units.Zerg.Drone):
          if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
            return self.build_structure(actions.FUNCTIONS.Build_SpawningPool_screen)
      drones = self.get_units_by_type(obs, units.Zerg.Drone)
      if len(drones) > 0:
        drone = random.choice(drones)
        return actions.FUNCTIONS.select_point("select_all_type", (drone.x,drone.y))

    evolution_chambers = self.get_units_by_type(obs, units.Zerg.EvolutionChamber)
    if len(evolution_chambers) == 0:
        if self.first_atack:
          if self.unit_type_is_selected(obs, units.Zerg.Drone):
            if self.can_do(obs, actions.FUNCTIONS.Build_EvolutionChamber_screen.id):
                return self.build_structure(actions.FUNCTIONS.Build_EvolutionChamber_screen)
          drones = self.get_units_by_type(obs, units.Zerg.Drone)
          if len(drones) > 0:
            drone = random.choice(drones)

            return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
                                                                      drone.y))

    if self.unit_type_is_selected(obs, units.Zerg.Larva):
      free_supply = (obs.observation.player.food_cap -
                     obs.observation.player.food_used)
      if free_supply == 0:
        if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
          return actions.FUNCTIONS.Train_Overlord_quick("now")

      if self.can_do(obs, actions.FUNCTIONS.Train_Roach_quick.id):
          return actions.FUNCTIONS.Train_Roach_quick("now")

      if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
        return actions.FUNCTIONS.Train_Zergling_quick("now")
    
    larvae = self.get_units_by_type(obs, units.Zerg.Larva)
    if len(larvae) > 0:
      larva = random.choice(larvae)
      return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))

    # i can't figure out why this doesn't work
    hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)

    if len(queens) < len(hatcheries):
        if self.can_do(obs, actions.FUNCTIONS.Train_Queen_quick):
          return actions.FUNCTIONS.Train_Queen_quick.id("now")

    if len(hatcheries) > 0:
        hatchery = random.choice(hatcheries)
        return actions.FUNCTIONS.select_point("select_all_type", (hatchery.x, hatchery.y))




    #default no-op
    return actions.FUNCTIONS.no_op()

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
