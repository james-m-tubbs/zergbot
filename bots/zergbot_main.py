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
ACTION_ATTACK_ZERGLINGS = 'attack_zerglings'
ACTION_ATTACK_DRONES = 'attack_drones'
ACTION_ATTACK_QUEENS = 'attack_queens'

basic_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK_ZERGLINGS,
    ACTION_ATTACK_DRONES,
    ACTION_ATTACK_QUEENS
]

ACTION_SELECT_HATCHERY = 'selecthatchery'
ACTION_SELECT_LARVA = 'selectlarva'
ACTION_TRAIN_QUEEN = 'trainqueen'
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
ACTION_SELECT_ZERGLING = 'selectzergling'
ACTION_SELECT_SPAWNING_POOL = 'selectspawningpool'
ACTION_RESEARCH_ADRENAL_GLANDS = 'researchadrenalglands'
ACTION_RESEARCH_METABOLIC_BOOST = 'researchmetabolicboost'

zergling_actions = [
    ACTION_BUILD_SPAWNING_POOL,
    ACTION_TRAIN_ZERGLING,
    ACTION_SELECT_SPAWNING_POOL,
    ACTION_RESEARCH_ADRENAL_GLANDS,
    ACTION_RESEARCH_METABOLIC_BOOST
]

smart_actions = basic_actions + hatchery_actions + zergling_actions

#####################
### Reward Values ###
#####################

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.6

TOTAL_VALUE_UNITS_REWARD = 0.5
TOTAL_VALUE_STRUCTURES_REWARD = 0.5

MINERAL_COLLECTION_REWARD = 0.7
VESPENE_COLLECTION_REWARD = 0.7
MINERAL_SPENT_REWARD = 0.4
VESPENE_SPENT_REWARD = 0.4
DRONE_CREATE_REWARD = 0.3

ZERGLING_CREATE_REWARD = 1

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
  def __init__(self, actions, learning_rate=0.1, reward_decay=0.8, e_greedy=0.9):
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

      print(xmean)
      print(ymean)

      if xmean <= 40 and ymean <= 40:
        self.attack_coordinates = (200, 200)
      else:
        self.attack_coordinates = (12, 16)

    ###############################
    ### Capture Game State ###
    ###############################

    # Get the current game "score" #
    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    total_value_units = obs.observation['score_cumulative'][3]
    total_value_structures = obs.observation['score_cumulative'][4]

    mineral_collection_rate = obs.observation['score_cumulative'][9]
    vespene_collection_rate = obs.observation['score_cumulative'][10]
    minerals_spent = obs.observation['score_cumulative'][11]
    vespene_spent = obs.observation['score_cumulative'][12]

    ##### capture current game state #####
    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]

    # Building Counts #
    spawning_pool_count = len(self.get_units_by_type(obs, units.Zerg.SpawningPool))

    # Unit Counts
    overlord_count = len(self.get_units_by_type(obs, units.Zerg.Overlord))
    queen_count = len(self.get_units_by_type(obs, units.Zerg.Queen))
    zergling_count = len(self.get_units_by_type(obs, units.Zerg.Zergling))
    drone_count = len(self.get_units_by_type(obs, units.Zerg.Drone))
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
      drone_count,
      mineral_collection_rate,
      minerals_spent,
      vespene_collection_rate,
      vespene_spent
    ]

    if self.previous_action is not None:
      reward = 0

      if killed_unit_score > self.previous_killed_unit_score:
        reward += KILL_UNIT_REWARD

      if killed_building_score > self.previous_killed_building_score:
        reward += KILL_BUILDING_REWARD

      if mineral_collection_rate > self.prevoius_mineral_collection_rate:
          reward += MINERAL_COLLECTION_REWARD

      if mineral_collection_rate < self.prevoius_mineral_collection_rate:
          reward -= MINERAL_COLLECTION_REWARD

      if vespene_collection_rate > self.prevoius_vespene_collection_rate:
          reward += VESPENE_COLLECTION_REWARD

      if vespene_collection_rate > self.prevoius_vespene_collection_rate:
          reward -= VESPENE_COLLECTION_REWARD

      if minerals_spent > self.prevoius_minerals_spent:
          reward += MINERAL_SPENT_REWARD

      if vespene_spent > self.prevoius_vespene_spent:
        reward += VESPENE_SPENT_REWARD

      if total_value_units > self.previous_total_value_units:
        reward += TOTAL_VALUE_UNITS_REWARD

      if total_value_structures > self.previous_total_value_structures:
          reward += TOTAL_VALUE_STRUCTURES_REWARD

      if drone_count != self.previous_drone_count:
        reward += DRONE_CREATE_REWARD * (drone_count - self.previous_drone_count)

      if zergling_count != self.previous_zergling_count:
          reward += ZERGLING_CREATE_REWARD * (zergling_count - self.previous_zergling_count)

      self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

    rl_action = self.qlearn.choose_action(str(current_state))
    smart_action = smart_actions[rl_action]

    self.previous_killed_unit_score = killed_unit_score
    self.previous_killed_building_score = killed_building_score

    self.previous_total_value_units = total_value_units
    self.previous_total_value_structures = total_value_structures

    self.prevoius_mineral_collection_rate = mineral_collection_rate
    self.prevoius_vespene_collection_rate = vespene_collection_rate

    self.prevoius_minerals_spent = minerals_spent
    self.prevoius_vespene_spent = vespene_spent

    self.previous_drone_count = drone_count

    self.previous_zergling_count = zergling_count

    self.previous_state = current_state
    self.previous_action = rl_action

    print(smart_action)

    ###############################
    ### Basic Actions ###
    ###############################
    if smart_action == ACTION_DO_NOTHING:
        return actions.FunctionCall(_NO_OP, [])

    elif smart_action == ACTION_SELECT_ARMY:
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) > 1:
            zergling = random.choice(zerglings)
            return actions.FUNCTIONS.select_point("select_all_type", (zergling.x,
                                                                      zergling.y))

    elif smart_action == ACTION_ATTACK_ZERGLINGS:
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if self.unit_type_is_selected(obs, units.Zerg.Zergling):
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now",
                                                        self.attack_coordinates)

    elif smart_action == ACTION_ATTACK_DRONES:
        drones = self.get_units_by_type(obs, units.Zerg.Drone)
        if len(drones) >= 10:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",
                                                            self.attack_coordinates)
    elif smart_action == ACTION_ATTACK_QUEENS:
        queens = self.get_units_by_type(obs, units.Zerg.Queen)
        if len(queens) >= 10:
            if self.unit_type_is_selected(obs, units.Zerg.Queen):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",
                                                            self.attack_coordinates)



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
        else:
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
                larva = random.choice(larvae)
                return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                          larva.y))

    elif smart_action == ACTION_TRAIN_DRONE:
        drones = self.get_units_by_type(obs, units.Zerg.Drone)
        if len(drones) < 16:
          if self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id):
            return actions.FUNCTIONS.Train_Drone_quick("now")
          else:
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
                larva = random.choice(larvae)
                return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                          larva.y))

    elif smart_action == ACTION_SELECT_HATCHERY:
      hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
      if len(hatcheries) > 0:
        hatchery = random.choice(hatcheries)
        return actions.FUNCTIONS.select_point("select_all_type", (hatchery.x,
                                                                  hatchery.y))

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
        else:
          drones = self.get_units_by_type(obs, units.Zerg.Drone)
          if len(drones) > 0:
            drone = random.choice(drones)
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x,
                                                                      drone.y))
      else:
          larvae = self.get_units_by_type(obs, units.Zerg.Larva)
          if len(larvae) > 0:
              larva = random.choice(larvae)
              return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                        larva.y))



    elif smart_action == ACTION_TRAIN_QUEEN:
        if len(self.get_units_by_type(obs, units.Zerg.Queen)) < 2*(len(self.get_units_by_type(obs, units.Zerg.Hatchery))):
          if self.can_do(obs, actions.FUNCTIONS.Train_Queen_quick.id):
            return actions.FUNCTIONS.Train_Queen_quick("now")
          else:
            hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatcheries) > 0:
                hatchery = random.choice(hatcheries)
                return actions.FUNCTIONS.select_point("select_all_type", (hatchery.x,
                                                                          hatchery.y))

    elif smart_action == ACTION_TRAIN_ZERGLING:
        if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
            return actions.FUNCTIONS.Train_Zergling_quick("now")
        else:
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
                larva = random.choice(larvae)
                return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                          larva.y))

    elif smart_action == ACTION_SELECT_SPAWNING_POOL:
        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        if len(spawning_pools) > 1:
            spawning_pool = random.choice(spawning_pools)
            return actions.FUNCTIONS.select_point("select_all_type", (spawning_pool.x,
                                                                      spawning_pool.y))

    elif smart_action == ACTION_RESEARCH_ADRENAL_GLANDS:
      if self.unit_type_is_selected(obs, units.Zerg.Drone):
        if self.can_do(obs, actions.FUNCTIONS.Research_ZerglingAdrenalGlands_quick.id):
          return actions.FUNCTIONS.Research_ZerglingAdrenalGlands_quick("now")

    elif smart_action == ACTION_RESEARCH_METABOLIC_BOOST:
        if self.unit_type_is_selected(obs, units.Zerg.Drone):
            if self.can_do(obs, actions.FUNCTIONS.Research_ZerglingMetabolicBoost_quick.id):
                return actions.FUNCTIONS.Research_ZerglingMetabolicBoost_quick("now")

    #######################
    ### Default NO_OP   ###
    #######################

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
              feature_dimensions=features.Dimensions(screen=168, minimap=128),
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
