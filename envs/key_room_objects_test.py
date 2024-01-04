from __future__ import annotations
from enum import Enum
from absl import logging

import copy
from gymnasium import spaces
from typing import Optional, Union, List, Dict, Tuple
import itertools

try:
  import wandb
except:
   pass
import dm_env
import matplotlib.pyplot as plt
import numpy as np

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.core.world_object import Ball, Box, Key, Floor

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel, RejectSampling
from minigrid.envs.babyai.core.verifier import Instr

try:
  from library.utils import LevelAvgObserver
except:
   LevelAvgObserver = object

Number = Union[int, float, np.float32]

class DummyInstr(Instr):
    """
    Dummy Instruction Class.
    """
    def verify(self, action): del action
    def reset_verifier(self, env): del env
    def surface(self, env): return ''
    def update_objs_poss(self): pass
    def verify_action(self): pass

def construct(shape: str, color: str):
  if shape == 'ball':
      return Ball(color)
  elif shape == 'box':
    return Box(color)
  elif shape == 'key':
    return Key(color)
  elif shape == 'floor':
    return Floor(color)
  else:
    raise NotImplementedError(f"shape {shape} not supported")

def name(o):
   return f"{o.color} {o.type}"

class ObjectTestTaskFeatures:
  def __init__(self, colors, types):
    self.thing2indx = {}
    for idx, (color, shape) in enumerate(itertools.product(colors, types)):
        self.thing2indx[(color, shape)] = idx


  @property
  def things(self):
     return self.thing2indx.keys()

  def task(self, obj2reward: dict):
    vector = np.zeros(len(self.thing2indx))
    for obj, reward in obj2reward.items():
      vector[self.thing2indx[obj]] = reward

    return vector

  def empty_state_features(self):
    vector = np.zeros(len(self.thing2indx))
    return vector

  def index(self, color: str, type: str):
    key = (color, type)
    if key in self.thing2indx:
      return self.thing2indx[key]
    else:
      return -1

  def state_features(self, color: str, type: str):
    vector = np.zeros(len(self.thing2indx))
    vector[self.thing2indx[(color, type)]] = 1.0
    return vector

class ObjectTestTask:
    def __init__(self,
                 floor,
                 init,
                 w,
                 source: str = 'color',
                 floor2task_color: dict = None,
                 type2indx: dict = None,
                 task_reward: float = 1.0,
                 ):
        self.floor = floor
        self.init = init
        self.w = w
        self.task_reward = task_reward

        assert source in ['shape', 'color']
        self.source = source
        self.types = ['key', 'room', 'box', 'ball']

        type2indx = type2indx or {
           'key': 0,
           'room': 1,
           'ball': 2,
           'box': 3,
        }
        floor2task_color = floor2task_color or {
            'blue': 'red',
            'yellow': 'green',
        }

        colors = floor2task_color.values()

        self.feature_cnstr = ObjectTestTaskFeatures(colors, self.types)

        self.type2indx = type2indx
        self.task_colors = list(floor2task_color.values())
        self.floor2task_color = floor2task_color
        self.shape2task_color = {
            'ball': 'red',
            'box': 'green',
        }

    def task_vector(self):
       vector = self.feature_cnstr.task(obj2reward={
        (self.goal_color(), 'key'): .1,
        (self.goal_color(), 'room'): .5,
        self.goal(): self.task_reward
       })
       return vector

    def initial_object(self):
      return construct(shape=self.init, color='grey')

    def initial_floor(self):
       return construct(shape='floor', color=self.floor)

    def color_goal(self):
        return (self.floor2task_color[self.floor], self.w)

    def type_goal(self):
        return (self.shape2task_color[self.init], self.w)

    def goal(self):
       if self.source == 'shape':
         return self.type_goal()
       elif self.source == 'color':
         return self.color_goal()
       else:
         raise NotImplementedError(f"source {self.source} not supported")

    def goal_name(self):
      color, type = self.goal()
      return f"{color} {type}"

    def goal_color(self):
       return self.goal()[0]

    def goal_object(self):
       return self.goal()[1]

    def desc(self):
      def name(c, t): return f"{c} {t}"
      if self.source == 'color':
        n = name(*self.color_goal())
        return f"floor={self.floor}, init={self.init}, w={self.w} |-> {n}"
      elif self.source == 'shape':
        n = name(*self.type_goal())
        return f"init={self.init}, floor={self.floor}, w={self.w} |-> {n}"

    def reset(self,
        color_to_room: Dict[str, Tuple[int]],
        start_room_color: str):

       self.color_to_room = color_to_room
       self.room_to_color = {room: color for color, room in color_to_room.items()}

       self.start_room_color = start_room_color
       self.feature_counts = self.feature_cnstr.empty_state_features()
       self.prior_counts = self.feature_cnstr.empty_state_features()
       self.timestep = 0


    def step(self, env):
      """
      Things we'll check:
      1. did the agent pick up an object?
        - held prior vs. held now
        - at end of each time-step, store self.prior_carrying = env.carrying
        - at beginning of next time-step, compare: (self.prior_carrying, env.carrying)
          - if env.carrying and different:
            increase pick-up count for object type
      2. did the agent enter a room
        - room_prior vs. room_now
        - at end of each time-step, store self.prior_room = current_room
        - at beginning, if diff: increase count
      
      3. state-features = (new instance picking up/entering room)*(1st time doing this).
        - essentially a "first-occupancy" variant of state-features

      """
      self.timestep += 1
      #=======================
      # Did the agent pick up a object?
      #=======================
      if env.carrying and (env.carrying != self.prior_carrying):
        idx = self.feature_cnstr.index(
          env.carrying.color, env.carrying.type)
        if idx >=0:
          self.feature_counts[idx] += 1
        # print(f"{self.timestep}: picked up {env.carrying.color} {env.carrying.type}")

      #=======================
      # Did the agent enter a new room?
      #=======================
      agent_room = env.room_from_pos(*env.agent_pos)
      room_color = self.room_to_color[agent_room]
      if room_color != self.start_room_color and (
        self.prior_room_color != room_color):
        idx = self.feature_cnstr.index(
          room_color, "room")
        if idx >=0:
          self.feature_counts[idx] += 1
        # print(f"{self.timestep}: entered {room_color} room")

      #=======================
      # Did the agent enter a new room
      #=======================
      first = (self.feature_counts == 1).astype(np.float32)
      difference = self.feature_counts - self.prior_counts
      state_features = first*difference

      #=======================
      # update
      #=======================
      self.prior_carrying = env.carrying
      self.prior_room_color = room_color
      self.prior_counts = np.array(self.feature_counts)

      #=======================
      # is the task over?
      #=======================
      terminal = False
      color, type = self.goal()
      if (env.carrying is not None and 
          env.carrying.type == type and 
          env.carrying.color == color):
         terminal = True

      return state_features, terminal

class KeyRoomObjectTest(LevelGen):
    """
    KeyRoomObjectTest is a class for generating grid-based environments with tasks involving keys, doors, and objects
    of different colors.

    Args:
        tasks (List[ObjectTestTask]): A list of tasks to be performed in the environment.
        room_size (int, optional): The size of each room in the grid. Defaults to 7.
        num_rows (int, optional): The number of rows of rooms in the grid. Defaults to 3.
        num_cols (int, optional): The number of columns of rooms in the grid. Defaults to 3.
        num_dists (int, optional): The number of distractor objects. Defaults to 0.
        locations (bool, optional): Whether to randomize object locations. Defaults to False.
        unblocking (bool, optional): Whether unblocking is required. Defaults to False.
        rooms_locked (bool, optional): Whether rooms are initially locked. Defaults to True.
        include_task_signals (bool, optional): Whether to include task-specific signals. Defaults to True.
        max_steps_per_room (int, optional): The maximum number of steps per room. Defaults to 100.
        implicit_unlock (bool, optional): Whether implicit unlocking is allowed. Defaults to True.
        room_colors (List[str], optional): The colors of the rooms in the grid. Defaults to ['blue', 'yellow', 'red', 'green'].

    Attributes:
        tasks (List[ObjectTestTask]): A list of tasks to be performed in the environment.
        rooms_locked (bool): Whether rooms are initially locked.
        room_colors (List[str]): The colors of the rooms in the grid.
        include_task_signals (bool): Whether to include task-specific signals.
    """
    def __init__(self,
      tasks: List[ObjectTestTask],
      room_size=7,
      num_rows=3,
      num_cols=3,
      num_dists=0,
      locations=False,
      unblocking=False,
      rooms_locked=True,
      include_task_signals=True,
      max_steps_per_room: int = 100,
      implicit_unlock=True,
      room_colors: List[str] = ['blue', 'yellow', 'red', 'green'],
      **kwargs):
      """
      Initialize a KeyRoomObjectTest environment with customizable parameters.

      NOTE: this environment assumes that the objects present in a task are present across ALL tasks. This is used for keeping maintaining track of which objects are picked up.

      Args:
          tasks (List[ObjectTestTask]): A list of tasks to be performed in the environment.
          room_size (int, optional): The size of each room in the grid. Defaults to 7.
          num_rows (int, optional): The number of rows of rooms in the grid. Defaults to 3.
          num_cols (int, optional): The number of columns of rooms in the grid. Defaults to 3.
          num_dists (int, optional): The number of distractor objects. Defaults to 0.
          locations (bool, optional): Whether to randomize object locations. Defaults to False.
          unblocking (bool, optional): Whether unblocking is required. Defaults to False.
          rooms_locked (bool, optional): Whether rooms are initially locked. Defaults to True.
          include_task_signals (bool, optional): Whether to include task-specific signals. Defaults to True.
          max_steps_per_room (int, optional): The maximum number of steps per room. Defaults to 100.
          implicit_unlock (bool, optional): Whether implicit unlocking is allowed. Defaults to True.
          room_colors (List[str], optional): The colors of the rooms in the grid. Defaults to ['blue', 'yellow', 'red', 'green'].
          **kwargs: Additional keyword arguments for customization.

      Attributes:
          tasks (List[ObjectTestTask]): A list of tasks to be performed in the environment.
          rooms_locked (bool): Whether rooms are initially locked.
          room_colors (List[str]): The colors of the rooms in the grid.
          include_task_signals (bool): Whether to include task-specific signals (i.e. room color, indicator object).
      """

      self.tasks = tasks
      self.rooms_locked = rooms_locked
      self.room_colors = room_colors
      self.include_task_signals = include_task_signals
      
      dummy_task_vector = self.tasks[0].task_vector()
      self.train_task_vectors = np.array([t.task_vector() for t in tasks])

      super().__init__(
          room_size=room_size,
          num_rows=num_rows,
          num_cols=num_cols,
          num_dists=num_dists,
          locations=locations, # randomize locations?
          unblocking=unblocking,  # require need to unblock
          implicit_unlock=implicit_unlock,
          **kwargs,
      )
      self._max_steps = max_steps_per_room*self.num_navs_needed()
      cumulants_space = spaces.Box(
          low=0,
          high=100.0,
          shape=dummy_task_vector.shape,  # number of cells
          dtype="float32",
      )
      train_task_vectors = spaces.Box(
          low=0,
          high=100.0,
          shape=self.train_task_vectors.shape,  # number of cells
          dtype="float32",
      )
      self.observation_space = spaces.Dict(
          {**self.observation_space.spaces,
           "state_features": cumulants_space,
           "task": copy.deepcopy(cumulants_space),  # equivalent specs
           "train_tasks": train_task_vectors,
           }
      )

    def gen_mission(self):
      """_summary_

      
      # Returns:
      #     _type_: _description_
      # """

      ###########################
      # Place agent in center room
      ###########################
      center_room = (self.num_rows//2, self.num_cols//2)
      self.place_agent(*center_room)
      _ = self.room_from_pos(*self.agent_pos)

      ###########################################
      # Place keys, door, and ball of same color
      ###########################################
      self.all_objects = []
      # colors = ['red', 'green', 'yellow', 'grey']
      colors = self.room_colors
      # starts to the right for some reason
      goal_room_coords = [(2,1), (1, 2), (0, 1), (1,0)]
      colors__to__room = {}
      # use 'start' key for start room coords
      start_room_color = 'start'
      colors__to__room[start_room_color] = self.get_room(*center_room)

      def generate_room_objects(color: str):
        return (Key(color), Ball(color), Box(color))

      for room_idx, color in enumerate(colors):
        room = goal_room_coords[room_idx]
        colors__to__room[color] = self.get_room(*room)

        key, ball, box = generate_room_objects(color)
        self.all_objects.extend([key, ball, box])
        self.place_in_room(*center_room, key)
        self.place_in_room(*room, ball)
        self.place_in_room(*room, box)
        _, _ = self.add_door(
            *center_room,
            door_idx=room_idx,
            color=color,
            locked=self.rooms_locked)

      task_idx = np.random.randint(len(self.tasks))
      self.task = self.tasks[task_idx]
      task_room_coord = goal_room_coords[task_idx]

      self.task.reset(
        color_to_room=colors__to__room,
        start_room_color=start_room_color,
        )

      ###########################################
      # Place extra objects in main room
      ###########################################
      if self.include_task_signals:
        init_object = self.task.initial_object()
        self.place_in_room(*center_room, init_object)
        self.all_objects.extend([init_object])

        init_floor = self.task.initial_floor()
        self.place_in_room(*center_room, init_floor)
        self.place_in_room(*center_room, init_floor)
        self.place_in_room(*center_room, init_floor)

      self.instrs = DummyInstr()
      self.task_vector = self.task.task_vector()

    def update_obs(self, obs, state_features):
      obs['task'] = self.task_vector
      obs['state_features'] = state_features
      obs['train_tasks'] = self.train_task_vectors


    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.update_obs(obs,
                        state_features=np.zeros_like(self.task_vector))

        # reset object counts
        self.object_counts = {name(o):0 for o in self.all_objects}
        self.carrying_prior = None
        return obs, info

    def step(self, action, **kwargs):
      obs, _, _, _, info = super().step(action, **kwargs)
      state_features, terminated = self.task.step(self)
      self.update_obs(obs, state_features)

      reward = (obs['state_features']*obs['task']).sum()

      truncated = False
      if self.step_count >= self._max_steps:
          truncated = True
          terminated = True

      # was not carrying an object before but am now
      if self.carrying_prior is None and self.carrying:
         self.object_counts[name(self.carrying)] += 1

      # update object counts
      self.carrying_prior = self.carrying

      return obs, reward, terminated, truncated, info

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super(RoomGridLevel, self)._gen_grid(width, height)

                # Generate the mission
                self.gen_mission()

                # # Validate the instructions
                # self.validate_instrs(self.instrs)

            except RecursionError as error:
                print("Timeout during mission generation:", error)
                continue

            except RejectSampling as error:
                print("Sampling rejected:", error)
                continue

            break

        # Generate the surface form for the instructions
        # self.surface = self.instrs.surface(self)
        # self.mission = self.surface

    def num_navs_needed(self, instr: str = None) -> int:
        return 2


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

class ObjectCountObserver(LevelAvgObserver):
  """Log object counts over episodes and create barplot."""

  def __init__(self, *args, agent_name: str = '', **kwargs):
    super(ObjectCountObserver, self).__init__(*args, **kwargs)
    self.agent_name = agent_name

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state __after__ reset. Increments by 1, logs task_name."""
    self.idx += 1
    self.task_name = self.current_task(env)

  def observe(self,
              env: dm_env.Environment,
              timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """At last time-step, record object counts."""
    if timestep.last():
       self.results[self.task_name].append(env.unwrapped.object_counts)

  def get_metrics(self) -> Dict[str, Number]:
    """Every K episodes, create and log a barplot to wandb with object counts."""
    if not self.logging: return {}

    if self.idx % self.reset == 0 and self.logging:

      for key, counts in self.results.items():
        # {name: counts}
        data = dict_mean(counts)

        # Mapping of color names to colors
        color_mapping = {
            'red': 'r',
            'green': 'g',
            'blue': 'b',
            'purple': '#701FC3',  # Use hex color code for custom colors
            'yellow': 'y',
            'grey': '#646464',    # Use hex color code for custom colors
        }

        # Mapping of shape names to Unicode characters
        shape_mapping = {
            'box': '\u25A1',    # Unicode for a square shape
            'ball': '\u25CF',   # Unicode for a circle shape
            'key': '\u272A'     # Unicode for a star shape
        }

        # Create subplots with fig and ax
        width, height = 8, 8
        fig, ax = plt.subplots(figsize=(width, height))
        values = data.values()
        ax.bar(np.arange(len(values)), values)

        objs = sorted(list(data.keys()))
        # Customize key labels with colors and shapes
        for i, obj in enumerate(objs):
            color_name, shape_name = obj.split()
            color = color_mapping.get(color_name, 'black')  # Default to black if color not found
            shape = shape_mapping.get(shape_name, '')  # Empty string if shape not found
            ax.text(i, -0.1, shape, rotation=0,
                    ha='center', va='top', color=color)

        ax.set_xticks([])
        ax.set_ylabel("Counts", fontsize=12)
        ax.set_title(f"{self.agent_name}: {str(key)}", fontsize=14)
        fig.tight_layout()

        try:
          wandb.log({f"{self.prefix}/counts_{key}": wandb.Image(fig)})
          plt.close(fig)
        except wandb.errors.Error as e:
          self.logging = False
          logging.warning(f"{self.prefix}: turning off logging.")


    return {}
