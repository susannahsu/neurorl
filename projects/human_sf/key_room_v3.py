"""
"""
from __future__ import annotations
from enum import Enum
from absl import logging

import copy
from gymnasium import spaces
from typing import Optional, Union, List, Dict, Tuple
import itertools
import random

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


class BaseTaskRep:
  def __init__(
      self,
      types,
      colors,
      target,
      target_room_color: Optional[str] = None,
      colors_to_room: Optional[dict] = None,
      first_instance: bool = True):
      self.types = types
      self.colors = colors
      self.target_type, self.target_color  = target
      self.target_room_color = target_room_color

      colors_to_room = colors_to_room or dict()
      self.room_to_color = {room: color for color,
                            room in colors_to_room.items()}

      self.first_instance = first_instance
      self.reset()

  def get_carrying(self, env):
    if env.carrying:
        color, shape = env.carrying.color, env.carrying.type
        return color, shape
    return None, None

  def current_room(self, env):
    agent_room = env.room_from_pos(*env.agent_pos)
    return self.room_to_color[agent_room]

  @property
  def task_array(self): return self._task_array

  @property
  def state_features(self): return self._state_features

  def reset(self):
     self._task_array = self.make_task_array()
     self._state_features = self.empty_array()
     self.feature_counts = self.empty_array()
     self.prior_feature_counts = self.empty_array()

  def step(self, env):
    current_counts = self.current_state(env)
    difference = (current_counts - 
                 self.prior_feature_counts).astype(np.float32)
    different = (difference > 0.0).astype(np.float32)

    # whichever state-features have changed, add them to the counts
    self.feature_counts += difference*different

    if self.first_instance:
      # in 1 setting, state-feature is only active the 1st time the count goes to 1
      first = (self.feature_counts == 1).astype(np.float32)
      state_features = first*difference
    else:
      # in this setting, the feature difference
      state_features = difference

    self.prior_feature_counts = self.feature_counts
    self.state_features = state_features

  def empty_array(self):
    raise NotImplementedError

  def make_task_array(self):
    raise NotImplementedError

  def current_state(self, env):
    raise NotImplementedError

class FlatTaskRep(BaseTaskRep):

  def empty_array(self):
    return np.zeros(len(self.colors) * len(self.types))

  def get_vector_index(self, color, shape):
    return self.colors.index(color) * len(self.types) + self.types.index(shape)

  def make_task_array(self):
    vector = np.zeros(len(self.colors) * len(self.types))

    target_idx = self.get_vector_index(self.target_color, self.target_type)
    vector[target_idx] = 1.0

    if self.target_room_color:
      target_room_idx = self.get_vector_index(self.target_room_color, 'room')
      vector[target_room_idx] = 0.5

      target_key_idx = self.get_vector_index(self.target_room_color, 'key')
      vector[target_key_idx] = 0.1

    return vector

  def print_task_array(self):
    # Retrieve the task array
    task_vector = self.make_task_array()

    # Iterate over each element in the vector and print it with the corresponding color and shape
    for index in range(len(task_vector)):
        # Integer division to get color index
        color_idx = index // len(self.types)
        shape_idx = index % len(self.types)   # Modulo to get shape index
        color = self.colors[color_idx]
        shape = self.types[shape_idx]
        val = task_vector[index]
        print(f'{color}, {shape} = {val}')

  def current_state(self, env):
    state_vector = self.empty_array()

    carrying_color, carrying_shape = self.get_carrying(env)
    if carrying_color and carrying_shape:
        idx = self.get_vector_index(carrying_color, carrying_shape)
        state_vector[idx] = 1

    current_room_color = self.current_room(env)
    if current_room_color:
        room_idx = self.get_vector_index(current_room_color, 'room')
        state_vector[room_idx] = 1

    return state_vector

class StructuredTaskRep(BaseTaskRep):

  def empty_array(self):
    # Initialize the array with zeros
    return np.zeros((len(self.colors), len(self.types)))

  def make_task_array(self):
    # Initialize the array with zeros
    array = np.zeros((len(self.colors), len(self.types)))

    # Set the target shape and color to 1
    target_color_idx = self.colors.index(self.target_color)
    target_shape_idx = self.types.index(self.target_type)
    array[target_color_idx, target_shape_idx] = 1.0

    # Set the room and key for the target room color to 0.5 and 0.1 respectively
    if self.target_room_color:
      target_room_color_idx = self.colors.index(self.target_room_color)
      room_shape_idx = self.types.index('room')
      key_shape_idx = self.types.index('key')
      array[target_room_color_idx, room_shape_idx] = 0.5
      array[target_room_color_idx, key_shape_idx] = 0.1

    return array

  def print_task_array(self):
    # Create an empty matrix with the appropriate dimensions
    matrix = [['' for _ in self.types] for _ in self.colors]

    # Fill the matrix with combinations of rows and columns
    task_array = self.make_task_array()
    for i, row in enumerate(self.colors):
        for j, col in enumerate(self.types):
            val = task_array[i, j]
            matrix[i][j] = f'{row}, {col} = {val}'

    # Printing the matrix
    for row in matrix:
        print(row)

  def current_state(self, env):
    # Reset the state array to all zeros
    state_array = self.empty_array()

    # Check if the agent is carrying something
    carrying_color, carrying_shape = self.get_carrying(env)
    if carrying_color and carrying_shape:
        color_idx = self.colors.index(carrying_color)
        shape_idx = self.types.index(carrying_shape)
        state_array[color_idx, shape_idx] = 1

    # Check the color of the current room
    current_room_color = self.current_room(env)
    if current_room_color:
        room_color_idx = self.colors.index(current_room_color)
        room_shape_idx = self.types.index('room')
        state_array[room_color_idx, room_shape_idx] = 1

    return state_array

ObjectNames = List[List[str]]

def swap_test_pairs(maze_config):
  maze_config = copy.deepcopy(maze_config)
  maze_config['pairs'][0][1], maze_config['pairs'][1][1] = maze_config['pairs'][1][1], maze_config['pairs'][0][1]
  maze_config['pairs'][2][1], maze_config['pairs'][3][1] = maze_config['pairs'][3][1], maze_config['pairs'][2][1]
  return maze_config


class KeyRoom(LevelGen):
    """
    KeyRoom is a class for generating grid-based environments with tasks involving keys, doors, and objects
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
    def __init__(
      self,
      maze_config: Dict[str:ObjectNames],
      flat_task: bool = True,
      room_size=7,
      num_rows=3,
      num_cols=3,
      num_dists=0,
      locations=False,
      unblocking=False,
      rooms_locked=True,
      swap_episodes: int = 100_000,
      color_rooms: bool = True,
      training=True,
      # include_task_signals=False,
      max_steps_per_room: int = 100,
      implicit_unlock=True,
      room_colors: List[str] = ['blue', 'yellow', 'red', 'green'],
      **kwargs):
      """
      Initialize a KeyRoom environment with customizable parameters.

      Args:
          maze (Dict[str:ObjectNames]): A dictionary describing the objects in the environment.
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

      self.maze_config = maze_config
      self.maze_swap_config = swap_test_pairs(maze_config)
      self.flat_task = flat_task
      self.swap_episodes = swap_episodes
      self.training = training
      self.room_size = room_size
      self.train_objects = [p[0] for p in maze_config['pairs']]
      self.test_objects = [p[1] for p in maze_config['pairs']]
      self.all_final_objects = self.train_objects + self.test_objects
      self.color_rooms = color_rooms
      self.types = set(['key', 'room'])
      self.colors = set(['start'])
      for key in maze_config['keys']:
         self.colors.add(key[1])
      for pairs in maze_config['pairs']:
         for p in pairs:
          self.colors.add(p[1])
          self.types.add(p[0])

      self.colors = list(self.colors)
      self.types = list(self.types)

      self.rooms_locked = rooms_locked
      self.room_colors = room_colors

      all_tasks = []
      train_tasks = []
      for idx, (type, color) in enumerate(maze_config['keys']):
        train_object, test_object = maze_config['pairs'][idx]

        train_task = self.task_class(
            types=self.types,
            colors=self.colors,
            target=train_object)
        train_tasks.append(train_task)
        all_tasks.append(train_task)

        test_task = self.task_class(
            types=self.types,
            colors=self.colors,
            target=test_object)
        all_tasks.append(test_task)

      self.all_task_arrays = np.array([t.task_array for t in all_tasks])
      self.train_task_arrays = np.array([t.task_array for t in train_tasks])
      self.episodes = 0

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
      task_array = self.all_task_arrays[0]
      cumulants_space = spaces.Box(
          low=0,
          high=100.0,
          shape=task_array.shape,  # number of cells
          dtype="float32",
      )
      train_task_arrays = spaces.Box(
          low=0,
          high=100.0,
          shape=self.train_task_arrays.shape,  # number of cells
          dtype="float32",
      )
      all_task_arrays = spaces.Box(
          low=0,
          high=100.0,
          shape=self.all_task_arrays.shape,  # number of cells
          dtype="float32",
      )
      self.observation_space = spaces.Dict(
          {**self.observation_space.spaces,
           "state_features": cumulants_space,
           "task": copy.deepcopy(cumulants_space),  # equivalent specs
           "train_tasks": train_task_arrays,
           "all_tasks": all_task_arrays,
           }
      )

    @property
    def task_class(self):
      if self.flat_task:
          return FlatTaskRep
      else:
          return StructuredTaskRep

    def gen_mission(self):
      """_summary_

      
      # Returns:
      #     _type_: _description_
      # """
      if self.training:
        choice = random.choice([0, 1, 2])
        if choice == 0:
           self.single_room_placement()
        else:
           self.multi_room_placement()
      else:
        if self.episodes >= self.swap_episodes:
           self.multi_room_placement(
              maze_config=self.maze_swap_config)
        else:
          self.multi_room_placement()

    def single_room_placement(self):
      self.instrs = DummyInstr()
      ###########################
      # Place agent in center room
      ###########################
      center_room_coords = (self.num_rows//2, self.num_cols//2)
      self.place_agent(*center_room_coords)
      _ = self.room_from_pos(*self.agent_pos)

      ###########################################
      # Place objects
      ###########################################
      # starts to the right for some reason
      self.all_objects = []
      for object in self.train_objects+self.test_objects:
       # Assuming construct is a function that creates an object based on the type and color
        obj = construct(*object)
        self.all_objects.append(obj)
        self.place_in_room(*center_room_coords, obj)

      # Sample random task object
      all_choices = self.train_objects + self.test_objects
      task_object = random.choice(all_choices)

      # Assuming 'colors_to_room' is defined elsewhere or is not needed for flat tasks
      self.task = self.task_class(
          types=self.types,
          colors=self.colors,
          target=task_object,
      )

    def multi_room_placement(self, maze_config=None):
      self.instrs = DummyInstr()
      maze_config = maze_config or self.maze_config
      ###########################
      # Place agent in center room
      ###########################
      center_room_coords = (self.num_rows//2, self.num_cols//2)
      self.place_agent(*center_room_coords)
      _ = self.room_from_pos(*self.agent_pos)

      ###########################################
      # Place objects
      ###########################################
      # starts to the right for some reason
      self.all_objects = []
      goal_room_coords = [(2, 1), (1, 2), (0, 1), (1, 0)]
      keys = maze_config['keys']
      room_colors = [k[1] for k in keys]
      pairs = maze_config['pairs']
      indices = list(range(len(keys)))
      colors_to_room = {}

      start_room_color = 'start'
      center_room = self.get_room(*center_room_coords)
      colors_to_room[start_room_color] = center_room

      for idx in indices:
          key_color = keys[idx][1]
          room = goal_room_coords[idx]
          colors_to_room[key_color] = self.get_room(*room)


          key = Key(key_color)
          self.all_objects.append(key)
          self.place_in_room(*center_room_coords, key)
          door, door_pos = self.add_door(*center_room_coords, door_idx=idx, color=key_color, locked=self.rooms_locked)
          self.all_objects.append(door)

          pair1, pair2 = pairs[idx]
          obj1 = construct(*pair1)
          obj2 = construct(*pair2)
          self.all_objects.extend([obj1, obj2])
          self.place_in_room(*room, obj1)
          self.place_in_room(*room, obj2)

          init_floor = construct(shape='floor', color=key_color)

          width = self.room_size - 3
          if self.color_rooms:
            for _ in range(width*width):
              self.place_in_room(*room, init_floor)

      task_idx = random.sample(indices, 1)[0]
      if self.training:
         task_object = self.train_objects[task_idx]
      else:
         task_object = self.test_objects[task_idx]

      target_room_color = room_colors[task_idx] if self.training else None
      self.task = self.task_class(
          types=self.types,
          colors=self.colors,
          target=task_object,
          target_room_color=target_room_color,
          colors_to_room=colors_to_room)

    def update_obs(self, obs):
      obs['task'] = self.task.task_array
      obs['state_features'] = self.task.state_features
      obs['train_tasks'] = self.train_task_arrays
      obs['all_tasks'] = self.all_task_arrays

    def reset(self, *args, **kwargs):
      obs, info = super().reset(*args, **kwargs)
      self.update_obs(obs)

      # reset object counts
      self.object_counts = {name(o):0 for o in self.all_objects}
      self.carrying_first_time = False
      self.episodes += 1
      return obs, info

    def step(self, action, **kwargs):
      obs, _, _, _, info = super().step(action, **kwargs)
      self.task.step(self)
      self.update_obs(obs)

      reward = (obs['state_features']*obs['task']).sum()

      # was not carrying an object before but am now
      if not self.carrying_first_time and self.carrying:
         self.carrying_first_time = True
         # first time, log it
         self.object_counts[name(self.carrying)] += 1

      if self.carrying:
         # if carrying and one of the "final" objects, terminate
         shape, color = self.carrying.type, self.carrying.color
         if [shape, color] in self.all_final_objects:
            terminated = True

      truncated = False
      if self.step_count >= self._max_steps:
          truncated = True
          terminated = True

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
        mean_dict[key] = sum(d.get(key, 0) for d in dict_list) / len(dict_list)
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
        # Customize key labels with colors and types
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
