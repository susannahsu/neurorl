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
from minigrid.core.world_object import Ball, Box, Key, Floor, WorldObj
from minigrid.core.world_object import Point, WorldObj


from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel, RejectSampling
from minigrid.envs.babyai.core.verifier import Instr
from minigrid.minigrid_env import MiniGridEnv

Color = str
Shape = str

def reject_next_to(env: MiniGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """

    # sx, sy = env.agent_pos
    x, y = pos
    all_empty = True
    for dx in (-1, 1):
      for dy in (-1, 1):
        sx = x - dx
        sy = y - dy
        empty = env.grid.get(sy, sx) is None
        all_empty = all_empty and empty
    reject = all_empty = False
    return reject

try:
  from library.utils import LevelAvgObserver
except:
   LevelAvgObserver = object

Number = Union[int, float, np.float32]
START_ROOM_COLOR = 'start'

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

def name(color: str, type: str):
   return f"{color} {type}"


class BaseTaskRep:
  def __init__(
      self,
      types,
      colors,
      target,
      key_room_rewards: Tuple[float] = (.1, .25),
      target_room_color: Optional[str] = None,
      colors_to_room: Optional[dict] = None,
      first_instance: bool = True):
      self.types = types
      self.colors = colors
      self.target_type, self.target_color  = target
      self._task_name = f"{self.target_color} {self.target_type}"
      self.target_room_color = target_room_color
      self.key_room_rewards = key_room_rewards
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
  def task_name(self): return self._task_name

  @property
  def task_array(self):
     return np.array(self._task_array)

  @property
  def state_features(self):
     return np.array(self._state_features)

  def reset(self):
     self._task_array = self.make_task_array()
     self._state_features = self.empty_array()
     self._prior_features = self.empty_array()
     self.feature_counts = self.empty_array()
     self.prior_feature_counts = self.empty_array()

  def step(self, env):
    # any features that changed, add them to feature counts
    current_features = self.current_state(env)
    difference = (current_features -
                  self._prior_features).astype(np.float32)
    positive_difference = difference*(difference > 0).astype(np.float32)
    self.feature_counts += positive_difference

    if self.first_instance:
      # in 1 setting, state-feature is only active the 1st time the count goes  0 -> +
      first = (self.feature_counts == 1).astype(np.float32)
      state_features = first*positive_difference
    else:
      # in this setting, whenever goes 0 -> +
      state_features = positive_difference

    self._prior_features = current_features
    self._state_features = state_features

  def empty_array(self):
    raise NotImplementedError

  def make_task_array(self):
    raise NotImplementedError

  def current_state(self, env):
    raise NotImplementedError

class FlatTaskRep(BaseTaskRep):

  def populate_array(self, objs: List[Shape, Color]):
    array = self.empty_array()
    for (shape, color) in objs:
      idx = self.get_vector_index(color=color, shape=shape)
      array[idx] = 1
    return array

  def empty_array(self):
    return np.zeros(len(self.colors) * len(self.types))

  def get_vector_index(self, color, shape):
    return self.colors.index(color) * len(self.types) + self.types.index(shape)

  def make_task_array(self):
    vector = np.zeros(len(self.colors) * len(self.types))

    target_idx = self.get_vector_index(self.target_color, self.target_type)
    vector[target_idx] = 1.0

    if self.target_room_color:
      target_key_idx = self.get_vector_index(self.target_room_color, 'key')
      vector[target_key_idx] = self.key_room_rewards[0]

      target_room_idx = self.get_vector_index(self.target_room_color, 'room')
      vector[target_room_idx] = self.key_room_rewards[1]

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
    if current_room_color != START_ROOM_COLOR:
        room_idx = self.get_vector_index(current_room_color, 'room')
        state_vector[room_idx] = 1

    return state_vector

class StructuredTaskRep(BaseTaskRep):

  def populate_array(self, objs: List[str, str]):
     raise NotImplementedError('need to generalize to matrix form')
    # array = self.empty_array()
    # for (color, shape) in objs:
    #   idx = self.get_vector_index(color=color, shape=shape)
    #   array[idx] = 1
    # return array

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
      array[target_room_color_idx, key_shape_idx] = self.key_room_rewards[0]
      array[target_room_color_idx, room_shape_idx] = self.key_room_rewards[1]

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
    if current_room_color != START_ROOM_COLOR:
        room_color_idx = self.colors.index(current_room_color)
        room_shape_idx = self.types.index('room')
        state_array[room_color_idx, room_shape_idx] = 1
        env.carrying = None  # remove object

    return state_array

ObjectNames = List[List[str]]

def swap_test_pairs(maze_config):
  maze_config = copy.deepcopy(maze_config)
  maze_config['pairs'][0][1], maze_config['pairs'][1][1] = maze_config['pairs'][1][1], maze_config['pairs'][0][1]
  try:
    maze_config['pairs'][2][1], maze_config['pairs'][3][1] = maze_config['pairs'][3][1], maze_config['pairs'][2][1]
  except:
     pass
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
      room_size=7,
      num_rows=3,
      num_cols=3,
      num_dists=0,
      locations=False,
      unblocking=False,
      rooms_locked=True,
      basic_only: int = 0,
      flat_task: bool = True,
      swap_episodes: int = 0,
      terminate_failure: bool = True,
      num_task_rooms: int = 2,
      color_rooms: bool = False,
      ignore_task: bool = False,
      training=True,
      test_itermediary_rewards: bool = True,
      key_room_rewards: Tuple[float] = (.1, .25),
      train_basic_objects: bool = True,
      # include_task_signals=False,
      max_steps_per_room: int = 100,
      implicit_unlock=True,
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
          **kwargs: Additional keyword arguments for customization.

      Attributes:
          tasks (List[ObjectTestTask]): A list of tasks to be performed in the environment.
          rooms_locked (bool): Whether rooms are initially locked.
          include_task_signals (bool): Whether to include task-specific signals (i.e. room color, indicator object).
      """
      assert color_rooms is False
      ###############
      # preparing maze
      ###############
      # limit number of rooms
      maze_config['keys'] = maze_config['keys'][:num_task_rooms]
      maze_config['pairs'] = maze_config['pairs'][:num_task_rooms]

      self.maze_config = maze_config

      # create swap if will be swapping objects after some number of episodes
      if swap_episodes:
        assert len(maze_config['keys']) > 1, f'increase num_task_rooms. currently {num_task_rooms}'
        self.maze_swap_config = swap_test_pairs(maze_config)

      # summarizing maze objects
      self.train_objects = [p[0] for p in maze_config['pairs']]
      self.test_objects = [p[1] for p in maze_config['pairs']]
      self.all_finalroom_objects = self.train_objects + self.test_objects

      self.all_possible_task_objects = self.all_finalroom_objects + maze_config['keys']
      self.all_possible_objects = self.all_possible_task_objects + [['door', k[1]] for k in maze_config['keys']]

      # gather full list of possible objects and types
      colors = ["red", "green", "blue", "purple", "yellow", "grey"]
      object_types = ["key", "ball", "box"]
      all_pairs = [(obj_type, color) for obj_type in object_types for color in colors]

      self.potential_distractors = [pair for pair in all_pairs if pair not in self.all_possible_task_objects]

      self.color_rooms = color_rooms
      self.num_task_rooms = num_task_rooms
      self.types = set(['key', 'room'])
      self.colors = set([START_ROOM_COLOR])
      for key in maze_config['keys']:
         self.colors.add(key[1])
      for pairs in maze_config['pairs']:
         for p in pairs:
          self.colors.add(p[1])
          self.types.add(p[0])

      # sort them so always have same order across processed
      self.colors = list(self.colors)
      self.types = list(self.types)

      self.colors.sort()
      self.types.sort()
      self.key_room_rewards = key_room_rewards

      ###############
      # task settings
      ###############
      self.flat_task = flat_task
      self.test_itermediary_rewards = test_itermediary_rewards
      self.task_setting = 'multi'

      ###############
      # training settings
      ###############
      self.ignore_task = ignore_task
      self.train_basic_objects = train_basic_objects
      self.swap_episodes = swap_episodes
      self.training = training
      self.room_size = room_size
      self.num_dists = num_dists
      self.terminate_failure = terminate_failure

      self.rooms_locked = rooms_locked
      self.basic_only = basic_only
      self.episodes = 0
      self.max_steps_per_room = int(room_size/6 * max_steps_per_room)

      self.initiated = False
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
      self.reset()
      self.initiated = True
      # self.colors_to_room_coords = self.get_colors_to_room_coords(
      #    maze_config, num_rows=num_rows, num_cols=num_cols)

      ####################
      # Collect the target colors associated with different objects
      ####################
      self.object_to_target_color = dict()
      for idx, (type, room_color) in enumerate(maze_config['keys']):
        train_object, test_object = maze_config['pairs'][idx]

        self.object_to_target_color[tuple(train_object)] = room_color  # train
        self.object_to_target_color[tuple(test_object)] = room_color  # test
        self.object_to_target_color[(type, room_color)] = room_color  # key

      ####################
      # Collect the train tasks
      ####################
      train_tasks = []
      for idx, (type, room_color) in enumerate(maze_config['keys']):
        train_object, test_object = maze_config['pairs'][idx]

        if self.train_basic_objects:
          train_tasks.append(
            self.make_task(task_object=train_object, intermediary_reward=False))
          train_tasks.append(
            self.make_task(task_object=test_object, intermediary_reward=False))

        train_tasks.append(
           self.make_task(task_object=train_object, intermediary_reward=True))

      self.task_set = np.array([t.task_array for t in train_tasks])

      ####################
      # cumulant mask
      ####################
      possible_cumulant_objects = self.all_possible_task_objects + [['room', k[1]] for k in maze_config['keys']]
      self.cumulant_mask = train_tasks[0].populate_array(possible_cumulant_objects)

      ####################
      # action space settings
      ####################
      self.primitive_actions = [
        self.actions.left,
        self.actions.right,
        self.actions.forward,
        self.actions.pickup,
        self.actions.drop,
        self.actions.toggle,
        self.actions.done,  # does nothing
      ]
      self.action_names = [a.name for a in self.primitive_actions]

      ####################
      # gym spaces
      ####################
      task_array = self.task_set[0]
      cumulants_space = spaces.Box(
          low=0,
          high=100.0,
          shape=task_array.shape,  # number of cells
          dtype="float32",
      )
      train_task_arrays = spaces.Box(
          low=0,
          high=100.0,
          shape=self.task_set.shape,  # number of cells
          dtype="float32",
      )
      context_array = spaces.Box(
          low=0,
          high=100.0,
          shape=(2,),  # number of cells
          dtype="float32",
      )
      self.observation_space = spaces.Dict(
          {**self.observation_space.spaces,
           "state_features": cumulants_space,
           "task": copy.deepcopy(cumulants_space),  # equivalent specs
           "cumulant_mask": copy.deepcopy(cumulants_space),  # equivalent specs
           "train_tasks": train_task_arrays,
           "context": context_array,
           }
      )

    @property
    def env_objects(self):
      return self.all_possible_objects

    def make_task(
          self,
          task_object: Union[List[str], Tuple[str, str]],
          colors_to_room: Optional[Dict] = None,
          intermediary_reward: bool = True):
      if self.flat_task:
          TaskCls = FlatTaskRep
      else:
          TaskCls = StructuredTaskRep

      target_room_color = self.object_to_target_color[tuple(task_object)]
      target_room_color = target_room_color if intermediary_reward else None
      return TaskCls(
        types=self.types,
        colors=self.colors,
        colors_to_room=colors_to_room,
        target=task_object,
        key_room_rewards=self.key_room_rewards,
        target_room_color=target_room_color
    )

    def gen_mission(self):
      """_summary_

      
      # Returns:
      #     _type_: _description_
      # """

      if self.basic_only > 0:
         return self.single_room_placement()

      if self.training:
        choice = random.choice([0, 1, 2])
        if choice == 0 and self.train_basic_objects:
          return self.single_room_placement()
        else:
          return self.multi_room_placement()
      else:
        if self.swapped:
          return self.multi_room_placement(
            maze_config=self.maze_swap_config)
        else:
          return self.multi_room_placement()

    @property
    def task_name(self):
      name = self.task.task_name
      name += f" - {self.task_setting}"
      if self.swapped:
        name += " - swapped"
      return name

    @property
    def swapped(self):
      return self.swap_episodes and self.episodes >= self.swap_episodes

    def get_colors_to_room_coords(self, maze_config, num_rows, num_cols):
      center_room_coords = (num_rows//2, num_cols//2)
      # right room, bottom room, left room, upper room. in that order
      goal_room_coords = [(2, 1), (1, 2), (0, 1), (1, 0)]
      keys = maze_config['keys']
      room_colors = [k[1] for k in keys]
      indices = list(range(len(keys)))
      colors_to_room_coords = {}

      start_room_color = START_ROOM_COLOR
      colors_to_room_coords[start_room_color] = center_room_coords

      for idx in indices:
          room_color = room_colors[idx]
          room_coord = goal_room_coords[idx]
          colors_to_room_coords[room_color] = room_coord

      return colors_to_room_coords

    def single_room_placement(self):
      self.context = np.zeros((2))
      self.context[1] = 1
      self.task_setting = '1room'
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
      potential_objects = self.train_objects+self.test_objects
      if self.basic_only == 2:
         raise NotImplementedError
         potential_objects = potential_objects[:1]

      for object in potential_objects:
       # Assuming construct is a function that creates an object based on the type and color
        obj = construct(*object)
        self.place_in_room(*center_room_coords, obj)

      # Sample random task object
      task_object = random.choice(potential_objects)

      # Assuming 'colors_to_room' is defined elsewhere or is not needed for flat tasks
      self.task = self.make_task(
         task_object=task_object,
         intermediary_reward=False,
         colors_to_room={START_ROOM_COLOR:
                         self.get_room(*center_room_coords)})

      offtask_goal_object = [o for o in potential_objects if o != task_object][0]
      self.offtask_goal = self.make_task(
         task_object=offtask_goal_object,
         intermediary_reward=False,
         colors_to_room={START_ROOM_COLOR:
                         self.get_room(*center_room_coords)})

    def place_in_room(
        self, i: int, j: int, obj: WorldObj,
        task_obj_index: int = 0,
    ) -> tuple[WorldObj, tuple[int, int]]:
        """
        Add an existing object to room (i, j)
        """

        room = self.get_room(i, j)

        if task_obj_index:
          pos = self.place_task_obj(
              obj, room.top, room.size, reject_fn=reject_next_to, max_tries=1000,
              task_obj_index=task_obj_index,
          )
        else:
           pos = self.place_obj(
            obj, room.top, room.size, reject_fn=reject_next_to, max_tries=1000)

        room.objs.append(obj)

        return obj, pos

    def place_task_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        task_obj_index: int = 0,
        max_tries=np.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            raise RuntimeError
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
           raise RuntimeError
            # size = (self.grid.width, self.grid.height)

        num_tries = 0

        # decrease size by 1 on each border
        size = (size[0] - 2, size[1] - 2)

        # add 1 to each. so not next to door
        top = (top[0] + 2, top[1] + 2)

        # midway point
        halfsize = ((size[0]//2), (size[1]//2))
        mid = ((top[0] + halfsize[0]-1), (top[1] + halfsize[1]-1))

        # import ipdb; ipdb.set_trace()
        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1
            def make_pos(_top, _size):
               return (
                  self._rand_int(
                     _top[0], min(_top[0] + _size[0], self.grid.width)),
                  self._rand_int(
                     _top[1], min(_top[1] + _size[1], self.grid.height)),
              )

            if task_obj_index == 1:
               pos = make_pos(top, halfsize)
            elif task_obj_index == 2:
               pos = make_pos(mid, halfsize)
            else:
               raise RuntimeError

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def multi_room_placement(self, maze_config=None):
      self.task_setting = 'multi'
      self.context = np.zeros((2))
      self.context[0] = 1
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
      goal_room_coords = [(2, 1), (1, 2), (0, 1), (1, 0)]
      keys = maze_config['keys']
      room_colors = [k[1] for k in keys]
      pairs = maze_config['pairs']
      indices = list(range(len(keys)))
      colors_to_room = {}

      start_room_color = START_ROOM_COLOR
      center_room = self.get_room(*center_room_coords)
      colors_to_room[start_room_color] = center_room

      for idx in indices:
          key_color = keys[idx][1]
          room = goal_room_coords[idx]
          colors_to_room[key_color] = self.get_room(*room)


          key = Key(key_color)
          self.place_in_room(*center_room_coords, key)
          door, door_pos = self.add_door(*center_room_coords, door_idx=idx, color=key_color, locked=self.rooms_locked)

          pair1, pair2 = pairs[idx]
          obj1 = construct(*pair1)
          obj2 = construct(*pair2)
          self.place_in_room(*room, obj1, task_obj_index=1)
          self.place_in_room(*room, obj2, task_obj_index=2)

          init_floor = construct(shape='floor', color=key_color)

          width = self.room_size - 3
          if self.color_rooms:
            for _ in range(width*width):
              self.place_in_room(*room, init_floor)

          # if self.num_dists and not self.training:
          #   # Place distractors in the room
          #   for _ in range(self.num_dists):
          #       # Assuming you have a method 'construct_distractor' to create a distractor
          #       distractor_type, distractor_color = random.choice(self.potential_distractors)
          #       distractor = construct(distractor_type, distractor_color)
          #       self.place_in_room(*room, distractor)

      room_idxs = range(len(keys))
      room_idx = random.sample(room_idxs, 1)[0]
      target_room_color = room_colors[room_idx]
      if self.training:
        task_object = self.train_objects[room_idx]
        offtask_goal_object = self.test_objects[room_idx]
      else:
        if random.sample((0, 1), 1)[0] == 0:  # TRAINING
          task_object = self.train_objects[room_idx]
          offtask_goal_object = self.test_objects[room_idx]
        else:  # TESTING
          task_object = self.test_objects[room_idx]
          offtask_goal_object = self.train_objects[room_idx]

      assert self.object_to_target_color[tuple(task_object)] == target_room_color

      self.task = self.make_task(
         task_object=task_object,
         colors_to_room=colors_to_room,
         intermediary_reward=self.training or self.test_itermediary_rewards)

      self.offtask_goal = self.make_task(
         task_object=offtask_goal_object,
         colors_to_room=colors_to_room,
         intermediary_reward=self.test_itermediary_rewards)

    def update_obs(self, obs):
      obs['task'] = self.task.task_array
      obs['cumulant_mask'] = self.cumulant_mask
      obs['state_features'] = self.task.state_features
      obs['train_tasks'] = self.task_set
      obs['context'] = np.array(self.context)

      if self.ignore_task:
         obs['task'] = obs['task']*0.0
         raise RuntimeError("test removal and delete")

    def reset(self, *args, **kwargs):
      obs, info = super().reset(*args, **kwargs)
      if not self.initiated:
        return obs, info

      self.update_obs(obs)

      # reset object counts
      self.object_counts = {
         name(color=o[1], type=o[0]):0 for o in self.all_possible_task_objects}
      self.carrying_first_time = False
      self.episodes += 1
      return obs, info

    def step(self, action, **kwargs):
      obs, _, _, _, info = super().step(action, **kwargs)
      if action == self.actions.toggle:
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell:
          if fwd_cell.type == 'door' and fwd_cell.is_open:
              self.carrying = None

      self.task.step(self)
      self.update_obs(obs)


      reward = (obs['state_features']*obs['task']).sum()

      # was not carrying an object before but am now
      if not self.carrying_first_time and self.carrying:
        self.carrying_first_time = True
        # first time, log it
        key = name(type=self.carrying.type, color=self.carrying.color)
        self.object_counts[key] += 1
        if self.object_counts[key] > 1:
           raise RuntimeError

      terminated = False
      if self.carrying:
        # if carrying and one of the "final" objects, terminate
        shape, color = self.carrying.type, self.carrying.color
        if self.terminate_failure:
          if [shape, color] in self.all_finalroom_objects:
            terminated = True
        else:
          target_shape = self.task.target_type
          target_color = self.task.target_color
          if shape == target_shape and color == target_color:
            terminated = True

      truncated = False
      if self.step_count >= self.max_episode_steps:
          truncated = True
          # terminated = True

      return obs, reward, terminated, truncated, info

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super(RoomGridLevel, self)._gen_grid(width, height)
                self.instrs = DummyInstr()
                if not self.initiated:
                   return

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
        if self.task_setting == 'multi':
           return 2
        elif self.task_setting == '1room':
           return 1
        else: 
           raise NotImplementedError

    @property
    def max_episode_steps(self):
       return self.max_steps_per_room*self.num_navs_needed()

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
          pass


    return {}
