from __future__ import annotations
from enum import Enum

import copy
from gymnasium import spaces
from typing import Optional, Union, List
import itertools
import numpy as np

from minigrid.envs.babyai import goto
from minigrid.envs.babyai.core.levelgen import LevelGen

import numpy as np

from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Door, Key, WorldObj, Floor
from minigrid.minigrid_env import MiniGridEnv

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel, RejectSampling
from minigrid.envs.babyai.core.verifier import PickupInstr, ObjDesc, Instr

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
        self.types = ['key', 'task_room', 'box', 'ball']

        type2indx = type2indx or {
           'key': 0,
           'task_room': 1,
           'ball': 2,
           'box': 3,
        }
        self.type2indx = type2indx
        floor2task_color = floor2task_color or {
            'blue': 'red',
            'yellow': 'green',
        }
        self.task_colors = list(floor2task_color.values())
        self.floor2task_color = floor2task_color
        self.shape2task_color = {
            'ball': 'red',
            'box': 'green',
        }

    def task_type_vector(self):
       vector = [.1, .5, 0., 0]
       vector[self.type2indx[self.goal_object()]] = self.task_reward
       return np.array(vector)

    def task_color_vector(self):
       vector = [float(t == self.goal_color()) for t in self.task_colors]
       return np.array(vector)

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

    def reset(self):
       self.picked_up_key = False
       self.entered_task_room = False
       self.task_room = None

    def set_room(self, room):
       self.task_room = room

    def agent_in_task_room(self, env):
      agent_room = env.room_from_pos(*env.agent_pos)
      return agent_room == self.task_room

    def step(self, env):
      color, type = self.goal()

      def onehot(type: str = None):
        x = np.zeros(len(self.types))
        x[self.type2indx[type]] = 1
        return x

      terminal = False

      # check # 1: is the task over?
      if (env.carrying is not None and 
          env.carrying.type == type and 
          env.carrying.color == color):
         terminal = True
         state_features = onehot(type)
         return state_features, terminal

      # check # 2: have you entered the task room for 1st time
      if (not self.entered_task_room and 
            self.agent_in_task_room(env)):
         self.entered_task_room = True
         state_features = onehot('task_room')
         return state_features, terminal

      # check # 3: have you picked up the task key
      if (not self.picked_up_key and 
            env.carrying is not None and
            env.carrying.type == 'key' and
            env.carrying.color == color):
         self.picked_up_key = True
         state_features = onehot('key')
         return state_features, terminal

      state_features = np.zeros(len(self.types))
      return state_features, terminal

class KeyRoomObjectTest(LevelGen):
    """

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
      """Keyroom.

      Args:
          room_size (int, optional): _description_. Defaults to 7.
          num_rows (int, optional): _description_. Defaults to 3.
          num_cols (int, optional): _description_. Defaults to 3.
          num_dists (int, optional): _description_. Defaults to 10.
          locations (bool, optional): _description_. Defaults to False.
          unblocking (bool, optional): _description_. Defaults to False.
          implicit_unlock (bool, optional): _description_. Defaults to False.
          fixed_door_locs (bool, optional): _description_. Defaults to True.
      """
      self.tasks = tasks
      self.rooms_locked = rooms_locked
      self.room_colors = room_colors
      self.include_task_signals = include_task_signals
      
      dummy_task_vector = self.tasks[0].task_type_vector()
      self.train_task_vectors = np.array([t.task_type_vector() for t in tasks])

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
          shape=(len(dummy_task_vector),),  # number of cells
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
           #  "state_features": cumulants_space,
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

      ###########################
      # Place keys, door, and ball of same color
      ###########################
      # colors = ['red', 'green', 'yellow', 'grey']
      colors = self.room_colors
      # starts to the right for some reason
      goal_room_coords = [(2,1), (1, 2), (0, 1), (1,0)]

      def generate_room_objects(color: str):
        return (Key(color), Ball(color), Box(color))

      for room_idx, color in enumerate(colors):
        room = goal_room_coords[room_idx]
        key, ball, box = generate_room_objects(color)
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

      self.task.reset()
      self.task.set_room(self.get_room(*task_room_coord))

      if self.include_task_signals:
        init_object = self.task.initial_object()
        self.place_in_room(*center_room, init_object)

        init_floor = self.task.initial_floor()
        self.place_in_room(*center_room, init_floor)
        self.place_in_room(*center_room, init_floor)
        self.place_in_room(*center_room, init_floor)

      self.instrs = DummyInstr()
      self.task_vector = self.task_type_vector = self.task.task_type_vector()
      # self.task_matrix = np.array(
      #    [self.task_vector,
      #     self.task.task_color_vector()])

    def update_obs(self, obs, state_features):
      obs['task'] = self.task_vector
      obs['state_features'] = state_features
      obs['train_tasks'] = self.train_task_vectors


    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.update_obs(obs, np.zeros_like(self.task_vector))
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

