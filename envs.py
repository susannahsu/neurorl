from __future__ import annotations

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
from minigrid.core.world_object import Ball, Box, Door, Key, WorldObj
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

def rotate(array, n=1):
  """Rotate array by n."""
  length = len(array)

  # Rotate the array to the right by one position
  return[array[length - n]] + array[:length - n]

def make_name(color, type):
    return f"{color} {type}"

class KeyRoom(LevelGen):
    """

    """

    def __init__(self,
        room_size=7,
        num_rows=3,
        num_cols=3,
        num_dists=10,
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        training: bool = True,
        fixed_door_locs:bool=True,
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
        self._fixed_door_locs = fixed_door_locs
        self._training = training
        self._goal_objects = []
        self._non_goal_objects = []

        # object_types = OBJECT_TO_IDX.keys()
        object_types = ['key', 'ball', 'box']
        color_types = itertools.product(COLOR_NAMES, object_types)
        self.token_2_idx = {make_name(color, type): idx for idx, (color, type) in enumerate(color_types)}
        self.idx_2_token = {idx: name for name, idx in self.token_2_idx.items()}

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
        cumulants_space = spaces.Box(
            low=0,
            high=100.0,
            shape=(self.nfeatures,),  # number of cells
            dtype="float32",
        )

        self._train_tasks = list(itertools.product(self.train_colors, self.train_objects))
        self._ntrain_tasks = len(self._train_tasks)
        self._train_task_vectors = np.identity(self._ntrain_tasks)

        train_task_vectors = spaces.Box(
            low=0,
            high=100.0,
            shape=(self._ntrain_tasks, self.nfeatures,),  # number of cells
            dtype="float32",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
             "state_features": cumulants_space,
             "task_vector": copy.deepcopy(cumulants_space),  # equivalent specs
             "train_task_vectors": train_task_vectors,
            }
        )

    @property
    def train_colors(self):
      return COLOR_NAMES[:4]

    @property
    def train_objects(self):
      return ['ball']

    @property
    def nfeatures(self):
        return len(self.token_2_idx)

    def make_features(self, idx: Optional[Union[List, int]]=None):
        state_features = np.zeros(self.nfeatures)
        if idx is not None:
            if isinstance(idx, list):
              for i in idx:
                state_features[i] = 1
            elif isinstance(idx, int):
              state_features[idx] = 1
            else:
                raise NotImplementedError(f"idx type {type(idx)} not supported")
        return state_features

    def gen_mission(self):
        """_summary_

        
        Returns:
            _type_: _description_
        """
        self._goal_objects = []
        self._non_goal_objects = []
        self.task_vector = np.zeros(self.nfeatures)

        ###########################
        # Place agent in center room
        ###########################
        center_room = (self.num_rows//2, self.num_cols//2)
        self.place_agent(*center_room)
        _ = self.room_from_pos(*self.agent_pos)

        ###########################
        # Place keys, door, and ball of same color
        ###########################
        # place keys in main room
        key_colors = self.train_colors
        key_idxs = list(range(len(key_colors)))
        goal_room_idxs = [0, 1, 2, 3]
        # starts to the right for some reason
        goal_room_coords = [(2,1), (1, 2), (0, 1), (1,0)]

        if not self._fixed_door_locs:
            key_idxs = np.random.permutation(key_idxs)

        def permute(x):
            return [x[i] for i in key_idxs]


        goal_room_idxs = permute(goal_room_idxs)
        goal_room_coords = permute(goal_room_coords)

        for i in range(len(key_colors)):
            key = Key(key_colors[i])
            self.place_in_room(*center_room, key)
            door, pos = self.add_door(
                *center_room, 
                door_idx=goal_room_idxs[i],
                color=key_colors[i],
                locked=True)

            ball = Ball(key_colors[i])
            self.place_in_room(*goal_room_coords[i], ball)

            self._goal_objects.append(ball)
            self._non_goal_objects.append(key)

        ###########################
        # place non-task object
        ###########################
        # to have consistent __other__ colors for objects in room, simply rotate this matrix by 1
        box_colors = rotate(key_colors)
        for i in range(len(key_colors)):
            box = Box(box_colors[i])
            self.place_in_room(*goal_room_coords[i], box)

        # Generate random instructions
        if self._training:
            idx = np.random.randint(len(self._goal_objects))
            obj = self._goal_objects[idx]
            obj_desc = ObjDesc(obj.type, obj.color)
        else:
            idx = np.random.randint(len(self._non_goal_objects))
            obj = self._goal_objects[idx]
            obj_desc = ObjDesc(obj.type, obj.color)

        obj_idx = self.token_2_idx[make_name(obj.color, obj.type)]
        self.task_vector[obj_idx] = 1

        self.instrs = DummyInstr()
        self.instruction = PickupInstr(obj_desc)
        self.instruction.reset_verifier(self)

    def update_obs(self, obs):
      state_features = self.make_features()
      if self.carrying:
          obj_idx = self.token_2_idx[
              make_name(self.carrying.color, self.carrying.type)]
          state_features[obj_idx] = 1

      obs['task_vector'] = self.task_vector
      obs['train_task_vectors'] = self._train_task_vectors
      obs['state_features'] = state_features

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.update_obs(obs)
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = super().step(action, **kwargs)

        self.update_obs(obs)
        reward = float(self.instruction.verify(action) == "success")
        return obs, reward, terminated, truncated, info

    def all_types(self):
        return set(self._goal_objects+self._non_goal_objects)

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

    def num_navs_needed(self, instr) -> int:
        return 2

