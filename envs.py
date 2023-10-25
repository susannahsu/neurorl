from __future__ import annotations

import itertools
import numpy as np

from minigrid.envs.babyai import goto
from minigrid.envs.babyai.core.levelgen import LevelGen

import numpy as np

from minigrid.core.constants import COLOR_NAMES
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

    def gen_mission(self):
        """_summary_

        
        Returns:
            _type_: _description_
        """
        self._goal_objects = []
        self._non_goal_objects = []

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
        key_colors = COLOR_NAMES[:4]
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
        box_colors = rotate(key_colors)
        for i in range(len(key_colors)):
            box = Box(box_colors[i])
            self.place_in_room(*goal_room_coords[i], box)
            # self._non_goal_objects.append(box)

        # Generate random instructions
        if self._training:
            idx = np.random.randint(len(self._goal_objects))
            obj = self._goal_objects[idx]
            obj_desc = ObjDesc(obj.type, obj.color)
        else:
            idx = np.random.randint(len(self._non_goal_objects))
            obj = self._goal_objects[idx]
            obj_desc = ObjDesc(obj.type, obj.color)
        self.instrs = DummyInstr()
        self.instruction = PickupInstr(obj_desc)

    def get_objects(self):
      objects = np.array(
            [OBJECT_TO_IDX[o.type] if o is not None else -1 for o in self.grid.grid]
        )
      return sorted(list(set(objects)))

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

