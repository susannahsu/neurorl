import copy
from typing import NamedTuple, Tuple

from gymnasium.core import Wrapper
from gymnasium import spaces
import numpy as np

from minigrid.utils import baby_ai_bot as bot_lib

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX, IDX_TO_COLOR, IDX_TO_OBJECT


from minigrid.envs.babyai.core.verifier import PickupInstr

class GlobalObjDesc(NamedTuple):
    color: int
    category: str
    type: int
    state: int
    global_pos: Tuple[int, int]
    local_pos: Tuple[int, int]

def matrix_to_row(row, column, num_columns):
    unique_index = row * num_columns + column
    return unique_index

def flat_to_matrix(unique_index, num_columns):
    row = unique_index // num_columns
    col = unique_index % num_columns
    return int(row), int(col)


class GoNextToSubgoal(bot_lib.Subgoal):
    """Simplified version of BabyAI subgoal.

    only accepts tuples for pos
    """

    def replan_before_action(self):
        target_obj = None
        target_pos = tuple(self.datum)

        # Suppore we are walking towards the door that we would like to open,
        # it is locked, and we don't have the key. What do we do? If we are carrying
        # something, it makes to just continue, as we still need to bring this object
        # close to the door. If we are not carrying anything though, then it makes
        # sense to change the plan and go straight for the required key.

        # The position we are on is the one we should go next to
        # -> Move away from it

        if np.array_equal(target_pos, self.fwd_pos):
            self.bot.stack.pop()
            return

        # We are still far from the target
        # -> try to find a non-blocker path
        path, _, _ = self.bot._shortest_path(
            lambda pos, cell: pos == target_pos,
        )

        # No non-blocker path found and
        # reexploration within the room is not allowed or there is nothing to explore
        # -> Look for blocker paths
        if not path:
            path, _, _ = self.bot._shortest_path(
                lambda pos, cell: pos == target_pos, try_with_blockers=True
            )

        # So there is a path (blocker, or non-blockers)
        # -> try following it
        next_cell = np.asarray(path[0])

        # Choose the action in the case when the forward cell
        # is the one we should go next to
        if np.array_equal(next_cell, self.fwd_pos):
            if self.fwd_cell:
               return self.actions.done
            else:
                return self.actions.forward

        # The forward cell is not the one we should go to
        # -> turn towards the direction we need to go
        if np.array_equal(next_cell - self.pos, self.right_vec):
            return self.actions.right
        elif np.array_equal(next_cell - self.pos, -self.right_vec):
            return self.actions.left

        # If we reach this point in the code,  then the cell is behind us.
        # Instead of choosing left or right randomly,
        # let's do something that might be useful:
        # Because when we're GoingNextTo for the purpose of exploring,
        # things might change while on the way to the position we're going to, we should
        # pick this right or left wisely.
        # The simplest thing we should do is: pick the one
        # that doesn't lead you to face a non empty cell.
        # One better thing would be to go to the direction
        # where the closest wall/door is the furthest
        distance_right = self.bot._closest_wall_or_door_given_dir(
            self.pos, self.right_vec
        )
        distance_left = self.bot._closest_wall_or_door_given_dir(
            self.pos, -self.right_vec
        )
        if distance_left > distance_right:
            return self.actions.left
        return self.actions.right

    def replan_after_action(self, action_taken):
        if action_taken in [
            self.actions.pickup,
            self.actions.drop,
            self.actions.toggle,
        ]:
            self._plan_undo_action(action_taken)

    def is_exploratory(self):
        return self.reason == "Explore"


class GotoBot(bot_lib.BabyAIBot):
  """"""
  def __init__(self, env, loc):

    # Mission to be solved
    self.mission = mission = env

    # Visibility mask. True for explored/seen, false for unexplored.
    self.vis_mask = np.zeros(shape=(mission.unwrapped.width, mission.unwrapped.height), dtype=bool)

    # Stack of tasks/subtasks to complete (tuples)
    # self.subgoals = subgoals = self.mission.task.subgoals()
    self.loc = loc
    self.goal = GoNextToSubgoal(self, loc, reason='none')
    self.stack = [self.goal]
    self.stack.reverse()

    # How many BFS searches this bot has performed
    self.bfs_counter = 0

    # How many steps were made in total in all BFS searches
    # performed by this bot
    self.bfs_step_counter = 0

  def generate_trajectory(self, action_taken=None):

    # steps_left = len(self.stack)
    env = self.mission

    all_obs = []
    all_action = []
    all_reward = []
    all_truncated = []
    all_done = []
    all_info = []

    def step_update(_action):
      _obs, _reward, _done, _trunc, _info = env.step(_action)
      all_obs.append(_obs)
      all_action.append(_action)
      all_reward.append(_reward)
      all_truncated.append(_trunc)
      all_done.append(_done)
      all_info.append(_info)

    idx = 0
    while self.stack:
      idx += 1
      if idx > 1000:
        raise RuntimeError(f"generate_trajectory taking too long {idx}")

      action = self.replan(action_taken)

      # need to do extra step to complete, exit
      if len(self.stack) > 1:
         break
      # -----------------------
      # done??
      # -----------------------
      if action == env.actions.done:
        break

      # -----------------------
      # take actions
      # -----------------------
      step_update(action)
      action_taken = action

    return (
      all_action,
      all_obs,
      all_reward,
      all_truncated,
      all_done,
      all_info,
    )

  def replan(self, action_taken=None):
    """Replan and suggest an action.

    Call this method once per every iteration of the environment.

    Args:
        action_taken: The last action that the agent took. Can be `None`, in which
        case the bot assumes that the action it suggested was taken (or that it is
        the first iteration).

    Returns:
        suggested_action: The action that the bot suggests. Can be `done` if the
        bot thinks that the mission has been accomplished.

    """
    self._process_obs()

    # Check that no box has been opened
    self._check_erroneous_box_opening(action_taken)

    # TODO: instead of updating all subgoals, just add a couple
    # properties to the `Subgoal` class.
    for subgoal in self.stack:
        subgoal.update_agent_attributes()

    if self.stack:
        self.stack[-1].replan_after_action(action_taken)
    
    suggested_action = None
    idx = 0
    while self.stack:
        idx += 1
        if idx > 1000:
          import ipdb; ipdb.set_trace()
          raise RuntimeError(f"replan taking too long {idx}")
        subgoal = self.stack[-1]
        suggested_action = subgoal.replan_before_action()
        # If is not clear what can be done for the current subgoal
        # (because it is completed, because there is blocker,
        # or because exploration is required), keep replanning
        if suggested_action is not None:
            break
    if not self.stack:
        suggested_action = self.mission.unwrapped.actions.done

    self._remember_current_state()

    return suggested_action

  def _check_erroneous_box_opening(self, action):
    # ignore this
    pass

class GotoOptionsWrapper(Wrapper):
    """
    Wrapper that swaps actions for options.
    Primitive actions: {left, right, forward, pickup, drop}
    options: goto x

    for each object, get: (location, type, color, state)
    - given in N x 4 matrix where N is over objects.
    """

    def __init__(self,
                 env,
                 max_options: int = 1,
                 use_options: bool = True,
                 partial_obs: bool = True):
        """
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.prior_action = None
        self.use_options = use_options
        self.primitive_actions = [
          self.actions.left,
          self.actions.right,
          self.actions.forward,
          self.actions.pickup,
          self.actions.drop,
          self.actions.toggle,
          self.actions.done,  # does nothing
        ]
        self.prior_objects = None
        self.prior_visible_objects = None
        self.prior_object_mask = None

        max_options = max(max_options, len(self.unwrapped.env_objects))

        self.object_types = [tuple(o) for o in self.unwrapped.env_objects]
        self.object_types.sort()
        self.object2idx = { o : i for i, o in  enumerate(self.object_types)}

        self.primitive_actions_arr = np.array(
           [int(a) for a in self.primitive_actions], dtype=np.uint8)
        nactions = len(self.primitive_actions_arr)
        self.max_options = max_options

        if not partial_obs:
            raise NotImplementedError
        else:
            self.num_cols = self.agent_view_size

        actions_space = spaces.Box(
            low=0, high=255, # doesn't matter
            shape=(len(self.primitive_actions),),  # number of cells
            dtype="uint8",
        )
        self.max_dims_per_feature = 100
        nfeatures = 5  # x-pos, y-pos, color, type, state
        objects_space = spaces.Box(
            low=0, high=255, # doesn't matter
            shape=(max_options, self.max_dims_per_feature*nfeatures),  # number of cells
            dtype="uint8",
        )
        objects_mask_space = spaces.Box(
            low=0, high=255, # doesn't matter
            shape=(max_options,),  # number of cells
            dtype="uint8",
        )
        action_mask_space = spaces.Box(
            low=0, high=255, # doesn't matter
            shape=(nactions+max_options,),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
             "actions": actions_space,
             "objects": objects_space,
             "objects_mask": objects_mask_space,
             "action_mask": action_mask_space,
             }
        )

    def get_visible_objects(self):
        grid, _ = self.gen_obs_grid()

        visible_objects = []
        types_ignore = ['wall', 'unseen', 'empty', 'floor']
        for idx, object in enumerate(grid.grid):
          if object is None: continue
          if object.type in types_ignore: continue
          type, color, state = object.encode()
          obj = GlobalObjDesc(
            type=type,
            color=color,
            state=state,
            category=(IDX_TO_COLOR[color], IDX_TO_OBJECT[type]),
            global_pos=object.cur_pos,
            local_pos=flat_to_matrix(
               idx, self.num_cols),
          )
          visible_objects.append(obj)

        return visible_objects

    def post_env_iter_update(self, obs, info):
      """Update after every env.reset() or env.step().

      This will update the observation and env with relevant object information."""
      #############
      # Get visible objects
      #############
      objects = self.get_visible_objects()
      nobjects = len(objects)

      #############
      # Create matrix with object information
      #############
      def make_onehot(idx):
        x = np.zeros(self.max_dims_per_feature, dtype=np.uint8)
        x[idx] = 1
        return x
      object_info = np.zeros(
         (self.max_options, 5*self.max_dims_per_feature),
         dtype=np.uint8)

      object_mask = np.zeros((self.max_options), dtype=np.uint8)

      for _, obj in enumerate(objects):
        color, type = obj.category
        key = (type, color)
        idx = self.object2idx[key]
        object_mask[idx] = 1
        object_info[idx] = np.concatenate(
           (make_onehot(obj.local_pos[0]),
            make_onehot(obj.local_pos[1]),
            make_onehot(obj.type),
            make_onehot(obj.color),
            make_onehot(obj.state)))

      #############
      # Update observation
      #############
      obs['actions'] = np.array(self.primitive_actions_arr)
      obs['objects'] = np.array(object_info)
      obs['objects_mask'] = np.array(object_mask)
      obs['action_mask'] = np.concatenate(
         (np.ones(len(self.primitive_actions_arr)), object_mask)).astype(np.uint8)

      # info['nactions'] = len(self.primitive_actions_arr) + nobjects
      # info['nobjects'] = nobjects

      # info['actions'] = self.primitive_actions + [
      #     f'go to {IDX_TO_COLOR[o.color]} {IDX_TO_OBJECT[o.type]}' for o in objects
      # ]
      # info['actions'] = {idx: action for idx, action in enumerate(info['actions'])}
      #############
      # Update environment variables
      #############
      del self.prior_objects
      del self.prior_visible_objects
      del self.prior_object_mask

      self.prior_objects = objects
      self.prior_visible_objects = objects
      self.prior_object_mask = object_mask

    def reset(self, *args, **kwargs):
      self.prior_action = None
      obs, info = self.env.reset(*args, **kwargs)

      self.post_env_iter_update(obs, info)
      return obs, info

    def execute_option(self, action):
        option_idx = action - len(self.primitive_actions)

        # obj = self.prior_visible_objects[option_idx]
        obj = self.object_types[option_idx]

        shape, color = obj
        def match(x):
           return x.category == (color, shape)

        # print(f"EXECUTING OPTION {option_idx}: go to", color, shape)

        match = [o for o in self.prior_visible_objects if match(o)]
        assert len(match) == 1, f'mismatches: {match}'
        match = match[0]

        # if position in front is already goal position, do nothing
        front_pos = self.unwrapped.front_pos
        global_pos = match.global_pos
        if front_pos[0] == global_pos[0] and front_pos[1] == global_pos[1]:
          # don't update prior primitive action
          # self.prior_action = action
          return self.env.step(self.actions.done)

        if self.env.carrying:
          if self.env.carrying.color == color and self.env.carrying.type == shape:
            return self.env.step(self.actions.done)
        # otherwise, generate a trajectory to the goal position

        bot = GotoBot(self.env, match.global_pos)
        actions, obss, rewards, truncateds, dones, infos = bot.generate_trajectory()

        # not doable
        if len(actions) == 0:
           return self.env.step(self.actions.done)

        self.prior_action = actions[-1]

        return (obss[-1],
                sum(rewards),
                sum(dones) > 0,
                sum(truncateds) > 0,
                infos[-1])

    def step(self, action, *args, **kwargs):
        """Steps through the environment with `action`."""

        if action in self.primitive_actions:
          obs, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)
          self.prior_action = action
        else:
          if self.use_options:
            assert len(self.prior_visible_objects), "impossible"
            obs, reward, terminated, truncated, info = self.execute_option(action)
          else:
            # ignore the action
            obs, reward, terminated, truncated, info = self.env.step(self.actions.done, *args, **kwargs)
            self.prior_action = action

        self.post_env_iter_update(obs, info)
        return obs, reward, terminated, truncated, info

def main():
  from projects.human_sf.key_room_v3 import KeyRoom
  import minigrid
  import random
  from pprint import pprint
  import json
  json_file = 'projects/human_sf/maze_pairs.json'
  with open(json_file, 'r') as file:
      maze_configs = json.load(file)

  env = KeyRoom(maze_config=maze_configs[0])
  env = minigrid.wrappers.DictObservationSpaceWrapper(env)
  env = GotoOptionsWrapper(env)
  env = minigrid.wrappers.RGBImgObsWrapper(env, tile_size=12)



  for idx in range(500):
    print(idx)
    obs, info = env.reset()
    for t in range(100):
        mask = obs['action_mask']
        action = np.random.multinomial(1, pvals=mask/mask.sum(), size=1)
        action = action[0].argmax(-1)
        obs, reward, done, truncated, info = env.step(action)

if __name__ == "__main__":
  main()
