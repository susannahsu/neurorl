import numpy as np
from gymnasium import spaces

import minigrid

class DictObservationSpaceWrapper(minigrid.wrappers.DictObservationSpaceWrapper):

  def __init__(self, env, max_words_in_mission=50, word_dict=None):
      """
      Main change is to KEEP prior pieces of observation space and only over-ride the mission one.
      """
      super(minigrid.wrappers.DictObservationSpaceWrapper, self).__init__(env)

      if word_dict is None:
        word_dict = self.get_minigrid_words()

      self.max_words_in_mission = max_words_in_mission
      self.word_dict = word_dict

      self.observation_space = spaces.Dict(
          {
             **self.observation_space.spaces,
             "mission": spaces.MultiDiscrete(
                [len(self.word_dict.keys())] * max_words_in_mission
            ),
          }
      )
  def observation(self, obs):
    obs["mission"] = self.string_to_indices(obs["mission"])
    assert len(obs["mission"]) < self.max_words_in_mission
    obs["mission"] += [0] * (self.max_words_in_mission - len(obs["mission"]))
    obs["mission"] = np.array(obs["mission"])
    return obs