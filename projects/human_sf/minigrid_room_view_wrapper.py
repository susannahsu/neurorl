import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX

def one_hot(index, n):
    if index >= n or index < 0:
        raise ValueError("index must be within the range 0 to n-1")
    
    # Initialize a vector of zeros with length n
    vector = np.zeros(n)
    
    # Set the element at the specified index to 1
    vector[index] = 1
    
    return vector
class RGBImgRoomObsWrapper(ObservationWrapper):
    """
    Wrapper to display only the current room the agent is in as an RGB image.
    This can be useful for tasks requiring focus on local room-level features,
    without the distraction of other parts of the environment.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)
        self.tile_size = tile_size

        # Assuming a maximum room size to define the observation space.
        # You might need a more dynamic approach depending on the environment.
        example_room = env.get_room(0, 0)
        # Adjust based on your environment's specifications
        max_room_width = example_room.size[0]
        # Adjust based on your environment's specifications
        max_room_height = example_room.size[1]
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                max_room_width * tile_size,
                max_room_height * tile_size,
                3,
            ),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        grid, agent_pos = self.gen_room_obs_grid()
        img = grid.render(
            self.tile_size,
            agent_pos,
            self.agent_dir,
        )

        return {**obs, "image": img}



    def gen_room_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """
        agent_view_size = agent_view_size or self.agent_view_size

        room = self.room_from_pos(*self.agent_pos)
        topX, topY = room.top
        grid = self.grid.slice(topX, topY, room.size[0], room.size[1])

        agent_pos = (self.agent_pos[0] - topX, self.agent_pos[1] - topY)

        # grid.set(*agent_pos, self.carrying)
        # if self.carrying:
        # else:
        # grid.set(*agent_pos, None)

        return grid, agent_pos


class OneHotRoomObsWrapper(ObservationWrapper):
    """
    Wrapper to display only the current room the agent is in as an RGB image.
    This can be useful for tasks requiring focus on local room-level features,
    without the distraction of other parts of the environment.
    """

    def __init__(self, env, tile_size=8, include_images=True):
        super().__init__(env)
        self.tile_size = tile_size
        self.include_images = include_images

        new_spaces = {}
        ###########################
        # Images
        ###########################
        if include_images:
          # Assuming a maximum room size to define the observation space.
          # You might need a more dynamic approach depending on the environment.
          example_room = env.get_room(0, 0)
          # Adjust based on your environment's specifications
          max_room_width = example_room.size[0]
          # Adjust based on your environment's specifications
          max_room_height = example_room.size[1]
          new_image_space = spaces.Box(
              low=0,
              high=255,
              shape=(
                  max_room_width * tile_size,
                  max_room_height * tile_size,
                  3,
              ),
              dtype="uint8",
          )
          new_spaces.update({"image": new_image_space})

        ###########################
        # Symbols
        ###########################
        obs_shape = env.observation_space["image"].shape
        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        symbol_space = spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        new_spaces.update({
          "symbols": symbol_space,
          "direction": spaces.Box(low=0, high=255, shape=(5,), dtype="uint8")
        })

        ###########################
        # Obs Space
        ###########################
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
             **new_spaces,}
        )

    def observation(self, obs):
        #-------------------
        # generate symbolic grid of room
        #-------------------
        room = self.room_from_pos(*self.agent_pos)
        topX, topY = room.top
        grid = self.grid.slice(topX, topY, room.size[0], room.size[1])
        img = grid.encode()

        # -------------------
        # generate 1-hot rep
        # -------------------
        out = np.zeros(
            self.observation_space.spaces["symbols"].shape,
            dtype="uint8")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        updates = {
          "symbols": out,
          "direction": one_hot(obs['direction'], n=5).astype(np.uint8),
        }
        if self.include_images:
          agent_pos = (self.agent_pos[0] - topX, self.agent_pos[1] - topY)
          updates['image'] = grid.render(
              self.tile_size,
              agent_pos,
              self.agent_dir,
          )
          
        return {
            **obs,
            **updates}

