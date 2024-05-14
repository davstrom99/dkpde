import pyroomacoustics as pa
import torch
from .SimulatedDatasetAbstract import SimulatedDatasetAbstract
import matplotlib.pyplot as plt
import numpy as np

class SimulatedDataset3D(SimulatedDatasetAbstract):
    def get_microphone_positions(self,N, mic_pos_center = None, mic_pos_side= None, seed = None):
        """
        Generates a grid of equally spaced microphone positions in 3D space.

        Args:
            mic_pos_center: Center of the microphone grid (X, Y, Z).
            mic_pos_dimensions: Dimensions of the microphone grid (X, Y, Z).
            array_size: Number of microphones.

        Returns:
            List of microphone positions as tuples (X, Y, Z).
        """

        if mic_pos_center is None:
            mic_pos_center = self.mic_pos_center
        if mic_pos_side is None:
            mic_pos_side = self.mic_pos_side


        if seed is not None:
            # Temporary change the random seed (so that the signal still is randomized when creating a new instance of the dataset)
            state = np.random.get_state()
            np.random.seed(seed)
            mic_positions = np.random.rand(N, 3) * mic_pos_side + mic_pos_center - mic_pos_side / 2
            np.random.set_state(state)
        else:
            # Randomly generate microphone positions in a cube
            mic_positions = np.random.rand(N, 3) * mic_pos_side + mic_pos_center - mic_pos_side / 2

        return np.array(mic_positions)
    
    def get_collocation_points(self,N_collocation_time,N_collocation_space):
        """
        Generates a grid of randomly positioned  collocation points in 3D space and time.

        Args:
            mic_pos_center: Center of the microphone grid (X, Y, Z).
            mic_pos_dimensions: Dimensions of the microphone grid (X, Y, Z).
            array_size: Number of microphones.

        Returns:
            List of collocation points (X, Y, Z, T), i.e., a matrix of N_collocation_time*N_collocation_space rows and 4 columns.
        """

        mic_pos_center = self.mic_pos_center
        mic_pos_side = self.mic_pos_side    

        # Randomly generate microphone positions in a cube
        mic_positions = np.random.rand(N_collocation_space, 3) * mic_pos_side + mic_pos_center - mic_pos_side / 2

        time_grid = np.random.rand(N_collocation_time)*self.N_time/self.fs

        mic_positions = np.repeat(mic_positions, N_collocation_time, axis=0)
        collocation_points = np.concatenate((mic_positions, np.tile(time_grid, N_collocation_space).reshape(-1,1)), axis=1)

        return np.array(collocation_points)

    def get_collocation_points_fixed(self):
        """
        Generates a grid of collocation points which is the same as the training and validation grid.

        Args:
            mic_pos_center: Center of the microphone grid (X, Y, Z).
            mic_pos_dimensions: Dimensions of the microphone grid (X, Y, Z).
            array_size: Number of microphones.

        Returns:
            List of collocation points (X, Y, Z, T), i.e., a matrix of N_collocation_time*N_collocation_space rows and 4 columns.
        """

        mic_pos_center = self.mic_pos_center
        mic_pos_side = self.mic_pos_side    

        # Randomly generate microphone positions in a cube
        mic_positions = np.random.rand(N_collocation_space, 3) * mic_pos_side + mic_pos_center - mic_pos_side / 2

        time_grid = np.random.rand(N_collocation_time)*self.N_time/self.fs

        mic_positions = np.repeat(mic_positions, N_collocation_time, axis=0)
        collocation_points = np.concatenate((mic_positions, np.tile(time_grid, N_collocation_space).reshape(-1,1)), axis=1)

        return np.array(collocation_points)


