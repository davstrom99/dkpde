import pyroomacoustics as pa
import torch
import matplotlib.pyplot as plt
import numpy as np

class SimulatedDatasetAbstract(torch.utils.data.Dataset):
    def __init__(self, room_dim = [5,4,3], fs = 8000, source_pos = [3,3,1.5],source_signal = "white", array_size = 100,N_time = 10,mic_pos_center = [1,1,1.5], 
                 mic_pos_side = 0.5,rt60 = 0.4, frequency = 500,SNR = 20,dtype=torch.float32,device = 'cpu',cutoff_low = 50,cutoff_high = 1000, seed = 0):
        # Load parameters from config
        self.room_dim = room_dim # Room dimensions
        self.fs = fs # Sampling frequency
        self.source_pos = source_pos # Source position
        self.mic_pos_center = mic_pos_center # Center of microphone positions
        self.mic_pos_side = mic_pos_side  # Radius of microphone positions
        self.array_size = array_size # Number of microphones
        self.rt60 = rt60 # Reverberation time
        self.frequency = frequency
        self.SNR = SNR
        self.dtype = dtype
        self.device = device
        self.N_time = N_time
        self.source_signal = source_signal
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high

        self.mic_positions = self.get_microphone_positions(self.array_size, seed = seed) 
        
        self.room = self.generate_room()
        self.RIRs,self.signals,self.space_time_grid,dump1,dump2 = self.generate_data(self.room, self.mic_positions, self.N_time)
        self.space_time_grid = torch.tensor(self.space_time_grid, dtype = self.dtype,device = self.device)
        self.signals = torch.tensor(self.signals, dtype = self.dtype,device = self.device)  

        self.Nh = self.RIRs.shape[0] 
        self.mic_positions = torch.tensor(self.mic_positions, dtype = self.dtype,device = self.device)

    def __len__(self):
        return self.array_size*self.N_time 

    def __getitem__(self, idx):
        return (self.space_time_grid[idx,:],self.signals[idx])
    

    def get_microphone_positions(self,N):
        raise NotImplementedError("Abstract method - implement in subclass")

    def generate_data(self,room,mic_positions,N_time,Nh = None):
        """
        Generates signals for each microphone in the room.

        Returns:
            List of signals and RIRs for each microphone position.
        """
        array_size = mic_positions.shape[0]
        
        room.add_microphone_array(pa.MicrophoneArray(mic_positions.T, self.fs))
        room.compute_rir()

        if Nh is None:
            shortest_rir_length = min(room.rir[i][0].shape[0] for i in range(array_size))-30
        else:
            shortest_rir_length = Nh

        
        RIRs = np.zeros((shortest_rir_length, array_size))  # Initialize with the correct shape
        for i in range(array_size):
            RIRs[:, i] = room.rir[i][0][0:shortest_rir_length]
            room.rir[i][0] = room.rir[i][0][0:shortest_rir_length]

        room.simulate(snr = self.SNR)  # Add white Gaussian noise

        # Retreive a snapshot of the data
        signals = room.mic_array.signals[:,shortest_rir_length:-shortest_rir_length]
        signals = signals[:,signals.shape[0]//2:(signals.shape[0]//2+N_time)]  # Take a sequence from middle snapshot
        samples = range(signals.shape[1])

        # Check if self.mic_positions is a tensor, then convert to numpy
        if isinstance(mic_positions, torch.Tensor):
            mic_positions = mic_positions.cpu().numpy()

        # Flatten the signals and the grid
        signals_flat = np.zeros(signals.shape[0]*signals.shape[1])
        grid_flat = np.zeros((signals.shape[0]*signals.shape[1],4))
        grid = np.zeros((signals.shape[0],signals.shape[1],4))
        for m in range(signals.shape[0]):
            for n in range(signals.shape[1]):
                signals_flat[m*signals.shape[1]+n] = signals[m,n]
                grid_flat[m*signals.shape[1]+n,0:3] = mic_positions[m,:]
                grid_flat[m*signals.shape[1]+n,3] = samples[n]/self.fs
                grid[m,n,0:3] = mic_positions[m,:]
                grid[m,n,3] = samples[n]/self.fs


        return RIRs, signals_flat, grid_flat, signals, grid
    
    def generate_room(self):
        """
        Generates a room with a source.

        Returns:
            Room object.
        """

        signal = self.get_source_signal()

        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pa.inverse_sabine(self.rt60, self.room_dim)

        # Create the room
        room = pa.ShoeBox(self.room_dim, fs=self.fs, materials=pa.Material(e_absorption), max_order=max_order,
                          use_rand_ism = False, max_rand_disp = 0.05)
        room.add_source(self.source_pos, signal = signal) 
        return room
    
    def get_source_signal(self):
        duration = 2 # Seconds. We only need a short signal for the training
        if self.source_signal == "white":
            from scipy.signal import butter, lfilter       

            # Generate a bandpass white noise signal     
            signal = np.random.normal(0,1,self.fs*duration)
            cutoff_low = self.cutoff_low
            cutoff_high = self.cutoff_high
            N  = 5
            b,a = butter(N,[cutoff_low, cutoff_high], fs = self.fs,btype='bandpass')
            signal = lfilter(b,a,signal)

            # Normalize signal
            signal = signal/10

        elif self.source_signal == "sine":
            amplitude = 1 
            time = np.arange(self.fs * duration) / self.fs
            signal = amplitude * np.sin(2 * np.pi * self.frequency * time)
        else:
            ValueError(f"Source signal {self.source_signal} not supported")     
        return signal

    def plot_geometry(self):
        # Plot room and microphone positions
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot shoebox room 
        x_dim, y_dim, z_dim = self.room_dim
        # Lines for the walls
        ax.plot([0, x_dim, x_dim, 0, 0], [0, 0, y_dim, y_dim, 0], [0, 0, 0, 0, 0], c='gray')  # Floor
        ax.plot([0, x_dim, x_dim, 0, 0], [0, 0, y_dim, y_dim, 0], [z_dim, z_dim, z_dim, z_dim, z_dim], c='gray') # Ceiling
        ax.plot([0, 0], [0, 0], [0, z_dim], c='gray')  # Wall 1
        ax.plot([x_dim, x_dim], [0, 0], [0, z_dim], c='gray')  # Wall 2
        ax.plot([0, 0], [y_dim, y_dim], [0, z_dim], c='gray')  # Wall 3   
        ax.plot([x_dim, x_dim], [y_dim, y_dim], [0, z_dim], c='gray')  # Wall 4        

        # plot source position
        ax.scatter(self.source_pos[0], self.source_pos[1], self.source_pos[2], c='r',s = 5, marker='x', label='Source')

        # plot microphone positions
        for i in range(self.array_size):
            mic_pos = self.mic_positions[i,:].cpu()
            print(mic_pos[2].item())
            ax.scatter(mic_pos[0].item(), mic_pos[1].item(), mic_pos[2].item(), c='b', marker='o',s = 1, label=f'Microphone' if i == 0 else "")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        return fig

    def plot_RIRs(self, num_rirs = 3):
        # Plot RIRs
        plt.figure()
        for i in range(num_rirs):
            plt.plot(self.RIRs[:, i], label=f'mic {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
