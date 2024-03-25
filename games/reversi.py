import datetime
import pathlib

import math
import numpy
import torch
import copy
import random

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (3, 8, 8)  # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(8 * 8))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 2  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 2  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 65  # Maximum number of moves if game is not finished before
        self.num_simulations = 400  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 65   # Number of game moves to keep for every batch element
        self.td_steps = 65   # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Reversi()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action
    
    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)


class Reversi:
    def __init__(self):
        self.board_size = 8
        self.player = 1
        self.board_size_range = [4, 6, 8]
        self.board = self.set_board()
        self._pass = self.board_size * self.board_size
    
    def set_board(self):
        """
        generate board
        -----
        @return
            new_board (int array)
        """
        def get_board_size():
            """
            @return
                (int)
            """
            return random.choice(self.board_size_range)
        
        def set_center_koma(active_board):
            """ 
            Place white and black in the center 4 squares of the board
            -----
            """
            base_y = math.floor(self.height / 2)
            base_x = math.floor(self.width / 2)

            active_board[base_y-1][base_x-1] = 1
            active_board[base_y][base_x] = 1
            active_board[base_y-1][base_x] = -1
            active_board[base_y][base_x-1] = -1
        

        self.height = get_board_size()
        self.width = get_board_size()

        all_board = numpy.full((self.board_size, self.board_size), 999, dtype="int32")
        active_board = numpy.zeros((self.height, self.width), dtype="int32")
        set_center_koma(active_board)
        
        w1, h1 = all_board.shape
        w2, h2 = active_board.shape
        w, h = (w1-w2)//2, (h1-h2)//2
        new_board = copy.deepcopy(all_board)
        new_board[w:(-w if w else None), h:(-h if h else None)] = active_board

        return new_board

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = self.set_board()
        self.player = 1
        return self.get_observation()
    
    def step(self, action):
        if action != self._pass:
            y = math.floor(action / self.board_size)
            x = action % self.board_size
            reverse = self.is_legal_action_xy(y, x)[1]
            self.board[y][x] = self.player
            for r in reverse:
                ry = math.floor(r / self.board_size)
                rx = r % self.board_size
                self.board[ry][rx] = self.player
        
        done = self.is_finished()
        reward = 1 if done else 0
        self.player *= -1
        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == -1, 0.0, self.board)
        board_cp = numpy.where(self.board == 1, 0.0, self.board)
        board_player2 = numpy.where(board_cp == -1, 1.0, board_cp)
        board_to_play = numpy.full((self.board_size, self.board_size), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        """
        return legal move list
        -----
        @return
            (int array): List of legal moves, return _pass if there is no legal move (to pass)
        """
        legal = []
        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.is_legal_action_xy(y, x)[0]:
                    legal.append(y * self.board_size + x)
        
        if legal == []:
            return [self._pass]
        return legal

    def is_legal_action_xy(self, y, x):
        """
        Returns an array that indicates whether or not the input value is a legal move and flips it over.
        -----
        @param
            y (int): 1D element number of board
            x (int): 2D element number of board
        @return
            True or False: Whether or not there is a legal hand
            flip (int array): flip list
        """
        search_args_y = [0, 0, -1, 1, -1, 1, -1, 1]
        search_args_x = [-1, 1, 0, 0, -1, -1, 1, 1]
        flip = []

        if self.board[y][x] == 0:   # Can input values be placed on the board?
            for dy, dx in zip(search_args_y, search_args_x):
                py, px = y, x       # Current search position
                reverse_list = []
                while True:
                    py, px = py+dy, px+dx
                    if (py < 0) or (self.board_size-1 < py) or (px < 0) or (self.board_size-1 < px):
                        break
                    p_state = self.board[py][px]
                    if (p_state == 0) or (p_state == 999):
                        break
                    elif (p_state == self.player):
                        if reverse_list == []:
                            break
                        else:
                            flip.extend(reverse_list)
                            break
                    else:
                        reverse_list.append(py * self.board_size + px)

        if flip != []:
            return True, flip
        return False, []

    def is_finished(self):
        """
        Exit conditions
        Ends if there are no legal moves twice in a row
        -----
        @return
            True or False: End / Continue
        """
        legal_action = []
        for _ in range(2):
            legal_action.extend(self.legal_actions())
            self.player *= -1
        if legal_action[0] == self._pass and legal_action[1] == self._pass:
            return True
        return False

    def render(self):
        """ indicate """
        print("○" if self.player == 1 else "●", "'s Turn")
        marker = "  "
        rs, cs = math.floor((self.board_size-self.height)/2), math.floor((self.board_size-self.width)/2)
        re, ce = rs + self.height, cs + self.width
        for i in range(self.width):
            marker = marker + str(i) + " "
        print(marker)
        for row in range(rs, re):
            print(str(row-rs), end=" ")
            for col in range(cs, ce):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("○", end=" ")
                elif ch == -1:
                    print("●", end=" ")
            print()

    def human_input_to_action(self):
        """
        people make behavioral choices
        -----
        @return
            True or False: Input value correct/incorrect
            action (int): Action value, returns _pass if it is a pass
        """
        human_input = input("Enter an action. [yx] :")
        if (
            len(human_input) == 2
            and int(human_input[0]) in range(self.height)
            and int(human_input[1]) in range(self.width)
        ):
            y = int(human_input[0]) + math.floor((self.board_size - self.height) / 2)
            x = int(human_input[1]) + math.floor((self.board_size - self.width) / 2)
            if self.is_legal_action_xy(y, x)[0]:
                return True, y*self.board_size+x

        if sum(self.legal_actions()) == self._pass: # If the input value is inappropriate, search for legal moves and determine the pass
            return True, self._pass
        return False, -1

    def expert_action(self):
        return numpy.random.choice(self.legal_actions())

    def action_to_human_input(self, action):
        if action == self._pass:
            return "pass"
            
        y = math.floor(action / self.board_size)
        x = action % self.board_size
        y -= math.floor((self.board_size-self.height)/2)
        x -= math.floor((self.board_size-self.width)/2)
        return str(y)+str(x)