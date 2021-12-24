import random
import numpy as np

class ReplayBUffer:
    def __init__(self, args):
        random.seed(args.seed)
        self.size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.buffer = []
        self.current_idx = 0

    def remember(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.current_idx] = (state, action, reward, next_state, done)
        self.current_idx = (self.current_idx + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)