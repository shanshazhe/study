import numpy as np

class SimpleRNN:
    """
    A simple character-level RNN implementation using NumPy
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        
        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
    def forward(self, inputs, h_prev):
        """
        Forward pass through the RNN
        inputs: list of input vectors (one-hot encoded)
        h_prev: previous hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        # Forward pass
        for t in range(len(inputs)):
            xs[t] = inputs[t]
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # softmax
            
        return xs, hs, ps
    
    def backward(self, xs, hs, ps, targets):
        """
        Backward pass (BPTT - Backpropagation Through Time)
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        
        loss = 0
        
        # Backward pass
        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh
            
            dbh += dh_raw
            dWxh += np.dot(dh_raw, xs[t].T)
            dWhh += np.dot(dh_raw, hs[t-1].T)
            
            dh_next = np.dot(self.Whh.T, dh_raw)
            
            loss += -np.log(ps[t][targets[t], 0])
        
        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(xs)-1]
    
    def update_params(self, dWxh, dWhh, dWhy, dbh, dby):
        """
        Update parameters using gradient descent
        """
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
    
    def sample(self, h, seed_ix, n):
        """
        Sample a sequence from the model
        """
        x = np.zeros((self.Wxh.shape[1], 1))
        x[seed_ix] = 1
        indices = []
        
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.Wxh.shape[1]), p=p.ravel())
            x = np.zeros((self.Wxh.shape[1], 1))
            x[ix] = 1
            indices.append(ix)
        
        return indices


def train_rnn_example():
    """
    Example: Train RNN on a simple text sequence
    """
    # Sample data
    data = "hello world hello deep learning"
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    
    print(f"Data has {data_size} characters, {vocab_size} unique.")
    
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Hyperparameters
    hidden_size = 100
    seq_length = 10
    learning_rate = 0.1
    
    # Initialize RNN
    rnn = SimpleRNN(vocab_size, hidden_size, vocab_size, learning_rate)
    
    n, p = 0, 0
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length
    
    # Training loop
    for iteration in range(5000):
        # Prepare inputs and targets
        if p + seq_length + 1 >= len(data) or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            p = 0
        
        inputs = [np.zeros((vocab_size, 1)) for _ in range(seq_length)]
        targets = []
        
        for t in range(seq_length):
            inputs[t][char_to_ix[data[p + t]]] = 1
            targets.append(char_to_ix[data[p + t + 1]])
        
        # Forward pass
        xs, hs, ps = rnn.forward(inputs, h_prev)
        
        # Backward pass
        loss, dWxh, dWhh, dWhy, dbh, dby, h_prev = rnn.backward(xs, hs, ps, targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
        # Update parameters
        rnn.update_params(dWxh, dWhh, dWhy, dbh, dby)
        
        # Sample from the model periodically
        if n % 500 == 0:
            print(f"\nIteration {n}, loss: {smooth_loss:.4f}")
            sample_ix = rnn.sample(h_prev, char_to_ix[data[p]], 50)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f"Sample: {txt}")
        1
        p += seq_length
        n += 1
    
    print("\nTraining complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("=" * 50)
    print("Simple RNN Implementation with NumPy")
    print("=" * 50)
    
    train_rnn_example()
