import numpy as np

np.random.seed(0)

# Toy task: given a binary sequence, predict if sum >= T/2
# Sequence length T, hidden size H.
T = 6
H = 16
lr = 0.1
epochs = 200
batch_size = 32
steps_per_epoch = 50

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_batch(batch_size, T):
    # X shape: (batch_size, T), each element is 0/1
    X = np.random.randint(0, 2, size=(batch_size, T)).astype(np.float32)
    # y shape: (batch_size,), binary label based on sequence sum
    y = (X.sum(axis=1) >= (T / 2)).astype(np.float32)
    return X, y

# Parameters (RNN with scalar input at each time step)
# Wx: input -> hidden, Wh: hidden -> hidden, Wy: hidden -> output
Wx = np.random.randn(H).astype(np.float32) * 0.1
Wh = np.random.randn(H, H).astype(np.float32) * 0.1
bh = np.zeros(H, dtype=np.float32)
Wy = np.random.randn(H).astype(np.float32) * 0.1
by = np.float32(0.0)

def forward(seq):
    # hs stores hidden states; hs[0] is h0
    hs = [np.zeros(H, dtype=np.float32)]
    for x in seq:
        h_prev = hs[-1]
        # x is a scalar (0/1), so Wx * x is elementwise scaling
        h = np.tanh(Wx * x + Wh @ h_prev + bh)
        hs.append(h)
    # Use final hidden state for classification
    y_logit = Wy @ hs[-1] + by
    y_hat = sigmoid(y_logit)
    cache = (seq, hs)  # cache for BPTT
    return y_hat, cache

def backward(y, y_hat, cache):
    # Backprop through time (BPTT) for a single sequence
    seq, hs = cache
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbh = np.zeros_like(bh)
    dWy = np.zeros_like(Wy)
    dby = np.float32(0.0)

    # dL/dy_logit for sigmoid + BCE
    dy = y_hat - y
    dWy += dy * hs[-1]
    dby += dy

    # Backprop into last hidden state
    dh = dy * Wy
    for t in reversed(range(T)):
        h = hs[t + 1]
        h_prev = hs[t]
        # tanh' = 1 - h^2
        dt = (1.0 - h * h) * dh
        dbh += dt
        dWx += dt * seq[t]
        dWh += np.outer(dt, h_prev)
        # Propagate gradient to previous hidden state
        dh = Wh.T @ dt

    return dWx, dWh, dbh, dWy, dby

for epoch in range(1, epochs + 1):
    losses = []
    for _ in range(steps_per_epoch):
        X, y = generate_batch(batch_size, T)
        for i in range(batch_size):
            # Per-sample SGD for clarity (not the most efficient)
            y_hat, cache = forward(X[i])
            # Binary cross-entropy loss
            loss = -(y[i] * np.log(y_hat + 1e-8) + (1 - y[i]) * np.log(1 - y_hat + 1e-8))
            dWx, dWh, dbh, dWy, dby = backward(y[i], y_hat, cache)

            # SGD update
            Wx -= lr * dWx
            Wh -= lr * dWh
            bh -= lr * dbh
            Wy -= lr * dWy
            by -= lr * dby

            losses.append(loss)

    if epoch % 20 == 0:
        print(f"epoch {epoch:03d} | loss {np.mean(losses):.4f}")

# Quick sanity check
X_test, y_test = generate_batch(10, T)
for i in range(3):
    y_hat, _ = forward(X_test[i])
    print(f"seq={X_test[i].astype(int)} sum={int(X_test[i].sum())} pred={float(y_hat):.3f} label={int(y_test[i])}")
