import numpy as np


def batch_norm_forward(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    momentum=0.9,
    eps=1e-5,
    training=True,
):
    if training:
        # Batch statistics for current mini-batch.
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        # Normalize using batch stats, then apply scale and shift.
        x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
        out = gamma * x_hat + beta
        # Update running stats for inference.
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_var
    else:
        # Use running stats at inference time.
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    return out, running_mean, running_var


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(4, 3)
    gamma = np.ones(3)
    beta = np.zeros(3)
    running_mean = np.zeros(3)
    running_var = np.ones(3)

    # Training forward pass updates running stats.
    y, running_mean, running_var = batch_norm_forward(
        x, gamma, beta, running_mean, running_var, training=True
    )
    # Inference uses the updated running stats.
    y_infer, _, _ = batch_norm_forward(
        x, gamma, beta, running_mean, running_var, training=False
    )
    print("train:", y)
    print("infer:", y_infer)
