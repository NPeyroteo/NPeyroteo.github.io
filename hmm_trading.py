import numpy as np
import pandas as pd


def forward(obs, a, b, pi):
    """Forward algorithm."""
    T = len(obs)
    M = a.shape[0]
    alpha = np.zeros((T, M))
    alpha[0] = pi * b[:, obs[0]]
    for t in range(1, T):
        for j in range(M):
            alpha[t, j] = np.dot(alpha[t - 1], a[:, j]) * b[j, obs[t]]
    return alpha


def backward(obs, a, b):
    """Backward algorithm."""
    T = len(obs)
    M = a.shape[0]
    beta = np.zeros((T, M))
    beta[T - 1] = 1
    for t in range(T - 2, -1, -1):
        for j in range(M):
            beta[t, j] = np.dot(beta[t + 1] * b[:, obs[t + 1]], a[j])
    return beta


def baum_welch(obs, a, b, pi, n_iter=100):
    """Baum-Welch algorithm to estimate HMM parameters."""
    M = a.shape[0]
    T = len(obs)

    for _ in range(n_iter):
        alpha = forward(obs, a, b, pi)
        beta = backward(obs, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denom = np.dot(np.dot(alpha[t], a) * b[:, obs[t + 1]], beta[t + 1])
            for i in range(M):
                numer = alpha[t, i] * a[i] * b[:, obs[t + 1]] * beta[t + 1]
                xi[i, :, t] = numer / denom

        gamma = xi.sum(axis=1)
        a = xi.sum(2) / gamma.sum(axis=1, keepdims=True)

        gamma = np.hstack((gamma, xi[:, :, -1].sum(axis=0, keepdims=True).T))
        denom = gamma.sum(axis=1)
        for k in range(b.shape[1]):
            b[:, k] = gamma[:, obs == k].sum(axis=1)
        b = b / denom[:, None]
    return a, b


def viterbi(obs, a, b, pi):
    """Viterbi algorithm returning most likely hidden state sequence."""
    T = len(obs)
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0] = np.log(pi * b[:, obs[0]])
    prev = np.zeros((T - 1, M), dtype=int)

    for t in range(1, T):
        for j in range(M):
            prob = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, obs[t]])
            prev[t - 1, j] = np.argmax(prob)
            omega[t, j] = np.max(prob)

    S = np.zeros(T, dtype=int)
    S[-1] = np.argmax(omega[-1])
    for t in range(T - 2, -1, -1):
        S[t] = prev[t, S[t + 1]]
    return S


def main():
    data = pd.read_csv('data_python.csv')
    V = data['Visible'].to_numpy()

    # Two-state HMM
    a = np.full((2, 2), 0.5)
    b = np.array([[1, 3, 5], [2, 4, 6]], dtype=float)
    b /= b.sum(axis=1, keepdims=True)
    pi = np.array([0.5, 0.5])

    a, b = baum_welch(V, a, b, pi)
    hidden_states = viterbi(V, a, b, pi)

    print('Estimated transition matrix:\n', a)
    print('Estimated emission matrix:\n', b)
    print('Most likely hidden states:')
    print(hidden_states)


if __name__ == '__main__':
    main()
