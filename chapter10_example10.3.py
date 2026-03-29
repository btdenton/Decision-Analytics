#Chapter 10, Example 10.3: Mountain car
import time
import numpy as np
import gymnasium as gym
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation

# Input Data
ENV_ID = "MountainCar-v0"
SEED = 0                    #random number generator seed
DATA_STEPS = 1000           # Total number of state-decision transitions
RBF_COMPONENTS = 30         # Number of radial basis functions
L2_REG = 1e-2               # keep
RBF_SIGMA = 1.0             # hyperparameter for RBFs
LSPI_MAX_ITERS = 100        # max allowable iterations for LSPI
LSPI_TOL = 1e-6             # stopping criterion for LSPI
DISCOUNT = 0.99             #discount factor
MAX_EPISODE_STEPS = 200
BEHAVIOR_RANDOM_PROB = 0.5  # 50% random, 50% heuristic "push with velocity"
EVAL_EPISODES = 5

# Animation settings
ANIMATE_EPISODE_SEED = 123
ANIMATE_MAX_STEPS = 200     # None uses env default
ANIMATE_INTERVAL_MS = 30     # lower=faster playback
GIF_PATH = "MountainCar_lspi_policy.gif" #animation file

#Define mapping from states to radial basis functions
def fit_feature_map(states: np.ndarray,rbf_components: int, seed: int):
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)
    rbf_gamma = 1/(2*RBF_SIGMA**2)
    nys = Nystroem(kernel="rbf",gamma=rbf_gamma,n_components=rbf_components,
                   random_state=seed)
    nys.fit(states_scaled)
    def phi_batch(S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        S_scaled = scaler.transform(S)
        return nys.transform(S_scaled)  # (N,d)
    def phi_single(s: np.ndarray) -> np.ndarray:
        return phi_batch(np.asarray(s, dtype=float).reshape(1, -1)).ravel()
    d = rbf_components
    return phi_single, phi_batch, d

#Convert state features into state-decision features
def build_decision_features(phi_s: np.ndarray, decisions: np.ndarray, 
                            n_decisions: int) -> np.ndarray:
    N, d = phi_s.shape
    D = n_decisions * d
    X = np.zeros((N, D), dtype=float)
    rows = np.arange(N)
    start = decisions * d  # starting column per row
    for k in range(d): # Fill X[rows, start + k] = phi_s[:,k] for k=0..d-1
        X[rows, start + k] = phi_s[:, k]
    return X

#Collect greedy decisions for many states at once
def greedy_decisions_batch(theta: np.ndarray, phi_s: np.ndarray, 
                           n_decisions: int) -> np.ndarray:
    N, d = phi_s.shape
    W = theta.reshape(n_decisions, d)        # (A,d)
    Q = phi_s @ W.T                          # (N,A)
    return np.argmax(Q, axis=1).astype(int)

#Collect decision when there is only a single state (during evaluation)
def greedy_decision(theta: np.ndarray,phi_s: np.ndarray,n_decisions: int) -> int:
    d = phi_s.shape[0]
    W = theta.reshape(n_decisions, d)
    q = W @ phi_s
    return int(np.argmax(q))

#Collect state transitions using random or heuristic decisions
def collect_transitions(env_id: str, n_steps: int, seed: int,
                        max_episode_steps:int, behavior_random_prob: float = 1.0):
    env = gym.make(env_id)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    transitions = []
    for t in range(n_steps):
        if rng.random() < behavior_random_prob:
            a = int(rng.integers(env.action_space.n))
        else:
            # Heuristic: push with velocity sign (0=left, 1=coast, 2=right)
            a = 2 if obs[1] > 0 else 0
        next_obs, r, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)
        transitions.append((obs.copy(), a, float(r), next_obs.copy(), done))
        if done: obs, info = env.reset(seed=seed + t + 1)
        else: obs = next_obs
    env.close()
    return transitions

# Least squares policy iteration (LSPI)
def lspi(transitions, phi_batch_fn, d: int, n_decisions: int,
                    discount: float, l2_reg: float, max_iters: int, tol: float):
    # Unpack transitions
    states  = np.array([tr[0] for tr in transitions], dtype=float)
    decisions = np.array([tr[1] for tr in transitions], dtype=int)
    rewards = np.array([tr[2] for tr in transitions], dtype=float)
    next_s  = np.array([tr[3] for tr in transitions], dtype=float)
    done    = np.array([tr[4] for tr in transitions], dtype=bool)
    # Precompute features
    phi_s  = phi_batch_fn(states)   
    phi_sp = phi_batch_fn(next_s)   
    X = build_decision_features(phi_s, decisions, n_decisions)  
    D = X.shape[1]
    b = X.T @ rewards  #     # Fixed b
    theta = np.zeros(D, dtype=float)
    eye = np.eye(D, dtype=float)

    for it in range(max_iters):
        ap = greedy_decisions_batch(theta, phi_sp, n_decisions)        # (N,)
        Y = build_decision_features(phi_sp, ap, n_decisions)           # (N,D)
        Y[done, :] = 0.0
        A = X.T @ (X - discount * Y)                               # (D,D)
        A_reg = A + l2_reg * eye
        theta_new = np.linalg.solve(A_reg, b)
        diff = np.linalg.norm(theta_new - theta)
        theta = theta_new
        print(f"LSPI iter {it + 1:2d}: ||Delta|| = {diff:.3e}")
        if diff < tol: break
    return theta

# Policy evaluation + animation of policy application
def evaluate_policy(env_id: str, theta: np.ndarray, phi_single_fn, n_decisions: int,
                    episodes: int, seed: int, max_episode_steps:int):
    env = gym.make(env_id)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            phi_s = phi_single_fn(obs)
            a = greedy_decision(theta, phi_s, n_decisions)
            obs, r, terminated, truncated, info = env.step(a)
            total += float(r)
            done = bool(terminated or truncated)
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))

#Run one episode with render_mode='rgb_array' and collect frames.
def rollout_frames(env_id: str, theta: np.ndarray, phi_single_fn, n_decisions: int,
                   seed: int, max_episode_steps: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    obs, info = env.reset(seed=seed)
    frames = [env.render()]  # initial frame
    done = False; total = 0.0; steps = 0
    while not done:
        phi_s = phi_single_fn(obs)
        a = greedy_decision(theta, phi_s, n_decisions)
        obs, r, terminated, truncated, info = env.step(a)
        total += float(r)
        done = bool(terminated or truncated)
        frames.append(env.render())
        steps += 1
    env.close()
    return frames, total, steps

def animate_frames(frames, interval_ms: int = 30, title: str = "", gif_path: str = "out.gif"):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    im = ax.imshow(frames[0])
    if title: ax.set_title(title)
    def update(i):
        im.set_data(frames[i])
        return (im,)
    animation.FuncAnimation(fig, update, frames=len(frames),interval=interval_ms, blit=True)
    plt.show()
    imageio.mimsave(gif_path, frames, duration=interval_ms / 1000.0)
    print(f"Saved GIF to: {gif_path}")
 
# Run LSPI for training, evaluation; generate animation GIF
env0 = gym.make(ENV_ID)
n_decisions = env0.action_space.n
env0.close()
print(f"Collecting {DATA_STEPS} transitions from {ENV_ID}...")
t0 = time.perf_counter()
transitions = collect_transitions(ENV_ID, DATA_STEPS, SEED,MAX_EPISODE_STEPS,
                                  behavior_random_prob=BEHAVIOR_RANDOM_PROB)
print(f"Collected in {time.perf_counter() - t0:.2f}s")
states = np.array([tr[0] for tr in transitions], dtype=float)
print("Fitting Nyström feature map...")
t0 = time.perf_counter()
phi_single, phi_batch, d = fit_feature_map(states, RBF_COMPONENTS, SEED)
print(f"Fit in {time.perf_counter() - t0:.2f}s; d={d}, decision-aug dim={n_decisions * d}")
print("Running LSPI ...")
t0 = time.perf_counter()
theta = lspi(transitions=transitions,phi_batch_fn=phi_batch,d=d,n_decisions=
     n_decisions,discount=DISCOUNT,l2_reg=L2_REG,max_iters=LSPI_MAX_ITERS,tol=LSPI_TOL)
print(f"LSPI done in {time.perf_counter() - t0:.2f}s")

# Evaluate
mean_r, std_r = evaluate_policy(ENV_ID, theta, phi_single, n_decisions, episodes=
EVAL_EPISODES, seed=SEED, max_episode_steps=MAX_EPISODE_STEPS)
print(f"Evaluation over {EVAL_EPISODES} episodes: mean return = {mean_r:.1f}, std = {std_r:.1f}")

# Animate a single rollout
print("Generating animation...")
frames, total, steps = rollout_frames(ENV_ID, theta, phi_single, n_decisions,seed=
                     ANIMATE_EPISODE_SEED,max_episode_steps=ANIMATE_MAX_STEPS)
title = f"LSPI policy | return={total:.0f}, steps={steps}"
animate_frames(frames,interval_ms=ANIMATE_INTERVAL_MS,title=title,gif_path=GIF_PATH)
