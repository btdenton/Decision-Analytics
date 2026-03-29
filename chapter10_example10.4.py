#Chapter 10, Example 10.4: Training an agent to pilot a lunar lander
import os, imageio
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

#Input data
ENV_ID = "LunarLander-v3"
TRAIN_TIMESTEPS = 20_0000
SEED = 0
GIF_OUTPUT_DIR = "GIFS"
GIF_NAME = "LunarLander_PPO_policy"
GIF_MINUTES = 1
GIF_FPS = 30
NET_SIZE = [128, 128]# PPO network size 

# Record a GIF animation of back-to-back episodes until time elapses
def record_gif(model, env_id, out_dir="videos", gif_name="LunaryLander_PPO_policy",
               minutes=1, fps=30, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    target_frames = int(minutes * 60 * fps)
    frames, total_return, episodes = [], 0.0, 0
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while len(frames) < target_frames:
        decision, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(decision)
        total_return += reward
        frames.append(env.render())
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset(seed=seed + episodes)
            terminated = truncated = False
    env.close()

    gif_path = os.path.join(out_dir, f"{gif_name}.gif")
    # duration is seconds per frame
    imageio.mimsave(gif_path, frames, duration=1.0 / fps)
    print(f"Saved {gif_path}  (~{minutes} min, {episodes+1} episodes, total return: {total_return:.1f})")

# Train proximal policy optimization and record animation
print(f"Training PPO on {ENV_ID} for {TRAIN_TIMESTEPS} timesteps...")
env = Monitor(gym.make(ENV_ID))
model = PPO("MlpPolicy", env, seed=SEED, verbose=0,policy_kwargs=dict(net_arch=NET_SIZE))
model.learn(total_timesteps=TRAIN_TIMESTEPS)
env.close()
print("Recording animation of LunarLandero...")
record_gif(model, ENV_ID, out_dir=GIF_OUTPUT_DIR, gif_name=GIF_NAME,
         minutes=GIF_MINUTES, fps=GIF_FPS, seed=SEED)  

