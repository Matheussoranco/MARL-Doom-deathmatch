import argparse
import glob
import math
import multiprocessing as mp
import os
import platform
import queue
import random
import sys
import time
import warnings
import zipfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import vizdoom as vzd

warnings.filterwarnings('ignore')

OBS_RES = (84, 84)
FRAME_STACK = 4
FRAME_SKIP = 4
HIDDEN_SIZE = 256
VIDEO_FPS = max(1, round(35 / FRAME_SKIP))
MAX_REC_FRAMES = 4000
EVAL_REC_FRAMES = 8000

TRAIN_SCREEN_RES = vzd.ScreenResolution.RES_640X480
EVAL_SCREEN_RES = vzd.ScreenResolution.RES_1280X720

BUTTONS = [
    vzd.Button.ATTACK,
    vzd.Button.SPEED,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.SELECT_NEXT_WEAPON,
]

ACTIONS = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
]
NUM_ACTIONS = len(ACTIONS)

GAME_VARS = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.DAMAGECOUNT,
    vzd.GameVariable.DAMAGE_TAKEN,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.ITEMCOUNT,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.DEATHCOUNT,
]


class DeathmatchEnv:
    def __init__(self, rank, n_players, port=5029, scenario='cig',
                 episode_minutes=4.0, frame_skip=FRAME_SKIP,
                 frame_stack=FRAME_STACK, resolution=OBS_RES,
                 screen_res=TRAIN_SCREEN_RES, record=False, max_rec=MAX_REC_FRAMES,
                 reward_kwargs=None):
        self.rank = rank
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.resolution = resolution
        self.record = record
        self.max_rec = max_rec
        rk = reward_kwargs or {}
        self.frag_reward = rk.get('frag_reward', 1.0)
        self.damage_reward = rk.get('damage_reward', 0.01)
        self.damage_taken_penalty = rk.get('damage_taken_penalty', -0.005)
        self.hit_reward = rk.get('hit_reward', 0.02)
        self.item_reward = rk.get('item_reward', 0.05)
        self.health_reward = rk.get('health_reward', 0.001)
        self.death_penalty = rk.get('death_penalty', 0.5)
        self.ammo_reward = rk.get('ammo_reward', 0.005)
        self.living_bonus = rk.get('living_bonus', 0.0005)

        self.game = vzd.DoomGame()
        cfg_path = Path(vzd.scenarios_path) / f'{scenario}.cfg'
        if not cfg_path.is_file():
            raise FileNotFoundError(f'Scenario config not found: {cfg_path}')
        self.game.load_config(str(cfg_path))

        self.game.set_available_buttons(BUTTONS)
        self.game.set_available_game_variables(GAME_VARS)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(screen_res)
        self.game.set_window_visible(False)
        self.game.set_console_enabled(False)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_doom_skill(4)

        if rank == 0:
            self.game.add_game_args(
                f'-host {n_players} -port {port} -netmode 0 -deathmatch '
                f'+timelimit {episode_minutes} '
                '+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 '
                '+sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 2 '
                '+viz_nocheat 1 +sv_losefrag 1')
            self.game.add_game_args('+name AGENT0 +colorset 0')
        else:
            self.game.add_game_args(f'-join 127.0.0.1 -port {port}')
            self.game.add_game_args(f'+name AGENT{rank} +colorset {rank % 8}')

        self.game.init()

        self.actions = ACTIONS
        self.num_actions = NUM_ACTIONS
        self.frames = deque(maxlen=frame_stack)
        self.recorded_frames = []
        self._last = {v: 0.0 for v in GAME_VARS}

    def _var(self, var):
        try:
            return float(self.game.get_game_variable(var))
        except Exception:
            return 0.0

    def _current_screen(self):
        state = self.game.get_state()
        return state.screen_buffer if state is not None else None

    def _preprocess(self, frame):
        if frame is None:
            return np.zeros(self.resolution, dtype=np.uint8)
        frame = np.asarray(frame)
        if frame.ndim == 3:
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]),
                           interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def _grab(self):
        if len(self.recorded_frames) >= self.max_rec:
            return
        buf = self._current_screen()
        if buf is None:
            return
        buf = np.asarray(buf)
        if buf.ndim == 3 and buf.shape[0] == 3:
            buf = np.transpose(buf, (1, 2, 0))
        if buf.ndim == 3 and buf.shape[-1] >= 3:
            self.recorded_frames.append(
                np.ascontiguousarray(buf[..., :3], dtype=np.uint8).copy())

    def _snapshot_vars(self):
        return {v: self._var(v) for v in GAME_VARS}

    def start_recording(self):
        self.recorded_frames = []
        self.record = True

    def stop_recording(self):
        self.record = False
        return self.recorded_frames

    def reset(self):
        self.game.new_episode()
        self._last = self._snapshot_vars()
        frame = self._preprocess(self._current_screen())
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        if self.record:
            self._grab()
        return np.stack(self.frames, axis=0)

    def step(self, action_idx):
        self.game.make_action(self.actions[action_idx], self.frame_skip)
        done = self.game.is_episode_finished()
        dead = self.game.is_player_dead()

        cur = self._snapshot_vars()
        d_frag = cur[vzd.GameVariable.FRAGCOUNT] - self._last[vzd.GameVariable.FRAGCOUNT]
        d_damage = cur[vzd.GameVariable.DAMAGECOUNT] - self._last[vzd.GameVariable.DAMAGECOUNT]
        d_taken = cur[vzd.GameVariable.DAMAGE_TAKEN] - self._last[vzd.GameVariable.DAMAGE_TAKEN]
        d_hit = cur[vzd.GameVariable.HITCOUNT] - self._last[vzd.GameVariable.HITCOUNT]
        d_item = cur[vzd.GameVariable.ITEMCOUNT] - self._last[vzd.GameVariable.ITEMCOUNT]
        d_health = cur[vzd.GameVariable.HEALTH] - self._last[vzd.GameVariable.HEALTH]
        d_ammo = cur[vzd.GameVariable.SELECTED_WEAPON_AMMO] - self._last[vzd.GameVariable.SELECTED_WEAPON_AMMO]

        reward = self.living_bonus
        reward += d_frag * self.frag_reward
        reward += max(0.0, d_damage) * self.damage_reward
        reward += max(0.0, d_taken) * self.damage_taken_penalty
        reward += max(0.0, d_hit) * self.hit_reward
        reward += max(0.0, d_item) * self.item_reward
        if d_health > 0:
            reward += d_health * self.health_reward
        reward += max(0.0, d_ammo) * self.ammo_reward
        if dead:
            reward -= self.death_penalty
        reward = float(np.clip(reward, -3.0, 5.0))

        self._last = cur

        if dead and not done:
            self.game.respawn_player()
            self._last = self._snapshot_vars()

        if done:
            frame = np.zeros(self.resolution, dtype=np.uint8)
        else:
            frame = self._preprocess(self._current_screen())
            if self.record:
                self._grab()
        self.frames.append(frame)
        info = {
            'frags': cur[vzd.GameVariable.FRAGCOUNT],
            'damage': cur[vzd.GameVariable.DAMAGECOUNT],
            'hits': cur[vzd.GameVariable.HITCOUNT],
            'deaths': cur[vzd.GameVariable.DEATHCOUNT],
        }
        return np.stack(self.frames, axis=0), reward, done, info

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass


class RecurrentActorCritic(nn.Module):
    def __init__(self, in_ch=FRAME_STACK, n_actions=NUM_ACTIONS,
                 hidden=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(inplace=True),
        )
        self.flatten_dim = 64 * 7 * 7
        self.fc = nn.Linear(self.flatten_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def encode(self, obs):
        x = obs.float() / 255.0
        z = self.conv(x).flatten(1)
        return F.relu(self.fc(z), inplace=True)

    def forward(self, obs, h):
        z = self.encode(obs)
        h_new = self.gru(z, h)
        logits = self.actor(h_new)
        value = self.critic(h_new).squeeze(-1)
        return logits, value, h_new

    def init_hidden(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.hidden_size, device=device)


def _kill_stale():
    if platform.system() != 'Windows':
        os.system('pkill -9 -f vizdoom > /dev/null 2>&1')
        time.sleep(1.5)


def _try_load_state(net, path):
    try:
        ckpt = torch.load(path, map_location='cpu')
        net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        return os.path.getmtime(path)
    except Exception:
        return None


def _pick_snapshot(snapshot_dir):
    snaps = sorted(glob.glob(os.path.join(snapshot_dir, 'snapshot_*.pt')))
    if not snaps:
        return None
    pool = snaps[-10:] if len(snaps) > 10 else snaps
    return random.choice(pool)


def actor_worker(rank, n_players, port, scenario, episode_minutes,
                 model_path, snapshot_dir, video_dir, seed,
                 rollout_q, video_q, stop_event,
                 rollout_length, contribute, league_prob, refresh_every,
                 reward_kwargs, ready_event=None):
    random.seed(seed + rank * 1000)
    np.random.seed(seed + rank * 1000)
    torch.manual_seed(seed + rank * 1000)
    torch.set_num_threads(1)

    try:
        env = DeathmatchEnv(rank=rank, n_players=n_players, port=port,
                            scenario=scenario, episode_minutes=episode_minutes,
                            screen_res=TRAIN_SCREEN_RES, record=True,
                            max_rec=MAX_REC_FRAMES, reward_kwargs=reward_kwargs)
    except Exception as exc:
        print(f'[agent {rank}] env init failed: {exc}', flush=True)
        if ready_event is not None:
            ready_event.set()
        return

    if ready_event is not None:
        ready_event.set()

    net = RecurrentActorCritic().eval()
    loaded_mtime = None
    best_frags = -1.0
    video_path = os.path.join(video_dir, f'agent{rank}_best.mp4')

    def reload_latest():
        nonlocal loaded_mtime
        if not os.path.isfile(model_path):
            return False
        try:
            mtime = os.path.getmtime(model_path)
            if mtime != loaded_mtime:
                mt = _try_load_state(net, model_path)
                if mt is not None:
                    loaded_mtime = mt
            return True
        except Exception:
            return False

    state = env.reset()
    h = net.init_hidden(1)
    using_league = False
    step = 0
    ep_reward = 0.0

    obs_buf = np.zeros((rollout_length, FRAME_STACK, *OBS_RES), dtype=np.uint8)
    act_buf = np.zeros(rollout_length, dtype=np.int64)
    lp_buf = np.zeros(rollout_length, dtype=np.float32)
    rew_buf = np.zeros(rollout_length, dtype=np.float32)
    done_buf = np.zeros(rollout_length, dtype=np.float32)
    val_buf = np.zeros(rollout_length, dtype=np.float32)
    h_init = h.squeeze(0).detach().cpu().numpy().copy()
    rollout_step = 0

    try:
        while not stop_event.is_set():
            if step % refresh_every == 0 and not using_league:
                reload_latest()

            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(state)).unsqueeze(0).float()
                logits, value, h_new = net(obs_t, h)
                dist = Categorical(logits=logits)
                action_t = dist.sample()
                logp_t = dist.log_prob(action_t)
                action = int(action_t.item())
                logp = float(logp_t.item())
                v = float(value.item())

            next_state, reward, done, info = env.step(action)

            if contribute and not using_league:
                obs_buf[rollout_step] = state
                act_buf[rollout_step] = action
                lp_buf[rollout_step] = logp
                rew_buf[rollout_step] = reward
                done_buf[rollout_step] = float(done)
                val_buf[rollout_step] = v
                rollout_step += 1

                if rollout_step == rollout_length:
                    with torch.no_grad():
                        ns_t = torch.from_numpy(np.asarray(next_state)).unsqueeze(0).float()
                        h_for_bootstrap = h_new * (1.0 - float(done))
                        _, last_v, _ = net(ns_t, h_for_bootstrap)
                        last_value = float(last_v.item())
                    payload = {
                        'obs': obs_buf.copy(),
                        'actions': act_buf.copy(),
                        'logprobs': lp_buf.copy(),
                        'rewards': rew_buf.copy(),
                        'dones': done_buf.copy(),
                        'values': val_buf.copy(),
                        'h_init': h_init.copy(),
                        'last_value': last_value,
                        'last_done': float(done),
                        'rank': rank,
                    }
                    try:
                        rollout_q.put(payload, timeout=2.0)
                    except queue.Full:
                        pass
                    rollout_step = 0
                    h_init = h_new.squeeze(0).detach().cpu().numpy().copy()

            if done:
                h = net.init_hidden(1)
            else:
                h = h_new

            state = next_state
            step += 1
            ep_reward += reward

            if done:
                frags = info.get('frags', 0.0)
                frames = env.stop_recording()
                if frags > best_frags and frames:
                    best_frags = frags
                    if video_q is not None:
                        try:
                            video_q.put_nowait((rank, video_path, frames, float(frags), float(ep_reward)))
                        except queue.Full:
                            pass
                ep_reward = 0.0
                if league_prob > 0.0 and random.random() < league_prob:
                    chosen = _pick_snapshot(snapshot_dir)
                    if chosen is not None:
                        _try_load_state(net, chosen)
                        using_league = True
                    else:
                        using_league = False
                        reload_latest()
                else:
                    using_league = False
                    reload_latest()
                rollout_step = 0
                h_init = np.zeros_like(h_init)
                env.start_recording()
                state = env.reset()
    except Exception as exc:
        print(f'[agent {rank}] stopped: {exc}', flush=True)
    finally:
        env.close()


def _save_video(frames, path, fps=VIDEO_FPS):
    try:
        import imageio
    except ImportError:
        print('[video] imageio not installed, skipping', flush=True)
        return False
    if not frames:
        return False
    clean, shape0 = [], None
    for f in frames:
        a = np.asarray(f, dtype=np.uint8)
        if a.ndim != 3 or a.shape[-1] < 3:
            continue
        a = np.ascontiguousarray(a[..., :3])
        if shape0 is None:
            shape0 = a.shape
        if a.shape != shape0:
            continue
        clean.append(a)
    if not clean:
        return False
    tmp = path + '.tmp.mp4'
    try:
        imageio.mimsave(tmp, clean, fps=fps, codec='libx264', quality=9,
                        macro_block_size=1,
                        output_params=['-pix_fmt', 'yuv420p',
                                       '-preset', 'medium',
                                       '-crf', '18'])
        os.replace(tmp, path)
        return True
    except Exception as exc:
        print(f'[video] save failed: {exc}', flush=True)
        return False


def video_writer_proc(video_q, stop_event):
    while not stop_event.is_set():
        try:
            rank, path, frames, frags, ep_reward = video_q.get(timeout=1.0)
        except queue.Empty:
            continue
        ok = _save_video(frames, path)
        if ok:
            print(f'[video] agent{rank} frags={frags:.0f} reward={ep_reward:.2f} -> {os.path.basename(path)}',
                  flush=True)


class PPOLearner:
    def __init__(self, device, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip=0.1, vf_coef=0.5, ent_coef=0.005, max_grad_norm=0.5,
                 ppo_epochs=4, minibatch_rollouts=4, value_clip=0.2):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_rollouts = minibatch_rollouts
        self.value_clip = value_clip
        self.net = RecurrentActorCritic().to(device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr,
                                     eps=1e-5, weight_decay=1e-6)
        self.updates = 0

    def save(self, path):
        torch.save({
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'updates': self.updates,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
        self.updates = ckpt.get('updates', 0)

    def publish(self, path):
        tmp = path + '.tmp'
        torch.save({'model': {k: v.detach().cpu()
                              for k, v in self.net.state_dict().items()}}, tmp)
        os.replace(tmp, path)

    def _compute_gae(self, rollouts):
        for r in rollouts:
            T = len(r['rewards'])
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            next_value = r['last_value']
            next_nonterm = 1.0 - r['last_done']
            for t in reversed(range(T)):
                delta = r['rewards'][t] + self.gamma * next_value * next_nonterm - r['values'][t]
                gae = delta + self.gamma * self.gae_lambda * next_nonterm * gae
                adv[t] = gae
                next_value = r['values'][t]
                next_nonterm = 1.0 - r['dones'][t]
            r['advantages'] = adv
            r['returns'] = adv + r['values']

    def update(self, rollouts):
        self._compute_gae(rollouts)
        all_adv = np.concatenate([r['advantages'] for r in rollouts])
        adv_mean = float(all_adv.mean())
        adv_std = float(all_adv.std()) + 1e-8

        B = len(rollouts)
        T = len(rollouts[0]['rewards'])
        device = self.device

        obs = torch.from_numpy(np.stack([r['obs'] for r in rollouts])).to(device)
        actions = torch.from_numpy(np.stack([r['actions'] for r in rollouts])).to(device)
        old_logp = torch.from_numpy(np.stack([r['logprobs'] for r in rollouts])).to(device)
        adv = torch.from_numpy(np.stack([r['advantages'] for r in rollouts])).to(device)
        returns = torch.from_numpy(np.stack([r['returns'] for r in rollouts])).to(device)
        dones = torch.from_numpy(np.stack([r['dones'] for r in rollouts])).to(device)
        old_values = torch.from_numpy(np.stack([r['values'] for r in rollouts])).to(device)
        h_init = torch.from_numpy(np.stack([r['h_init'] for r in rollouts])).to(device)
        adv = (adv - adv_mean) / adv_std

        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
                 'approx_kl': 0.0, 'clipfrac': 0.0}
        n_minibatches = 0
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, self.minibatch_rollouts):
                idx = perm[start:start + self.minibatch_rollouts]
                if idx.numel() == 0:
                    continue
                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_old_logp = old_logp[idx]
                mb_adv = adv[idx]
                mb_ret = returns[idx]
                mb_dones = dones[idx]
                mb_old_v = old_values[idx]
                mb_h = h_init[idx]

                mb_B = idx.numel()
                logits_seq = []
                value_seq = []
                h = mb_h
                for t in range(T):
                    if t > 0:
                        mask = (1.0 - mb_dones[:, t - 1]).unsqueeze(-1)
                        h = h * mask
                    logits, value, h = self.net(mb_obs[:, t], h)
                    logits_seq.append(logits)
                    value_seq.append(value)
                logits_all = torch.stack(logits_seq, dim=1)
                values_all = torch.stack(value_seq, dim=1)

                dist = Categorical(logits=logits_all)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = mb_old_v + torch.clamp(values_all - mb_old_v,
                                                   -self.value_clip, self.value_clip)
                vloss_unclipped = (values_all - mb_ret).pow(2)
                vloss_clipped = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(vloss_unclipped, vloss_clipped).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.clip).float().mean().item()
                stats['policy_loss'] += float(policy_loss.item())
                stats['value_loss'] += float(value_loss.item())
                stats['entropy'] += float(entropy.item())
                stats['approx_kl'] += float(approx_kl)
                stats['clipfrac'] += float(clipfrac)
                n_minibatches += 1
        n_minibatches = max(1, n_minibatches)
        for k in stats:
            stats[k] /= n_minibatches
        self.updates += 1
        return stats


def _spawn_worker(ctx, **kw):
    p = ctx.Process(target=actor_worker, kwargs=kw, daemon=True)
    p.start()
    return p


def train(cfg):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = save_dir / 'snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    video_dir = save_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(save_dir / 'policy_latest.pt')

    _kill_stale()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch={torch.__version__} device={device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        torch.backends.cudnn.benchmark = True

    learner = PPOLearner(
        device=device, lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip=cfg.clip, vf_coef=cfg.vf_coef, ent_coef=cfg.ent_coef,
        ppo_epochs=cfg.ppo_epochs, minibatch_rollouts=cfg.minibatch_rollouts,
    )
    if cfg.resume and os.path.isfile(cfg.resume):
        learner.load(cfg.resume)
        print(f'Resumed from {cfg.resume} at update {learner.updates}')
    learner.publish(model_path)
    learner.publish(str(snapshot_dir / 'snapshot_0000.pt'))

    ctx = mp.get_context('fork' if platform.system() != 'Windows' else 'spawn')
    rollout_q = ctx.Queue(maxsize=cfg.queue_size)
    video_q = ctx.Queue(maxsize=64)
    stop_event = ctx.Event()

    reward_kwargs = {
        'frag_reward': cfg.frag_reward,
        'damage_reward': cfg.damage_reward,
        'damage_taken_penalty': cfg.damage_taken_penalty,
        'hit_reward': cfg.hit_reward,
        'item_reward': cfg.item_reward,
        'health_reward': cfg.health_reward,
        'death_penalty': cfg.death_penalty,
        'ammo_reward': cfg.ammo_reward,
        'living_bonus': cfg.living_bonus,
    }

    video_proc = ctx.Process(target=video_writer_proc,
                             args=(video_q, stop_event), daemon=True)
    video_proc.start()

    n_players = cfg.n_players
    n_contrib = cfg.n_contributors
    procs = []
    ready_events = []

    for rank in range(n_players):
        contribute = rank < n_contrib
        league_prob = 0.0 if contribute else cfg.league_prob
        rev = ctx.Event()
        ready_events.append(rev)
        p = _spawn_worker(
            ctx,
            rank=rank, n_players=n_players, port=cfg.port,
            scenario=cfg.scenario, episode_minutes=cfg.episode_minutes,
            model_path=model_path, snapshot_dir=str(snapshot_dir),
            video_dir=str(video_dir), seed=cfg.seed,
            rollout_q=rollout_q, video_q=video_q, stop_event=stop_event,
            rollout_length=cfg.rollout_length, contribute=contribute,
            league_prob=league_prob, refresh_every=cfg.refresh_every,
            reward_kwargs=reward_kwargs, ready_event=rev,
        )
        procs.append(p)
        role = 'CONTRIB' if contribute else f'LEAGUE(p={league_prob:.2f})'
        print(f'  spawned agent {rank} ({role})')
        if rank == 0:
            # Host opens the listen socket inside game.init(); give the cold
            # process time to import torch/cv2/vizdoom before joiners connect.
            time.sleep(6.0)
        else:
            time.sleep(0.4)

    # Barrier: every worker's ViZDoom env must come up. The host's ready_event
    # only fires after the full multiplayer handshake completes, so if any
    # joiner failed to connect we'd otherwise deadlock here forever. Detect it
    # and abort loudly instead.
    handshake_deadline = time.time() + 120
    for rank, rev in enumerate(ready_events):
        if not rev.wait(timeout=max(0.0, handshake_deadline - time.time())):
            print(f'[train] agent {rank} never initialized (multiplayer '
                  f'handshake failed); aborting.', flush=True)
            stop_event.set()
            for p in procs:
                if p.is_alive():
                    p.terminate()
            for p in procs:
                p.join(timeout=5)
            if video_proc.is_alive():
                video_proc.terminate()
            return model_path
    print('All agents connected; deathmatch underway.')

    print(f'\nGathering {cfg.rollouts_per_update} rollouts per PPO update.')
    print(f'Snapshotting every {cfg.snapshot_every} updates, '
          f'training for {cfg.total_updates} updates.\n')

    pending = []
    t_start = time.time()
    total_steps = 0
    snapshot_idx = 1
    recent_loss = deque(maxlen=20)
    recent_kl = deque(maxlen=20)
    recent_ent = deque(maxlen=20)

    try:
        while learner.updates < cfg.total_updates:
            try:
                roll = rollout_q.get(timeout=5.0)
                pending.append(roll)
                total_steps += cfg.rollout_length
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    print('[train] all workers died; aborting', flush=True)
                    break
                continue

            if len(pending) >= cfg.rollouts_per_update:
                batch = pending[:cfg.rollouts_per_update]
                pending = pending[cfg.rollouts_per_update:]
                stats = learner.update(batch)
                recent_loss.append(stats['policy_loss'])
                recent_kl.append(stats['approx_kl'])
                recent_ent.append(stats['entropy'])
                learner.publish(model_path)

                if learner.updates % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    sps = total_steps / max(elapsed, 1e-6)
                    print(
                        f'upd={learner.updates:>5} steps={total_steps:>9} '
                        f'sps={sps:6.0f} '
                        f'pi_loss={np.mean(recent_loss):+.4f} '
                        f'v_loss={stats["value_loss"]:.4f} '
                        f'ent={np.mean(recent_ent):.3f} '
                        f'kl={np.mean(recent_kl):.4f} '
                        f'clipf={stats["clipfrac"]:.2f} '
                        f'qsize={rollout_q.qsize()}',
                        flush=True)

                if learner.updates % cfg.snapshot_every == 0:
                    snap_path = snapshot_dir / f'snapshot_{snapshot_idx:04d}.pt'
                    learner.publish(str(snap_path))
                    snapshot_idx += 1
                    snaps = sorted(glob.glob(str(snapshot_dir / 'snapshot_*.pt')))
                    while len(snaps) > cfg.max_snapshots:
                        try:
                            os.remove(snaps[0])
                        except OSError:
                            pass
                        snaps = snaps[1:]
                    print(f'[snapshot] saved {snap_path.name}', flush=True)

                if learner.updates % cfg.checkpoint_every == 0:
                    learner.save(str(save_dir / 'checkpoint_latest.pt'))
                    print(f'[ckpt] update {learner.updates}', flush=True)
    finally:
        stop_event.set()
        learner.save(str(save_dir / 'checkpoint_final.pt'))
        learner.publish(model_path)
        time.sleep(1.0)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)
        if video_proc.is_alive():
            video_proc.terminate()
        video_proc.join(timeout=5)
        print('Training shutdown complete.')

    return model_path


def eval_worker(rank, n_players, port, scenario, episode_minutes,
                model_path, video_dir, seed, n_episodes,
                video_q, stop_event, ready_event=None):
    random.seed(seed + rank * 7919)
    np.random.seed(seed + rank * 7919)
    torch.manual_seed(seed + rank * 7919)
    torch.set_num_threads(1)

    try:
        env = DeathmatchEnv(rank=rank, n_players=n_players, port=port,
                            scenario=scenario, episode_minutes=episode_minutes,
                            screen_res=EVAL_SCREEN_RES, record=True,
                            max_rec=EVAL_REC_FRAMES, reward_kwargs=None)
    except Exception as exc:
        print(f'[eval {rank}] env init failed: {exc}', flush=True)
        if ready_event is not None:
            ready_event.set()
        return

    if ready_event is not None:
        ready_event.set()

    net = RecurrentActorCritic().eval()
    _try_load_state(net, model_path)

    best_per_agent = -1.0
    try:
        for ep in range(n_episodes):
            env.start_recording()
            state = env.reset()
            h = net.init_hidden(1)
            done = False
            ep_reward = 0.0
            while not done and not stop_event.is_set():
                with torch.no_grad():
                    obs_t = torch.from_numpy(np.asarray(state)).unsqueeze(0).float()
                    logits, _, h = net(obs_t, h)
                    if random.random() < 0.04:
                        action = int(Categorical(logits=logits).sample().item())
                    else:
                        action = int(logits.argmax(dim=1).item())
                state, reward, done, info = env.step(action)
                ep_reward += reward
            frags = float(info.get('frags', 0.0))
            frames = env.stop_recording()
            if frags > best_per_agent and frames:
                best_per_agent = frags
                path = os.path.join(video_dir, f'eval_agent{rank}_ep{ep}_frags{int(frags)}.mp4')
                if video_q is not None:
                    try:
                        video_q.put_nowait((rank, path, frames, frags, ep_reward))
                    except queue.Full:
                        pass
            print(f'[eval {rank}] ep {ep} frags={frags:.0f} reward={ep_reward:.2f}', flush=True)
    except Exception as exc:
        print(f'[eval {rank}] stopped: {exc}', flush=True)
    finally:
        env.close()


def evaluate(cfg, model_path):
    print('\n========== EVALUATION PASS (1280x720) ==========')
    save_dir = Path(cfg.save_dir)
    eval_video_dir = save_dir / 'eval_videos'
    eval_video_dir.mkdir(parents=True, exist_ok=True)

    _kill_stale()

    ctx = mp.get_context('fork' if platform.system() != 'Windows' else 'spawn')
    video_q = ctx.Queue(maxsize=64)
    stop_event = ctx.Event()

    video_proc = ctx.Process(target=video_writer_proc,
                             args=(video_q, stop_event), daemon=True)
    video_proc.start()

    n_players = cfg.n_players
    procs = []
    for rank in range(n_players):
        rev = ctx.Event()
        p = ctx.Process(
            target=eval_worker,
            kwargs=dict(
                rank=rank, n_players=n_players, port=cfg.eval_port,
                scenario=cfg.scenario, episode_minutes=cfg.eval_episode_minutes,
                model_path=model_path, video_dir=str(eval_video_dir),
                seed=cfg.seed + 12345, n_episodes=cfg.eval_episodes,
                video_q=video_q, stop_event=stop_event, ready_event=rev,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)
        if rank == 0:
            time.sleep(2.5)
        else:
            time.sleep(0.4)

    for p in procs:
        p.join()

    time.sleep(2.0)
    stop_event.set()
    if video_proc.is_alive():
        video_proc.terminate()
    video_proc.join(timeout=5)
    print('========== EVALUATION COMPLETE ==========\n')


def bundle_results(save_dir):
    save_dir = Path(save_dir)
    out_zip = save_dir / 'doom_marl_results.zip'
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for mp4 in sorted((save_dir / 'eval_videos').glob('*.mp4')):
            z.write(mp4, f'eval_videos/{mp4.name}')
        for mp4 in sorted((save_dir / 'videos').glob('*.mp4')):
            z.write(mp4, f'train_videos/{mp4.name}')
        final = save_dir / 'checkpoint_final.pt'
        if final.is_file():
            z.write(final, 'checkpoint_final.pt')
        latest = save_dir / 'policy_latest.pt'
        if latest.is_file():
            z.write(latest, 'policy_latest.pt')
    return str(out_zip)


def build_argparser():
    p = argparse.ArgumentParser(description='8-agent MARL ViZDoom deathmatch (recurrent PPO + league).')
    p.add_argument('--scenario', type=str, default='cig')
    p.add_argument('--n_players', type=int, default=8,
                   help='Total ViZDoom players (max 8 for cig map).')
    p.add_argument('--n_contributors', type=int, default=5,
                   help='Players whose rollouts train the policy; rest are league opponents.')
    p.add_argument('--league_prob', type=float, default=0.6,
                   help='Per-episode chance a league worker loads a fresh snapshot.')
    p.add_argument('--port', type=int, default=5029)
    p.add_argument('--eval_port', type=int, default=5039)
    p.add_argument('--episode_minutes', type=float, default=4.0)
    p.add_argument('--eval_episode_minutes', type=float, default=5.0)
    p.add_argument('--total_updates', type=int, default=4000)
    p.add_argument('--rollout_length', type=int, default=128)
    p.add_argument('--rollouts_per_update', type=int, default=16)
    p.add_argument('--queue_size', type=int, default=64)
    p.add_argument('--ppo_epochs', type=int, default=4)
    p.add_argument('--minibatch_rollouts', type=int, default=4)
    p.add_argument('--lr', type=float, default=2.5e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--vf_coef', type=float, default=0.5)
    p.add_argument('--ent_coef', type=float, default=0.005)
    p.add_argument('--refresh_every', type=int, default=200,
                   help='Worker weight reload cadence (in env steps).')
    p.add_argument('--save_dir', type=str, default='./doom_marl_runs/run01')
    p.add_argument('--log_every', type=int, default=5)
    p.add_argument('--snapshot_every', type=int, default=50)
    p.add_argument('--checkpoint_every', type=int, default=100)
    p.add_argument('--max_snapshots', type=int, default=20)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--frag_reward', type=float, default=1.0)
    p.add_argument('--damage_reward', type=float, default=0.01)
    p.add_argument('--damage_taken_penalty', type=float, default=-0.005)
    p.add_argument('--hit_reward', type=float, default=0.02)
    p.add_argument('--item_reward', type=float, default=0.05)
    p.add_argument('--health_reward', type=float, default=0.001)
    p.add_argument('--death_penalty', type=float, default=0.5)
    p.add_argument('--ammo_reward', type=float, default=0.005)
    p.add_argument('--living_bonus', type=float, default=0.0005)
    p.add_argument('--skip_train', action='store_true',
                   help='Skip training and only run evaluation on --resume.')
    p.add_argument('--skip_eval', action='store_true')
    p.add_argument('--eval_episodes', type=int, default=3)
    return p


def main():
    cfg = build_argparser().parse_args()
    assert 2 <= cfg.n_players <= 8, 'cig scenario seats 2..8 players.'
    assert 1 <= cfg.n_contributors <= cfg.n_players

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(save_dir / 'policy_latest.pt')

    if not cfg.skip_train:
        model_path = train(cfg)

    if not cfg.skip_eval:
        eval_source = model_path if os.path.isfile(model_path) else (cfg.resume or model_path)
        evaluate(cfg, eval_source)

    bundle = bundle_results(cfg.save_dir)
    print(f'\nBundle written: {bundle}')


if __name__ == '__main__':
    main()
