"""Standalone native Breakout signals and three local reward variants."""
import math
import gymnasium as gym
import ale_py
import numpy as np


def locate(frame):
    if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
        raise ValueError("Expected native 210x160 uint8 RGB")
    court = frame[93:193, 8:152]
    red = (court[:, :, 0] == 200) & (court[:, :, 1] == 72) & (court[:, :, 2] == 72)
    ys, xs = np.nonzero(red[:96])
    ball = None
    if len(xs) and xs.max()-xs.min() < 6 and ys.max()-ys.min() < 8 and len(xs) <= 48:
        ball = (int(xs.min()+8), int(ys.min()+93), int(xs.max()+8), int(ys.max()+93))
    best = None
    for row in red[96:]:
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            width = int(right-left)
            if 4 <= width <= 32 and (best is None or width > best[0]):
                best = (width, int(left+8), int(right+7))
    paddle = None if best is None else (best[1], 189, best[2], 192)
    return ball, paddle


class PaddleBounceDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_ball, self.contact = None, None
        self.last_frame, self.direction, self.last_bounce = 0, 0, -100

    def update(self, ball, paddle, frame):
        if ball is None:
            if frame-self.last_frame > 8:
                self.last_ball, self.contact, self.direction = None, None, 0
            return False
        bounced = False
        if self.last_ball is not None:
            previous = self.last_ball
            elapsed = frame-self.last_frame
            dx = (ball[0]+ball[2]-previous[0]-previous[2])/2
            dy = (ball[1]+ball[3]-previous[1]-previous[3])/2
            if elapsed <= 8 and max(abs(dx), abs(dy)) <= 8*elapsed:
                if (dy >= 0 and self.direction >= 0 and ball[3] >= 185 and paddle is not None
                        and ball[2] >= paddle[0]-2 and ball[0] <= paddle[2]+2):
                    self.contact = frame
                bounced = (dy < 0 and self.direction > 0 and self.contact is not None
                           and frame-self.contact <= 8 and max(ball[3], previous[3]) >= 184
                           and frame-self.last_bounce > 8)
                if bounced:
                    self.last_bounce, self.contact = frame, None
                if dy:
                    self.direction = 1 if dy > 0 else -1
            else:
                self.direction, self.contact = 0, None
        self.last_ball, self.last_frame = ball, frame
        return bool(bounced)


def landing_potential(history, ball, paddle, distance_scale=40.0):
    if ball is None or paddle is None or len(history) < 5:
        return 0.0
    y = (ball[1]+ball[3])/2
    if not 100 <= y <= 187:
        return 0.0
    n = len(history)
    mt, mx, my = [sum(row[k] for row in history)/n for k in range(3)]
    denominator = sum((row[0]-mt)**2 for row in history)
    if denominator <= 0 or history[-1][0]-history[0][0] < 4:
        return 0.0
    vx = sum((t-mt)*(px-mx) for t, px, py in history)/denominator
    vy = sum((t-mt)*(py-my) for t, px, py in history)/denominator
    dx = [b[1]-a[1] for a, b in zip(history, history[1:])]
    dy = [b[2]-a[2] for a, b in zip(history, history[1:])]
    turn = any(v > 0 for v in dx) and any(v < 0 for v in dx)
    residual = max(max(abs(px-mx-vx*(t-mt)), abs(py-my-vy*(t-mt))) for t, px, py in history)
    if turn or any(v < 0 for v in dy) or not 0 < vy <= 8 or abs(vx) > 8 or residual > 1.25:
        return 0.0
    projected = (ball[0]+ball[2])/2 + vx*max(0.0, (187-y)/vy)
    folded = (projected-9.0) % 282.0
    landing = 9.0 + (folded if folded <= 141 else 282-folded)
    margin = (ball[2]-ball[0]+1)/2
    distance = max(paddle[0]+margin-landing, landing-(paddle[2]-margin), 0.0)
    return max(0.0, min(1.0, 1.0-distance/distance_scale))


class NativeSignals(gym.Wrapper):
    """Track events on every emulator frame while returning the raw game reward."""
    def __init__(self, env):
        super().__init__(env)
        self.detector = PaddleBounceDetector()
        self.total_bonus, self.bounces, self.frame = 0.0, 0, 0
        self.lives, self.history, self.phi = None, [], 0.0
        self.rally_points, self.active = 0.0, False

    def clear_life(self):
        self.detector.reset()
        self.history, self.phi, self.rally_points, self.active = [], 0.0, 0.0, False

    def observe(self, observation, reward, lives, terminal=False):
        self.frame += 1
        lost = self.lives is not None and lives is not None and lives < self.lives
        self.lives = lives
        if lost:
            self.clear_life()
            return
        ball, paddle = locate(observation)
        if self.detector.update(ball, paddle, self.frame):
            self.bounces += 1
            self.total_bonus += 1.0
            self.rally_points, self.active = 0.0, True
        if self.active and reward > 0:
            before = min(2.0, 1.0+0.2*self.rally_points)
            self.rally_points += reward
            self.total_bonus += min(2.0, 1.0+0.2*self.rally_points)-before
        if ball is None:
            self.history = []
        else:
            point = (self.frame, (ball[0]+ball[2])/2, (ball[1]+ball[3])/2)
            self.history = (self.history+[point])[-7:]
        self.phi = landing_potential(self.history, ball, paddle)
        if terminal:
            self.clear_life()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.total_bonus, self.bounces, self.frame, self.lives = 0.0, 0, 0, None
        self.clear_life()
        self.observe(observation, 0.0, info.get("lives"))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.observe(observation, reward, info.get("lives"), terminated)
        return observation, reward, terminated, truncated, info


def extra_reward(variant, native_bonus, phi, next_phi, terminal, gamma):
    if variant == "raw":
        return 0.0
    if variant == "small-bounce":
        return 0.05*native_bonus
    if variant == "landing":
        return 0.1*(gamma*(0.0 if terminal else next_phi)-phi)
    raise ValueError(f"Unknown reward variant: {variant}")


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, signals, variant, gamma):
        super().__init__(env)
        if variant not in ("raw", "small-bounce", "landing"):
            raise ValueError("Unknown reward variant")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and between zero and one")
        self.signals, self.variant, self.gamma = signals, variant, gamma
        self.raw_score, self.lives = 0.0, None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.raw_score, self.lives = 0.0, info.get("lives")
        return observation, info

    def step(self, action):
        phi, bonus, bounces = self.signals.phi, self.signals.total_bonus, self.signals.bounces
        observation, raw, terminated, truncated, info = self.env.step(action)
        lives = info.get("lives")
        lost = self.lives is not None and lives is not None and lives < self.lives
        learning_terminal = bool(terminated or lost)
        aux = extra_reward(self.variant, self.signals.total_bonus-bonus, phi,
                           self.signals.phi, learning_terminal, self.gamma)
        self.raw_score += float(raw)
        self.lives = lives
        info = dict(info, raw_reward=float(raw), aux_reward=float(aux), raw_score=self.raw_score,
                    learning_terminal=learning_terminal, paddle_bounces=self.signals.bounces-bounces)
        return observation, float(raw)+aux, terminated, truncated, info


def make_env(variant="raw", mode=0, difficulty=0, frame_skip=4, sticky=0.25, gamma=0.995):
    gym.register_envs(ale_py)
    native = gym.make("ALE/Breakout-v5", mode=mode, difficulty=difficulty, frameskip=1,
                      repeat_action_probability=sticky)
    signals = NativeSignals(native)
    env = gym.wrappers.AtariPreprocessing(signals, frame_skip=frame_skip, screen_size=84,
                                        grayscale_obs=True, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return RewardWrapper(env, signals, variant, gamma)
