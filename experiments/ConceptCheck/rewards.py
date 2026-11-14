"""Optional bounded rally bonuses; the environment's game reward stays raw.

Place ``RallyRewardTracker`` around native RGB ALE, before AtariPreprocessing.
It observes every emulator frame, so a paddle contact cannot disappear between
agent decisions. Read differences of ``info['rally_reward']['bonus_total']`` at
decision boundaries; reset/no-op frames can then be excluded from training.

The rally counter is history-dependent auxiliary information, not an additional
CNN input. This is an experimental reward objective, not policy-invariant shaping.
"""

from copy import deepcopy
import math

import gymnasium as gym
import numpy as np


def _number(value, name, minimum=0.0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number >= {minimum}") from error
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return result


class PaddleBounceDetector:
    """Detect a downward-to-upward reversal at the native Breakout paddle.

    The six selected point-scoring ROM modes use the same red ball/paddle and
    court coordinates in 210x160 RGB frames. The paddle width is measured, not
    assumed (difficulty and ceiling contact change it). A short contact cache
    bridges frames where ball and paddle overlap; side/brick bounces do not arm
    it. No RAM coordinates or reward events are used to infer a paddle bounce.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.frame = 0
        self.last_ball = None
        self.last_ball_frame = 0
        self.direction = 0
        self.contact_frame = None
        self.last_bounce_frame = -100

    @staticmethod
    def locate(observation):
        if (not isinstance(observation, np.ndarray)
                or observation.shape != (210, 160, 3)
                or observation.dtype != np.uint8):
            raise ValueError("rally rewards require native 210x160 uint8 RGB Breakout frames")
        court = observation[93:193, 8:152]
        red = ((court[:, :, 0] == 200) & (court[:, :, 1] == 72)
               & (court[:, :, 2] == 72))
        ys, xs = np.nonzero(red[:96])
        ball = None
        if (len(xs) and xs.max() - xs.min() < 6
                and ys.max() - ys.min() < 8 and len(xs) <= 48):
            ball = (int(xs.min() + 8), int(ys.min() + 93),
                    int(xs.max() + 8), int(ys.max() + 93))
        # A separate ball beside the paddle must not widen the paddle's box.
        best = None
        for row in red[96:100]:
            edges = np.diff(np.r_[False, row, False].astype(np.int8))
            for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                width = int(right - left)
                if 4 <= width <= 32 and (best is None or width > best[0]):
                    best = (width, int(left + 8), int(right + 7))
        paddle = None if best is None else (best[1], 189, best[2], 192)
        return ball, paddle

    def observe(self, observation):
        self.frame += 1
        ball, paddle = self.locate(observation)
        if ball is None:
            if self.frame - self.last_ball_frame > 8:
                self.last_ball = None
                self.direction = 0
                self.contact_frame = None
            return False
        bounced = False
        if self.last_ball is not None:
            previous = self.last_ball
            elapsed = self.frame - self.last_ball_frame
            dy = (ball[1] + ball[3] - previous[1] - previous[3]) / 2.0
            dx = (ball[0] + ball[2] - previous[0] - previous[2]) / 2.0
            continuous = elapsed <= 8 and max(abs(dx), abs(dy)) <= 8 * elapsed
            if continuous:
                if (dy >= 0 and self.direction >= 0 and ball[3] >= 185
                        and paddle is not None and ball[2] >= paddle[0] - 2
                        and ball[0] <= paddle[2] + 2):
                    self.contact_frame = self.frame
                bounced = bool(
                    dy < 0 and self.direction > 0
                    and self.contact_frame is not None
                    and self.frame - self.contact_frame <= 8
                    and max(ball[3], previous[3]) >= 184
                    and self.frame - self.last_bounce_frame > 8
                )
                if bounced:
                    self.last_bounce_frame = self.frame
                    self.contact_frame = None
                if dy:
                    self.direction = 1 if dy > 0 else -1
            else:
                self.direction = 0
                self.contact_frame = None
        self.last_ball = ball
        self.last_ball_frame = self.frame
        return bounced

    def get_state(self):
        return deepcopy(vars(self))

    def set_state(self, state):
        if not isinstance(state, dict) or set(state) != set(vars(self)):
            raise ValueError("invalid paddle detector state")
        for name in ("frame", "last_ball_frame", "last_bounce_frame"):
            if isinstance(state[name], bool) or not isinstance(state[name], int):
                raise ValueError("invalid paddle detector frame counter")
        if state["direction"] not in (-1, 0, 1) or isinstance(state["direction"], bool):
            raise ValueError("invalid paddle detector direction")
        if state["contact_frame"] is not None and (
                isinstance(state["contact_frame"], bool)
                or not isinstance(state["contact_frame"], int)):
            raise ValueError("invalid paddle detector contact counter")
        ball = state["last_ball"]
        if ball is not None and (not isinstance(ball, (tuple, list)) or len(ball) != 4
                                 or any(not isinstance(v, int) or isinstance(v, bool) for v in ball)):
            raise ValueError("invalid paddle detector ball coordinates")
        self.__dict__.update(deepcopy(state))


class RallyRewardTracker(gym.Wrapper):
    """Track an additive, capped bonus while returning original game rewards.

    A real paddle bounce pays ``bounce_bonus``. Subsequent positive game points
    increase that rally's multiplier from 1 to ``rally_bonus_max_multiplier``;
    only its increment is paid. Thus *all auxiliary reward* between two paddle
    bounces is at most ``bounce_bonus * rally_bonus_max_multiplier``. Original
    brick rewards are neither clipped nor capped by this helper.
    """

    def __init__(self, env, *, bounce_bonus=1.0, rally_bonus_step=0.2,
                 rally_bonus_max_multiplier=2.0):
        super().__init__(env)
        self.bounce_bonus = _number(bounce_bonus, "bounce_bonus")
        self.rally_bonus_step = _number(rally_bonus_step, "rally_bonus_step")
        self.rally_bonus_max_multiplier = _number(
            rally_bonus_max_multiplier, "rally_bonus_max_multiplier", 1.0)
        if not math.isfinite(self.bounce_bonus * self.rally_bonus_max_multiplier):
            raise ValueError("maximum rally bonus must be finite")
        self.detector_version = 1
        self.detector = PaddleBounceDetector()
        self.bonus_total = 0.0
        self.raw_reward_total = 0.0
        self.bounces_total = 0
        self.rally_points = 0.0
        self.rally_active = False
        self.previous_lives = None

    def _clear_rally(self):
        self.rally_points = 0.0
        self.rally_active = False
        self.detector.reset()

    def _multiplier(self, points):
        return min(self.rally_bonus_max_multiplier, 1.0 + self.rally_bonus_step * points)

    def snapshot(self):
        return {
            "bonus_total": self.bonus_total, "raw_reward_total": self.raw_reward_total,
            "bounces_total": self.bounces_total, "rally_points": self.rally_points,
            "rally_active": self.rally_active,
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._clear_rally()
        self.previous_lives = info.get("lives")
        if self.previous_lives is None:
            raise ValueError("rally rewards require Breakout's lives counter")
        self.detector.observe(observation)
        if self.detector.locate(observation)[1] is None:
            raise ValueError("rally rewards could not identify the native Breakout paddle")
        return observation, dict(info, rally_reward=self.snapshot())

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        reward = _number(raw_reward, "Breakout game reward")
        lives = info.get("lives")
        if lives is None:
            raise ValueError("rally rewards require Breakout's lives counter")
        self.raw_reward_total += reward
        lost_life = self.previous_lives is not None and lives < self.previous_lives
        if lost_life:
            self._clear_rally()
            bounced = False
        else:
            bounced = self.detector.observe(observation)
        bonus = 0.0
        if bounced:
            self.bounces_total += 1
            self.rally_active = True
            self.rally_points = 0.0
            bonus += self.bounce_bonus
        if self.rally_active and reward > 0:
            before = self._multiplier(self.rally_points)
            self.rally_points += reward
            bonus += self.bounce_bonus * (self._multiplier(self.rally_points) - before)
        self.bonus_total += bonus
        self.previous_lives = lives
        if terminated or truncated:
            self._clear_rally()
        return observation, raw_reward, terminated, truncated, dict(info, rally_reward=self.snapshot())

    def get_state(self):
        return self.snapshot() | {
            "version": self.detector_version,
            "config": {name: getattr(self, name) for name in (
                "bounce_bonus", "rally_bonus_step", "rally_bonus_max_multiplier")},
            "previous_lives": self.previous_lives,
            "detector": self.detector.get_state(),
        }

    def set_state(self, state):
        expected = self.get_state()
        if (not isinstance(state, dict) or set(state) != set(expected)
                or state["version"] != self.detector_version
                or state["config"] != expected["config"]):
            raise ValueError("rally reward checkpoint/configuration mismatch")
        numbers = {name: _number(state[name], name) for name in (
            "bonus_total", "raw_reward_total", "rally_points")}
        count = state["bounces_total"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid rally bounce counter")
        if not isinstance(state["rally_active"], bool):
            raise ValueError("invalid rally active flag")
        lives = state["previous_lives"]
        if lives is not None and (isinstance(lives, bool) or not isinstance(lives, int) or lives < 0):
            raise ValueError("invalid rally lives counter")
        detector = PaddleBounceDetector()
        detector.set_state(state["detector"])
        self.__dict__.update(numbers)
        self.bounces_total = count
        self.rally_active = state["rally_active"]
        self.previous_lives = lives
        self.detector = detector
