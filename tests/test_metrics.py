"""Metric reconstruction and comparable checkpoints, without training or GPU use."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.ConceptCheck import metrics


class MetricTests(unittest.TestCase):
    def report(self, step=1_000_000, rewards=(0, 2, 4, 10)):
        seeds = list(range(90173, 90173 + len(rewards)))
        return {"completed_at_utc": "2026-11-14T00:00:00Z", "checkpoint_step": step,
                "experiment_algorithm": "double-dqn", "evaluation_seed": seeds[0],
                "episode_seeds": seeds, "policy_epsilon": 0.0, "max_steps": 27000,
                "auto_fire_after_life_loss": True, "environment": {"mode": 0, "difficulty": 0},
                "activation": "leaky-relu", "deterministic": True,
                "summary": {"episodes": len(rewards), "mean_reward": 999},
                "episodes": [{"seed": seed, "reward": reward, "steps": 100 + i,
                              "hit_step_limit": i == len(rewards) - 1}
                             for i, (seed, reward) in enumerate(zip(seeds, rewards))]}

    def write(self, root, report):
        path = root / f"eval-{report['checkpoint_step']:09d}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_recomputes_population_metrics_and_quartiles_from_raw_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.report())
            row, = metrics.collect_metrics(root)
        self.assertEqual(row["mean_reward"], 4)
        self.assertEqual(row["median_reward"], 3)
        self.assertAlmostEqual(row["std_reward"], math.sqrt(14))
        self.assertEqual((row["q25"], row["q75"]), (1.5, 5.5))
        self.assertEqual(row["hit_step_limit"], 1)
        self.assertEqual(row["mean_steps"], 101.5)

    def test_score_threshold_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.report(rewards=(0, 864, 900)))
            rows = metrics.collect_metrics(root)
            metrics.write_csv(rows, root / "metrics.csv")
            with (root / "metrics.csv").open(newline="") as handle:
                saved, = list(csv.DictReader(handle))
            self.assertEqual(saved["wins_864"], "2")
            self.assertAlmostEqual(float(saved["win_rate_864"]), 2 / 3)

    def test_skips_incomplete_reports_nonmillion_steps_and_nested_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, self.report())
            self.write(root, self.report(500_000))
            incomplete = self.report(2_000_000)
            incomplete.pop("completed_at_utc")
            self.write(root, incomplete)
            nested = root / "preflight"
            nested.mkdir()
            self.write(nested, self.report(3_000_000))
            self.assertEqual([r["decisions"] for r in metrics.collect_metrics(root)], [1_000_000])

    def test_rejects_different_seeds_or_game_protocol(self):
        for field, value in (("environment", {"mode": 4, "difficulty": 0}),
                             ("evaluation_seed", 7741), ("policy_epsilon", .1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write(root, self.report())
                other = self.report(2_000_000)
                other[field] = value
                self.write(root, other)
                with self.assertRaisesRegex(ValueError, "protocol or seeds changed"):
                    metrics.collect_metrics(root)

    def test_rejects_invalid_or_incomplete_episode_data(self):
        for change in (lambda r: r["episodes"][0].update(reward=float("nan")),
                       lambda r: r["episodes"][0].update(steps=0),
                       lambda r: r["episodes"][0].update(hit_step_limit=1),
                       lambda r: r["episodes"].pop(),
                       lambda r: r["episodes"][0].update(seed=1)):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                report = self.report()
                change(report)
                self.write(root, report)
                with self.assertRaises(ValueError):
                    metrics.collect_metrics(root)

    def test_rejects_checkpoint_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write(root, self.report())
            report = self.report(2_000_000)
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint step"):
                metrics.collect_metrics(root)

    def test_completed_evaluation_before_controller_tags_are_added(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.report()
            self.write(root, report)
            report.pop("checkpoint_step")
            report.pop("experiment_algorithm")
            (root / "eval-001000000.json").write_text(json.dumps(report), encoding="utf-8")
            (root / "experiment.config.json").write_text(json.dumps({"algorithm": "double-dqn"}))
            row, = metrics.collect_metrics(root)
            self.assertEqual((row["decisions"], row["algorithm"]), (1_000_000, "double-dqn"))


if __name__ == "__main__":
    unittest.main()
