"""Collect completed million-step evaluations and draw their learning curve."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics as stats


FIELDS = ("decisions", "algorithm", "mode", "difficulty", "episodes", "mean_reward", "median_reward",
          "min_reward", "max_reward", "std_reward", "wins_864", "win_rate_864",
          "hit_step_limit", "mean_steps", "q25", "q75")
PROTOCOL = ("evaluation_seed", "episode_seeds", "policy_epsilon", "max_steps",
            "auto_fire_after_life_loss", "environment", "activation", "deterministic")


def collect_metrics(run_dir):
    """Recompute metrics from raw games; one fixed protocol and seed set per curve."""
    rows, protocol, steps_seen = [], None, set()
    config_path = Path(run_dir) / "experiment.config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    for path in sorted(Path(run_dir).glob("eval-*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report.get("completed_at_utc"):
                continue
            match = re.fullmatch(r"eval-(\d+)\.json", path.name)
            step = report.get("checkpoint_step", int(match[1]) if match else None)
            if (not match or type(step) is not int or step <= 0
                    or step != int(match[1]) or step in steps_seen):
                raise ValueError("invalid or duplicate checkpoint step")
            if step % 1_000_000:
                continue
            games, seeds = report["episodes"], report["episode_seeds"]
            if (not games or len(games) != report["summary"]["episodes"]
                    or [game["seed"] for game in games] != seeds
                    or len(seeds) != len(set(seeds))):
                raise ValueError("episode count or seed list is inconsistent")
            algorithm = report.get("experiment_algorithm", config.get("algorithm"))
            if algorithm not in ("dqn", "double-dqn"):
                raise ValueError("missing algorithm; retain experiment.config.json beside reports")
            current = [report[key] for key in PROTOCOL] + [algorithm]
            if protocol is not None and current != protocol:
                raise ValueError("evaluation protocol or seeds changed between checkpoints")
            rewards, lengths = [], []
            for game in games:
                reward, length = game["reward"], game["steps"]
                if (isinstance(reward, bool) or not isinstance(reward, (float, int))
                        or not math.isfinite(reward) or type(length) is not int or length <= 0
                        or type(game["hit_step_limit"]) is not bool):
                    raise ValueError("invalid reward, episode length or step-cap flag")
                rewards.append(reward)
                lengths.append(length)
            quartiles = stats.quantiles(rewards, n=4, method="inclusive") if len(games) > 1 else rewards * 3
            wins = sum(value >= 864 for value in rewards)
            rows.append(dict(zip(FIELDS, (step, algorithm, report["environment"]["mode"],
                report["environment"]["difficulty"], len(games),
                stats.mean(rewards), stats.median(rewards), min(rewards), max(rewards),
                stats.pstdev(rewards), wins, wins / len(games),
                sum(game["hit_step_limit"] for game in games), stats.mean(lengths),
                quartiles[0], quartiles[2]))))
            protocol = current
            steps_seen.add(step)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{path.name}: {error}") from error
    if not rows:
        raise ValueError("No completed million-step evaluation reports found")
    return sorted(rows, key=lambda row: row["decisions"])


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_metrics(rows, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    x = [row["decisions"] / 1_000_000 for row in rows]
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    ax.fill_between(x, [r["q25"] for r in rows], [r["q75"] for r in rows],
                    color="#2563eb", alpha=.12, label="25–75% wyników gier")
    ax.plot(x, [r["mean_reward"] for r in rows], "o-", color="#2563eb", lw=2, ms=4, label="Średnia")
    ax.plot(x, [r["median_reward"] for r in rows], color="#0f766e", lw=1.8, label="Mediana")
    ax.plot(x, [r["max_reward"] for r in rows], "--", color="#94a3b8", lw=1.2, label="Maksimum")
    best = max(rows, key=lambda row: row["mean_reward"])
    ax.annotate(f"Najlepsza średnia: {best['mean_reward']:.2f}\n{best['decisions'] / 1e6:g} mln",
                (best["decisions"] / 1e6, best["mean_reward"]), xytext=(-70, 28),
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#475569"},
                bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "none", "alpha": .9})
    label = {"dqn": "DQN", "double-dqn": "Double DQN"}.get(rows[0]["algorithm"], rows[0]["algorithm"])
    ax.set(title=f"{label} — Breakout, tryb {rows[0]['mode']}, trudność {rows[0]['difficulty']}",
           xlabel="Decyzje agenta [mln]", ylabel="Punkty w grze", ylim=(0, None))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if ax.get_ylim()[1] >= 864:
        ax.axhline(864, color="#b45309", lw=1, ls=":")
    target = " (poza skalą)" if ax.get_ylim()[1] < 864 else ""
    note = (f"Ocena co 1 mln decyzji: {rows[0]['episodes']} gier, te same seedy. "
            f"Próg 864{target}: {rows[-1]['wins_864']}/{rows[-1]['episodes']} gier w ostatniej ocenie.\n"
            "Wstęga oznacza kwartyle wyników, nie przedział ufności. Jeden przebieg treningu.")
    fig.suptitle(note, fontsize=9, color="#475569", y=-.01)
    for extension in ("png", "svg"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    svg = prefix.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, help="Output prefix, without an extension")
    args = parser.parse_args(argv)
    prefix = args.output_prefix or args.run_dir / "metrics"
    if prefix.with_suffix(".csv").name == "comparison.csv":
        parser.error("Use a separate output prefix; comparison.csv belongs to the training controller")
    try:
        rows = collect_metrics(args.run_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    write_csv(rows, prefix.with_suffix(".csv"))
    plot_metrics(rows, prefix)
    print(f"{len(rows)} checkpoints; latest: {rows[-1]['decisions']:,} decisions")
    for extension in ("csv", "png", "svg"):
        print(prefix.with_suffix("." + extension))
    return rows


if __name__ == "__main__":
    main()
