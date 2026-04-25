import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass

PHASES = ["NS_straight", "NS_left", "EW_straight", "EW_left"]
MIN_GREEN = 10
MAX_GREEN = 60
STATE_SIZE = 6
ACTION_SIZE = 2


@dataclass
class Sample:
    state: list[float]
    label: int


class TinyMLP:
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, seed: int = 12345):
        self.rng = random.Random(seed)
        dims = [input_size] + hidden_sizes + [output_size]
        self.layers = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            std = math.sqrt(2.0 / fan_in)
            w = [[self.rng.gauss(0.0, std) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [0.0 for _ in range(fan_out)]
            self.layers.append({"W": w, "b": b})

    def forward(self, x: list[float]):
        activations = [x[:]]
        pre_acts = []
        cur = x[:]
        for li, layer in enumerate(self.layers):
            rows = len(layer["W"])
            cols = len(layer["b"])
            out = [0.0 for _ in range(cols)]
            for j in range(cols):
                s = layer["b"][j]
                for i in range(rows):
                    s += cur[i] * layer["W"][i][j]
                out[j] = s
            pre_acts.append(out)
            if li < len(self.layers) - 1:
                cur = [max(0.0, v) for v in out]
            else:
                cur = out
            activations.append(cur)
        return activations, pre_acts

    def predict(self, x: list[float]) -> int:
        acts, _ = self.forward(x)
        q = acts[-1]
        return 1 if q[1] > q[0] else 0

    def train(self, samples: list[Sample], epochs: int = 25, lr: float = 0.0015, batch_size: int = 128):
        for _ in range(epochs):
            self.rng.shuffle(samples)
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                if not batch:
                    continue

                gW = []
                gB = []
                for layer in self.layers:
                    rows = len(layer["W"])
                    cols = len(layer["b"])
                    gW.append([[0.0 for _ in range(cols)] for _ in range(rows)])
                    gB.append([0.0 for _ in range(cols)])

                for sample in batch:
                    acts, pre = self.forward(sample.state)
                    logits = acts[-1]
                    m = max(logits)
                    exps = [math.exp(v - m) for v in logits]
                    z = exps[0] + exps[1]
                    probs = [exps[0] / z, exps[1] / z]

                    delta = probs[:]
                    delta[sample.label] -= 1.0

                    for li in reversed(range(len(self.layers))):
                        a_prev = acts[li]
                        rows = len(self.layers[li]["W"])
                        cols = len(self.layers[li]["b"])

                        for j in range(cols):
                            gB[li][j] += delta[j]
                            for i in range(rows):
                                gW[li][i][j] += delta[j] * a_prev[i]

                        if li > 0:
                            next_delta = [0.0 for _ in range(rows)]
                            for i in range(rows):
                                s = 0.0
                                for j in range(cols):
                                    s += delta[j] * self.layers[li]["W"][i][j]
                                # ReLU gate from previous layer pre-activation
                                s = s if pre[li - 1][i] > 0.0 else 0.0
                                next_delta[i] = s
                            delta = next_delta

                scale = lr / max(1, len(batch))
                for li, layer in enumerate(self.layers):
                    rows = len(layer["W"])
                    cols = len(layer["b"])
                    for j in range(cols):
                        layer["b"][j] -= scale * gB[li][j]
                    for i in range(rows):
                        for j in range(cols):
                            layer["W"][i][j] -= scale * gW[li][i][j]


class SyntheticTraffic:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.queues = [self.rng.uniform(4.0, 14.0), self.rng.uniform(4.0, 14.0), self.rng.uniform(1.0, 6.0), self.rng.uniform(1.0, 6.0)]
        self.phase = 0
        self.elapsed = 0
        self.step_idx = 0
        self.regime = self.rng.choice(("ns_straight", "ns_left", "ew_straight", "ew_left"))

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _arrival_profile(self):
        if self.step_idx % 40 == 0:
            self.regime = self.rng.choice(("ns_straight", "ns_left", "ew_straight", "ew_left"))

        base = 0.10 + 0.04 * math.sin(self.step_idx / 90.0)
        ns_s = base + self.rng.uniform(0.00, 0.10)
        ew_s = base + self.rng.uniform(0.00, 0.10)
        ns_l = base * 0.6 + self.rng.uniform(0.00, 0.07)
        ew_l = base * 0.6 + self.rng.uniform(0.00, 0.07)

        if self.regime == "ns_straight":
            ns_s += 0.95
            ns_l += 0.18
        elif self.regime == "ns_left":
            ns_l += 0.95
            ns_s += 0.20
        elif self.regime == "ew_straight":
            ew_s += 0.95
            ew_l += 0.18
        else:
            ew_l += 0.95
            ew_s += 0.20

        return (
            self._clip(ns_s, 0.02, 1.20),
            self._clip(ew_s, 0.02, 1.20),
            self._clip(ns_l, 0.01, 1.00),
            self._clip(ew_l, 0.01, 1.00),
        )

    def _state_norm(self):
        qn = 60.0
        return [
            self._clip(self.queues[0] / qn, 0.0, 1.0),
            self._clip(self.queues[1] / qn, 0.0, 1.0),
            self._clip(self.queues[2] / qn, 0.0, 1.0),
            self._clip(self.queues[3] / qn, 0.0, 1.0),
            self.phase / 3.0,
            self._clip(self.elapsed / MAX_GREEN, 0.0, 1.0),
        ]

    def _pressure(self, idx: int) -> float:
        return self.queues[idx]

    def _best_alt(self, idx: int) -> float:
        vals = [self.queues[i] for i in range(4) if i != idx]
        return max(vals)

    def clone(self):
        other = SyntheticTraffic(0)
        other.rng.setstate(self.rng.getstate())
        other.queues = self.queues[:]
        other.phase = self.phase
        other.elapsed = self.elapsed
        other.step_idx = self.step_idx
        other.regime = self.regime
        return other

    def _heuristic_action(self) -> int:
        if self.elapsed < MIN_GREEN:
            return 0
        if self.elapsed >= MAX_GREEN:
            return 1

        cur = self._pressure(self.phase)
        alt = self._best_alt(self.phase)
        elapsed_norm = self.elapsed / MAX_GREEN
        hold_score = 1.05 * cur - 0.62 * alt + 0.55 * (1.0 - elapsed_norm)
        switch_score = 1.35 * (alt - cur) + 1.20 * elapsed_norm
        return 1 if switch_score > hold_score else 0

    def teacher_action(self) -> int:
        if self.elapsed < MIN_GREEN:
            return 0
        if self.elapsed >= MAX_GREEN:
            return 1

        def estimate_value(action: int, horizon: int = 14) -> float:
            sim = self.clone()
            total = 0.0
            discount = 1.0
            for h in range(horizon):
                a = action if h == 0 else sim._heuristic_action()
                reward, _, _ = sim.step(a)
                total += discount * reward
                discount *= 0.95
            return total

        hold_v = estimate_value(0)
        switch_v = estimate_value(1)
        return 1 if switch_v > hold_v else 0

    def step(self, action: int):
        if self.elapsed < MIN_GREEN:
            action = 0
        if self.elapsed >= MAX_GREEN:
            action = 1

        switched = 1 if action == 1 else 0
        before = sum(self.queues)

        ns_arr, ew_arr, ns_left_arr, ew_left_arr = self._arrival_profile()
        # Arrivals
        self.queues[0] += ns_arr * 2.8
        self.queues[2] += ns_left_arr * 2.2
        self.queues[1] += ew_arr * 2.8
        self.queues[3] += ew_left_arr * 2.2

        cleared = 0.0
        if switched:
            self.phase = (self.phase + 1) % 4
            self.elapsed = 0
        else:
            self.elapsed += 1
            cap = 3.4 if self.phase in (0, 2) else 2.8
            idx = self.phase
            c = min(self.queues[idx], cap)
            self.queues[idx] -= c
            cleared += c

        after = sum(self.queues)
        ns_pressure = self.queues[0] + self.queues[2]
        ew_pressure = self.queues[1] + self.queues[3]
        queue_delta = before - after
        reward = (0.66 * queue_delta) + (0.32 * cleared) - (0.015 * after) - (0.012 * abs(ns_pressure - ew_pressure)) - (0.025 * switched)

        for i in range(4):
            self.queues[i] = self._clip(self.queues[i], 0.0, 120.0)

        self.step_idx += 1
        return reward, after, switched


def build_dataset(seed: int, samples: int) -> list[Sample]:
    rng = random.Random(seed)
    data: list[Sample] = []
    env = SyntheticTraffic(seed)
    while len(data) < samples:
        # Blend realistic trajectories with random state coverage.
        if rng.random() < 0.75:
            state = env._state_norm()
            label = env.teacher_action()
            env.step(label)
        else:
            env.queues = [rng.uniform(0.0, 60.0), rng.uniform(0.0, 60.0), rng.uniform(0.0, 40.0), rng.uniform(0.0, 40.0)]
            env.phase = rng.randint(0, 3)
            env.elapsed = rng.randint(0, MAX_GREEN)
            env.regime = rng.choice(("ns_straight", "ns_left", "ew_straight", "ew_left"))
            state = env._state_norm()
            label = env.teacher_action()

        data.append(Sample(state=state, label=label))
    return data


def evaluate_policy(model: TinyMLP, episodes: int, steps: int, seed: int):
    def run(policy_name: str):
        totals = {"queue": 0.0, "reward": 0.0, "switches": 0.0}
        for ep in range(episodes):
            env = SyntheticTraffic(seed + ep * 17)
            ep_q = 0.0
            ep_r = 0.0
            ep_s = 0.0
            for _ in range(steps):
                if policy_name == "fixed":
                    action = 1 if env.elapsed >= 30 else 0
                else:
                    action = model.predict(env._state_norm())
                reward, total_q, switched = env.step(action)
                ep_q += total_q
                ep_r += reward
                ep_s += switched
            totals["queue"] += ep_q / steps
            totals["reward"] += ep_r / steps
            totals["switches"] += ep_s
        for k in totals:
            totals[k] /= episodes
        return totals

    fixed = run("fixed")
    rl = run("rl")
    queue_improvement = ((fixed["queue"] - rl["queue"]) / max(1e-8, fixed["queue"])) * 100.0
    reward_improvement = ((rl["reward"] - fixed["reward"]) / max(1e-8, abs(fixed["reward"]))) * 100.0
    return {
        "fixed": fixed,
        "rl": rl,
        "queue_improvement_pct": queue_improvement,
        "reward_improvement_pct": reward_improvement,
    }


def aggregate_policy_states(model: TinyMLP, seed: int, episodes: int, steps: int) -> list[Sample]:
    samples: list[Sample] = []
    for ep in range(episodes):
        env = SyntheticTraffic(seed + 30000 + ep * 13)
        for _ in range(steps):
            state = env._state_norm()
            label = env.teacher_action()
            samples.append(Sample(state=state, label=label))
            action = model.predict(state)
            env.step(action)
    return samples


def export_model(model: TinyMLP, output_path: str, total_steps: int, metrics: dict, seed: int):
    layers = []
    for layer in model.layers:
        rows = len(layer["W"])
        cols = len(layer["b"])
        flat_w = []
        for i in range(rows):
            for j in range(cols):
                flat_w.append(layer["W"][i][j])
        layers.append({"W": flat_w, "b": layer["b"][:]})

    payload = {
        "version": 1,
        "stateSize": STATE_SIZE,
        "actionSize": ACTION_SIZE,
        "hiddenSizes": [32, 32],
        "trainedAt": int(time.time() * 1000),
        "totalSteps": total_steps,
        "trainingSource": "offline-synthetic-data",
        "seed": seed,
        "metrics": metrics,
        "layers": layers,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Train an RL-compatible traffic model offline using synthetic data.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--samples", type=int, default=18000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=str, default=os.path.join("__pycache__", "rl_offline_model.json"))
    args = parser.parse_args()

    data = build_dataset(args.seed, args.samples)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    model = TinyMLP(STATE_SIZE, [32, 32], ACTION_SIZE, seed=args.seed)
    model.train(train_data, epochs=args.epochs, lr=0.0015, batch_size=128)

    # DAgger-style improvement: collect states visited by current policy,
    # label with teacher, and continue training on aggregated data.
    aggregated = list(train_data)
    for _ in range(4):
        policy_states = aggregate_policy_states(model, args.seed, episodes=8, steps=700)
        aggregated.extend(policy_states)
        model.train(aggregated, epochs=10, lr=0.0012, batch_size=128)

    correct = 0
    for s in val_data:
        if model.predict(s.state) == s.label:
            correct += 1
    val_acc = correct / max(1, len(val_data))

    metrics = evaluate_policy(model, episodes=16, steps=1500, seed=args.seed)
    metrics["validation_accuracy"] = val_acc

    payload = export_model(
        model=model,
        output_path=args.output,
        total_steps=args.epochs * len(train_data),
        metrics=metrics,
        seed=args.seed,
    )

    print("Offline model training complete")
    print(f"Output: {args.output}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Queue improvement vs fixed: {metrics['queue_improvement_pct']:.2f}%")
    print(f"Reward improvement vs fixed: {metrics['reward_improvement_pct']:.2f}%")
    print(f"Model version: {payload['version']} | Hidden: {payload['hiddenSizes']}")


if __name__ == "__main__":
    main()
