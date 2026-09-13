import math
import random
import statistics

import torch

from fly_mb_policy import ACTIONS
from trainable_torch_mbon_policy_16 import TrainableTorchMBONPolicy16


STEP_DEG = 5.0
BIN_DEG = 22.5


class PopulationAngleProbe(TrainableTorchMBONPolicy16):
    """
    Experimental continuous-angle readout built on the existing trained
    16-state fly.

    It does NOT modify training or memory.

    The 16 disjoint KC groups are treated as 16 preferred-angle populations
    spaced every 22.5 degrees. For an angle between two populations, a
    proportional subset of KCs from each population is recruited.

    The full 138k-neuron CUDA connectome is then simulated. Exact target dx/dy
    is never used to construct the output vector. The output vector is decoded
    only from opponent MBON scores:

        x = RIGHT - LEFT
        y = DOWN - UP
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.angle_state = {}
        self.state_angle = {}
        self.population_kcs = {}

        # Build deterministic preferred-angle populations.
        for state_key, info in self.rich_patterns.items():
            sx, sy = [int(v) for v in info["direction"]]
            priority = str(info["priority"])

            if sx == 0 or sy == 0:
                vx = float(sx)
                vy = float(sy)
            elif priority == "X_DOMINANT":
                vx = float(sx)
                vy = float(sy) * math.tan(math.radians(22.5))
            elif priority == "Y_DOMINANT":
                vx = float(sx) * math.tan(math.radians(22.5))
                vy = float(sy)
            else:
                vx = float(sx)
                vy = float(sy)

            angle = math.degrees(math.atan2(vy, vx)) % 360.0
            bin_index = int(round(angle / BIN_DEG)) % 16

            if bin_index in self.angle_state:
                raise RuntimeError(
                    f"Two KC states mapped to angular bin {bin_index}: "
                    f"{self.angle_state[bin_index]} and {state_key}"
                )

            self.angle_state[bin_index] = state_key
            self.state_angle[state_key] = angle

            # Fixed shuffle so fractional recruitment does not depend on the
            # original JSON ordering.
            ids = [int(x) for x in info["kc_indices"]]
            seed = sum((i + 1) * ord(ch) for i, ch in enumerate(state_key))
            rng = random.Random(seed)
            rng.shuffle(ids)
            self.population_kcs[state_key] = ids

        if len(self.angle_state) != 16:
            raise RuntimeError(
                f"Expected 16 preferred-angle populations, got "
                f"{len(self.angle_state)}"
            )

        # Concatenate every state's direct KC->MBON correction plan.
        # The 16-state class already verifies that KC groups are disjoint.
        self.population_baseline_plan = self._combine_plans(
            self.rich_baseline_plan
        )
        self.population_learned_plan = self._combine_plans(
            self.rich_learned_plan
        )

        print()
        print("Preferred-angle KC populations:")
        for i in range(16):
            key = self.angle_state[i]
            print(
                f"  {i * BIN_DEG:6.1f} deg -> "
                f"{key:28s} "
                f"KCs={len(self.population_kcs[key]):2d}"
            )
        print()

    def _combine_plans(self, source):
        combined = {}

        for action in ACTIONS:
            all_pre = []
            all_delta = []
            post_idx = None

            for state_key in self.rich_states:
                pre, delta, this_post = source[state_key][action]
                all_pre.append(pre)
                all_delta.append(delta)

                if post_idx is None:
                    post_idx = int(this_post)
                elif post_idx != int(this_post):
                    raise RuntimeError(
                        f"Inconsistent postsynaptic MBON for {action}"
                    )

            combined[action] = (
                torch.cat(all_pre)
                if all_pre
                else torch.empty(
                    0, dtype=torch.long, device=self.device
                ),
                torch.cat(all_delta)
                if all_delta
                else torch.empty(
                    0, dtype=torch.float32, device=self.device
                ),
                post_idx,
            )

        return combined

    def _population_mix(self, angle_deg):
        """
        Return the two adjacent preferred-angle populations and linear weights.
        Example: 37 deg blends the 22.5 and 45 deg KC groups.
        """
        angle = float(angle_deg) % 360.0
        position = angle / BIN_DEG

        lower_bin = int(math.floor(position)) % 16
        frac = position - math.floor(position)
        upper_bin = (lower_bin + 1) % 16

        lower_state = self.angle_state[lower_bin]
        upper_state = self.angle_state[upper_bin]

        return (
            (lower_state, 1.0 - frac),
            (upper_state, frac),
        )

    def _active_population_kcs(self, angle_deg):
        active = []
        mix = self._population_mix(angle_deg)

        for state_key, strength in mix:
            ids = self.population_kcs[state_key]

            # Population coding: encode strength by how much of the real
            # anatomical KC population is recruited.
            count = int(round(strength * len(ids)))
            count = max(0, min(len(ids), count))

            active.extend(ids[:count])

        # The populations are disjoint, but unique() also protects against
        # accidental duplication at exact wraparound boundaries.
        active = sorted(set(active))

        return torch.tensor(
            active,
            dtype=torch.long,
            device=self.device,
        ), mix

    @torch.no_grad()
    def _run_population(self, angle_deg, learned):
        active_kcs, mix = self._active_population_kcs(angle_deg)

        conductance, delay_buffer, spikes, v, refrac = (
            self.model.state_init()
        )

        if active_kcs.numel() > 0:
            # Same stimulation mechanism as the existing validated policy.
            v[:, active_kcs] = -44.0

        peak = torch.full(
            (len(self.action_order),),
            -float("inf"),
            device=self.device,
        )

        plan = (
            self.population_learned_plan
            if learned
            else self.population_baseline_plan
        )

        for _ in range(self.num_steps):
            weighted = torch.matmul(
                spikes,
                self.weights.transpose(0, 1),
            )

            self._apply_correction(
                weighted,
                spikes,
                plan,
            )

            conductance, delay_buffer, spikes, v, refrac = (
                self.model.neurons(
                    self.model.scale * weighted,
                    conductance,
                    delay_buffer,
                    spikes,
                    v,
                    refrac,
                )
            )

            peak = torch.maximum(
                peak,
                conductance[0, self.action_indices],
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        values = peak.cpu().tolist()

        return {
            action: float(values[i])
            for i, action in enumerate(self.action_order)
        }, mix, int(active_kcs.numel())

    def get_population_scores(self, angle_deg):
        learned, mix, kc_count = self._run_population(
            angle_deg,
            learned=True,
        )
        baseline, _, _ = self._run_population(
            angle_deg,
            learned=False,
        )

        scores = {
            action: learned[action] - baseline[action]
            for action in self.action_order
        }

        return scores, mix, kc_count

    @staticmethod
    def scores_to_vector(scores):
        # IMPORTANT: exact target geometry is NOT used here.
        x = float(scores["RIGHT"]) - float(scores["LEFT"])
        y = float(scores["DOWN"]) - float(scores["UP"])

        length = math.hypot(x, y)

        if length < 1e-12:
            return 0.0, 0.0

        return x / length, y / length

    @staticmethod
    def vector_angle(vx, vy):
        if abs(vx) < 1e-12 and abs(vy) < 1e-12:
            return None
        return math.degrees(math.atan2(vy, vx)) % 360.0

    @staticmethod
    def angular_error(target, predicted):
        if predicted is None:
            return 180.0
        return abs((predicted - target + 180.0) % 360.0 - 180.0)


def main():
    print("Loading continuous population-angle probe...")
    print("This does NOT change your trained memory.")
    print()

    brain = PopulationAngleProbe(
        run_time_ms=3,
        device="cuda",
    )

    results = []
    angle = 0.0

    print(
        "target   predicted   error    vector           KCs   KC blend"
    )
    print("-" * 86)

    while angle < 360.0 - 1e-9:
        scores, mix, kc_count = brain.get_population_scores(angle)
        vx, vy = brain.scores_to_vector(scores)
        predicted = brain.vector_angle(vx, vy)
        error = brain.angular_error(angle, predicted)

        blend_text = (
            f"{mix[0][0]} {mix[0][1]:.2f} + "
            f"{mix[1][0]} {mix[1][1]:.2f}"
        )

        predicted_text = (
            f"{predicted:8.2f}"
            if predicted is not None
            else "    NONE"
        )

        print(
            f"{angle:6.1f}°  "
            f"{predicted_text}°  "
            f"{error:6.2f}°   "
            f"({vx:+.3f},{vy:+.3f})   "
            f"{kc_count:3d}   "
            f"{blend_text}"
        )

        results.append(
            {
                "target": angle,
                "predicted": predicted,
                "error": error,
                "vx": vx,
                "vy": vy,
                "kc_count": kc_count,
                "scores": scores,
                "mix": mix,
            }
        )

        angle += STEP_DEG

    errors = [r["error"] for r in results]

    print()
    print("============================================================")
    print("CONTINUOUS NEURAL ANGLE PROBE")
    print("============================================================")
    print(f"Samples:      {len(errors)}")
    print(f"Mean error:   {statistics.mean(errors):.2f} deg")
    print(f"Median error: {statistics.median(errors):.2f} deg")
    print(f"Max error:    {max(errors):.2f} deg")
    print()

    under_5 = sum(e <= 5.0 for e in errors)
    under_10 = sum(e <= 10.0 for e in errors)
    under_15 = sum(e <= 15.0 for e in errors)

    print(f"Within  5°:   {under_5}/{len(errors)}")
    print(f"Within 10°:   {under_10}/{len(errors)}")
    print(f"Within 15°:   {under_15}/{len(errors)}")
    print()

    print(
        "If the error is already low, the existing learned KC->MBON "
        "weights interpolate surprisingly well."
    )
    print(
        "If it is poor, that is expected: the current memory was trained "
        "for categorical winners, not MBON score ratios. The next step is "
        "continuous-angle dopamine training."
    )


if __name__ == "__main__":
    main()
