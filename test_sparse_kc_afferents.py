from chat_with_fly import run_simulation


TESTS = {
    "VA3_A": 720575940656090017,
    "DM2_A": 720575940630024566,
    "DL5_A": 720575940617207185,
    "DM2_B": 720575940638633535,
    "VM6":   720575940632698797,
    "DL1_A": 720575940616079035,
    "VC1":   720575940637526190,
    "VA3_B": 720575940629009513,
}


patterns = {}


for name, neuron_id in TESTS.items():

    print()
    print("=" * 70)
    print(name)
    print("Stimulating:", neuron_id)
    print("=" * 70)

    result = run_simulation(
        neuron_ids=[neuron_id],
        freq_hz=[50],
        duration_sec=0.1,
        use_memory=False
    )

    mb = result.get(
        "mushroom_body",
        {}
    )

    active_kcs = set(
        mb.get(
            "active_kc_indices",
            []
        )
    )

    patterns[name] = active_kcs

    print(
        "Active KCs:",
        len(active_kcs)
    )

    print(
        "First KC indices:",
        sorted(active_kcs)[:30]
    )

    print(
        "Total active neurons:",
        result["active_neurons"]
    )


# -------------------------------------
# Compare pattern overlap
# -------------------------------------

print()
print("=" * 70)
print("KC PATTERN OVERLAP")
print("=" * 70)

names = list(patterns)

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        a_name = names[i]
        b_name = names[j]

        a = patterns[a_name]
        b = patterns[b_name]

        if not a or not b:
            overlap = 0.0
        else:
            overlap = (
                len(a & b)
                / len(a | b)
            )

        print(
            f"{a_name:8s} vs "
            f"{b_name:8s}: "
            f"{overlap:.3f}"
        )