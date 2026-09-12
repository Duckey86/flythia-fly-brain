from chat_with_fly import run_simulation


TESTS = {
    # UP candidates
    "UP_MBON17": [
        720575940638774606
    ],

    "UP_MBON18_A": [
        720575940624539284
    ],

    "UP_MBON18_B": [
        720575940622997453
    ],

    "UP_MBON28": [
        720575940614892182
    ],

    # DOWN candidates
    "DOWN_MBON26_B": [
        720575940629981440
    ],

    "DOWN_MBON32": [
        720575940609959637
    ],

    "DOWN_MBON35_A": [
        720575940632943277
    ],

    "DOWN_MBON35_B": [
        720575940637902938
    ],
}


def rate(out, name):
    return out.get(name, {}).get("rate_hz", 0.0)


for name, neurons in TESTS.items():

    print()
    print("=" * 70)
    print(name)
    print("Stimulating:", neurons)
    print("=" * 70)

    result = run_simulation(
        neuron_ids=neurons,
        freq_hz=[200],
        duration_sec=0.1,
        use_memory=False
    )

    out = result["output_neuron_activity"]

    print(
        "DNa01 L/R:",
        rate(out, "DNa01_left"),
        rate(out, "DNa01_right")
    )

    print(
        "DNa02 L/R:",
        rate(out, "DNa02_left"),
        rate(out, "DNa02_right")
    )

    print(
        "P9 L/R:",
        rate(out, "P9_oDN1_left"),
        rate(out, "P9_oDN1_right")
    )

    print(
        "MDNs:",
        rate(out, "MDN_1"),
        rate(out, "MDN_2"),
        rate(out, "MDN_3"),
        rate(out, "MDN_4")
    )

    print("ACTIVE OUTPUTS:")

    for neuron, data in out.items():
        if data.get("rate_hz", 0) > 0:
            print(
                f"  {neuron:20s}"
                f"{data['rate_hz']:6.1f} Hz"
            )