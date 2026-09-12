from chat_with_fly import run_simulation

ACTIONS = {
    "LEFT": [
        720575940616012061,   # AOTU025 left
    ],

    "RIGHT": [
        720575940639182424,   # AOTU025 right
    ],

    "UP_FORWARD": [
        720575940627652358,
        720575940635872101,
    ],

    "DOWN_BACKWARD": [
        720575940616026939,
        720575940631082808,
        720575940640331472,
        720575940610236514,
    ],
}


def rate(out, name):
    return out.get(name, {}).get("rate_hz", 0.0)


for action, neurons in ACTIONS.items():

    print("\n" + "=" * 70)
    print(action)
    print("stimulating:", neurons)
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

    print("ACTIVE:")

    for name, data in out.items():
        if data.get("rate_hz", 0) > 0:
            print(
                f"{name:20s}"
                f"{data['rate_hz']:6.1f} Hz"
            )