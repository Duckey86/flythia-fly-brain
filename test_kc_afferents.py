from chat_with_fly import run_simulation


TESTS = {
    "DP1m_A": 720575940618308825,
    "DP1m_B": 720575940622726271,

    "DM1_A": 720575940630779042,
    "DM1_B": 720575940619071005,

    "DC1_A": 720575940637056887,
    "DC1_B": 720575940621529435,

    "VA2": 720575940611079236,

    "DM4": 720575940615366055,
}


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

    active_kcs = mb.get(
        "active_kc_indices",
        []
    )

    print(
        "Active KCs:",
        len(active_kcs)
    )

    print(
        "KC indices:",
        active_kcs[:30]
    )

    print(
        "Total active neurons:",
        result["active_neurons"]
    )