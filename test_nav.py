from chat_with_fly import run_simulation


TESTS = {
    "LEFT_AOTU025": [
        720575940616012061
    ],

    "RIGHT_AOTU025": [
        720575940639182424
    ],

    "FORWARD_CB0283": [
        720575940621319924,
        720575940612990321
    ],

    "FORWARD_LT51": [
        720575940622364598,
        720575940637106032
    ],
}

def rate(output, name):
    return output.get(name, {}).get("rate_hz", 0.0)


for name, neurons in TESTS.items():

    print()
    print("=" * 70)
    print(name)
    print("stimulating:", neurons)
    print("=" * 70)

    result = run_simulation(
        neuron_ids=neurons,
        freq_hz=[200],
        duration_sec=0.1,
        use_memory=False
    )

    out = result["output_neuron_activity"]

    print("DNa01 L/R:",
          rate(out, "DNa01_left"),
          rate(out, "DNa01_right"))

    print("DNa02 L/R:",
          rate(out, "DNa02_left"),
          rate(out, "DNa02_right"))

    print("P9 L/R:",
          rate(out, "P9_oDN1_left"),
          rate(out, "P9_oDN1_right"))

    print("MDNs:",
          rate(out, "MDN_1"),
          rate(out, "MDN_2"),
          rate(out, "MDN_3"),
          rate(out, "MDN_4"))

    print("Active outputs:")

    for neuron, data in out.items():
        if data.get("rate_hz", 0) > 0:
            print(
                f"  {neuron:20s}"
                f"{data['rate_hz']:6.1f} Hz"
            )