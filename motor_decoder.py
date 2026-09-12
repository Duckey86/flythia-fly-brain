def get_rate(output, name):
    neuron = output.get(name, {})
    return neuron.get("rate_hz", 0.0)


def decode_motor(output):
    # Horizontal movement
    left_turn = get_rate(output, "DNa02_left")
    right_turn = get_rate(output, "DNa02_right")

    dx = right_turn - left_turn

    # Forward movement
    forward_left = get_rate(output, "p9_oDN1_left")
    forward_right = get_rate(output, "p9_oDN1_right")

    forward = (forward_left + forward_right) / 2

    # Backward / escape movement
    backward_names = [
        "MDN_1",
        "MDN_2",
        "MDN_3",
        "MDN_4",
    ]

    backward_rates = [
        get_rate(output, name)
        for name in backward_names
    ]

    backward = sum(backward_rates) / len(backward_rates)

    # Screen coordinates:
    # positive x = right
    # positive y = down
    dy = backward - forward

    return dx, dy