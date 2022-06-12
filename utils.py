import numpy as np


def make_buttons(num_buttons, size_buttons):
    buttons = []
    for i in range(num_buttons):
        buttons.append([0] * size_buttons)
    return np.array(buttons)


def define_buttons_pressed(pressed, no_op=True):
    buttons = make_buttons(len(pressed) + no_op, len(pressed) + 1)
    for i in range(len(pressed)):
        buttons[i + no_op][pressed[i]] = 1
    return buttons


def base_buttons(num_discrete):
    return define_buttons_pressed(np.arange(num_discrete) + 1)


def base_buttons_with_concat(num_discrete, concat):
    buttons = base_buttons(num_discrete)
    for i in range(len(concat)):
        buttons = np.append(buttons, [np.zeros(buttons[i].shape)], axis=0)
        buttons[-1][concat[i]] = 1
    return buttons.astype(np.compat.long)


def create_action(mouse, button, buttons):
    action = buttons[button].copy()
    action[0] = mouse[0]
    return np.array(action)
