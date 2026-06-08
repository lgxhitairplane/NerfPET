"""PET scanner geometry presets.

The values mirror the old UPETScanner::InitXXX methods, but are stored as
plain dictionaries so PETScanner can be configured without hard-coded init
methods.
"""

from copy import deepcopy


def _scanner_config(
    *,
    crystal_num,
    crystal_size,
    crystal_pitch,
    block_num,
    block_size_x,
    block_pitch,
    module_num,
    module_size_x,
    module_pitch_extra_z=0.0,
    module_pitch_equals_size=False,
    panel_num,
    crystal_clockwise_offset,
    scanner_radius,
):
    block_size = [
        block_size_x,
        crystal_pitch[1] * crystal_num[1],
        crystal_pitch[2] * crystal_num[2],
    ]
    module_size = [
        module_size_x,
        block_pitch[1] * block_num[1],
        block_pitch[2] * block_num[2],
    ]
    if module_pitch_equals_size:
        module_pitch = module_size[:]
    else:
        module_pitch = [
            module_size_x,
            module_size[1],
            module_size[2] + module_pitch_extra_z,
        ]
    panel_size = [
        module_pitch[0],
        module_pitch[1] * module_num[1],
        module_pitch[2] * module_num[2],
    ]

    return {
        "crystal_num": crystal_num,
        "crystal_size": crystal_size,
        "crystal_pitch": crystal_pitch,
        "block_num": block_num,
        "block_size": block_size,
        "block_pitch": block_pitch,
        "module_num": module_num,
        "module_size": module_size,
        "module_pitch": module_pitch,
        "panel_num": panel_num,
        "panel_size": panel_size,
        "panel_pitch": panel_size[:],
        "crystal_clockwise_offset": crystal_clockwise_offset,
        "scanner_radius": scanner_radius,
    }


SCANNER_CONFIGS = {
    "bio4panel": _scanner_config(
        crystal_num=[1, 13, 53],
        crystal_size=[13, 1.89, 1.89],
        crystal_pitch=[13, 2.0, 2.0],
        block_num=[1, 4, 1],
        block_size_x=13,
        block_pitch=[13, 26.5, 106],
        module_num=[1, 1, 1],
        module_size_x=13,
        module_pitch_equals_size=True,
        panel_num=4,
        crystal_clockwise_offset=0,
        scanner_radius=61.1,
    ),
    "D80": _scanner_config(
        crystal_num=[1, 13, 13],
        crystal_size=[13, 1.89, 1.89],
        crystal_pitch=[13, 2.0, 2.0],
        block_num=[1, 1, 4],
        block_size_x=13.3,
        block_pitch=[13.3, 26.5, 26.5],
        module_num=[1, 1, 1],
        module_size_x=20,
        module_pitch_extra_z=2,
        panel_num=12,
        crystal_clockwise_offset=1,
        scanner_radius=105.8 / 2,
    ),
    "D180": _scanner_config(
        crystal_num=[1, 13, 13],
        crystal_size=[13, 1.89, 1.89],
        crystal_pitch=[13, 2.0, 2.0],
        block_num=[1, 1, 4],
        block_size_x=13.3,
        block_pitch=[13.3, 26.5, 26.5],
        module_num=[1, 1, 1],
        module_size_x=20,
        module_pitch_extra_z=2,
        panel_num=24,
        crystal_clockwise_offset=1,
        scanner_radius=106.5,
    ),
    "E180": _scanner_config(
        crystal_num=[1, 13, 13],
        crystal_size=[13, 1.89, 1.89],
        crystal_pitch=[13, 2.0, 2.0],
        block_num=[1, 1, 4],
        block_size_x=13.3,
        block_pitch=[13.3, 26.5, 26.5],
        module_num=[1, 1, 2],
        module_size_x=20,
        module_pitch_extra_z=2.0,
        panel_num=24,
        crystal_clockwise_offset=1,
        scanner_radius=106.5,
    ),
    "Brain": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 1, 4],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 2],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=44,
        crystal_clockwise_offset=0,
        scanner_radius=188.5,
    ),
    "DPET100": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 1, 4],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 1],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=88,
        crystal_clockwise_offset=0,
        scanner_radius=378.5,
    ),
    "DPET200": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 1, 4],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 2],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=88,
        crystal_clockwise_offset=0,
        scanner_radius=378.5,
    ),
    "DigitMI2D": _scanner_config(
        crystal_num=[1, 6, 1],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 2, 1],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 1],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=24,
        crystal_clockwise_offset=0,
        scanner_radius=201.9
    ),
    "DigitMI930": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 2, 4],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 3],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=48,
        crystal_clockwise_offset=0,
        scanner_radius=807.6 / 2,
    ),
    "DigitMI925": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 2, 2],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 5],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=48,
        crystal_clockwise_offset=0,
        scanner_radius=807.6 / 2,
    ),
    "DigitMI920": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 2, 4],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 2],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=48,
        crystal_clockwise_offset=0,
        scanner_radius=807.6 / 2,
    ),
    "DigitMI930_24": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 2, 4],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 2, 3],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=24,
        crystal_clockwise_offset=0,
        scanner_radius=821.8 / 2,
    ),
    "DigitMIi30": _scanner_config(
        crystal_num=[1, 6, 6],
        crystal_size=[20, 3.9, 3.9],
        crystal_pitch=[20, 4.2, 4.2],
        block_num=[1, 2, 2],
        block_size_x=20,
        block_pitch=[20, 25.5, 25.5],
        module_num=[1, 1, 5],
        module_size_x=20,
        module_pitch_equals_size=True,
        panel_num=24,
        crystal_clockwise_offset=0,
        scanner_radius=403.7 / 2,
    ),
}


def get_scanner_config(name):
    """Return a copy of one scanner config by name."""
    if name not in SCANNER_CONFIGS:
        options = ", ".join(sorted(SCANNER_CONFIGS))
        raise KeyError(f"Unknown scanner config '{name}'. Options: {options}")
    return deepcopy(SCANNER_CONFIGS[name])


def list_scanner_configs():
    return sorted(SCANNER_CONFIGS)


DEFAULT_SCANNER_NAME = "D80"
DEFAULT_SCANNER_CONFIG = get_scanner_config(DEFAULT_SCANNER_NAME)
