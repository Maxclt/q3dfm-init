from typing import List
import numpy as np


def set_frequencies(str_frq: List[str]) -> np.ndarray:
    """Translate a string frequency (e.g., "y" for yearly) into an integer
    representing its equivalent frequency value in the mix.

    Args:
        str_frq (List[str]): List encoding each series frequency as
            string from ("y", "q", "m", "b", "w", "d")

    Returns:
        np.ndarray: Array encoding each series frequency as an integer
    """

    freq_map = {"d": 1, "w": 7, "b": 14, "m": 28, "q": 84, "y": 336}
    frq = np.array([freq_map.get(freq, 28) for freq in str_frq])
    min_frq = int(min(frq))
    frq = frq / min_frq  # Scale by the minimum frequency

    # Finalizing replacements
    replace_map = {
        1: {28: 31, 84: 91, 336: 365},
        7: {12: 13, 48: 52},
        14: {24: 26},
    }

    # Apply replacements if min_frq matches
    if min_frq in replace_map:
        frq = np.array([replace_map[min_frq].get(f, f) for f in frq])

    return frq
