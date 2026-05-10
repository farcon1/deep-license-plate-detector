from __future__ import annotations

from typing import Iterable, List


PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O",
]

ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "O",
]

ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "O",
]


def get_num_classes() -> List[int]:
    return [len(PROVINCES), len(ALPHABETS), len(ADS), len(ADS), len(ADS), len(ADS), len(ADS)]


def validate_plate_indices(indices: Iterable[int]) -> List[int]:
    out = [int(x) for x in indices]
    if len(out) != 7:
        raise ValueError(f"Expected 7 indices for CCPD plate, got {len(out)}: {out}")

    if not (0 <= out[0] < len(PROVINCES)):
        raise ValueError(f"Province index out of range: {out[0]}")
    if not (0 <= out[1] < len(ALPHABETS)):
        raise ValueError(f"Alphabet index out of range: {out[1]}")
    for i in range(2, 7):
        if not (0 <= out[i] < len(ADS)):
            raise ValueError(f"ADS index out of range at pos={i}: {out[i]}")
    return out


def decode_plate_indices(indices: Iterable[int]) -> str:
    x = validate_plate_indices(indices)
    chars = [
        PROVINCES[x[0]],
        ALPHABETS[x[1]],
        ADS[x[2]],
        ADS[x[3]],
        ADS[x[4]],
        ADS[x[5]],
        ADS[x[6]],
    ]
    return "".join(chars)