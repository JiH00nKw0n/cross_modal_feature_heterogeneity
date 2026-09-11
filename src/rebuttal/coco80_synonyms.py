"""COCO's 80 object categories, plus caption synonyms.

Two consumers:
  * ``build_coco80_labels.py`` uses ``COCO_80`` to map COCO's non-contiguous
    ``category_id`` values onto 0..79 index positions.
  * The caption-derived label fallback uses ``PATTERNS`` when the instance
    annotations are unavailable.

The synonyms exist because a caption rarely uses COCO's category name verbatim:
photos of a "tv" are captioned "television" or "monitor", "couch" appears as
"sofa", and "sports ball" essentially never appears as that phrase. Matching on
the bare category name alone drops those categories entirely.
"""

from __future__ import annotations

import re

# Order matters: index i in this list is column i of the label matrix.
COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Extra surface forms, beyond the category name itself. Kept deliberately tight:
# a loose pattern ("ball" for "sports ball") trades false negatives for false
# positives, and for this test a false positive is the more damaging error.
SYNONYMS: dict[str, list[str]] = {
    "motorcycle": ["motorbike"],
    "airplane": ["plane", "aeroplane", "jet"],
    "sports ball": ["soccer ball", "baseball", "basketball", "tennis ball", "volleyball"],
    "baseball bat": ["bat"],
    "baseball glove": ["glove", "mitt"],
    "tennis racket": ["racket", "racquet"],
    "wine glass": ["wineglass"],
    "hot dog": ["hotdog"],
    "couch": ["sofa"],
    "potted plant": ["houseplant", "plant in a pot"],
    "dining table": ["table"],
    "tv": ["television", "monitor", "tv screen"],
    "cell phone": ["phone", "cellphone", "smartphone", "mobile phone"],
    "remote": ["remote control"],
    "hair drier": ["hair dryer", "blow dryer", "blowdryer"],
    "donut": ["doughnut"],
    "refrigerator": ["fridge"],
    "teddy bear": ["teddy", "stuffed bear"],
    "person": ["man", "woman", "boy", "girl", "people", "guy", "lady", "child"],
}


def _compile(term: str) -> str:
    return r"\b" + re.escape(term.lower()) + r"\b"


# category -> one compiled regex matching the name or any synonym
PATTERNS: dict[str, re.Pattern] = {
    c: re.compile("|".join(_compile(t) for t in [c, *SYNONYMS.get(c, [])]))
    for c in COCO_80
}


def matches(caption: str, category: str) -> bool:
    """True if the caption mentions the category (word-boundary, lowercased)."""
    return bool(PATTERNS[category].search(caption.lower()))
