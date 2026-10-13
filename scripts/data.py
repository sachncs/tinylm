"""Downloads the pinned BabyLM 2026 corpora and evaluation data once."""

import pathlib

from tinylm.fetch import Fetch, Pin

ROOT = pathlib.Path("data")
MANIFESTS = pathlib.Path("manifests")
PINS = (
    Pin(
        name="strict",
        repo="BabyLM-community/BabyLM-2026-Strict",
        revision="9e57baaaa91ac3c638746be14d1d5fa6c789f4cf",
        patterns=("*.train.txt",),
    ),
    Pin(
        name="small",
        repo="BabyLM-community/BabyLM-2026-Strict-Small",
        revision="c92ab16b4f08858304b0815706065b3354d8fc0a",
        patterns=("*.train.txt",),
    ),
    Pin(
        name="evals",
        repo="BabyLM-community/BabyLM-2026-Strict-Evals",
        revision="8d52da9424a9ff30b9e8266c4f751aba9c504233",
        patterns=(
            "evaluation_data/full_eval/blimp_filtered/*",
            "evaluation_data/fast_eval/blimp_fast/*",
            "evaluation_data/full_eval/supplement_filtered/*",
            "evaluation_data/full_eval/entity_tracking/*",
            "evaluation_data/full_eval/comps/*",
        ),
    ),
    Pin(
        name="extra",
        repo="BabyLM-community/BabyLM-2026-Strict-Evals",
        revision="8d52da9424a9ff30b9e8266c4f751aba9c504233",
        patterns=(
            "evaluation_data/fast_eval/ewok_fast.zip",
            "evaluation_data/full_eval/reading/*",
        ),
    ),
)


def main() -> None:
    """Fetches (or verifies) every pin."""
    fetch = Fetch(ROOT, MANIFESTS)
    for pin in PINS:
        fetch.run(pin)
        print(f"{pin.name}: ok")


if __name__ == "__main__":
    main()
