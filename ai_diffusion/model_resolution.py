"""Resolve model references (checkpoints, LoRAs, VAE) against the models a server provides.

A reference is matched in order of decreasing certainty:
1. Exact (case-insensitive, path-separator-normalized) name match.
2. Hash match, if identity hashes are known for both sides.
3. Basename match, preferring paths closest to the reference (e.g. `foo.safetensors`
   wins over `sub/foo.safetensors`; among several subfolders the most similar full path wins).
4. Fuzzy match by edit distance, only when explicitly allowed, with a tight threshold.
"""

from __future__ import annotations


def sanitize_name(name: str) -> str:
    return name.replace("\\", "/").lower()


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _basename(name: str) -> str:
    return sanitize_name(name).rsplit("/", 1)[-1]


def _stem(name: str) -> str:
    sanitized = sanitize_name(name)
    if "." in sanitized.rsplit("/", 1)[-1]:
        sanitized = sanitized.rsplit(".", 1)[0]
    return sanitized


def _path_depth(name: str) -> int:
    return sanitize_name(name).count("/")


def _fuzzy_threshold(reference: str) -> int:
    return max(2, len(_stem(reference)) // 8)


def resolve_model_name(
    reference: str,
    available_names: list[str],
    reference_hash: str | None = None,
    available_hashes: dict[str, str] | None = None,
    allow_fuzzy: bool = True,
) -> str | None:
    """Find the available model which the reference is most likely pointing at."""

    def distance(name: str) -> int:
        return edit_distance(_stem(reference), _stem(name))

    exact = {sanitize_name(n): n for n in available_names}
    if found := exact.get(sanitize_name(reference)):
        return found

    if reference_hash and available_hashes:
        for name, model_hash in available_hashes.items():
            if model_hash == reference_hash and name in exact.values():
                return name

    basename = _basename(reference)
    basename_matches = [n for n in available_names if _basename(n) == basename]
    if basename_matches:
        return min(basename_matches, key=lambda n: (distance(n), _path_depth(n)))

    if allow_fuzzy:
        best = min(available_names, key=lambda n: (distance(n), len(n)), default=None)
        if best is not None and distance(best) <= _fuzzy_threshold(reference):
            return best

    return None
