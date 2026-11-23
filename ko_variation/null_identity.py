"""Compatibility fingerprint for persistent nulls, excluding score-only code."""
import ast
import hashlib
import json
from pathlib import Path

LEGACY_GLMM_HASHES = {
    "ba518210a483d820aee8d73d8e8ebe69f787da2727daff99d6d678df04a51041",
    "8b06c2661605486b884b950b261282f86685f94d8d41135d6ad7258b11a183e1",
    "ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4",
    "0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081",
}
LEGACY_NULL_SOURCE = "24ea0e0a34d41640706f06e343211ff4e6a76535420bedb97dc91e6f9d85fe45"


_V094_LAYOUT = {
    "discovery.py": (
        "d69521e7bfc009fd17867d83a253c07dc1b35ea8ae9ee710600a0bd6ae3fbf62",
        "0c90b46674abf127242d6088564ae94e237311ab06d92dfd666f7dbca380d515",
    ),
    "discovery_eligibility.py": (
        "b98409612eecbdc42c9d9157cef25c261da4761a6c1d83045f2fb7334ce02f1a",
        "de0c9017ad641b726c8e1d7f1d77c598329e47b4f0642ea678945284cff4ab12",
    ),
    "discovery_store.py": (
        "884f92bbacb96fb9aa85d952109e3de7a65be236c88205858cf6ffd2e13ed3b0",
        "173f5c20a94c1a98352ce55ee7a5a3885a2b3adeffb1b61c9653c8b625d1b9d0",
    ),
    "discovery_execution.py": (
        "2d5bdcdb80f30505a35256b48be02d0cf1cef0e6179c9d46e185589da02ba529",
        "7b2c1a5e0e85c46b41b6cd1d67d9b31255da33f98cc92209a2175d0328e812d4",
    ),
    "discovery_score.py": (
        "a7497e476046bb600d4ba02c878ed3e1266504238ec32d40d62fe171e6564a13",
        "4393b5b4f64b85106300c59217e5de478168fa2e28d3f63afe2c5471bc52ba94",
    ),
    "discovery_nulls.py": (
        "9ab4baa3f136c1824a2bc660ac4b1df3b4a71c18c2720f66cd8f051afc486fee",
        "3057be51666a34ac90d09e0400af542592b16fdfb61d2f00805b23c2891fdf4b",
    ),
    "coarse_screen.py": (
        "a75a329809ba747f80e12def1b4e77e987e99ed1f5d9e92a2bfcf6d41e123d37",
        "d109b01d82caa052306e97fb1a038fe061d3bdeab512462691ee050188d1a230",
    ),
    "null_identity.py": (
        "d64daabfbbca0b5c65cdf08beac13627349291bbeda5962cd66374fd20ba155e",
        "6fe3265a6c0e5ef0ce131c5029e5916e81ce9bb3da8dc96d3976356653069e46",
    ),
}

_REFACTORED_LAYOUT = {
    "discovery.py": "de1e3045071f9293ea9f94e16de740eb9bd8294346f603281443b00b0280bb3a",
    "discovery_search.py": "9d297678a40fcd1184441f9cd4b76d4fdfe0a5478c7b9c2597bb3e364f0eb5ea",
    "discovery_store.py": "be41fc65dd7b4063c1e78353c50c418fc6bbb4613a10a5ddfec6d2150f561832",
    "discovery_execution.py": "6fcdb5c5c254fa10011541314a16f0aa5339b0766dfa6ed0de4168fb7a80c809",
    "discovery_score.py": "f2815d27790fefbfe7784fd917e46f7567687304ed1e2ded3f725a3de3432604",
    "discovery_nulls.py": "50143bd57de9be61e3ef8a912cf6458477effa09a9bc1be81e3525a9cde7a35f",
    "coarse_screen.py": "4dd82ba00545ba26b5d3c5b05e561ba8e8b123ae6bcfc22a0e764536b6d6f0da",
}


def null_source_hash():
    source = Path(__file__).with_name("glmm.py").read_text(encoding="utf-8")
    excluded = set()
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {
                "score_predictor_block","score_predictor","neglog10"}:
            excluded.update(range(node.lineno,node.end_lineno+1))
    fitting = "".join(line for number,line in enumerate(source.splitlines(keepends=True),1) if number not in excluded)
    fitting += Path(__file__).with_name('tree_null.py').read_text(encoding='utf-8')
    return hashlib.sha256(fitting.encode()).hexdigest()


def _same_refactored_run(previous, expected):
    if previous.get("version") != "0.9.4":
        return False
    prior_code = previous.get("code", {})
    current_code = expected.get("code", {})
    if any(prior_code.get(name) not in hashes for name, hashes in _V094_LAYOUT.items()):
        return False
    for name, fingerprint in _REFACTORED_LAYOUT.items():
        source = Path(__file__).with_name(name).read_bytes()
        if hashlib.sha256(source).hexdigest() != current_code.get(name):
            return False
        if hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != fingerprint:
            return False
    old_unchanged = {k: v for k, v in prior_code.items() if k not in _V094_LAYOUT}
    new_unchanged = {k: v for k, v in current_code.items()
                     if k not in _REFACTORED_LAYOUT and k != "null_identity.py"}
    return old_unchanged == new_unchanged and dict(previous, code=current_code) == expected


# Full v0.9.4 production and development source manifests (LF and CRLF).
_TILED_PREDECESSORS = {
    "252c294dd83a2d544b8e070d4ecfe45dcd54739f3722cfb3b5f13d4b73be3d80",
    "a756ef907e9537c267631b8fd3dc6c70f91f3b43e2c7b21665658f90cdf1ccc2",
    "9086e70d6f319a18c0aa9d74f6ca6d6191890afac352a381bbec605b1c4c9164",
    "4933c8d8e175e7cceb27ac215f1081046967478cb8f75fbb8c60bfd86709508e",
}
_TILED_SOURCE = "59bd2ef8adec18d8f952ac3f1967ec1d41877483f27115e47ab395f932728ffc"


def _source_manifest_hash(code):
    return hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()


def _same_tiled_run(previous, expected):
    if (previous.get("version"), previous.get("execution_revision")) != ("0.9.4", "tree-null-1"):
        return False
    if expected.get("execution_revision") != "tiled-tree-1":
        return False
    if _source_manifest_hash(previous.get("code", {})) not in _TILED_PREDECESSORS:
        return False
    current = expected.get("code", {})
    normalized = {}
    for name, fingerprint in current.items():
        if name == "null_identity.py":
            continue
        content = Path(__file__).with_name(name).read_bytes()
        if hashlib.sha256(content).hexdigest() != fingerprint:
            return False
        normalized[name] = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    if _source_manifest_hash(normalized) != _TILED_SOURCE:
        return False
    return dict(previous, version=expected["version"], execution_revision=expected["execution_revision"],
                code=current) == expected


def check_identity(previous, expected):
    """Return whether a compatible manifest needs rewriting."""
    if previous == expected:
        return False
    if previous.get('null_model_source_sha256'):
        metadata_only=dict(previous,version=expected.get('version'))
        code=dict(previous.get('code',{}))
        code['null_identity.py']=expected.get('code',{}).get('null_identity.py')
        metadata_only['code']=code
        if metadata_only==expected:
            return True
        if _same_refactored_run(previous, expected) or _same_tiled_run(previous, expected):
            return True
    if previous.get('settings') != expected.get('settings'):
        raise ValueError('Scientific settings changed; use a new output')
    raise ValueError('Input/code/environment differs from this verified run; use a new output')
