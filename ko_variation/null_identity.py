"""Compatibility fingerprint for persistent nulls, excluding score-only code."""
import ast
import hashlib
from pathlib import Path

LEGACY_GLMM_HASHES = {
    "ba518210a483d820aee8d73d8e8ebe69f787da2727daff99d6d678df04a51041",
    "8b06c2661605486b884b950b261282f86685f94d8d41135d6ad7258b11a183e1",
    "ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4",
    "0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081",
}
LEGACY_NULL_SOURCE = "24ea0e0a34d41640706f06e343211ff4e6a76535420bedb97dc91e6f9d85fe45"


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


def migrate_identity(previous, expected, manifest):
    """Allow metadata-only releases; never migrate changed numerical state."""
    if previous == expected:
        return False
    if previous.get('null_model_source_sha256'):
        metadata_only=dict(previous,version=expected.get('version'))
        code=dict(previous.get('code',{}))
        code['null_identity.py']=expected.get('code',{}).get('null_identity.py')
        metadata_only['code']=code
        if metadata_only==expected:
            from .discovery_store import atomic_json
            atomic_json(manifest,expected)
            return True
    if previous.get('settings') != expected.get('settings'):
        raise ValueError('Scientific settings changed; use a new output')
    raise ValueError('Input/code/environment differs from this verified run; use a new output')
