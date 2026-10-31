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
    return hashlib.sha256(fitting.encode()).hexdigest()


V090_CODE = {
    'discovery.py': {'f7b73f6c0638da2e1667b918ba9bf04c418488c8144ddd141910e3da3d8e3414', '58815668cd72808476938323038f40bf9a9857982199b1381d7a3266605227a7'},
    'discovery_store.py': {'3d032490d2c9aeaf083c623cb5bfc7254da629a35cc07c03197761db74f3a358', '84669b63f25bb42eff0573404b94c456066de20179322bb25c784e45c4b84364'},
    'discovery_score.py': {'2841afde850491a6bc071bdf5d8060110ccba469e4c0ffb515ce602943fe74df', 'ad77b0edae27a4cbaf5aeefb30774f9a06800bb300f295091266bbeb13844193'},
    'null_identity.py': {'8498a97a83d407af0be6b218621e2a01d3c4cda72a58b451adba3dc9b7718ea2', 'df135b2b7513b9029d1f39704d99be47a3e1d54575af49c947a5e88b6de0d4ec'},
    'glmm.py': {'ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4', '0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081'},
    'kinship.py': {'35ac6734cd8976956cbcaf5d6b6f91af09a234eca19c5ead7425dae9f7a5eef7', 'e08b3b1a7926159b4a6d56b9a3600120df038c363f20934a9ebb89c9c43862c6'},
}

V091_CODE = {
    **V090_CODE,
    'discovery.py': {'53664cca3cd6885b746c128c63090386d32c2bd360cf1763e0e3b1748d3ac68c', '8a53201a544ac03b397f1ea3ee37b6cbd6f8699a4831b27f87ee999e36621768'},
    'discovery_store.py': {'bb82f663b2665472538ef680c287327ae382163b54dc6b7ad1f4a3d7d856df1f', '479fb111a52ff336de371bf5d9b54760d970cbd14443b2593bcbf70b2d92deec'},
    'discovery_score.py': {'3eb758648e4455c8408691fa64531afb8255bb5a6520ae14f48a0d5fbc483952', 'fe508a48de0cd46148f1973a48c8e06f64dbd08993efd6862f000bb57ded447f'},
    'discovery_execution.py': {'ea2f5c1ef74f026e369639146ba2db2e5b3eda52fb60927c178b95d057048b52', 'd369ebe7cc9050dea018c4c19fb6abab9c6181e73329582af90326379bbf33a7'},
    'memory_budget.py': {'03187b9171deed24e725c78477ac69fe6b6ac3cada7c2099d2db6d69b917fece', 'fa864be7581b9dcae14ab2abc2b5714873d68749079e3f287d450589ac014972'},
    'discovery_compat.py': {'dddeeee812771b6e79187d5f964d9ca6b9894aea77f8bc765bb21a5d95fbec1d', '076cbb1eb4e1030c9726dad405357d9c8d85cdb12edc40554b21264c2024af91'},
}

V092_CODE = {
    **V091_CODE,
    'discovery.py': {'4e13cebc29dea5d79be40e091b9166e7ec24a2ee78eb88e8e6293375f084b31d', '50f5697ada7a119ab8cc464bef3a2db7fecf3ca81780451f93a140069714e28c'},
    'discovery_score.py': {'44f75dab744c10029218f4be01b9ff88d5fe2d5bc1a73b6fd64ad8138de882fa', '6e490bb0d484303d515ad0eaebb7e2117d9fee9c549acc2ea4854d91eaa8466e'},
    'discovery_compat.py': {'abe0cca5a8ef56dd13f9165288b7e0edb33df6755f2c683ec57c7df1a2a0deda', 'a9ff42fead9748236fc76059a7c2fbd1c91b55be7cc2f9867e093e7306bb2ba2'},
    'discovery_eligibility.py': {'de0c9017ad641b726c8e1d7f1d77c598329e47b4f0642ea678945284cff4ab12', 'b98409612eecbdc42c9d9157cef25c261da4761a6c1d83045f2fb7334ce02f1a'},
}

PREFIT1_CODE = {
    **V092_CODE,
    'discovery.py': {'257a63bc9a16288c52b9ac5a431baa503df33b143d004579c9a98b78658f9350', '3492d03f774ca54f7a986d1a02de64b95d57cd3d14108d6f57c24e6285613540'},
    'discovery_score.py': {'ff3d8f3930111d5950115ccd2eeb2b52abd588007cac5e47011c89f2fc401596', '3cb3a605e2575ee56e70e76c623b9e9e7b836ae321a043757328e7bdbbb48565'},
    'discovery_execution.py': {'7b2c1a5e0e85c46b41b6cd1d67d9b31255da33f98cc92209a2175d0328e812d4', '2d5bdcdb80f30505a35256b48be02d0cf1cef0e6179c9d46e185589da02ba529'},
    'memory_budget.py': {'9ce1c1c87a4a304faf844b4b375c812c2368d8d89150a0259efd9ff4a60a93b9', 'f0c80afb3a4a23e9a82a91019b8f087054999f997e16addad84ae01fc368a56e'},
    'discovery_compat.py': {'00430cc681ea0950bfa349f9563a34cd4f15fb2ad5b25d9fce4149f3e6853593', '147735566192965beeac4fcd18e66bdb9b93e0ed7cb33ef0d699e7ed0b9bd2f7'},
    'discovery_null_cache.py': {'a80535e60f450f9c555a86ae57e1f783f5faa0550ca7fcbaa006dda3c019b61a', '4ff615aa10a4b8e2121b353d3813246c996cd840b113a8459db669b3f17e3357'},
    'discovery_prefit.py': {'64228e885803dab771f7dd4040f12f82f1c04b7e36cb65622841ce171ae2abb2', '629f484618acb26ce561fbe6db8e5a55b43778d15913f0f321d817d9f7312690'},
}

PREFIT2_CODE = {
    **PREFIT1_CODE,
    'discovery.py': {'4a1d88c5cd186247c6ea6d8f4ac3ce259fbbf1b955c5cfc4bbe20c50f3bb3991', 'c47add04b98cadabc8c91868ea10f6a115bb64f12cfd60ffa13caea834e3afb6'},
    'discovery_score.py': {'88b9b9c971479f2f80b3f6205cfbd05f8237884611e614d96a639ac3394be826', 'e31fc966426a9f5c037f6151018d3fee701e365516a0e86c7202a7ab00bb58d3'},
    'discovery_compat.py': {'f494a9c20fbd1c218eb8c7ba8b08747259b0b364b412790a51a2b47d4aa59c4c', 'b119bc831538fd7cb3245fc086c148f27d750b3660540ba8728debd475cddba9'},
    'discovery_null_cache.py': {'2151d7c8fb8727087d33bbad388622dfc61f02130835327399d61c18481cea97', '6fe8beb612aae14d69492c1ed9ebfcfd6b3c74e3b2d385ce68e1bf4c7f019af4'},
    'discovery_prefit.py': {'d333e1fcbed1dd15d5494ca0821c0cf00ca4cb2543796f82b675868dbd9d3fd3', '37ed3d9264387722c02534a1003bc80da5bd7fca779a73d01e0544f7bf9cb3d7'},
    'discovery_inventory.py': {'e41e662d5a6772778cb6564cf6d0e4d68c544a16025d3cb2ac7830cce4f4ff8c', '07697966f8f87eec6e4af07e7396ca9b84ecce9f55021c22eb1fefe41d97a54f'},
    'discovery_import.py': {'c07634e6d19421c13ba8b0fd5c2096bbb79976d8746085c2e063e8c280889499', '1062fc7f616aee711a5a6713f4fb26e36cda639dc50118ed9a020686902c37fd'},
}


def migrate_identity(previous, expected, manifest):
    """Accept only known release code with identical scientific inputs/settings."""
    from .discovery_store import atomic_json
    if previous == expected:
        return False
    release = {"0.9.0":V090_CODE, "0.9.1":V091_CODE, "0.9.2":V092_CODE}.get(previous.get("version"))
    release={'prefit-compact-1':PREFIT1_CODE,'prefit-compact-2':PREFIT2_CODE}.get(
        previous.get('execution_revision'),release)
    if release is None or set(previous.get("code",{})) != set(release):
        raise ValueError("Input/code/statistical settings changed; use a new output")
    if any(previous["code"][key] not in values for key,values in release.items()):
        raise ValueError("Previous source is not a verified release; migration rejected")
    scientific={key:value for key,value in previous.get("settings",{}).items() if key!="batch_size"}
    if scientific != expected["settings"]:
        raise ValueError("Scientific settings changed; migration cannot reuse this traversal")
    for key in ("inputs","numpy","scipy","null_model_source_sha256"):
        if previous.get(key)!=expected.get(key):
            raise ValueError(f"Restart {key} differs; migration rejected")
    suffix=previous.get('execution_revision') if previous['version']=='0.9.2' else None
    backup=manifest.with_name('run_identity.v'+previous['version'].replace('.','')+
                              ('.'+suffix if suffix else '')+'.json')
    if not backup.exists():
        atomic_json(backup,previous)
    atomic_json(manifest,expected)
    return True
