"""Narrow restart migration from released v0.9.0/v0.9.1/original-v0.9.2.

Only known source snapshots are accepted. Inputs, scientific settings, fitting
source and NumPy/SciPy versions must still match; no guessed cache compatibility.
"""
from .discovery_store import atomic_json


V090_CODE = {
    "discovery.py": {"58815668cd72808476938323038f40bf9a9857982199b1381d7a3266605227a7", "f7b73f6c0638da2e1667b918ba9bf04c418488c8144ddd141910e3da3d8e3414"},
    "discovery_store.py": {"84669b63f25bb42eff0573404b94c456066de20179322bb25c784e45c4b84364", "3d032490d2c9aeaf083c623cb5bfc7254da629a35cc07c03197761db74f3a358"},
    "discovery_score.py": {"2841afde850491a6bc071bdf5d8060110ccba469e4c0ffb515ce602943fe74df", "ad77b0edae27a4cbaf5aeefb30774f9a06800bb300f295091266bbeb13844193"},
    "null_identity.py": {"df135b2b7513b9029d1f39704d99be47a3e1d54575af49c947a5e88b6de0d4ec", "8498a97a83d407af0be6b218621e2a01d3c4cda72a58b451adba3dc9b7718ea2"},
    "glmm.py": {"ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4", "0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081"},
    "kinship.py": {"35ac6734cd8976956cbcaf5d6b6f91af09a234eca19c5ead7425dae9f7a5eef7", "e08b3b1a7926159b4a6d56b9a3600120df038c363f20934a9ebb89c9c43862c6"},
}


V091_CODE = {
    'discovery.py': {'8a53201a544ac03b397f1ea3ee37b6cbd6f8699a4831b27f87ee999e36621768', '53664cca3cd6885b746c128c63090386d32c2bd360cf1763e0e3b1748d3ac68c'},
    'discovery_store.py': {'bb82f663b2665472538ef680c287327ae382163b54dc6b7ad1f4a3d7d856df1f', '479fb111a52ff336de371bf5d9b54760d970cbd14443b2593bcbf70b2d92deec'},
    'discovery_score.py': {'fe508a48de0cd46148f1973a48c8e06f64dbd08993efd6862f000bb57ded447f', '3eb758648e4455c8408691fa64531afb8255bb5a6520ae14f48a0d5fbc483952'},
    'discovery_execution.py': {'ea2f5c1ef74f026e369639146ba2db2e5b3eda52fb60927c178b95d057048b52', 'd369ebe7cc9050dea018c4c19fb6abab9c6181e73329582af90326379bbf33a7'},
    'memory_budget.py': {'03187b9171deed24e725c78477ac69fe6b6ac3cada7c2099d2db6d69b917fece', 'fa864be7581b9dcae14ab2abc2b5714873d68749079e3f287d450589ac014972'},
    'discovery_compat.py': {'076cbb1eb4e1030c9726dad405357d9c8d85cdb12edc40554b21264c2024af91', 'dddeeee812771b6e79187d5f964d9ca6b9894aea77f8bc765bb21a5d95fbec1d'},
    'null_identity.py': {'8498a97a83d407af0be6b218621e2a01d3c4cda72a58b451adba3dc9b7718ea2', 'df135b2b7513b9029d1f39704d99be47a3e1d54575af49c947a5e88b6de0d4ec'},
    'glmm.py': {'ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4', '0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081'},
    'kinship.py': {'35ac6734cd8976956cbcaf5d6b6f91af09a234eca19c5ead7425dae9f7a5eef7', 'e08b3b1a7926159b4a6d56b9a3600120df038c363f20934a9ebb89c9c43862c6'},
}


V092_CODE = {
    'discovery.py': {'50f5697ada7a119ab8cc464bef3a2db7fecf3ca81780451f93a140069714e28c', '4e13cebc29dea5d79be40e091b9166e7ec24a2ee78eb88e8e6293375f084b31d'},
    'discovery_store.py': {'479fb111a52ff336de371bf5d9b54760d970cbd14443b2593bcbf70b2d92deec', 'bb82f663b2665472538ef680c287327ae382163b54dc6b7ad1f4a3d7d856df1f'},
    'discovery_score.py': {'44f75dab744c10029218f4be01b9ff88d5fe2d5bc1a73b6fd64ad8138de882fa', '6e490bb0d484303d515ad0eaebb7e2117d9fee9c549acc2ea4854d91eaa8466e'},
    'discovery_execution.py': {'d369ebe7cc9050dea018c4c19fb6abab9c6181e73329582af90326379bbf33a7', 'ea2f5c1ef74f026e369639146ba2db2e5b3eda52fb60927c178b95d057048b52'},
    'memory_budget.py': {'03187b9171deed24e725c78477ac69fe6b6ac3cada7c2099d2db6d69b917fece', 'fa864be7581b9dcae14ab2abc2b5714873d68749079e3f287d450589ac014972'},
    'discovery_compat.py': {'a9ff42fead9748236fc76059a7c2fbd1c91b55be7cc2f9867e093e7306bb2ba2', 'abe0cca5a8ef56dd13f9165288b7e0edb33df6755f2c683ec57c7df1a2a0deda'},
    'discovery_eligibility.py': {'de0c9017ad641b726c8e1d7f1d77c598329e47b4f0642ea678945284cff4ab12', 'b98409612eecbdc42c9d9157cef25c261da4761a6c1d83045f2fb7334ce02f1a'},
    'null_identity.py': {'8498a97a83d407af0be6b218621e2a01d3c4cda72a58b451adba3dc9b7718ea2', 'df135b2b7513b9029d1f39704d99be47a3e1d54575af49c947a5e88b6de0d4ec'},
    'glmm.py': {'ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4', '0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081'},
    'kinship.py': {'35ac6734cd8976956cbcaf5d6b6f91af09a234eca19c5ead7425dae9f7a5eef7', 'e08b3b1a7926159b4a6d56b9a3600120df038c363f20934a9ebb89c9c43862c6'},
}


# Verified immutable v0.9.2-perf1 source, in LF and Windows CRLF form.
PREFIT1_CODE = {'discovery.py': {'257a63bc9a16288c52b9ac5a431baa503df33b143d004579c9a98b78658f9350',
                  '3492d03f774ca54f7a986d1a02de64b95d57cd3d14108d6f57c24e6285613540'},
 'discovery_store.py': {'479fb111a52ff336de371bf5d9b54760d970cbd14443b2593bcbf70b2d92deec',
                        'bb82f663b2665472538ef680c287327ae382163b54dc6b7ad1f4a3d7d856df1f'},
 'discovery_score.py': {'3cb3a605e2575ee56e70e76c623b9e9e7b836ae321a043757328e7bdbbb48565',
                        'ff3d8f3930111d5950115ccd2eeb2b52abd588007cac5e47011c89f2fc401596'},
 'discovery_execution.py': {'2d5bdcdb80f30505a35256b48be02d0cf1cef0e6179c9d46e185589da02ba529',
                            '7b2c1a5e0e85c46b41b6cd1d67d9b31255da33f98cc92209a2175d0328e812d4'},
 'memory_budget.py': {'9ce1c1c87a4a304faf844b4b375c812c2368d8d89150a0259efd9ff4a60a93b9',
                      'f0c80afb3a4a23e9a82a91019b8f087054999f997e16addad84ae01fc368a56e'},
 'discovery_compat.py': {'00430cc681ea0950bfa349f9563a34cd4f15fb2ad5b25d9fce4149f3e6853593',
                         '147735566192965beeac4fcd18e66bdb9b93e0ed7cb33ef0d699e7ed0b9bd2f7'},
 'discovery_eligibility.py': {'b98409612eecbdc42c9d9157cef25c261da4761a6c1d83045f2fb7334ce02f1a',
                              'de0c9017ad641b726c8e1d7f1d77c598329e47b4f0642ea678945284cff4ab12'},
 'discovery_null_cache.py': {'4ff615aa10a4b8e2121b353d3813246c996cd840b113a8459db669b3f17e3357',
                             'a80535e60f450f9c555a86ae57e1f783f5faa0550ca7fcbaa006dda3c019b61a'},
 'discovery_prefit.py': {'629f484618acb26ce561fbe6db8e5a55b43778d15913f0f321d817d9f7312690',
                         '64228e885803dab771f7dd4040f12f82f1c04b7e36cb65622841ce171ae2abb2'},
 'null_identity.py': {'8498a97a83d407af0be6b218621e2a01d3c4cda72a58b451adba3dc9b7718ea2',
                      'df135b2b7513b9029d1f39704d99be47a3e1d54575af49c947a5e88b6de0d4ec'},
 'glmm.py': {'0dc23fb4765055a12ffe8f173cef7ef30214a26a7ee98f7fdad9ddc2f3a87081',
             'ab8fecb0f4af96feb6c77d8d2ff3a054d026bae20ec4b005da939ebb5f4b82b4'},
 'kinship.py': {'35ac6734cd8976956cbcaf5d6b6f91af09a234eca19c5ead7425dae9f7a5eef7',
                'e08b3b1a7926159b4a6d56b9a3600120df038c363f20934a9ebb89c9c43862c6'}}


def migrate_identity(previous, expected, manifest):
    if previous == expected:
        return False
    release = {"0.9.0":V090_CODE, "0.9.1":V091_CODE, "0.9.2":V092_CODE}.get(previous.get("version"))
    if previous.get('execution_revision')=='prefit-compact-1':
        release=PREFIT1_CODE
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
