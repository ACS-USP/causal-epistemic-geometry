# First-divergence localization

Source: `mixed BF16 kernel, cache/reduction-order, and dtype effects; no sequence-semantic bug`.

BF16 first exceedance: `{'fixture_id': 'G12_1_FIXTURE_07', 'dtype': 'BF16', 'layer': '0', 'component': 'mlp', 'token_index': '0', 'max_abs_difference': '0.015625', 'rms_difference': '0.0008990354677140253'}`.

FP32 first exceedance: `{'fixture_id': 'G12_1_FIXTURE_07', 'dtype': 'FP32', 'layer': '0', 'component': 'mlp', 'token_index': '0', 'max_abs_difference': '2.09808349609375e-05', 'rms_difference': '5.963124626361969e-07'}`.

Prompt/continuation lengths, output positions, intervention masks, position IDs, cache positions, and causal-mask construction were checked explicitly. No semantic off-by-one or position bug was found.
