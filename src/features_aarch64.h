// This file is a part of Julia. License is MIT: https://julialang.org/license

// AArch64 features definition
// hwcap
JL_FEATURE_DEF(crypto, 3, 0)
JL_FEATURE_DEF(crc, 7, 0)
JL_FEATURE_DEF(lse, 8, 0)
JL_FEATURE_DEF(fullfp16, 9, 0)
JL_FEATURE_DEF(rdm, 12, 50000)
JL_FEATURE_DEF(jscvt, 13, UINT32_MAX)
JL_FEATURE_DEF(fcma, 14, UINT32_MAX)
JL_FEATURE_DEF(lrcpc, 15, UINT32_MAX)
// JL_FEATURE_DEF(ras, ???, 0)
// JL_FEATURE_DEF(sve, ???, 0)

// hwcap2
// JL_FEATURE_DEF(?, 32 + ?, 0)

// custom bits to match llvm model
JL_FEATURE_DEF(v8_1a, 32 * 2 + 0, 0)
JL_FEATURE_DEF(v8_2a, 32 * 2 + 1, 0)
// JL_FEATURE_DEF(v8_3a, 32 * 2 + 2, ?)
