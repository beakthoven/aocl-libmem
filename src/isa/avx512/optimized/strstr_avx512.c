/* Copyright (C) 2024-25 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "strlen_avx512.c"
#include "strchr_avx512.c"

/* This function compares two strings using AVX-512 vector instructions.
   Returns 0 if equal, non-zero otherwise. */
static inline int cmp_needle_avx512(const char *str1, const char *str2, size_t size)
{
    size_t offset = 0;
    __m512i z0, z1, z2, z3, z4, z5, z6, z7, z8;

    // Handle the case where the size is less than or equal 64B
    if (size <= ZMM_SZ)
    {
        z0 = _mm512_setzero_epi32();
        __mmask64 mask = ((uint64_t)-1) >> (ZMM_SZ - size);
        z1 = _mm512_mask_loadu_epi8(z0, mask, str1);
        z2 = _mm512_mask_loadu_epi8(z0, mask, str2);
        return _mm512_cmpneq_epu8_mask(z1, z2) != 0;
    }

    // Handle the case where the size lies between 65B-128B
    if (size <= 2 * ZMM_SZ)
    {
        z1 = _mm512_loadu_si512(str1);
        z2 = _mm512_loadu_si512(str2);
        if (_mm512_cmpneq_epu8_mask(z1, z2))
            return -1;
        z3 = _mm512_loadu_si512(str1 + size - ZMM_SZ);
        z4 = _mm512_loadu_si512(str2 + size - ZMM_SZ);
        return _mm512_cmpneq_epu8_mask(z3, z4) != 0;
    }

    // Handle the case where the size lies between 129B-2565B
    if (size <= 4* ZMM_SZ)
    {
        z1 = _mm512_loadu_si512(str1);
        z2 = _mm512_loadu_si512(str1 + ZMM_SZ);
        z5 = _mm512_loadu_si512(str2);
        z6 = _mm512_loadu_si512(str2 + ZMM_SZ);

        __m512i x1 = _mm512_xor_si512(z1, z5);
        __m512i x2 = _mm512_xor_si512(z2, z6);

        z3 = _mm512_loadu_si512(str1 + size - 2 * ZMM_SZ);
        z4 = _mm512_loadu_si512(str1 + size - ZMM_SZ);
        z7 = _mm512_loadu_si512(str2 + size - 2 * ZMM_SZ);
        z8 = _mm512_loadu_si512(str2 + size - ZMM_SZ);

        __m512i x3 = _mm512_xor_si512(z3, z7);
        __m512i x4 = _mm512_xor_si512(z4, z8);

        __m512i combined = _mm512_or_si512(_mm512_or_si512(x1, x2), _mm512_or_si512(x3, x4));
        return _mm512_test_epi8_mask(combined, combined) != 0;
    }

    // For sizes larger than 256B, process in chunks of 4 ZMM registers
    while ((size - offset) >= 4 * ZMM_SZ)
    {
        z1 = _mm512_loadu_si512(str1 + offset);
        z2 = _mm512_loadu_si512(str1 + offset + ZMM_SZ);
        z5 = _mm512_loadu_si512(str2 + offset);
        z6 = _mm512_loadu_si512(str2 + offset + ZMM_SZ);

        __m512i x1 = _mm512_xor_si512(z1, z5);
        __m512i x2 = _mm512_xor_si512(z2, z6);

        z3 = _mm512_loadu_si512(str1 + offset + 2 * ZMM_SZ);
        z4 = _mm512_loadu_si512(str1 + offset + 3 * ZMM_SZ);
        z7 = _mm512_loadu_si512(str2 + offset + 2 * ZMM_SZ);
        z8 = _mm512_loadu_si512(str2 + offset + 3 * ZMM_SZ);

        __m512i x3 = _mm512_xor_si512(z3, z7);
        __m512i x4 = _mm512_xor_si512(z4, z8);

        __m512i combined = _mm512_or_si512(_mm512_or_si512(x1, x2), _mm512_or_si512(x3, x4));
        if (_mm512_test_epi8_mask(combined, combined))
            return -1;

        offset += 4 * ZMM_SZ;
    }

    // Handle any remaining bytes that were not compared in the above loop
    size_t left_out = size - offset;
    if (left_out == 0)
        return 0;

    if (left_out <= ZMM_SZ)
    {
        z0 = _mm512_setzero_epi32();
        __mmask64 mask = ((uint64_t)-1) >> (ZMM_SZ - left_out);
        z1 = _mm512_mask_loadu_epi8(z0, mask, str1 + offset);
        z2 = _mm512_mask_loadu_epi8(z0, mask, str2 + offset);
        return _mm512_cmpneq_epu8_mask(z1, z2) != 0;
    }

    // Handle remainder between 65B-128B using overlapping loads
    if (left_out <= 2 * ZMM_SZ)
    {
        z1 = _mm512_loadu_si512(str1 + offset);
        z2 = _mm512_loadu_si512(str2 + offset);
        if (_mm512_cmpneq_epu8_mask(z1, z2))
            return -1;
        z3 = _mm512_loadu_si512(str1 + size - ZMM_SZ);
        z4 = _mm512_loadu_si512(str2 + size - ZMM_SZ);
        return _mm512_cmpneq_epu8_mask(z3, z4) != 0;
    }

    // Handle remainder between 129B-256B using 4 overlapping loads
    z1 = _mm512_loadu_si512(str1 + offset);
    z2 = _mm512_loadu_si512(str1 + offset + ZMM_SZ);
    z5 = _mm512_loadu_si512(str2 + offset);
    z6 = _mm512_loadu_si512(str2 + offset + ZMM_SZ);

    __m512i x1 = _mm512_xor_si512(z1, z5);
    __m512i x2 = _mm512_xor_si512(z2, z6);

    z3 = _mm512_loadu_si512(str1 + size - 2 * ZMM_SZ);
    z4 = _mm512_loadu_si512(str1 + size - ZMM_SZ);
    z7 = _mm512_loadu_si512(str2 + size - 2 * ZMM_SZ);
    z8 = _mm512_loadu_si512(str2 + size - ZMM_SZ);

    __m512i x3 = _mm512_xor_si512(z3, z7);
    __m512i x4 = _mm512_xor_si512(z4, z8);

    __m512i combined = _mm512_or_si512(_mm512_or_si512(x1, x2), _mm512_or_si512(x3, x4));
    return _mm512_test_epi8_mask(combined, combined) != 0;
}

/* This function is an optimized version of strstr using AVX-512 instructions.
It finds the first occurrence of the substring `needle` in the string `haystack`. */
static inline char * __attribute__((flatten)) _strstr_avx512(const char* haystack, const char* needle)
{
    __m512i z0, z1, z_first;
    __mmask64 match_mask, null_mask, null_pfx_mask;
    size_t offset, match_idx, null_idx;
    size_t needle_len;
    uint8_t first_char, last_char;

    // If the first character of the needle is the string terminator,
    // return the haystack (empty needle case)
    if (needle[0] == STR_TERM_CHAR)
        return (char*)haystack;

    // If the first character of the haystack is the string terminator,
    // return NULL (empty haystack case)
    if (haystack[0] == STR_TERM_CHAR)
        return NULL;

    // If the second character of the needle is the string terminator,
    // it is similar to finding a char in a string
    if (needle[1] == STR_TERM_CHAR)
        return (char*)_strchr_avx512(haystack, needle[0]);

    // Get needle length
    needle_len = _strlen_avx512(needle);

    // Use first and last characters for filtering
    first_char = (uint8_t)needle[0];
    last_char = (uint8_t)needle[needle_len - 1];

    // Initialize a zeroed AVX-512 register for comparisons against null terminators
    z_first = _mm512_set1_epi8(first_char);
    z0 = _mm512_setzero_si512 ();

    // Calculate the offset based on the alignment of the haystack pointer
    offset = (uintptr_t)haystack & (ZMM_SZ - 1);

    // Check for potential read beyond the page boundary
    if (unlikely((PAGE_SZ - ZMM_SZ) < ((PAGE_SZ - 1) & (uintptr_t)haystack)))
    {
        __m512i z_ff = _mm512_set1_epi8(0xff);
        __mmask64 load_mask = ALL_BITS_SET >> offset;
        z1 = _mm512_mask_loadu_epi8(z_ff, load_mask, haystack);
        null_mask = _mm512_cmpeq_epi8_mask(z1, z0);

        if (null_mask)
        {
            null_idx = _tzcnt_u64(null_mask);
            // Check if haystack is too short for needle
            if (null_idx < needle_len)
                return NULL;

            // Find first-char matches within valid range
            match_mask = _mm512_cmpeq_epi8_mask(z_first, z1);
            null_pfx_mask = (null_mask ^ (null_mask - 1));
            match_mask &= null_pfx_mask;

            while (match_mask)
            {
                match_idx = _tzcnt_u64(match_mask);
                // Check last character before full comparison
                if (match_idx + needle_len <= null_idx + 1 &&
                    *(haystack + match_idx + needle_len - 1) == last_char)
                {
                    if (!cmp_needle_avx512(haystack + match_idx, needle, needle_len))
                        return (char*)(haystack + match_idx);
                }
                match_mask = _blsr_u64(match_mask);
            }
            return NULL;
        }
    }
    else
    {
        z1 = _mm512_loadu_si512(haystack);
        null_mask = _mm512_cmpeq_epi8_mask(z1, z0);

        if (null_mask)
        {
            null_idx = _tzcnt_u64(null_mask);
            if (null_idx < needle_len)
                return NULL;

            match_mask = _mm512_cmpeq_epi8_mask(z_first, z1);
            null_pfx_mask = (null_mask ^ (null_mask - 1));
            match_mask &= null_pfx_mask;

            while (match_mask)
            {
                match_idx = _tzcnt_u64(match_mask);
                if (match_idx + needle_len <= null_idx + 1 &&
                    *(haystack + match_idx + needle_len - 1) == last_char)
                {
                    if (!cmp_needle_avx512(haystack + match_idx, needle, needle_len))
                        return (char*)(haystack + match_idx);
                }
                match_mask = _blsr_u64(match_mask);
            }
            return NULL;
        }
    }

    // If there is a match
    match_mask = _mm512_cmpeq_epi8_mask(z_first, z1);
    while (match_mask)
    {
        match_idx = _tzcnt_u64(match_mask);
        if (*(haystack + match_idx + needle_len - 1) == last_char)
        {
            if (!cmp_needle_avx512(haystack + match_idx, needle, needle_len))
                return (char*)(haystack + match_idx);
        }
        match_mask = _blsr_u64(match_mask);
    }

    // Adjust the offset to align further loads
    offset = ZMM_SZ - offset;

    // Loop until the end of the haystack or a null character is found
    while (1)
    {
        // Load current block
        z1 = _mm512_load_si512(haystack + offset);
        null_mask = _mm512_cmpeq_epi8_mask(z1, z0);

        // Find first-char matches
        match_mask = _mm512_cmpeq_epi8_mask(z1, z_first);

        // Mask out matches past null terminator
        if (null_mask)
        {
            null_idx = _tzcnt_u64(null_mask);
            null_pfx_mask = (null_mask ^ (null_mask - 1));
            match_mask &= null_pfx_mask;

            // Process matches with bounds checking
            while (match_mask)
            {
                match_idx = _tzcnt_u64(match_mask);
                size_t abs_pos = offset + match_idx;

                // Verify the needle fits within the remaining string
                if (match_idx + needle_len <= null_idx + 1 &&
                    *(haystack + abs_pos + needle_len - 1) == last_char)
                {
                    if (!cmp_needle_avx512(haystack + abs_pos, needle, needle_len))
                        return (char*)(haystack + abs_pos);
                }
                match_mask = _blsr_u64(match_mask);
            }
            return NULL;
        }

        // Process matches (no null terminator in this block)
        while (match_mask)
        {
            match_idx = _tzcnt_u64(match_mask);
            size_t abs_pos = offset + match_idx;

            // Check last character before expensive full compare
            if (*(haystack + abs_pos + needle_len - 1) == last_char)
            {
                if (!cmp_needle_avx512(haystack + abs_pos, needle, needle_len))
                    return (char*)(haystack + abs_pos);
            }
            match_mask = _blsr_u64(match_mask);
        }

        offset += ZMM_SZ;
    }
}
