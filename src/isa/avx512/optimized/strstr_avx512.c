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
   Returns 0 if equal, non-zero otherwise.
   Optimized using XOR+OR batching to stay in vector domain longer. */
static inline int cmp_needle_page_safe(const char *str1, const char *str2, size_t size)
{
    size_t offset = 0;
    __m512i z1, z2, z3, z4, z5, z6, z7, z8;

    // Handle the case where the size is less than or equal 64B
    if (likely(size <= ZMM_SZ))
    {
        __mmask64 mask = _bzhi_u64(UINT64_MAX, size);
        z1 = _mm512_maskz_loadu_epi8(mask, str1);
        z2 = _mm512_maskz_loadu_epi8(mask, str2);
        return !!(_mm512_cmpneq_epu8_mask(z1, z2));
    }

    // Handle the case where the size lies between 65B-128B
    if (likely(size <= 2 * ZMM_SZ))
    {
        z1 = _mm512_loadu_si512(str1);
        z2 = _mm512_loadu_si512(str2);
        if (_mm512_cmpneq_epu8_mask(z1, z2))
            return -1;

        offset = size - ZMM_SZ;
        z3 = _mm512_loadu_si512(str1 + offset);
        z4 = _mm512_loadu_si512(str2 + offset);
        return !!(_mm512_cmpneq_epu8_mask(z3, z4));
    }

    // Handle the case where the size lies between 129B-256B
    // Uses XOR+OR batching to reduce mask register transfers
    if (likely(size <= 4 * ZMM_SZ))
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
        return !!(_mm512_test_epi8_mask(combined, combined));
    }

    // For sizes larger than 256B, process in chunks of 4 ZMM registers
    // Uses XOR+OR batching to stay in vector domain longer
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

    // Handle any remaining bytes using overlapping loads
    size_t left_out = size - offset;
    if (left_out == 0)
        return 0;

    if (left_out <= ZMM_SZ)
    {
        __mmask64 mask = _bzhi_u64(UINT64_MAX, left_out);
        z1 = _mm512_maskz_loadu_epi8(mask, str1 + offset);
        z2 = _mm512_maskz_loadu_epi8(mask, str2 + offset);
        return !!(_mm512_cmpneq_epu8_mask(z1, z2));
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
        return !!(_mm512_cmpneq_epu8_mask(z3, z4));
    }

    // Handle remainder between 129B-256B using 4 overlapping loads with XOR+OR
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
    return !!(_mm512_test_epi8_mask(combined, combined));
}

// Handles page boundary, uses head/tail logic 
static inline int cmp_needle_page_cross(const char *str1, const char *str2, size_t size, size_t safe_bytes) 
{
    size_t offset = 0;
    while (offset < safe_bytes) {
        size_t safe_offset  = safe_bytes - offset;
        if (safe_offset > ZMM_SZ) safe_offset = ZMM_SZ;
        
        __mmask64 mask = (safe_offset >= ZMM_SZ) ? UINT64_MAX : _bzhi_u64(UINT64_MAX, safe_offset);
        __m512i z1 = _mm512_maskz_loadu_epi8(mask, str1 + offset);
        __m512i z2 = _mm512_maskz_loadu_epi8(mask, str2 + offset);
        if (_mm512_cmpneq_epu8_mask(z1, z2)) return -1;

        offset += safe_offset;
    }
    
    // For the rest, check again if tail is in page boundary
    size_t tail = size - offset;
    if (tail > 0) {
        size_t tail_offset1 = PAGE_SZ - ((uintptr_t)(str1 + offset) & (PAGE_SZ - 1));
        size_t tail_offset2 = PAGE_SZ - ((uintptr_t)(str2 + offset) & (PAGE_SZ - 1));
        size_t tail_safe_bytes = (tail_offset1 < tail_offset2) ? tail_offset1 : tail_offset2;

        if (tail_safe_bytes >= tail) {
            return cmp_needle_page_safe(str1 + offset, str2 + offset, tail);
        } else {
            return cmp_needle_page_cross(str1 + offset, str2 + offset, tail, tail_safe_bytes);
        }
    }
    return 0;
}

static inline int cmp_needle_avx512(const char *haystack, const char *needle, 
                                        size_t hay_idx, size_t needle_len)
{    
    size_t offset1 = PAGE_SZ - ((uintptr_t)(haystack + hay_idx) & (PAGE_SZ - 1));
    size_t offset2 = PAGE_SZ - ((uintptr_t)needle & (PAGE_SZ - 1));
    size_t safe_bytes = (offset1 < offset2) ? offset1 : offset2;

    if (likely(safe_bytes >= needle_len)) {
        return cmp_needle_page_safe(haystack + hay_idx, needle, needle_len);
    }
    return cmp_needle_page_cross(haystack + hay_idx, needle, needle_len, safe_bytes);
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
    if (unlikely(needle[0] == STR_TERM_CHAR))
        return (char*)haystack;

    // If the first character of the haystack is the string terminator,
    // return NULL (empty haystack case)
    if (unlikely(haystack[0] == STR_TERM_CHAR))
        return NULL;

    // If the second character of the needle is the string terminator,
    // it is similar to finding a char in a string
    if (unlikely(needle[1] == STR_TERM_CHAR))
        return (char*)_strchr_avx512(haystack, needle[0]);

    // Get needle length
    needle_len = _strlen_avx512(needle);

    // Use first and last characters for lightweight filtering
    first_char = (uint8_t)needle[0];
    last_char = (uint8_t)needle[needle_len - 1];

    z_first = _mm512_set1_epi8(first_char);
    z0 = _mm512_setzero_si512();

    // Calculate the offset based on the alignment of the haystack pointer
    offset = (uintptr_t)haystack & (ZMM_SZ - 1);

    // Check for potential read beyond the page boundary
    if (unlikely((PAGE_SZ - ZMM_SZ) < ((PAGE_SZ - 1) & (uintptr_t)haystack)))
    {
        __m512i z_ff = _mm512_set1_epi8(0xff);
        __mmask64 load_mask = UINT64_MAX >> offset;
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
                uintptr_t m_ptr = (uintptr_t)(haystack + match_idx);
                uintptr_t last_ptr = m_ptr + needle_len - 1;

                // If last char is on same page, scalar read is safe.
                // Otherwise, skip scalar read and go to dispatcher.
                if (likely((m_ptr ^ last_ptr) < PAGE_SZ)) {
                    if (*(char*)last_ptr == last_char) {
                        if (cmp_needle_avx512(haystack, needle, match_idx, needle_len) == 0)
                            return (char*)(haystack + match_idx);
                    }
                } else {
                    if (cmp_needle_avx512(haystack, needle, match_idx, needle_len) == 0)
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
                uintptr_t m_ptr = (uintptr_t)(haystack + match_idx);
                uintptr_t last_ptr = m_ptr + needle_len - 1;

                if (likely((m_ptr ^ last_ptr) < PAGE_SZ)) {
                    if (*(char*)last_ptr == last_char) {
                        if (cmp_needle_avx512(haystack, needle, match_idx, needle_len) == 0)
                            return (char*)(haystack + match_idx);
                    }
                } else {
                    if (cmp_needle_avx512(haystack, needle, match_idx, needle_len) == 0)
                        return (char*)(haystack + match_idx);
                }
                match_mask = _blsr_u64(match_mask);
            }
            return NULL;
        }
    }

    // If there is a match in first block (no null terminator found)
    match_mask = _mm512_cmpeq_epi8_mask(z_first, z1);
    while (match_mask)
    {
        match_idx = _tzcnt_u64(match_mask);
        uintptr_t m_ptr = (uintptr_t)(haystack + match_idx);
        uintptr_t last_ptr = m_ptr + needle_len - 1;

        if (likely((m_ptr ^ last_ptr) < PAGE_SZ)) {
            if (*(char*)last_ptr == last_char) {
                if (cmp_needle_avx512(haystack, needle, match_idx, needle_len) == 0)
                    return (char*)(haystack + match_idx);
            }
        } else {
            if (cmp_needle_avx512(haystack, needle, match_idx, needle_len) == 0)
                return (char*)(haystack + match_idx);
        }
        match_mask = _blsr_u64(match_mask);
    }

    // Adjust the offset to align further loads
    offset = ZMM_SZ - offset;

    // Main search loop using aligned loads
    while (1)
    {
        // Load current block (aligned)
        z1 = _mm512_load_si512(haystack + offset);
        null_mask = _mm512_cmpeq_epi8_mask(z1, z0);

        // Find first-char matches
        match_mask = _mm512_cmpeq_epi8_mask(z1, z_first);

        // Mask out matches past null terminator
        if (null_mask)
        {
            null_idx = _tzcnt_u64(null_mask);
            match_mask &= (null_mask ^ (null_mask - 1));

            // Process matches with bounds checking
            while (match_mask)
            {
                match_idx = _tzcnt_u64(match_mask);
                size_t abs_pos = offset + match_idx;

                // Verify the needle fits within the remaining string
                if (match_idx + needle_len <= null_idx + 1)
                {
                    uintptr_t m_ptr = (uintptr_t)(haystack + abs_pos);
                    uintptr_t last_ptr = m_ptr + needle_len - 1;

                    if (likely((m_ptr ^ last_ptr) < PAGE_SZ)) {
                        if (*(char*)last_ptr == last_char) {
                            if (cmp_needle_avx512(haystack, needle, abs_pos, needle_len) == 0)
                                return (char*)(haystack + abs_pos);
                        }
                    } else {
                        if (cmp_needle_avx512(haystack, needle, abs_pos, needle_len) == 0)
                            return (char*)(haystack + abs_pos);
                    }
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
            uintptr_t m_ptr = (uintptr_t)(haystack + abs_pos);
            uintptr_t last_ptr = m_ptr + needle_len - 1;

            if (likely((m_ptr ^ last_ptr) < PAGE_SZ)) {
                if (*(char*)last_ptr == last_char) {
                    if (cmp_needle_avx512(haystack, needle, abs_pos, needle_len) == 0)
                        return (char*)(haystack + abs_pos);
                }
            } else {
                if (cmp_needle_avx512(haystack, needle, abs_pos, needle_len) == 0)
                    return (char*)(haystack + abs_pos);
            }
            match_mask = _blsr_u64(match_mask);
        }

        offset += ZMM_SZ;
    }
}