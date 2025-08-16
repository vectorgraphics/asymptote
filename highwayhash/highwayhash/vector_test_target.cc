// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// WARNING: this is a "restricted" source file; avoid including any headers
// unless they are also restricted. See arch_specific.h for details.

#include "highwayhash/vector_test_target.h"

#include "highwayhash/arch_specific.h"

#ifndef HH_DISABLE_TARGET_SPECIFIC
#if HH_TARGET == HH_TARGET_AVX2
#include "highwayhash/vector256.h"
#elif HH_TARGET == HH_TARGET_SSE41
#include "highwayhash/vector128.h"
#elif HH_TARGET == HH_TARGET_NEON
#include "highwayhash/vector_neon.h"
#elif HH_TARGET == HH_TARGET_Portable
#include "highwayhash/scalar.h"
#else
#error "Unknown target, add its include here."
#endif

namespace highwayhash {
namespace HH_TARGET_NAME {
namespace {

#if HH_TARGET == HH_TARGET_AVX2
template <typename T>
using V = V256<T>;
#elif HH_TARGET == HH_TARGET_SSE41 || HH_TARGET == HH_TARGET_NEON
template <typename T>
using V = V128<T>;
#elif HH_TARGET == HH_TARGET_Portable
template <typename T>
using V = Scalar<T>;
#else
#error "Unknown target, add its vector typedef here."
#endif

template <class T>
void NotifyIfUnequal(const V<T>& v, const T expected, const size_t line,
                     const HHNotify notify) {
  HH_ALIGNAS(32) T lanes[V<T>::N];
  Store(v, lanes);
  for (size_t i = 0; i < V<T>::N; ++i) {
    if (lanes[i] != expected) {
      notify(TargetName(HH_TARGET), (line << 16) | (i << 8) | sizeof(T));
    }
  }
}

template <class T>
void NotifyIfUnequal(const T& t, const T expected, const size_t line,
                     const HHNotify notify) {
  if (t != expected) {
    notify(TargetName(HH_TARGET), (line << 16) | sizeof(T));
  }
}

// MaxValue<T>()() replaces std::numeric_limits<T>::max().
template <typename T>
struct MaxValue;
template <>
struct MaxValue<uint8_t> {
  constexpr uint8_t operator()() const { return 0xFFu; }
};
template <>
struct MaxValue<uint16_t> {
  constexpr uint16_t operator()() const { return 0xFFFFu; }
};
template <>
struct MaxValue<uint32_t> {
  constexpr uint32_t operator()() const { return 0xFFFFFFFFu; }
};
template <>
struct MaxValue<uint64_t> {
  constexpr uint64_t operator()() const { return 0xFFFFFFFFFFFFFFFFull; }
};

template <typename T>
void TestMembersAndBinaryOperatorsExceptShifts(const HHNotify notify) {
  // uninitialized
  V<T> v;

  // broadcast
  const V<T> v2(2);
  NotifyIfUnequal(v2, T(2), __LINE__, notify);

  // assign from V
  const V<T> v3(3);
  V<T> v3b;
  v3b = v3;
  NotifyIfUnequal(v3b, T(3), __LINE__, notify);

  // equal
  const V<T> veq(v3 == v3b);
  NotifyIfUnequal(veq, MaxValue<T>()(), __LINE__, notify);

  // Copying to, and constructing from intrinsic yields same result.
  typename V<T>::Intrinsic nv2 = v2;
  V<T> v2b(nv2);
  NotifyIfUnequal(v2b, T(2), __LINE__, notify);

  // .. assignment also works.
  V<T> v2c;
  v2c = nv2;
  NotifyIfUnequal(v2c, T(2), __LINE__, notify);

  const V<T> add = v2 + v3;
  NotifyIfUnequal(add, T(5), __LINE__, notify);

  const V<T> sub = v3 - v2;
  NotifyIfUnequal(sub, T(1), __LINE__, notify);

  const V<T> vand = v3 & v2;
  NotifyIfUnequal(vand, T(2), __LINE__, notify);

  const V<T> vor = add | v2;
  NotifyIfUnequal(vor, T(7), __LINE__, notify);

  const V<T> vxor = v3 ^ v2;
  NotifyIfUnequal(vxor, T(1), __LINE__, notify);
}

// SSE does not allow shifting uint8_t, so instantiate for all other types.
template <class T>
void TestShifts(const HHNotify notify) {
  const V<T> v1(1);
  // Shifting out of right side => zero
  NotifyIfUnequal(v1 >> 1, T(0), __LINE__, notify);

  // Simple left shift
  NotifyIfUnequal(v1 << 1, T(2), __LINE__, notify);

  // Sign bit
  constexpr int kSign = (sizeof(T) * 8) - 1;
  constexpr T max = MaxValue<T>()();
  constexpr T sign = ~(max >> 1);
  NotifyIfUnequal(v1 << kSign, sign, __LINE__, notify);

  // Shifting out of left side => zero. Arm cannot shift by the number of
  // lane bits, so start with 'left-shifting' by one (via adding).
  NotifyIfUnequal((v1 + v1) << kSign, T(0), __LINE__, notify);
}

template <class T>
void TestLoadStore(const HHNotify notify) {
  const size_t n = V<T>::N;
  HH_ALIGNAS(32) T lanes[2 * n];
  for (size_t i = 0; i < n; ++i) {
    lanes[i] = 4;
  }
  for (size_t i = n; i < 2 * n; ++i) {
    lanes[i] = 5;
  }
  // Aligned load
  const V<T> v4 = Load<V<T>>(lanes);
  NotifyIfUnequal(v4, T(4), __LINE__, notify);

  // Aligned store
  HH_ALIGNAS(32) T lanes4[n];
  Store(v4, lanes4);
  NotifyIfUnequal(Load<V<T>>(lanes4), T(4), __LINE__, notify);

  // Unaligned load
  const V<T> vu = LoadUnaligned<V<T>>(lanes + 1);
  Store(vu, lanes4);
  NotifyIfUnequal(lanes4[n - 1], T(5), __LINE__, notify);
  for (size_t i = 1; i < n - 1; ++i) {
    NotifyIfUnequal(lanes4[i], T(4), __LINE__, notify);
  }

  // Unaligned store
  StoreUnaligned(v4, lanes + n / 2);
  size_t i;
  for (i = 0; i < 3 * n / 2; ++i) {
    NotifyIfUnequal(lanes[i], T(4), __LINE__, notify);
  }
  // Subsequent values remain unchanged.
  for (; i < 2 * n; ++i) {
    NotifyIfUnequal(lanes[i], T(5), __LINE__, notify);
  }
}

void TestAll(const HHNotify notify) {
  TestMembersAndBinaryOperatorsExceptShifts<uint8_t>(notify);
  TestMembersAndBinaryOperatorsExceptShifts<uint16_t>(notify);
  TestMembersAndBinaryOperatorsExceptShifts<uint32_t>(notify);
  TestMembersAndBinaryOperatorsExceptShifts<uint64_t>(notify);

  TestShifts<uint16_t>(notify);
  TestShifts<uint32_t>(notify);
  TestShifts<uint64_t>(notify);

  TestLoadStore<uint8_t>(notify);
  TestLoadStore<uint16_t>(notify);
  TestLoadStore<uint32_t>(notify);
  TestLoadStore<uint64_t>(notify);
}

}  // namespace
}  // namespace HH_TARGET_NAME

template <TargetBits Target>
void VectorTest<Target>::operator()(const HHNotify notify) const {
  HH_TARGET_NAME::TestAll(notify);
}

// Instantiate for the current target.
template struct VectorTest<HH_TARGET>;

}  // namespace highwayhash
#endif  // HH_DISABLE_TARGET_SPECIFIC
