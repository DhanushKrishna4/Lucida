//! Build step 11: the GPU radix sort, against `slice::sort` as the oracle.
//!
//! Worth its own file rather than being checked indirectly through the tree it
//! feeds. A sort that is *almost* right produces a tree that is valid, renders
//! correctly, and is quietly worse than it should be — there is no image to look
//! at and nothing fails. Comparing against a known-correct sort is the only
//! cheap way to know.

use pt_gpu::{lbvh::sort_u32, Gpu};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("\n*** SKIPPING GPU TEST: {e} ***\n");
            None
        }
    }
}

/// Deterministic pseudo-random u32s. xorshift, local to the test — the
/// renderer's RNG is a separate thing and this must not depend on it.
fn xorshift(seed: u32, n: usize) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

fn check(gpu: &Gpu, keys: Vec<u32>, label: &str) {
    let values: Vec<u32> = (0..keys.len() as u32).collect();
    let (sorted_keys, sorted_values) = sort_u32(gpu, &keys, &values).expect("sort");

    // The oracle. Sorting by (key, original index) is what "stable" means, so
    // this expects the exact permutation, not merely a correct key ordering.
    let mut expect: Vec<(u32, u32)> = keys.iter().copied().zip(values.iter().copied()).collect();
    expect.sort();

    assert_eq!(sorted_keys.len(), keys.len(), "{label}: length changed");
    for (i, (k, v)) in expect.iter().enumerate() {
        assert_eq!(
            sorted_keys[i], *k,
            "{label}: key {i} is {} not {k}",
            sorted_keys[i]
        );
        assert_eq!(
            sorted_values[i], *v,
            "{label}: payload {i} is {} not {v} — the sort is not stable, which \
             destroys the ordering established by every earlier digit pass",
            sorted_values[i]
        );
    }
}

#[test]
fn sorts_random_keys() {
    let Some(gpu) = gpu() else { return };
    // Sizes straddling the 256-element workgroup: partial groups are where an
    // off-by-one in the bounds check or the histogram shows up.
    for n in [1usize, 2, 255, 256, 257, 511, 512, 1000, 4096, 100_000] {
        check(&gpu, xorshift(0x1234_5678, n), &format!("random n={n}"));
    }
}

#[test]
fn sorts_keys_with_many_duplicates() {
    let Some(gpu) = gpu() else { return };
    // Morton codes over a dense mesh collide often, and duplicates are exactly
    // where an unstable sort stops being detectable by checking keys alone.
    let n = 20_000;
    check(
        &gpu,
        xorshift(99, n).iter().map(|k| k % 64).collect(),
        "few distinct values",
    );
    check(&gpu, vec![7u32; n], "all identical");
}

#[test]
fn sorts_adversarial_orders() {
    let Some(gpu) = gpu() else { return };
    let n = 10_000;
    check(&gpu, (0..n as u32).collect(), "already sorted");
    check(&gpu, (0..n as u32).rev().collect(), "reversed");
    // Only the top digit differs, so seven of the eight passes are no-ops and
    // the last one does all the work.
    check(
        &gpu,
        (0..n as u32).map(|i| (i % 16) << 28).collect(),
        "high digit only",
    );
    // Only the bottom digit differs: the mirror case.
    check(
        &gpu,
        (0..n as u32).map(|i| i % 16).collect(),
        "low digit only",
    );
}

#[test]
fn sorts_the_full_u32_range() {
    let Some(gpu) = gpu() else { return };
    // Keys above 2^30, which real Morton codes never reach — the sort is a
    // general u32 sort and the top pass must not be skipped or mis-shifted.
    check(
        &gpu,
        vec![u32::MAX, 0, u32::MAX / 2, 1, u32::MAX - 1, 0x8000_0000],
        "extremes",
    );
}

#[test]
fn empty_input_is_not_an_error() {
    let Some(gpu) = gpu() else { return };
    let (k, v) = sort_u32(&gpu, &[], &[]).expect("empty sort");
    assert!(k.is_empty() && v.is_empty());
}
