use pyo3::prelude::*;
// 🚀 修改点 1：回归标准库的 HashMap 和 HashSet，PyO3 能完美识别它们
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
struct HeapItem {
    count: usize,
    pair: (u32, u32),
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count.cmp(&other.count)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[pyfunction]
fn train_bpe_rust(
    global_counts: HashMap<Vec<u8>, usize>,
    target_vocab_size: usize,
) -> PyResult<(Vec<(Vec<u8>, Vec<u8>)>, HashMap<usize, Vec<u8>>)> {
    
    let mut vocab: HashMap<u32, Vec<u8>> = (0..256)
        .map(|i| (i as u32, vec![i as u8]))
        .collect();
    let mut next_id = 256u32;
    let num_merges = target_vocab_size.saturating_sub(256);
    let mut merges = Vec::with_capacity(num_merges);

    let mut words: Vec<Vec<u32>> = Vec::new();
    let mut word_freqs: Vec<usize> = Vec::new();
    
    for (bytes, freq) in global_counts {
        words.push(bytes.into_iter().map(|b| b as u32).collect());
        word_freqs.push(freq);
    }

    let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut pair_to_words: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

    for (idx, word_ids) in words.iter().enumerate() {
        let freq = word_freqs[idx];
        for pair in word_ids.windows(2) {
            let p = (pair[0], pair[1]);
            *pair_counts.entry(p).or_insert(0) += freq;
            pair_to_words.entry(p).or_insert_with(HashSet::new).insert(idx);
        }
    }

    let mut heap: BinaryHeap<HeapItem> = pair_counts.iter()
        .map(|(&pair, &count)| HeapItem { count, pair })
        .collect();

    for _ in 0..num_merges {
        let mut best_pair = None;
        while let Some(top) = heap.pop() {
            if let Some(&current_count) = pair_counts.get(&top.pair) {
                if current_count == top.count {
                    best_pair = Some(top.pair);
                    break;
                }
            }
        }

        let p = match best_pair {
            Some(p) => p,
            None => break,
        };

        let p1_bytes = vocab.get(&p.0).unwrap().clone();
        let p2_bytes = vocab.get(&p.1).unwrap().clone();
        let mut new_token_bytes = p1_bytes;
        new_token_bytes.extend(p2_bytes);
        
        merges.push((vocab.get(&p.0).unwrap().clone(), vocab.get(&p.1).unwrap().clone()));
        vocab.insert(next_id, new_token_bytes);
        
        let new_id = next_id;
        next_id += 1;

        if let Some(word_indices) = pair_to_words.remove(&p) {
            pair_counts.remove(&p);

            for &word_idx in &word_indices {
                let freq = word_freqs[word_idx];
                let old_ids = &words[word_idx];
                let mut new_ids = Vec::new();
                
                let mut i = 0;
                while i < old_ids.len() {
                    if i < old_ids.len() - 1 && old_ids[i] == p.0 && old_ids[i+1] == p.1 {
                        new_ids.push(new_id);
                        i += 2;
                    } else {
                        new_ids.push(old_ids[i]);
                        i += 1;
                    }
                }
                
                for pair in old_ids.windows(2) {
                    let old_p = (pair[0], pair[1]);
                    if let Some(count) = pair_counts.get_mut(&old_p) {
                        *count -= freq;
                    }
                }

                words[word_idx] = new_ids;
                let updated_ids = &words[word_idx];
                for pair in updated_ids.windows(2) {
                    let new_p = (pair[0], pair[1]);
                    let count = pair_counts.entry(new_p).or_insert(0);
                    *count += freq;
                    pair_to_words.entry(new_p).or_insert_with(HashSet::new).insert(word_idx);
                    heap.push(HeapItem { count: *count, pair: new_p });
                }
            }
        }
    }

    let final_vocab = vocab.into_iter().map(|(k, v)| (k as usize, v)).collect();
    Ok((merges, final_vocab))
}

// 🚀 修改点 2：适配最新版 PyO3 的 Bound API 签名
#[pymodule]
fn fast_bpe(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_rust, m)?)?;
    Ok(())
}
