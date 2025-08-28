use std::time::Instant;
use std::fs::File;
use std::io::Read;
use std::collections::{HashMap, HashSet};
use rustc_hash::FxHashMap;

/// Ultra-optimized profiling version with advanced optimizations
struct UltraProfiler {
    timings: Vec<(String, f64)>,
}

impl UltraProfiler {
    fn new() -> Self {
        Self {
            timings: Vec::new(),
        }
    }
    
    fn time_operation<F, R>(&mut self, name: &str, f: F) -> R 
    where 
        F: FnOnce() -> R
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_secs_f64();
        self.timings.push((name.to_string(), elapsed));
        println!("⏱️  {}: {:.3}s", name, elapsed);
        result
    }
    
    fn print_summary(&self) {
        println!("\n📊 Ultra-Optimized Timing Summary");
        println!("=================================");
        
        let total_time: f64 = self.timings.iter().map(|(_, t)| t).sum();
        
        for (name, time) in &self.timings {
            let percentage = (time / total_time) * 100.0;
            println!("{:40} {:8.3}s ({:5.1}%)", name, time, percentage);
        }
        println!("{:-40} {:8.3}s", "TOTAL", total_time);
    }
}

/// Memory pool for Vec<i32> to avoid repeated allocations
struct VecPool {
    available: Vec<Vec<i32>>,
}

impl VecPool {
    fn new() -> Self {
        Self {
            available: Vec::with_capacity(1000),
        }
    }
    
    fn get(&mut self, capacity_hint: usize) -> Vec<i32> {
        if let Some(mut vec) = self.available.pop() {
            vec.clear();
            vec.reserve(capacity_hint);
            vec
        } else {
            Vec::with_capacity(capacity_hint)
        }
    }
    
    fn return_vec(&mut self, mut vec: Vec<i32>) {
        if vec.capacity() <= 1024 { // Don't keep very large vecs
            vec.clear();
            self.available.push(vec);
        }
    }
}

/// OPTIMIZATION 1: Inverted pair index for O(1) affected word lookup
struct InvertedPairIndex {
    /// Maps each pair to the set of word indices that contain it
    pair_to_words: FxHashMap<(i32, i32), HashSet<usize>>,
    /// Word index to word data mapping
    words: Vec<(Vec<i32>, u64)>,
}

impl InvertedPairIndex {
    fn new(word_freqs: &FxHashMap<Vec<i32>, u64>) -> Self {
        let mut pair_to_words: FxHashMap<(i32, i32), HashSet<usize>> = FxHashMap::default();
        let mut words: Vec<(Vec<i32>, u64)> = Vec::with_capacity(word_freqs.len());
        
        // Build index
        for (word_idx, (word_tokens, freq)) in word_freqs.iter().enumerate() {
            words.push((word_tokens.clone(), *freq));
            
            // Index all pairs in this word
            for i in 0..word_tokens.len().saturating_sub(1) {
                let pair = unsafe { 
                    (*word_tokens.get_unchecked(i), *word_tokens.get_unchecked(i + 1))
                };
                pair_to_words.entry(pair).or_insert_with(HashSet::new).insert(word_idx);
            }
        }
        
        Self { pair_to_words, words }
    }
    
    /// Get indices of words affected by a specific pair
    fn get_affected_words(&self, pair: (i32, i32)) -> Option<&HashSet<usize>> {
        self.pair_to_words.get(&pair)
    }
    
    /// Update index after a merge operation
    fn update_after_merge(&mut self, word_idx: usize, old_word: &[i32], new_word: Vec<i32>, freq: u64) {
        // Remove old pairs
        for i in 0..old_word.len().saturating_sub(1) {
            let pair = unsafe { 
                (*old_word.get_unchecked(i), *old_word.get_unchecked(i + 1))
            };
            if let Some(word_set) = self.pair_to_words.get_mut(&pair) {
                word_set.remove(&word_idx);
                if word_set.is_empty() {
                    self.pair_to_words.remove(&pair);
                }
            }
        }
        
        // Add new pairs
        for i in 0..new_word.len().saturating_sub(1) {
            let pair = unsafe { 
                (*new_word.get_unchecked(i), *new_word.get_unchecked(i + 1))
            };
            self.pair_to_words.entry(pair).or_insert_with(HashSet::new).insert(word_idx);
        }
        
        // Update word data
        self.words[word_idx] = (new_word, freq);
    }
    
    fn len(&self) -> usize {
        self.words.len()
    }
    
    fn get_word(&self, idx: usize) -> &(Vec<i32>, u64) {
        &self.words[idx]
    }
    
    fn to_word_freqs(&self) -> FxHashMap<Vec<i32>, u64> {
        let mut word_freqs = FxHashMap::with_capacity_and_hasher(
            self.words.len(),
            rustc_hash::FxBuildHasher::default()
        );
        
        for (word_tokens, freq) in &self.words {
            word_freqs.insert(word_tokens.clone(), *freq);
        }
        
        word_freqs
    }
}

/// OPTIMIZATION 2: SIMD-style consecutive pair detection (simulated with chunked processing)
#[inline]
fn contains_consecutive_pair_fast(word_tokens: &[i32], token1_id: i32, token2_id: i32) -> bool {
    if word_tokens.len() < 2 {
        return false;
    }
    
    // Process in chunks for better cache performance
    let mut i = 0;
    while i + 1 < word_tokens.len() {
        // Unroll loop for better performance
        if i + 3 < word_tokens.len() {
            unsafe {
                let chunk = [
                    *word_tokens.get_unchecked(i),
                    *word_tokens.get_unchecked(i + 1),
                    *word_tokens.get_unchecked(i + 2),
                    *word_tokens.get_unchecked(i + 3),
                ];
                
                if (chunk[0] == token1_id && chunk[1] == token2_id) ||
                   (chunk[1] == token1_id && chunk[2] == token2_id) ||
                   (chunk[2] == token1_id && chunk[3] == token2_id) {
                    return true;
                }
            }
            i += 3;
        } else {
            unsafe {
                if *word_tokens.get_unchecked(i) == token1_id && 
                   *word_tokens.get_unchecked(i + 1) == token2_id {
                    return true;
                }
            }
            i += 1;
        }
    }
    false
}

/// OPTIMIZATION 3: Ultra-fast merge application with memory pooling
fn apply_merge_ultra_fast(
    pool: &mut VecPool,
    word_tokens: &[i32],
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> Vec<i32> {
    // Get a pre-allocated vector from pool
    let mut result = pool.get(word_tokens.len());
    let mut i = 0;
    
    while i < word_tokens.len() {
        if i + 1 < word_tokens.len() {
            // Use unsafe for performance-critical path
            let current = unsafe { *word_tokens.get_unchecked(i) };
            let next = unsafe { *word_tokens.get_unchecked(i + 1) };
            
            if current == token1_id && next == token2_id {
                // Apply merge
                result.push(new_token_id);
                i += 2;
                continue;
            }
        }
        
        result.push(unsafe { *word_tokens.get_unchecked(i) });
        i += 1;
    }
    
    // Don't shrink here to avoid reallocation
    result
}

/// OPTIMIZATION 4: Batch pair delta calculation
fn calculate_pair_deltas_batch(
    old_words: &[(Vec<i32>, u64)],
    new_words: &[(Vec<i32>, u64)],
) -> FxHashMap<(i32, i32), i64> {
    let mut pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
    
    // Process in batches for better cache performance
    for ((old_word, freq), (new_word, _)) in old_words.iter().zip(new_words.iter()) {
        let freq_delta = *freq as i64;
        
        // Remove old pairs
        for i in 0..old_word.len().saturating_sub(1) {
            let pair = unsafe { 
                (*old_word.get_unchecked(i), *old_word.get_unchecked(i + 1))
            };
            *pair_deltas.entry(pair).or_insert(0) -= freq_delta;
        }
        
        // Add new pairs
        for i in 0..new_word.len().saturating_sub(1) {
            let pair = unsafe { 
                (*new_word.get_unchecked(i), *new_word.get_unchecked(i + 1))
            };
            *pair_deltas.entry(pair).or_insert(0) += freq_delta;
        }
    }
    
    pair_deltas
}

/// Ultra-optimized merge application
fn apply_merge_ultra_optimized(
    index: &mut InvertedPairIndex,
    pool: &mut VecPool,
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> FxHashMap<(i32, i32), i64> {
    use rayon::prelude::*;
    
    let target_pair = (token1_id, token2_id);
    
    // OPTIMIZATION 1: O(1) lookup of affected words using inverted index
    let affected_word_indices = match index.get_affected_words(target_pair) {
        Some(indices) => indices.iter().cloned().collect::<Vec<_>>(),
        None => return FxHashMap::default(),
    };
    
    println!("    📊 Affected words: {}/{} ({:.1}%)", 
             affected_word_indices.len(), 
             index.len(),
             (affected_word_indices.len() as f64 / index.len() as f64) * 100.0);
    
    if affected_word_indices.is_empty() {
        return FxHashMap::default();
    }
    
    // OPTIMIZATION 2: Parallel processing with better chunking
    let chunk_size = (affected_word_indices.len() / rayon::current_num_threads()).max(50);
    
    let results: Vec<_> = affected_word_indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_pool = VecPool::new();
            let mut old_words = Vec::with_capacity(chunk.len());
            let mut new_words = Vec::with_capacity(chunk.len());
            
            for &word_idx in chunk {
                let (word_tokens, freq) = index.get_word(word_idx);
                old_words.push((word_tokens.clone(), *freq));
                
                // Apply merge using ultra-fast method
                let new_word = apply_merge_ultra_fast(
                    &mut local_pool, 
                    word_tokens, 
                    token1_id, 
                    token2_id, 
                    new_token_id
                );
                new_words.push((new_word, *freq));
            }
            
            // Calculate pair deltas in batch
            let pair_deltas = calculate_pair_deltas_batch(&old_words, &new_words);
            
            (chunk.to_vec(), old_words, new_words, pair_deltas)
        })
        .collect();
    
    // OPTIMIZATION 3: Efficient index updates and pair delta merging
    let mut final_pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
    
    for (word_indices, old_words, new_words, chunk_deltas) in results {
        // Update index for each affected word
        for ((&word_idx, (old_word_tokens, _old_freq)), (new_word_tokens, new_freq)) in word_indices.iter()
            .zip(old_words.iter())
            .zip(new_words.into_iter()) {
            index.update_after_merge(word_idx, old_word_tokens, new_word_tokens, new_freq);
        }
        
        // Merge pair deltas
        for (pair, delta) in chunk_deltas {
            *final_pair_deltas.entry(pair).or_insert(0) += delta;
        }
    }
    
    final_pair_deltas
}

/// Ultra-optimized BPE training
fn train_bpe_ultra_optimized(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    max_merges: usize,
    vocab_size: usize,
    special_tokens: &[String],
) -> Result<(HashMap<i32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>> {
    
    let mut profiler = UltraProfiler::new();
    let mut pool = VecPool::new();
    
    // Initialize vocabulary with base bytes (0-255)
    let mut vocab: HashMap<i32, Vec<u8>> = profiler.time_operation("Initialize base vocabulary", || {
        (0..256).map(|i| (i as i32, vec![i as u8])).collect()
    });
    
    let mut next_token_id = 256i32;
    
    // Add special tokens to vocab
    profiler.time_operation("Add special tokens", || {
        for special_token in special_tokens {
            vocab.insert(next_token_id, special_token.as_bytes().to_vec());
            next_token_id += 1;
        }
    });
    
    if word_freqs.is_empty() {
        return Ok((vocab, Vec::new()));
    }
    
    let mut merges = Vec::new();
    let target_merges = vocab_size.saturating_sub(vocab.len()).min(max_merges);
    
    println!("Target merges: {}", target_merges);
    println!("Initial unique words: {}", word_freqs.len());
    
    // Build inverted pair index
    let mut index = profiler.time_operation("Build inverted pair index", || {
        InvertedPairIndex::new(&word_freqs)
    });
    
    // Initial pair counting with optimized approach
    let mut pair_counts: FxHashMap<(i32, i32), u64> = profiler.time_operation("Initial pair counting", || {
        let mut counts = FxHashMap::with_capacity_and_hasher(
            word_freqs.len() * 4,
            rustc_hash::FxBuildHasher::default()
        );
        
        for (word_tokens, freq) in &word_freqs {
            for i in 0..word_tokens.len().saturating_sub(1) {
                let pair = unsafe { 
                    (*word_tokens.get_unchecked(i), *word_tokens.get_unchecked(i + 1))
                };
                *counts.entry(pair).or_insert(0) += freq;
            }
        }
        counts
    });
    
    println!("Total unique pairs: {}", pair_counts.len());
    
    // Main merge loop with ultra optimization
    for iteration in 0..target_merges {
        if iteration % 5 == 0 {
            println!("Starting merge iteration {}/{}", iteration, target_merges);
        }
        
        if pair_counts.is_empty() {
            break;
        }
        
        // Find most frequent pair (already optimized)
        let best_pair = profiler.time_operation(&format!("Find best pair (iter {})", iteration), || {
            pair_counts
                .iter()
                .filter(|(_, &count)| count > 0)
                .max_by_key(|(_, &count)| count)
                .map(|(&pair, _)| pair)
        });
        
        let (token1_id, token2_id) = match best_pair {
            Some(pair) => pair,
            None => break,
        };
        
        if pair_counts[&(token1_id, token2_id)] == 0 {
            break;
        }
        
        // Create new token
        let new_token_id = next_token_id;
        next_token_id += 1;
        
        // Merge the tokens in vocabulary
        profiler.time_operation(&format!("Update vocabulary (iter {})", iteration), || {
            let token1_bytes = vocab[&token1_id].clone();
            let token2_bytes = vocab[&token2_id].clone();
            let mut new_token_bytes = token1_bytes.clone();
            new_token_bytes.extend_from_slice(&token2_bytes);
            
            vocab.insert(new_token_id, new_token_bytes);
            merges.push((token1_bytes, token2_bytes));
        });
        
        // ULTRA-OPTIMIZED: Apply merge with inverted index and advanced optimizations
        let pair_updates = profiler.time_operation(&format!("Apply merge & update pairs (iter {})", iteration), || {
            apply_merge_ultra_optimized(&mut index, &mut pool, token1_id, token2_id, new_token_id)
        });
        
        // Apply pair count updates
        profiler.time_operation(&format!("Update pair counts (iter {})", iteration), || {
            for (pair, delta) in pair_updates {
                let old_count = pair_counts.get(&pair).copied().unwrap_or(0);
                let new_count = (old_count as i64 + delta).max(0) as u64;
                
                if new_count == 0 {
                    pair_counts.remove(&pair);
                } else {
                    pair_counts.insert(pair, new_count);
                }
            }
        });
    }
    
    profiler.print_summary();
    
    println!("✅ Ultra-optimized BPE training completed with {} merges", merges.len());
    Ok((vocab, merges))
}

/// Load word frequencies from JSON file
fn load_word_frequencies(file_path: &str) -> Result<FxHashMap<Vec<i32>, u64>, Box<dyn std::error::Error>> {
    println!("📖 Loading word frequencies from {}...", file_path);
    let load_start = Instant::now();
    
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    let word_freqs_serialized: std::collections::BTreeMap<String, u64> = 
        serde_json::from_str(&contents)?;
    
    // Convert back to FxHashMap<Vec<i32>, u64>
    let mut word_freqs = FxHashMap::default();
    for (word_str, freq) in word_freqs_serialized {
        let word_tokens: Vec<i32> = word_str
            .split(',')
            .map(|s| s.parse::<i32>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse token IDs: {}", e))?;
        
        word_freqs.insert(word_tokens, freq);
    }
    
    let load_time = load_start.elapsed();
    println!("✓ Loaded {} unique word types in {:.3}s", word_freqs.len(), load_time.as_secs_f64());
    
    Ok(word_freqs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 4 {
        eprintln!("Usage: {} <word_freq_file> <num_merges> <vocab_size>", args[0]);
        std::process::exit(1);
    }
    
    let word_freq_file = &args[1];
    let num_merges = args[2].parse::<usize>().expect("Invalid num_merges");
    let vocab_size = args[3].parse::<usize>().expect("Invalid vocab_size");
    let special_tokens = vec!["<|endoftext|>".to_string()];
    
    println!("🚀 Ultra-Optimized BPE Merge Profiler");
    println!("======================================");
    println!("Word freq file: {}", word_freq_file);
    println!("Number of merges: {}", num_merges);
    println!("Vocab size: {}", vocab_size);
    println!("CPU cores available: {}", rayon::current_num_threads());
    println!();
    
    let total_start = Instant::now();
    
    // Load word frequencies
    let word_freqs = match load_word_frequencies(word_freq_file) {
        Ok(freqs) => freqs,
        Err(e) => {
            eprintln!("❌ Failed to load word frequencies: {}", e);
            std::process::exit(1);
        }
    };
    
    // Run ultra-optimized profiling
    let (_vocab, merges) = match train_bpe_ultra_optimized(
        word_freqs,
        num_merges,
        vocab_size,
        &special_tokens,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ BPE ultra-optimization failed: {}", e);
            std::process::exit(1);
        }
    };
    
    let total_time = total_start.elapsed();
    
    println!("\n🎯 Ultra-Optimized Results");
    println!("===========================");
    println!("✓ Merges completed: {}", merges.len());
    println!("✓ Total time: {:.3}s", total_time.as_secs_f64());
    println!("✓ Average time per merge: {:.1}ms", (total_time.as_secs_f64() * 1000.0) / merges.len() as f64);
}