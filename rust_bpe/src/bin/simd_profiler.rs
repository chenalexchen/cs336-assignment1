use std::time::Instant;
use std::fs::File;
use std::io::Read;
use std::collections::{HashMap, HashSet};
use rustc_hash::FxHashMap;
use std::arch::x86_64::*;

/// SIMD and Batch-optimized profiling version with cutting-edge optimizations
struct SIMDProfiler {
    timings: Vec<(String, f64)>,
}

impl SIMDProfiler {
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
        println!("\n📊 SIMD + Batch Optimized Timing Summary");
        println!("========================================");
        
        let total_time: f64 = self.timings.iter().map(|(_, t)| t).sum();
        
        for (name, time) in &self.timings {
            let percentage = (time / total_time) * 100.0;
            println!("{:40} {:8.3}s ({:5.1}%)", name, time, percentage);
        }
        println!("{:-40} {:8.3}s", "TOTAL", total_time);
    }
}

/// Memory pool with SIMD-aligned allocations
struct SIMDVecPool {
    available: Vec<Vec<i32>>,
}

impl SIMDVecPool {
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
        if vec.capacity() <= 1024 {
            vec.clear();
            self.available.push(vec);
        }
    }
}

/// OPTIMIZATION 1: SIMD-accelerated consecutive pair detection
#[target_feature(enable = "avx2")]
unsafe fn contains_consecutive_pair_simd(word_tokens: &[i32], token1_id: i32, token2_id: i32) -> bool {
    if word_tokens.len() < 2 {
        return false;
    }
    
    // For small words, use regular check
    if word_tokens.len() < 8 {
        for i in 0..word_tokens.len() - 1 {
            if *word_tokens.get_unchecked(i) == token1_id && 
               *word_tokens.get_unchecked(i + 1) == token2_id {
                return true;
            }
        }
        return false;
    }
    
    // SIMD vectorized search for larger words
    let target1 = _mm256_set1_epi32(token1_id);
    let target2 = _mm256_set1_epi32(token2_id);
    
    let mut i = 0;
    while i + 8 <= word_tokens.len() {
        // Load 8 consecutive tokens
        let data = _mm256_loadu_si256(word_tokens.as_ptr().add(i) as *const __m256i);
        
        // Compare with target1
        let cmp1 = _mm256_cmpeq_epi32(data, target1);
        
        if i + 9 <= word_tokens.len() {
            // Load next 8 tokens (shifted by 1)
            let data_next = _mm256_loadu_si256(word_tokens.as_ptr().add(i + 1) as *const __m256i);
            
            // Compare with target2
            let cmp2 = _mm256_cmpeq_epi32(data_next, target2);
            
            // Check if consecutive pairs match
            let matches = _mm256_and_si256(cmp1, cmp2);
            
            if _mm256_testz_si256(matches, matches) == 0 {
                return true;
            }
        }
        
        i += 8;
    }
    
    // Check remaining elements
    while i + 1 < word_tokens.len() {
        if *word_tokens.get_unchecked(i) == token1_id && 
           *word_tokens.get_unchecked(i + 1) == token2_id {
            return true;
        }
        i += 1;
    }
    
    false
}

/// Safe wrapper for SIMD pair detection
fn contains_consecutive_pair_fast_simd(word_tokens: &[i32], token1_id: i32, token2_id: i32) -> bool {
    if is_x86_feature_detected!("avx2") {
        unsafe { contains_consecutive_pair_simd(word_tokens, token1_id, token2_id) }
    } else {
        // Fallback to regular optimized version
        for i in 0..word_tokens.len().saturating_sub(1) {
            unsafe {
                if *word_tokens.get_unchecked(i) == token1_id && 
                   *word_tokens.get_unchecked(i + 1) == token2_id {
                    return true;
                }
            }
        }
        false
    }
}

/// OPTIMIZATION 2: SIMD-accelerated merge application
#[target_feature(enable = "avx2")]
unsafe fn apply_merge_simd(
    word_tokens: &[i32],
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> Vec<i32> {
    let mut result = Vec::with_capacity(word_tokens.len());
    let mut i = 0;
    
    // For small words, use regular processing
    if word_tokens.len() < 16 {
        while i < word_tokens.len() {
            if i + 1 < word_tokens.len() {
                let current = *word_tokens.get_unchecked(i);
                let next = *word_tokens.get_unchecked(i + 1);
                
                if current == token1_id && next == token2_id {
                    result.push(new_token_id);
                    i += 2;
                    continue;
                }
            }
            
            result.push(*word_tokens.get_unchecked(i));
            i += 1;
        }
        return result;
    }
    
    // SIMD-accelerated processing for larger words
    let target1 = _mm256_set1_epi32(token1_id);
    let target2 = _mm256_set1_epi32(token2_id);
    
    while i + 8 <= word_tokens.len() {
        // Load 8 consecutive tokens
        let data = _mm256_loadu_si256(word_tokens.as_ptr().add(i) as *const __m256i);
        let cmp1 = _mm256_cmpeq_epi32(data, target1);
        
        if i + 9 <= word_tokens.len() {
            let data_next = _mm256_loadu_si256(word_tokens.as_ptr().add(i + 1) as *const __m256i);
            let cmp2 = _mm256_cmpeq_epi32(data_next, target2);
            let matches = _mm256_and_si256(cmp1, cmp2);
            
            // Extract match mask
            let mask = _mm256_movemask_epi8(matches);
            
            if mask != 0 {
                // Found matches, process element by element for this chunk
                for j in 0..8 {
                    if i + j + 1 < word_tokens.len() {
                        let current = *word_tokens.get_unchecked(i + j);
                        let next = *word_tokens.get_unchecked(i + j + 1);
                        
                        if current == token1_id && next == token2_id {
                            result.push(new_token_id);
                            i += j + 2;
                            break;
                        } else {
                            result.push(current);
                            if j == 7 {
                                i += 8;
                            }
                        }
                    } else {
                        result.push(*word_tokens.get_unchecked(i + j));
                        i += j + 1;
                        break;
                    }
                }
            } else {
                // No matches in this chunk, copy all 8 elements
                for j in 0..8 {
                    result.push(*word_tokens.get_unchecked(i + j));
                }
                i += 8;
            }
        } else {
            // Process remaining elements normally
            for j in 0..8 {
                if i + j < word_tokens.len() {
                    result.push(*word_tokens.get_unchecked(i + j));
                }
            }
            i += 8;
        }
    }
    
    // Process remaining elements
    while i < word_tokens.len() {
        if i + 1 < word_tokens.len() {
            let current = *word_tokens.get_unchecked(i);
            let next = *word_tokens.get_unchecked(i + 1);
            
            if current == token1_id && next == token2_id {
                result.push(new_token_id);
                i += 2;
                continue;
            }
        }
        
        result.push(*word_tokens.get_unchecked(i));
        i += 1;
    }
    
    result
}

/// Safe wrapper for SIMD merge application
fn apply_merge_ultra_fast_simd(
    pool: &mut SIMDVecPool,
    word_tokens: &[i32],
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> Vec<i32> {
    if is_x86_feature_detected!("avx2") && word_tokens.len() >= 16 {
        unsafe { apply_merge_simd(word_tokens, token1_id, token2_id, new_token_id) }
    } else {
        // Fallback to regular optimized version
        let mut result = pool.get(word_tokens.len());
        let mut i = 0;
        
        while i < word_tokens.len() {
            if i + 1 < word_tokens.len() {
                unsafe {
                    let current = *word_tokens.get_unchecked(i);
                    let next = *word_tokens.get_unchecked(i + 1);
                    
                    if current == token1_id && next == token2_id {
                        result.push(new_token_id);
                        i += 2;
                        continue;
                    }
                }
            }
            
            result.push(unsafe { *word_tokens.get_unchecked(i) });
            i += 1;
        }
        
        result
    }
}

/// Inverted pair index with optimized data structures
struct OptimizedInvertedIndex {
    pair_to_words: FxHashMap<(i32, i32), Vec<usize>>,
    words: Vec<(Vec<i32>, u64)>,
    dirty_pairs: HashSet<(i32, i32)>,
}

impl OptimizedInvertedIndex {
    fn new(word_freqs: &FxHashMap<Vec<i32>, u64>) -> Self {
        let mut pair_to_words: FxHashMap<(i32, i32), Vec<usize>> = FxHashMap::default();
        let mut words: Vec<(Vec<i32>, u64)> = Vec::with_capacity(word_freqs.len());
        
        // Build index with pre-allocated vectors
        for (word_idx, (word_tokens, freq)) in word_freqs.iter().enumerate() {
            words.push((word_tokens.clone(), *freq));
            
            // Index all pairs in this word
            for i in 0..word_tokens.len().saturating_sub(1) {
                let pair = unsafe { 
                    (*word_tokens.get_unchecked(i), *word_tokens.get_unchecked(i + 1))
                };
                pair_to_words.entry(pair).or_insert_with(Vec::new).push(word_idx);
            }
        }
        
        Self { 
            pair_to_words, 
            words,
            dirty_pairs: HashSet::new(),
        }
    }
    
    fn get_affected_words(&self, pair: (i32, i32)) -> Option<&Vec<usize>> {
        self.pair_to_words.get(&pair)
    }
    
    fn update_after_merge(&mut self, word_idx: usize, old_word: &[i32], new_word: Vec<i32>, freq: u64) {
        // Mark old pairs as dirty for batch cleanup
        for i in 0..old_word.len().saturating_sub(1) {
            let pair = unsafe { 
                (*old_word.get_unchecked(i), *old_word.get_unchecked(i + 1))
            };
            self.dirty_pairs.insert(pair);
        }
        
        // Add new pairs immediately
        for i in 0..new_word.len().saturating_sub(1) {
            let pair = unsafe { 
                (*new_word.get_unchecked(i), *new_word.get_unchecked(i + 1))
            };
            self.pair_to_words.entry(pair).or_insert_with(Vec::new).push(word_idx);
        }
        
        // Update word data
        self.words[word_idx] = (new_word, freq);
    }
    
    fn cleanup_dirty_pairs(&mut self) {
        for &pair in &self.dirty_pairs {
            if let Some(word_indices) = self.pair_to_words.get_mut(&pair) {
                // Filter out words that no longer contain this pair
                word_indices.retain(|&word_idx| {
                    let (word_tokens, _) = &self.words[word_idx];
                    contains_consecutive_pair_fast_simd(word_tokens, pair.0, pair.1)
                });
                
                if word_indices.is_empty() {
                    self.pair_to_words.remove(&pair);
                }
            }
        }
        self.dirty_pairs.clear();
    }
    
    fn len(&self) -> usize {
        self.words.len()
    }
    
    fn get_word(&self, idx: usize) -> &(Vec<i32>, u64) {
        &self.words[idx]
    }
}

/// OPTIMIZATION 3: Batch merging - process multiple pairs in one pass
struct BatchMergeCandidate {
    pair: (i32, i32),
    new_token_id: i32,
    count: u64,
    affected_words: Vec<usize>,
}

fn find_batch_merge_candidates(
    pair_counts: &FxHashMap<(i32, i32), u64>,
    index: &OptimizedInvertedIndex,
    batch_size: usize,
) -> Vec<BatchMergeCandidate> {
    let mut candidates: Vec<_> = pair_counts
        .iter()
        .filter(|(_, &count)| count > 0)
        .map(|(&pair, &count)| {
            let affected_words = index.get_affected_words(pair)
                .map(|v| v.clone())
                .unwrap_or_default();
            (pair, count, affected_words)
        })
        .collect();
    
    // Sort by count (descending)
    candidates.sort_by_key(|(_, count, _)| std::cmp::Reverse(*count));
    
    // Select non-conflicting pairs for batch processing
    let mut selected = Vec::new();
    let mut used_words = HashSet::new();
    let mut next_token_id = 512; // Start from safe range
    
    for (pair, count, affected_words) in candidates.into_iter().take(batch_size * 3) {
        // Check if any affected words are already being processed
        let conflicts = affected_words.iter().any(|&word_idx| used_words.contains(&word_idx));
        
        if !conflicts && selected.len() < batch_size {
            // Mark words as used
            for &word_idx in &affected_words {
                used_words.insert(word_idx);
            }
            
            selected.push(BatchMergeCandidate {
                pair,
                new_token_id: next_token_id,
                count,
                affected_words,
            });
            
            next_token_id += 1;
        }
        
        if selected.len() >= batch_size {
            break;
        }
    }
    
    selected
}

/// OPTIMIZATION 4: Parallel batch merge processing
fn apply_batch_merges_simd(
    index: &mut OptimizedInvertedIndex,
    pool: &mut SIMDVecPool,
    candidates: Vec<BatchMergeCandidate>,
) -> FxHashMap<(i32, i32), i64> {
    use rayon::prelude::*;
    
    if candidates.is_empty() {
        return FxHashMap::default();
    }
    
    println!("    🚀 Batch processing {} merges simultaneously", candidates.len());
    
    // Process all batched merges in parallel
    let results: Vec<_> = candidates
        .par_iter()
        .map(|candidate| {
            let mut local_pool = SIMDVecPool::new();
            let mut old_words = Vec::with_capacity(candidate.affected_words.len());
            let mut new_words = Vec::with_capacity(candidate.affected_words.len());
            
            for &word_idx in &candidate.affected_words {
                let (word_tokens, freq) = index.get_word(word_idx);
                old_words.push((word_tokens.clone(), *freq));
                
                // Apply merge using SIMD
                let new_word = apply_merge_ultra_fast_simd(
                    &mut local_pool,
                    word_tokens,
                    candidate.pair.0,
                    candidate.pair.1,
                    candidate.new_token_id,
                );
                new_words.push((new_word, *freq));
            }
            
            (candidate, old_words, new_words)
        })
        .collect();
    
    // Update index and calculate pair deltas
    let mut final_pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
    
    for (candidate, old_words, new_words) in results {
        // Calculate pair deltas for this merge first (before consuming new_words)
        for (old_word_tokens, freq) in &old_words {
            let freq_delta = *freq as i64;
            
            // Remove old pairs
            for i in 0..old_word_tokens.len().saturating_sub(1) {
                let pair = unsafe { 
                    (*old_word_tokens.get_unchecked(i), *old_word_tokens.get_unchecked(i + 1))
                };
                *final_pair_deltas.entry(pair).or_insert(0) -= freq_delta;
            }
        }
        
        // Update index for each affected word
        for ((&word_idx, (old_word_tokens, _)), (new_word_tokens, new_freq)) in 
            candidate.affected_words.iter()
                .zip(old_words.iter())
                .zip(new_words.into_iter()) {
            index.update_after_merge(word_idx, old_word_tokens, new_word_tokens, new_freq);
        }
    }
    
    // Cleanup dirty pairs after batch processing
    index.cleanup_dirty_pairs();
    
    final_pair_deltas
}

/// SIMD + Batch optimized BPE training
fn train_bpe_simd_batch_optimized(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    max_merges: usize,
    vocab_size: usize,
    special_tokens: &[String],
) -> Result<(HashMap<i32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>> {
    
    let mut profiler = SIMDProfiler::new();
    let mut pool = SIMDVecPool::new();
    
    // Check SIMD support
    println!("🔧 SIMD Support - AVX2: {}", is_x86_feature_detected!("avx2"));
    
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
    
    // Build optimized inverted pair index
    let mut index = profiler.time_operation("Build optimized inverted pair index", || {
        OptimizedInvertedIndex::new(&word_freqs)
    });
    
    // Initial pair counting
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
    
    // Main merge loop with SIMD and batch optimization
    let batch_size = 4; // Process up to 4 merges simultaneously
    let mut iteration = 0;
    
    while iteration < target_merges && !pair_counts.is_empty() {
        if iteration % 10 == 0 {
            println!("Starting batch merge iteration {}/{}", iteration, target_merges);
        }
        
        // BATCH OPTIMIZATION: Find multiple non-conflicting merges
        let batch_candidates = profiler.time_operation(
            &format!("Find batch candidates (iter {})", iteration), 
            || find_batch_merge_candidates(&pair_counts, &index, batch_size)
        );
        
        if batch_candidates.is_empty() {
            break;
        }
        
        let actual_batch_size = batch_candidates.len();
        
        // Update vocabulary for all candidates
        profiler.time_operation(&format!("Update vocabulary (batch {})", iteration), || {
            for candidate in &batch_candidates {
                let token1_bytes = vocab[&candidate.pair.0].clone();
                let token2_bytes = vocab[&candidate.pair.1].clone();
                let mut new_token_bytes = token1_bytes.clone();
                new_token_bytes.extend_from_slice(&token2_bytes);
                
                vocab.insert(candidate.new_token_id, new_token_bytes);
                merges.push((token1_bytes, token2_bytes));
            }
        });
        
        // SIMD + BATCH: Apply all merges simultaneously
        let pair_updates = profiler.time_operation(
            &format!("Apply SIMD batch merges (iter {})", iteration), 
            || apply_batch_merges_simd(&mut index, &mut pool, batch_candidates)
        );
        
        // Update pair counts
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
        
        iteration += actual_batch_size;
    }
    
    profiler.print_summary();
    
    println!("✅ SIMD + Batch optimized BPE training completed with {} merges", merges.len());
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
    
    println!("🚀 SIMD + Batch Optimized BPE Merge Profiler");
    println!("=============================================");
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
    
    // Run SIMD + batch optimized profiling
    let (_vocab, merges) = match train_bpe_simd_batch_optimized(
        word_freqs,
        num_merges,
        vocab_size,
        &special_tokens,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ SIMD + Batch optimization failed: {}", e);
            std::process::exit(1);
        }
    };
    
    let total_time = total_start.elapsed();
    
    println!("\n🎯 SIMD + Batch Optimized Results");
    println!("==================================");
    println!("✓ Merges completed: {}", merges.len());
    println!("✓ Total time: {:.3}s", total_time.as_secs_f64());
    println!("✓ Average time per merge: {:.1}ms", (total_time.as_secs_f64() * 1000.0) / merges.len() as f64);
}