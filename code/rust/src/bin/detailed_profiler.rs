use std::time::Instant;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use rustc_hash::FxHashMap;

// Import the BPE training functions from lib.rs
use rust_bpe::{extract_word_frequencies_with_stats};

/// Detailed profiling version of BPE training with instrumentation
struct DetailedProfiler {
    timings: Vec<(String, f64)>,
}

impl DetailedProfiler {
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
        println!("\n📊 Detailed Timing Summary");
        println!("===========================");
        
        let total_time: f64 = self.timings.iter().map(|(_, t)| t).sum();
        
        for (name, time) in &self.timings {
            let percentage = (time / total_time) * 100.0;
            println!("{:30} {:8.3}s ({:5.1}%)", name, time, percentage);
        }
        println!("{:-30} {:8.3}s", "TOTAL", total_time);
    }
}

/// Manually instrumented BPE training function
fn train_bpe_detailed_profiling(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    max_merges: usize,
    vocab_size: usize,
    special_tokens: &[String],
) -> Result<(HashMap<i32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>> {
    
    let mut profiler = DetailedProfiler::new();
    
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
    
    // Initial pair counting with detailed timing
    let mut pair_counts: FxHashMap<(i32, i32), u64> = profiler.time_operation("Initial pair counting", || {
        let mut counts = FxHashMap::with_capacity_and_hasher(
            word_freqs.len() * 4,
            rustc_hash::FxBuildHasher::default()
        );
        
        for (word_tokens, freq) in &word_freqs {
            for window in word_tokens.windows(2) {
                let pair = (window[0], window[1]);
                *counts.entry(pair).or_insert(0) += freq;
            }
        }
        counts
    });
    
    println!("Total unique pairs: {}", pair_counts.len());
    
    let mut current_word_freqs = word_freqs;
    
    // Main merge loop with detailed timing
    for iteration in 0..target_merges {
        if iteration % 5 == 0 {
            println!("Starting merge iteration {}/{}", iteration, target_merges);
        }
        
        if pair_counts.is_empty() {
            break;
        }
        
        // Find most frequent pair
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
        
        // OPTIMIZED: Apply merge with affected word filtering and parallelization
        let (new_word_freqs, pair_updates) = profiler.time_operation(&format!("Apply merge & update pairs (iter {})", iteration), || {
            apply_merge_optimized(&current_word_freqs, token1_id, token2_id, new_token_id)
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
        
        current_word_freqs = new_word_freqs;
    }
    
    profiler.print_summary();
    
    println!("✅ Detailed BPE training completed with {} merges", merges.len());
    Ok((vocab, merges))
}

/// Apply a single merge to a word
fn apply_merge_to_word(
    word_tokens: &[i32],
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> Vec<i32> {
    let mut result = Vec::with_capacity(word_tokens.len());
    let mut i = 0;
    
    while i < word_tokens.len() {
        if i + 1 < word_tokens.len() 
            && word_tokens[i] == token1_id 
            && word_tokens[i + 1] == token2_id {
            // Apply merge
            result.push(new_token_id);
            i += 2;
        } else {
            result.push(word_tokens[i]);
            i += 1;
        }
    }
    
    result
}

/// OPTIMIZATION 1: Affected word filtering
/// Only process words that contain the target pair
fn apply_merge_optimized(
    word_freqs: &FxHashMap<Vec<i32>, u64>,
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> (FxHashMap<Vec<i32>, u64>, FxHashMap<(i32, i32), i64>) {
    use rayon::prelude::*;
    
    // Convert to vector for parallel processing
    let word_freqs_vec: Vec<_> = word_freqs.iter().collect();
    
    // OPTIMIZATION 1: Filter words that contain the target pair
    let (affected_words, unaffected_words): (Vec<_>, Vec<_>) = word_freqs_vec
        .par_iter()
        .partition(|(word_tokens, _)| {
            contains_consecutive_pair(word_tokens, token1_id, token2_id)
        });
    
    println!("    📊 Affected words: {}/{} ({:.1}%)", 
             affected_words.len(), 
             word_freqs.len(),
             (affected_words.len() as f64 / word_freqs.len() as f64) * 100.0);
    
    // OPTIMIZATION 2: Parallel processing of affected words only
    let (affected_results, pair_deltas) = if !affected_words.is_empty() {
        process_affected_words_parallel(&affected_words, token1_id, token2_id, new_token_id)
    } else {
        (Vec::new(), FxHashMap::default())
    };
    
    // OPTIMIZATION 3: Efficient reconstruction
    let mut new_word_freqs = FxHashMap::with_capacity_and_hasher(
        word_freqs.len(),
        rustc_hash::FxBuildHasher::default()
    );
    
    // Add unaffected words (no copying needed)
    for (word, freq) in unaffected_words {
        new_word_freqs.insert(word.clone(), *freq);
    }
    
    // Add processed affected words
    for (new_word, freq) in affected_results {
        new_word_freqs.insert(new_word, freq);
    }
    
    (new_word_freqs, pair_deltas)
}

/// Fast check if word contains consecutive pair
#[inline]
fn contains_consecutive_pair(word_tokens: &[i32], token1_id: i32, token2_id: i32) -> bool {
    // Optimized sliding window check
    if word_tokens.len() < 2 {
        return false;
    }
    
    // Use fast pointer arithmetic instead of windows()
    for i in 0..word_tokens.len() - 1 {
        if unsafe { 
            *word_tokens.get_unchecked(i) == token1_id && 
            *word_tokens.get_unchecked(i + 1) == token2_id 
        } {
            return true;
        }
    }
    false
}

/// OPTIMIZATION 2: Parallel processing of affected words with batch optimization
fn process_affected_words_parallel(
    affected_words: &[(&Vec<i32>, &u64)],
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> (Vec<(Vec<i32>, u64)>, FxHashMap<(i32, i32), i64>) {
    use rayon::prelude::*;
    
    // Choose chunk size based on workload
    let chunk_size = (affected_words.len() / rayon::current_num_threads()).max(100);
    
    // Process in parallel chunks
    let chunk_results: Vec<_> = affected_words
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_results = Vec::with_capacity(chunk.len());
            let mut local_pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
            
            for (word_tokens, freq) in chunk {
                // Apply merge to this word
                let new_word = apply_merge_to_word_fast(word_tokens, token1_id, token2_id, new_token_id);
                local_results.push((new_word.clone(), **freq));
                
                // Calculate pair deltas efficiently
                update_pair_deltas_fast(&mut local_pair_deltas, word_tokens, &new_word, **freq);
            }
            
            (local_results, local_pair_deltas)
        })
        .collect();
    
    // Merge results from all chunks
    let mut final_results = Vec::with_capacity(affected_words.len());
    let mut final_pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
    
    for (chunk_results, chunk_deltas) in chunk_results {
        final_results.extend(chunk_results);
        
        // Merge pair deltas
        for (pair, delta) in chunk_deltas {
            *final_pair_deltas.entry(pair).or_insert(0) += delta;
        }
    }
    
    (final_results, final_pair_deltas)
}

/// OPTIMIZATION 3: Fast merge application with pre-allocated capacity
#[inline]
fn apply_merge_to_word_fast(
    word_tokens: &[i32],
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> Vec<i32> {
    // Pre-allocate with worst-case size (no merges)
    let mut result = Vec::with_capacity(word_tokens.len());
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
    
    // Shrink to actual size
    result.shrink_to_fit();
    result
}

/// OPTIMIZATION 4: Fast pair delta calculation
#[inline]
fn update_pair_deltas_fast(
    pair_deltas: &mut FxHashMap<(i32, i32), i64>,
    old_word: &[i32],
    new_word: &[i32],
    freq: u64,
) {
    let freq_delta = freq as i64;
    
    // Remove old pairs using fast iteration
    if old_word.len() >= 2 {
        for i in 0..old_word.len() - 1 {
            let pair = unsafe { 
                (*old_word.get_unchecked(i), *old_word.get_unchecked(i + 1)) 
            };
            *pair_deltas.entry(pair).or_insert(0) -= freq_delta;
        }
    }
    
    // Add new pairs using fast iteration
    if new_word.len() >= 2 {
        for i in 0..new_word.len() - 1 {
            let pair = unsafe { 
                (*new_word.get_unchecked(i), *new_word.get_unchecked(i + 1)) 
            };
            *pair_deltas.entry(pair).or_insert(0) += freq_delta;
        }
    }
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
    
    println!("🔍 Detailed BPE Merge Profiler");
    println!("===============================");
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
    
    // Run detailed profiling
    let (_vocab, merges) = match train_bpe_detailed_profiling(
        word_freqs,
        num_merges,
        vocab_size,
        &special_tokens,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ BPE merge training failed: {}", e);
            std::process::exit(1);
        }
    };
    
    let total_time = total_start.elapsed();
    
    println!("\n🎯 Final Results");
    println!("================");
    println!("✓ Merges completed: {}", merges.len());
    println!("✓ Total time: {:.3}s", total_time.as_secs_f64());
    println!("✓ Average time per merge: {:.1}ms", (total_time.as_secs_f64() * 1000.0) / merges.len() as f64);
}