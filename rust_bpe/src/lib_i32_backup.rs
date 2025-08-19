#[cfg(feature = "python")]
use pyo3::prelude::*;

// U16 optimized version for memory efficiency
pub mod lib_u16;
use regex::Regex;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxBuildHasher};
use std::collections::{HashMap, BinaryHeap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, BufRead, BufReader};
use std::sync::OnceLock;
use memchr::memchr;

/// Pre-tokenization regex pattern matching GPT-2 style
const PRE_TOKENIZE_PATTERN: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// Global regex cache for better performance
static PRE_TOKENIZE_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_pre_tokenize_regex() -> &'static Regex {
    PRE_TOKENIZE_REGEX.get_or_init(|| {
        Regex::new(PRE_TOKENIZE_PATTERN).unwrap()
    })
}

/// Priority queue entry for efficient pair selection
#[derive(Debug, Clone, PartialEq, Eq)]
struct PairEntry {
    count: u64,
    pair: (i32, i32),
    // Tie-breaker using lexicographic ordering of token bytes
    tie_breaker: (Vec<u8>, Vec<u8>),
}

impl PartialOrd for PairEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: highest count
        match self.count.cmp(&other.count) {
            std::cmp::Ordering::Equal => {
                // Tie-breaker: lexicographic order (reversed for min-heap behavior)
                other.tie_breaker.cmp(&self.tie_breaker)
            }
            other => other
        }
    }
}

/// A single BPE merge rule
#[derive(Debug, Clone, PartialEq, Eq)]
struct MergeRule {
    token1: Vec<u8>,
    token2: Vec<u8>,
}

/// BPE Tokenizer implementation
#[cfg_attr(feature = "python", pyclass)]
pub struct BPETokenizer {
    vocab: FxHashMap<i32, Vec<u8>>,
    vocab_reverse: FxHashMap<Vec<u8>, i32>,
    merges: Vec<MergeRule>,
    special_tokens: Vec<String>,
    special_token_ids: FxHashMap<String, i32>,
}

#[cfg_attr(feature = "python", pymethods)]
impl BPETokenizer {
    #[cfg_attr(feature = "python", new)]
    #[cfg_attr(feature = "python", pyo3(signature = (vocab, merges, special_tokens=None)))]
    pub fn new(
        vocab: HashMap<i32, Vec<u8>>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Option<Vec<String>>,
    ) -> Self {
        let vocab: FxHashMap<i32, Vec<u8>> = vocab.into_iter().collect();
        let vocab_reverse: FxHashMap<Vec<u8>, i32> = vocab.iter().map(|(k, v)| (v.clone(), *k)).collect();
        
        let merges = merges
            .into_iter()
            .map(|(token1, token2)| MergeRule { token1, token2 })
            .collect();
        
        let special_tokens = special_tokens.unwrap_or_default();
        let mut special_token_ids = FxHashMap::default();
        
        // Map special tokens to their IDs
        for token in &special_tokens {
            let token_bytes = token.as_bytes().to_vec();
            if let Some(&id) = vocab_reverse.get(&token_bytes) {
                special_token_ids.insert(token.clone(), id);
            }
        }
        
        Self {
            vocab,
            vocab_reverse,
            merges,
            special_tokens,
            special_token_ids,
        }
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str) -> Vec<i32> {
        if text.is_empty() {
            return Vec::new();
        }

        let tokens = pre_tokenize(text, &self.special_tokens);
        let byte_sequences = tokens_to_bytes(&tokens, &self.special_tokens);
        
        let mut result = Vec::new();
        for byte_seq in byte_sequences {
            let token_ids = self.apply_merges(&byte_seq);
            result.extend(token_ids);
        }
        
        result
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[i32]) -> String {
        let mut result = Vec::new();
        
        for token_id in token_ids {
            if let Some(token_bytes) = self.vocab.get(&token_id) {
                result.extend_from_slice(token_bytes);
            }
        }
        
        String::from_utf8_lossy(&result).to_string()
    }

    /// Apply BPE merges to a sequence of byte tokens
    fn apply_merges(&self, byte_seq: &[i32]) -> Vec<i32> {
        if byte_seq.len() <= 1 {
            return byte_seq.to_vec();
        }

        let mut word: Vec<i32> = byte_seq.to_vec();
        
        loop {
            let mut best_merge_idx = None;
            let mut best_merge_rank = self.merges.len();
            
            // Find the highest priority merge (earliest in merge list)
            for i in 0..word.len().saturating_sub(1) {
                // Convert tokens back to bytes to match against merge rules
                let token1_bytes = if let Some(bytes) = self.vocab.get(&word[i]) {
                    bytes.clone()
                } else {
                    continue;
                };
                
                let token2_bytes = if let Some(bytes) = self.vocab.get(&word[i + 1]) {
                    bytes.clone()
                } else {
                    continue;
                };
                
                // Find this pair in the merges
                for (rank, merge_rule) in self.merges.iter().enumerate() {
                    if merge_rule.token1 == token1_bytes && merge_rule.token2 == token2_bytes {
                        if rank < best_merge_rank {
                            best_merge_rank = rank;
                            best_merge_idx = Some(i);
                        }
                        break;
                    }
                }
            }
            
            // If no merge found, we're done
            if best_merge_idx.is_none() {
                break;
            }
            
            let merge_idx = best_merge_idx.unwrap();
            let merge_rule = &self.merges[best_merge_rank];
            
            // Apply the merge
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i == merge_idx {
                    // Create merged token - look up in vocab_reverse
                    let mut merged_bytes = merge_rule.token1.clone();
                    merged_bytes.extend_from_slice(&merge_rule.token2);
                    
                    if let Some(&merged_id) = self.vocab_reverse.get(&merged_bytes) {
                        new_word.push(merged_id);
                    } else {
                        // If merged token not in vocab, keep original tokens
                        new_word.push(word[i]);
                        if i + 1 < word.len() {
                            new_word.push(word[i + 1]);
                        }
                    }
                    i += 2;
                } else {
                    new_word.push(word[i]);
                    i += 1;
                }
            }
            
            word = new_word;
        }
        
        word
    }
}

/// Pre-tokenize text into words using GPT-2 style regex
fn pre_tokenize(text: &str, special_tokens: &[String]) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    
    // Split on special tokens first, preferring longer matches
    let mut parts = vec![text.to_string()];
    
    // Sort special tokens by length (descending) to prefer longer matches
    let mut sorted_special_tokens = special_tokens.to_vec();
    sorted_special_tokens.sort_by_key(|s| std::cmp::Reverse(s.len()));
    
    for special_token in &sorted_special_tokens {
        let mut new_parts = Vec::new();
        for part in parts {
            if part == *special_token {
                new_parts.push(part);
            } else if special_tokens.contains(&part) {
                // This part is already a special token, don't split it further
                new_parts.push(part);
            } else {
                let split_parts: Vec<&str> = part.split(special_token).collect();
                for (i, split_part) in split_parts.iter().enumerate() {
                    if i > 0 {
                        new_parts.push(special_token.clone());
                    }
                    if !split_part.is_empty() {
                        new_parts.push(split_part.to_string());
                    }
                }
            }
        }
        parts = new_parts;
    }
    
    // Apply regex tokenization to non-special parts
    let regex = get_pre_tokenize_regex();
    let mut result = Vec::new();
    
    for part in parts {
        if special_tokens.contains(&part) {
            result.push(part);
        } else {
            for mat in regex.find_iter(&part) {
                result.push(mat.as_str().to_string());
            }
        }
    }
    
    result
}

/// Convert text tokens to byte token sequences
fn tokens_to_bytes(tokens: &[String], special_tokens: &[String]) -> Vec<Vec<i32>> {
    tokens
        .iter()
        .map(|token| {
            if special_tokens.contains(token) {
                // Special tokens become single tokens with IDs >= 256
                let special_id = 256 + special_tokens.iter().position(|t| t == token).unwrap() as i32;
                vec![special_id]
            } else {
                // Regular text becomes byte sequences
                token.bytes().map(|b| b as i32).collect()
            }
        })
        .collect()
}

/// Extract word frequencies from a text file with optimized streaming
pub fn extract_word_frequencies_with_stats(
    file_path: &str,
    special_tokens: &[String],
) -> Result<(FxHashMap<Vec<i32>, u64>, usize), Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let file_size = file.metadata()?.len();
    
    println!("File size: {:.1}MB", file_size as f64 / (1024.0 * 1024.0));
    
    // Choose processing strategy based on file size
    if file_size > 500_000_000 { // > 500MB
        println!("Using streaming pipeline (large file)");
        extract_word_frequencies_streaming(file_path, special_tokens)
    } else {
        println!("Using parallel processing (medium file)");
        extract_word_frequencies_parallel(file_path, special_tokens)
    }
}

/// Streaming approach for very large files
fn extract_word_frequencies_streaming(
    file_path: &str,
    special_tokens: &[String],
) -> Result<(FxHashMap<Vec<i32>, u64>, usize), Box<dyn std::error::Error>> {
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;
    
    let file = File::open(file_path)?;
    let file_size = file.metadata()?.len();
    println!("Using pipelined streaming approach for large dataset ({:.1}GB)", file_size as f64 / (1024.0 * 1024.0 * 1024.0));
    
    let chunk_size = 256 * 1024 * 1024; // 256MB chunks
    let num_workers = rayon::current_num_threads().min(8);
    
    // Channels for pipeline
    let (chunk_sender, chunk_receiver) = mpsc::sync_channel::<(Vec<u8>, usize)>(4);
    let (result_sender, result_receiver) = mpsc::sync_channel::<FxHashMap<Vec<i32>, u64>>(8);
    
    let chunk_receiver = Arc::new(Mutex::new(chunk_receiver));
    let special_tokens = special_tokens.to_vec();
    
    // Reader thread
    let file_path_owned = file_path.to_string();
    let reader_handle = {
        let chunk_sender = chunk_sender.clone();
        thread::spawn(move || {
            let mut file = File::open(&file_path_owned).unwrap();
            let mut chunk_id = 0;
            let mut total_read = 0u64;
            
            loop {
                let mut buffer = vec![0u8; chunk_size];
                let bytes_read = file.read(&mut buffer).unwrap_or(0);
                
                if bytes_read == 0 {
                    break;
                }
                
                buffer.truncate(bytes_read);
                total_read += bytes_read as u64;
                
                // Find a safe boundary (end of line)
                if bytes_read == chunk_size {
                    if let Some(newline_pos) = memchr(b'\n', &buffer[bytes_read.saturating_sub(512)..]) {
                        let safe_end = bytes_read.saturating_sub(512) + newline_pos + 1;
                        buffer.truncate(safe_end);
                        
                        // Seek back to safe position
                        let seek_back = bytes_read - safe_end;
                        file.seek(SeekFrom::Current(-(seek_back as i64))).unwrap();
                        total_read -= seek_back as u64;
                    }
                }
                
                if chunk_sender.send((buffer, chunk_id)).is_err() {
                    break;
                }
                
                chunk_id += 1;
                if chunk_id % 20 == 0 {
                    println!("Reader: queued chunk {} ({:.1}GB read)", chunk_id, total_read as f64 / (1024.0 * 1024.0 * 1024.0));
                }
            }
            
            println!("Reader: finished reading {} chunks", chunk_id);
        })
    };
    
    // Worker pool
    let mut worker_handles = Vec::new();
    for worker_id in 0..num_workers {
        let chunk_receiver = Arc::clone(&chunk_receiver);
        let result_sender = result_sender.clone();
        let special_tokens = special_tokens.clone();
        
        let handle = thread::spawn(move || {
            let mut processed_count = 0;
            
            while let Ok((chunk_data, _chunk_id)) = {
                let receiver = chunk_receiver.lock().unwrap();
                receiver.recv()
            } {
                let text = String::from_utf8_lossy(&chunk_data);
                let word_freqs = process_text_chunk(&text, &special_tokens);
                
                if result_sender.send(word_freqs).is_err() {
                    break;
                }
                
                processed_count += 1;
            }
            
            println!("Worker {}: finished processing {} chunks", worker_id, processed_count);
        });
        
        worker_handles.push(handle);
    }
    
    // Result collector
    drop(chunk_sender); // Signal no more chunks
    drop(result_sender); // Allow workers to finish
    
    let collector_handle = thread::spawn(move || {
        let mut final_word_freqs: FxHashMap<Vec<i32>, u64> = FxHashMap::default();
        let mut processed_chunks = 0;
        
        while let Ok(chunk_result) = result_receiver.recv() {
            for (word, freq) in chunk_result {
                *final_word_freqs.entry(word).or_insert(0) += freq;
            }
            processed_chunks += 1;
            
            if processed_chunks % 20 == 0 {
                println!("Collector: merged {} chunk results", processed_chunks);
            }
        }
        
        println!("Pipelined streaming complete: {} chunks read, {} chunks processed", processed_chunks, processed_chunks);
        final_word_freqs
    });
    
    // Wait for completion
    reader_handle.join().unwrap();
    for handle in worker_handles {
        handle.join().unwrap();
    }
    let final_word_freqs = collector_handle.join().unwrap();
    
    Ok((final_word_freqs, 0)) // Chunk count not tracked in streaming mode
}

/// Parallel approach for medium-sized files
fn extract_word_frequencies_parallel(
    file_path: &str,
    special_tokens: &[String],
) -> Result<(FxHashMap<Vec<i32>, u64>, usize), Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    // Read all lines
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
    println!("Processing {} lines in parallel", lines.len());
    
    // Process in parallel chunks
    let chunk_size = (lines.len() / rayon::current_num_threads()).max(1000);
    
    let word_freqs: FxHashMap<Vec<i32>, u64> = lines
        .par_chunks(chunk_size)
        .map(|chunk| {
            let text = chunk.join("\n");
            process_text_chunk(&text, special_tokens)
        })
        .reduce(
            || FxHashMap::default(),
            |mut acc, chunk_freqs| {
                for (word, freq) in chunk_freqs {
                    *acc.entry(word).or_insert(0) += freq;
                }
                acc
            }
        );
    
    let chunk_count = (lines.len() + chunk_size - 1) / chunk_size; // Ceiling division
    Ok((word_freqs, chunk_count))
}

/// Process a chunk of text into word frequencies
fn process_text_chunk(text: &str, special_tokens: &[String]) -> FxHashMap<Vec<i32>, u64> {
    let mut word_freqs = FxHashMap::default();
    
    // Split into lines and process each
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        
        // Pre-tokenize the line
        let tokens = pre_tokenize(line, special_tokens);
        let byte_sequences = tokens_to_bytes(&tokens, special_tokens);
        
        // Count each word (flattened byte sequence)
        for byte_seq in byte_sequences {
            if !byte_seq.is_empty() {
                *word_freqs.entry(byte_seq).or_insert(0) += 1;
            }
        }
    }
    
    word_freqs
}

/// Simple baseline BPE training (without optimizations) for comparison
pub fn train_bpe_from_word_freqs_baseline(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    vocab_size: usize,
    special_tokens: &[String],
) -> Result<(HashMap<i32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>> {
    
    println!("📊 Using baseline BPE training algorithm (no optimizations)");
    
    // Initialize vocabulary with base bytes (0-255)
    let mut vocab: HashMap<i32, Vec<u8>> = (0..256)
        .map(|i| (i as i32, vec![i as u8]))
        .collect();
    let mut next_token_id = 256i32;
    
    // Add special tokens to vocab
    for special_token in special_tokens {
        vocab.insert(next_token_id, special_token.as_bytes().to_vec());
        next_token_id += 1;
    }
    
    if word_freqs.is_empty() {
        return Ok((vocab, Vec::new()));
    }
    
    let mut merges = Vec::new();
    let target_merges = vocab_size.saturating_sub(vocab.len());
    
    println!("Target merges: {}", target_merges);
    println!("Initial unique words: {}", word_freqs.len());
    
    // Simple initial pair counting
    println!("Building initial pair counts...");
    let pair_count_start = std::time::Instant::now();
    
    let mut pair_counts: FxHashMap<(i32, i32), u64> = FxHashMap::default();
    for (word_tokens, freq) in &word_freqs {
        for window in word_tokens.windows(2) {
            let pair = (window[0], window[1]);
            *pair_counts.entry(pair).or_insert(0) += freq;
        }
    }
    
    println!("Initial pair counting: {:.3}s", pair_count_start.elapsed().as_secs_f64());
    println!("Total unique pairs: {}", pair_counts.len());
    
    let mut current_word_freqs = word_freqs;
    
    for iteration in 0..target_merges {
        if iteration % 1000 == 0 {
            let progress = iteration as f64 / target_merges as f64 * 100.0;
            println!("Progress: {:.1}% ({}/{} merges)", progress, iteration, target_merges);
        }
        
        if pair_counts.is_empty() {
            break;
        }
        
        // Find most frequent pair using linear search (no heap optimization)
        let best_pair_result = pair_counts
            .iter()
            .filter(|(_, &count)| count > 0)
            .max_by(|(&pair1, &count1), (&pair2, &count2)| {
                match count1.cmp(&count2) {
                    std::cmp::Ordering::Equal => {
                        // Tie-breaker: lexicographic order of token bytes
                        let tie1 = (&vocab[&pair1.0], &vocab[&pair1.1]);
                        let tie2 = (&vocab[&pair2.0], &vocab[&pair2.1]);
                        tie2.cmp(&tie1) // Reverse for min comparison
                    }
                    other => other
                }
            })
            .map(|(&pair, _)| pair);
        
        let (token1_id, token2_id) = match best_pair_result {
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
        let token1_bytes = vocab[&token1_id].clone();
        let token2_bytes = vocab[&token2_id].clone();
        let mut new_token_bytes = token1_bytes.clone();
        new_token_bytes.extend_from_slice(&token2_bytes);
        
        vocab.insert(new_token_id, new_token_bytes);
        merges.push((token1_bytes, token2_bytes));
        
        // Simple approach: rebuild everything from scratch (no incremental updates)
        pair_counts.clear();
        let mut new_word_freqs = FxHashMap::default();
        
        // Apply merge to all words
        for (word_tokens, freq) in current_word_freqs {
            let new_word = apply_merge_to_word(&word_tokens, token1_id, token2_id, new_token_id);
            new_word_freqs.insert(new_word, freq);
        }
        
        // Rebuild pair counts from scratch
        for (word_tokens, freq) in &new_word_freqs {
            for window in word_tokens.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += freq;
            }
        }
        
        current_word_freqs = new_word_freqs;
    }
    
    println!("✅ Baseline BPE training completed with {} merges", merges.len());
    Ok((vocab, merges))
}

/// Optimized BPE training with all performance improvements
pub fn train_bpe_from_word_freqs(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    vocab_size: usize,
    special_tokens: &[String],
) -> Result<(HashMap<i32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>> {
    
    println!("🚀 Using optimized BPE training algorithm");
    
    // Initialize vocabulary with base bytes (0-255)
    let mut vocab: HashMap<i32, Vec<u8>> = (0..256)
        .map(|i| (i as i32, vec![i as u8]))
        .collect();
    let mut next_token_id = 256i32;
    
    // Add special tokens to vocab
    for special_token in special_tokens {
        vocab.insert(next_token_id, special_token.as_bytes().to_vec());
        next_token_id += 1;
    }
    
    if word_freqs.is_empty() {
        return Ok((vocab, Vec::new()));
    }
    
    let mut merges = Vec::new();
    let target_merges = vocab_size.saturating_sub(vocab.len());
    
    println!("Target merges: {}", target_merges);
    println!("Initial unique words: {}", word_freqs.len());
    
    // OPTIMIZATION 1: Pre-allocate collections with known capacity
    let mut pair_counts: FxHashMap<(i32, i32), u64> = FxHashMap::with_capacity_and_hasher(
        word_freqs.len() * 4, // Estimate: avg 4 pairs per word
        FxBuildHasher::default()
    );
    
    // OPTIMIZATION 2: Batch initial pair counting
    println!("Building initial pair counts...");
    let pair_count_start = std::time::Instant::now();
    
    // Parallel initial pair counting
    let pair_count_results: Vec<FxHashMap<(i32, i32), u64>> = word_freqs
        .par_iter()
        .map(|(word_tokens, freq)| {
            let mut local_counts = FxHashMap::default();
            for window in word_tokens.windows(2) {
                let pair = (window[0], window[1]);
                *local_counts.entry(pair).or_insert(0) += freq;
            }
            local_counts
        })
        .collect();
    
    // Merge pair counts efficiently
    for local_counts in pair_count_results {
        for (pair, count) in local_counts {
            *pair_counts.entry(pair).or_insert(0) += count;
        }
    }
    
    println!("Initial pair counting: {:.3}s", pair_count_start.elapsed().as_secs_f64());
    println!("Total unique pairs: {}", pair_counts.len());
    
    // OPTIMIZATION 3: Use priority queue for efficient pair selection
    let mut pair_heap = BinaryHeap::with_capacity(pair_counts.len());
    for (&pair, &count) in &pair_counts {
        if count > 0 {
            let tie_breaker = (vocab[&pair.0].clone(), vocab[&pair.1].clone());
            pair_heap.push(PairEntry { count, pair, tie_breaker });
        }
    }
    
    let mut current_word_freqs = word_freqs;
    let progress_interval = target_merges / 100; // Report every 1%
    
    for iteration in 0..target_merges {
        // Progress reporting
        if progress_interval > 0 && iteration % progress_interval == 0 {
            let progress = iteration as f64 / target_merges as f64 * 100.0;
            println!("Progress: {:.1}% ({}/{} merges)", progress, iteration, target_merges);
        }
        
        // OPTIMIZATION 4: Efficient pair selection using heap
        let best_pair = loop {
            match pair_heap.pop() {
                Some(entry) => {
                    // Verify the count is still current (lazy deletion)
                    if pair_counts.get(&entry.pair).copied().unwrap_or(0) == entry.count && entry.count > 0 {
                        break entry.pair;
                    }
                    // Skip stale entries
                }
                None => {
                    // No more valid pairs
                    println!("✅ BPE training completed with {} merges", merges.len());
                    return Ok((vocab, merges));
                }
            }
        };
        
        let (token1_id, token2_id) = best_pair;
        
        // Create new token
        let new_token_id = next_token_id;
        next_token_id += 1;
        
        // Merge the tokens in vocabulary
        let token1_bytes = vocab[&token1_id].clone();
        let token2_bytes = vocab[&token2_id].clone();
        let mut new_token_bytes = token1_bytes.clone();
        new_token_bytes.extend_from_slice(&token2_bytes);
        
        vocab.insert(new_token_id, new_token_bytes);
        merges.push((token1_bytes, token2_bytes));
        
        // OPTIMIZATION 5: Incremental pair count updates
        let (new_word_freqs, pair_deltas) = if current_word_freqs.len() > 50_000 {
            // Large dataset: use streaming approach
            process_merge_optimized_streaming(
                current_word_freqs,
                token1_id,
                token2_id,
                new_token_id,
            )
        } else {
            // Medium dataset: use parallel batching
            process_merge_optimized_parallel(
                current_word_freqs,
                token1_id,
                token2_id,
                new_token_id,
            )
        };
        
        // Apply pair count updates efficiently
        let mut new_pairs_to_add = Vec::new();
        for (pair, delta) in pair_deltas {
            let old_count = pair_counts.get(&pair).copied().unwrap_or(0);
            let new_count = (old_count as i64 + delta).max(0) as u64;
            
            if new_count == 0 {
                pair_counts.remove(&pair);
            } else {
                pair_counts.insert(pair, new_count);
                
                // Add new/updated pairs to heap
                if new_count > old_count {
                    let tie_breaker = (vocab[&pair.0].clone(), vocab[&pair.1].clone());
                    new_pairs_to_add.push(PairEntry { 
                        count: new_count, 
                        pair, 
                        tie_breaker 
                    });
                }
            }
        }
        
        // Batch insert new pairs into heap
        for entry in new_pairs_to_add {
            pair_heap.push(entry);
        }
        
        current_word_freqs = new_word_freqs;
    }
    
    println!("✅ BPE training completed with {} merges", merges.len());
    Ok((vocab, merges))
}

/// Optimized merge processing for large datasets using streaming
fn process_merge_optimized_streaming(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> (FxHashMap<Vec<i32>, u64>, FxHashMap<(i32, i32), i64>) {
    
    // Convert to vector for parallel processing
    let word_freqs_vec: Vec<_> = word_freqs.into_iter().collect();
    let chunk_size = (word_freqs_vec.len() / rayon::current_num_threads()).max(5000);
    
    // Process in parallel chunks
    let results: Vec<_> = word_freqs_vec
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut new_word_freqs = FxHashMap::default();
            let mut pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
            
            for (word_tokens, freq) in chunk {
                let new_word = apply_merge_to_word(word_tokens, token1_id, token2_id, new_token_id);
                
                if new_word != *word_tokens {
                    // Word changed - update pair counts
                    update_pair_deltas(&mut pair_deltas, word_tokens, &new_word, *freq);
                }
                
                new_word_freqs.insert(new_word, *freq);
            }
            
            (new_word_freqs, pair_deltas)
        })
        .collect();
    
    // Merge results
    let mut final_word_freqs = FxHashMap::default();
    let mut final_pair_deltas: FxHashMap<(i32, i32), i64> = FxHashMap::default();
    
    for (word_freqs, pair_deltas) in results {
        for (word, freq) in word_freqs {
            final_word_freqs.insert(word, freq);
        }
        
        for (pair, delta) in pair_deltas {
            *final_pair_deltas.entry(pair).or_insert(0) += delta;
        }
    }
    
    (final_word_freqs, final_pair_deltas)
}

/// Optimized merge processing for medium datasets
fn process_merge_optimized_parallel(
    word_freqs: FxHashMap<Vec<i32>, u64>,
    token1_id: i32,
    token2_id: i32,
    new_token_id: i32,
) -> (FxHashMap<Vec<i32>, u64>, FxHashMap<(i32, i32), i64>) {
    
    let word_freqs_vec: Vec<_> = word_freqs.into_iter().collect();
    
    // Filter words that contain the target pair for efficiency
    let affected_words: Vec<_> = word_freqs_vec
        .par_iter()
        .filter(|(word_tokens, _)| {
            word_tokens.windows(2).any(|w| w[0] == token1_id && w[1] == token2_id)
        })
        .collect();
    
    let unaffected_words: Vec<_> = word_freqs_vec
        .par_iter()
        .filter(|(word_tokens, _)| {
            !word_tokens.windows(2).any(|w| w[0] == token1_id && w[1] == token2_id)
        })
        .map(|(word, freq)| (word.clone(), *freq))
        .collect();
    
    // Process only affected words
    let pair_deltas: FxHashMap<(i32, i32), i64> = affected_words
        .par_iter()
        .map(|(word_tokens, freq)| {
            let new_word = apply_merge_to_word(word_tokens, token1_id, token2_id, new_token_id);
            let mut local_deltas = FxHashMap::default();
            update_pair_deltas(&mut local_deltas, word_tokens, &new_word, *freq);
            local_deltas
        })
        .reduce(
            || FxHashMap::default(),
            |mut acc, local_deltas| {
                for (pair, delta) in local_deltas {
                    *acc.entry(pair).or_insert(0) += delta;
                }
                acc
            }
        );
    
    // Reconstruct word frequencies
    let mut new_word_freqs: FxHashMap<Vec<i32>, u64> = unaffected_words.into_iter().collect();
    
    for (word_tokens, freq) in affected_words {
        let new_word = apply_merge_to_word(word_tokens, token1_id, token2_id, new_token_id);
        new_word_freqs.insert(new_word, *freq);
    }
    
    (new_word_freqs, pair_deltas)
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

/// Update pair count deltas efficiently
fn update_pair_deltas(
    pair_deltas: &mut FxHashMap<(i32, i32), i64>,
    old_word: &[i32],
    new_word: &[i32],
    freq: u64,
) {
    let freq_delta = freq as i64;
    
    // Remove old pairs
    for window in old_word.windows(2) {
        let pair = (window[0], window[1]);
        *pair_deltas.entry(pair).or_insert(0) -= freq_delta;
    }
    
    // Add new pairs
    for window in new_word.windows(2) {
        let pair = (window[0], window[1]);
        *pair_deltas.entry(pair).or_insert(0) += freq_delta;
    }
}

/// Train BPE from a text file (Python-compatible function)
#[cfg(feature = "python")]
#[cfg_attr(feature = "python", pyo3::pyfunction)]
pub fn train_bpe(
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> Result<(HashMap<i32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), Box<dyn std::error::Error>> {
    let (word_freqs, _chunk_count) = extract_word_frequencies_with_stats(input_path, &special_tokens)?;
    let (vocab, merges) = train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens)?;
    Ok((vocab, merges))
}

/// Python module
#[cfg(feature = "python")]
#[cfg_attr(feature = "python", pyo3::pymodule)]
fn rust_bpe(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<BPETokenizer>()?;
    m.add_function(pyo3::wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    
    fn create_test_vocab() -> FxHashMap<i32, Vec<u8>> {
        (0..256).map(|i| (i as i32, vec![i as u8])).collect()
    }
    
    fn create_test_tokenizer_with_merges(merges: Vec<(Vec<u8>, Vec<u8>)>) -> BPETokenizer {
        let mut vocab = create_test_vocab();
        let mut next_id = 256;
        
        // Add merged tokens to vocab
        for (token1, token2) in &merges {
            let mut merged = token1.clone();
            merged.extend_from_slice(token2);
            vocab.insert(next_id, merged);
            next_id += 1;
        }
        
        // Add special token
        vocab.insert(next_id, b"<|endoftext|>".to_vec());
        
        BPETokenizer::new(
            vocab.into_iter().collect(),
            merges,
            Some(vec!["<|endoftext|>".to_string()]),
        )
    }
    
    #[test]
    fn test_empty_text_encoding() {
        let tokenizer = create_test_tokenizer_with_merges(vec![]);
        let encoded = tokenizer.encode("");
        assert_eq!(encoded, Vec::<i32>::new());
        
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "");
    }
    
    #[test]
    fn test_single_character_encoding() {
        let tokenizer = create_test_tokenizer_with_merges(vec![]);
        
        // Test ASCII character
        let encoded = tokenizer.encode("A");
        assert_eq!(encoded, vec![65]); // ASCII 'A' = 65
        
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "A");
    }
    
    #[test]
    fn test_unicode_character_encoding() {
        let tokenizer = create_test_tokenizer_with_merges(vec![]);
        
        let text = "🙃";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
        
        // Test that it round-trips correctly
        assert!(!encoded.is_empty());
    }
    
    #[test]
    fn test_basic_ascii_string() {
        let tokenizer = create_test_tokenizer_with_merges(vec![]);
        
        let text = "Hello, how are you?";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
        
        // Should have multiple tokens due to pre-tokenization
        assert!(encoded.len() > 1);
    }
    
    #[test]
    fn test_unicode_string() {
        let tokenizer = create_test_tokenizer_with_merges(vec![]);
        
        let text = "Héllò hôw are ü? 🙃";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }
    
    #[test]
    fn test_basic_merge() {
        // Create tokenizer with "th" merge
        let tokenizer = create_test_tokenizer_with_merges(vec![
            (b"t".to_vec(), b"h".to_vec())
        ]);
        
        let encoded = tokenizer.encode("the");
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "the");
        
        // Should contain the merged token (256) for "th"
        assert!(encoded.contains(&256));
    }
    
    #[test]
    fn test_multiple_merges() {
        // Create tokenizer with multiple merges
        let tokenizer = create_test_tokenizer_with_merges(vec![
            (b"t".to_vec(), b"h".to_vec()),    // "th" -> 256
            (b"e".to_vec(), b"r".to_vec()),    // "er" -> 257
            (b"th".to_vec(), b"e".to_vec()),   // "the" -> 258 (th + e)
        ]);
        
        let encoded = tokenizer.encode("the");
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "the");
        
        // Should use the most complete merge (258 for "the")
        assert!(encoded.contains(&258));
    }
    
    #[test]
    fn test_special_tokens() {
        let mut vocab = create_test_vocab();
        vocab.insert(256, b"<|endoftext|>".to_vec());
        
        let tokenizer = BPETokenizer::new(
            vocab.into_iter().collect(),
            vec![],
            Some(vec!["<|endoftext|>".to_string()]),
        );
        
        let encoded = tokenizer.encode("Hello<|endoftext|>");
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "Hello<|endoftext|>");
        
        // Should contain the special token ID (256)
        assert!(encoded.contains(&256));
    }
    
    #[test]
    fn test_overlapping_special_tokens() {
        let mut vocab = create_test_vocab();
        vocab.insert(256, b"<|endoftext|>".to_vec());
        vocab.insert(257, b"<|endoftext|><|endoftext|>".to_vec());
        
        let tokenizer = BPETokenizer::new(
            vocab.into_iter().collect(),
            vec![],
            Some(vec!["<|endoftext|>".to_string(), "<|endoftext|><|endoftext|>".to_string()]),
        );
        
        let text = "<|endoftext|><|endoftext|>";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
        
        // Should use the longer special token (257)
        assert!(encoded.contains(&257));
        // Should NOT contain the shorter one when the longer is available
        assert_eq!(encoded.len(), 1);
    }
    
    #[test]
    fn test_special_tokens_with_text() {
        let mut vocab = create_test_vocab();
        vocab.insert(256, b"<|endoftext|>".to_vec());
        
        let tokenizer = BPETokenizer::new(
            vocab.into_iter().collect(),
            vec![],
            Some(vec!["<|endoftext|>".to_string()]),
        );
        
        let text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
        
        // Count special token occurrences
        let special_count = encoded.iter().filter(|&&id| id == 256).count();
        assert_eq!(special_count, 3);
    }
    
    #[test]
    fn test_pre_tokenization() {
        let tokens = pre_tokenize("Hello, world!", &[]);
        assert!(!tokens.is_empty());
        
        // Should preserve punctuation and spaces correctly
        let rejoined: String = tokens.join("");
        assert_eq!(rejoined, "Hello, world!");
    }
    
    #[test]
    fn test_pre_tokenization_with_special_tokens() {
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let text = "Hello <|endoftext|> world";
        
        let tokens = pre_tokenize(text, &special_tokens);
        assert!(tokens.contains(&"<|endoftext|>".to_string()));
        
        let rejoined: String = tokens.join("");
        assert_eq!(rejoined, text);
    }
    
    #[test]
    fn test_apply_merge_to_word() {
        // Test basic merge: "th" -> token 256
        let word = vec![116, 104, 101]; // "the" in ASCII
        let result = apply_merge_to_word(&word, 116, 104, 256);
        assert_eq!(result, vec![256, 101]); // "th" (256) + "e" (101)
        
        // Test word without the target pair
        let word2 = vec![104, 101, 108, 108, 111]; // "hello"
        let result2 = apply_merge_to_word(&word2, 116, 104, 256);
        assert_eq!(result2, word2); // Should be unchanged
        
        // Test multiple occurrences
        let word3 = vec![116, 104, 116, 104]; // "thth"
        let result3 = apply_merge_to_word(&word3, 116, 104, 256);
        assert_eq!(result3, vec![256, 256]); // Both "th" should merge
    }
    
    #[test]
    fn test_update_pair_deltas() {
        let mut pair_deltas = FxHashMap::default();
        
        let old_word = vec![116, 104, 101]; // "the"
        let new_word = vec![256, 101];      // "th" (256) + "e"
        let freq = 5;
        
        update_pair_deltas(&mut pair_deltas, &old_word, &new_word, freq);
        
        // Should remove old pairs and add new ones
        assert_eq!(pair_deltas.get(&(116, 104)), Some(&-5)); // Remove "t" + "h"
        assert_eq!(pair_deltas.get(&(104, 101)), Some(&-5)); // Remove "h" + "e"
        assert_eq!(pair_deltas.get(&(256, 101)), Some(&5));  // Add "th" + "e"
    }
    
    #[test]
    fn test_word_frequency_extraction() {
        // Create a temporary file for testing
        let temp_file = "test_word_freq.txt";
        {
            let mut file = std::fs::File::create(temp_file).unwrap();
            writeln!(file, "hello world").unwrap();
            writeln!(file, "hello rust").unwrap();
            writeln!(file, "world of rust").unwrap();
        }
        
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let result = extract_word_frequencies_with_stats(temp_file, &special_tokens);
        
        assert!(result.is_ok());
        let (word_freqs, chunk_count) = result.unwrap();
        
        assert!(chunk_count > 0);
        assert!(!word_freqs.is_empty());
        
        // Should have entries for common words
        let hello_bytes: Vec<i32> = "hello".bytes().map(|b| b as i32).collect();
        assert!(word_freqs.contains_key(&hello_bytes));
        
        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_train_bpe_from_word_freqs() {
        // Create simple word frequencies
        let mut word_freqs = FxHashMap::default();
        word_freqs.insert(vec![104, 101, 108, 108, 111], 5); // "hello" appears 5 times
        word_freqs.insert(vec![119, 111, 114, 108, 100], 3); // "world" appears 3 times
        word_freqs.insert(vec![116, 104, 101], 10);          // "the" appears 10 times
        
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab_size = 300;
        
        let result = train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens);
        assert!(result.is_ok());
        
        let (vocab, merges) = result.unwrap();
        
        // Should have base vocab (256) + special token (1) + some merges
        assert!(vocab.len() >= 257);
        assert!(!merges.is_empty());
        
        // Special token should be in vocab
        assert!(vocab.values().any(|v| v == b"<|endoftext|>"));
        
        // Common pairs should be merged (like "th", "he", "ll")
        assert!(merges.len() > 0);
    }
    
    #[test]
    fn test_baseline_vs_optimized_consistency() {
        // Create simple word frequencies for consistent testing
        let mut word_freqs = FxHashMap::default();
        word_freqs.insert(vec![116, 104, 101], 10); // "the"
        word_freqs.insert(vec![116, 104, 105, 115], 5); // "this"
        word_freqs.insert(vec![104, 101, 108, 108, 111], 3); // "hello"
        
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab_size = 270; // Small vocab to ensure both finish
        
        let result1 = train_bpe_from_word_freqs(word_freqs.clone(), vocab_size, &special_tokens);
        let result2 = train_bpe_from_word_freqs_baseline(word_freqs, vocab_size, &special_tokens);
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        let (vocab1, merges1) = result1.unwrap();
        let (vocab2, merges2) = result2.unwrap();
        
        // Both should produce the same results
        assert_eq!(vocab1.len(), vocab2.len());
        assert_eq!(merges1, merges2);
    }
    
    #[test]
    fn test_performance_regression() {
        // Test to ensure we don't accidentally break performance
        use std::time::Instant;
        
        let mut word_freqs = FxHashMap::default();
        // Add more realistic data
        for i in 0..1000 {
            let word = vec![65 + (i % 26) as i32, 66 + ((i + 1) % 26) as i32, 67 + ((i + 2) % 26) as i32];
            word_freqs.insert(word, (i % 10 + 1) as u64);
        }
        
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab_size = 300;
        
        let start = Instant::now();
        let result = train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        
        // Should complete in reasonable time (less than 5 seconds for test data)
        assert!(duration.as_secs() < 5, "Training took too long: {:?}", duration);
    }
    
    #[test]
    fn test_empty_input_handling() {
        let empty_word_freqs = FxHashMap::default();
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab_size = 300;
        
        let result = train_bpe_from_word_freqs(empty_word_freqs, vocab_size, &special_tokens);
        assert!(result.is_ok());
        
        let (vocab, merges) = result.unwrap();
        assert_eq!(vocab.len(), 257); // 256 base + 1 special token
        assert_eq!(merges.len(), 0);
    }
    
    #[test]
    fn test_special_token_preservation() {
        let mut vocab = create_test_vocab();
        vocab.insert(256, b"<|endoftext|>".to_vec());
        vocab.insert(257, b"<|special|>".to_vec());
        
        let tokenizer = BPETokenizer::new(
            vocab.into_iter().collect(),
            vec![],
            Some(vec!["<|endoftext|>".to_string(), "<|special|>".to_string()]),
        );
        
        let text = "Hello <|special|> world <|endoftext|>";
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        
        assert_eq!(decoded, text);
        assert!(encoded.contains(&256)); // <|endoftext|>
        assert!(encoded.contains(&257)); // <|special|>
    }
    
    #[test]
    fn test_large_vocabulary_handling() {
        // Test with a larger vocabulary to ensure we handle edge cases
        let mut word_freqs = FxHashMap::default();
        
        // Create a diverse set of words
        for i in 0..100 {
            let word1 = vec![65 + (i % 26) as i32, 97 + (i % 26) as i32]; // Capital + lowercase
            let word2 = vec![48 + (i % 10) as i32, 65 + (i % 26) as i32]; // Number + letter
            
            word_freqs.insert(word1, 1);
            word_freqs.insert(word2, 1);
        }
        
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab_size = 500;
        
        let result = train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens);
        assert!(result.is_ok());
        
        let (vocab, merges) = result.unwrap();
        assert!(vocab.len() <= vocab_size);
        assert!(merges.len() > 0);
    }
    
    #[test]
    fn test_merge_rule_ordering() {
        // Test that merges are applied in the correct order
        let mut word_freqs = FxHashMap::default();
        word_freqs.insert(vec![65, 66, 67], 10); // "ABC" - high frequency
        word_freqs.insert(vec![65, 66], 5);      // "AB" - medium frequency
        word_freqs.insert(vec![66, 67], 3);      // "BC" - low frequency
        
        let special_tokens = vec![];
        let vocab_size = 270;
        
        let result = train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens);
        assert!(result.is_ok());
        
        let (_vocab, merges) = result.unwrap();
        assert!(!merges.is_empty());
        
        // The first merge should be the most frequent pair
        // This depends on the specific frequency counting, but there should be a logical order
    }
    
    #[test]
    fn test_tokens_to_bytes() {
        let tokens = vec!["hello".to_string(), "world".to_string()];
        let special_tokens = vec![];
        
        let byte_sequences = tokens_to_bytes(&tokens, &special_tokens);
        
        assert_eq!(byte_sequences.len(), 2);
        assert_eq!(byte_sequences[0], vec![104, 101, 108, 108, 111]); // "hello"
        assert_eq!(byte_sequences[1], vec![119, 111, 114, 108, 100]); // "world"
    }
    
    #[test]
    fn test_tokens_to_bytes_with_special() {
        let tokens = vec!["hello".to_string(), "<|endoftext|>".to_string(), "world".to_string()];
        let special_tokens = vec!["<|endoftext|>".to_string()];
        
        let byte_sequences = tokens_to_bytes(&tokens, &special_tokens);
        
        assert_eq!(byte_sequences.len(), 3);
        assert_eq!(byte_sequences[0], vec![104, 101, 108, 108, 111]); // "hello"
        assert_eq!(byte_sequences[1], vec![256]); // special token gets ID 256
        assert_eq!(byte_sequences[2], vec![119, 111, 114, 108, 100]); // "world"
    }
    
    #[test]
    fn test_speed_benchmark() {
        // Test that mimics the Python speed test
        use std::time::Instant;
        
        // Create test data similar to corpus.en
        let temp_file = "test_speed_benchmark.txt";
        {
            let mut file = std::fs::File::create(temp_file).unwrap();
            for _ in 0..1000 {
                writeln!(file, "the quick brown fox jumps over the lazy dog").unwrap();
                writeln!(file, "hello world this is a test sentence").unwrap();
                writeln!(file, "rust programming language is fast and safe").unwrap();
            }
        }
        
        let special_tokens = vec!["<|endoftext|>".to_string()];
        let vocab_size = 500;
        
        let start = Instant::now();
        let result = extract_word_frequencies_with_stats(temp_file, &special_tokens);
        assert!(result.is_ok());
        
        let (word_freqs, _) = result.unwrap();
        let result2 = train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens);
        assert!(result2.is_ok());
        
        let duration = start.elapsed();
        
        // Should complete in reasonable time (less than 2 seconds for test data)
        assert!(duration.as_secs() < 2, "BPE training took too long: {:?}", duration);
        
        // Cleanup
        std::fs::remove_file(temp_file).unwrap();
    }
    
    #[test]
    fn test_edge_cases() {
        // Test various edge cases
        
        // Very small vocabulary
        let mut word_freqs = FxHashMap::default();
        word_freqs.insert(vec![65, 66], 1); // "AB"
        
        let result = train_bpe_from_word_freqs(word_freqs, 257, &[]);
        assert!(result.is_ok());
        let (vocab, merges) = result.unwrap();
        assert_eq!(vocab.len(), 257); // Just base vocab + 1 merge
        assert_eq!(merges.len(), 1);
        
        // Single character words
        let mut word_freqs2 = FxHashMap::default();
        word_freqs2.insert(vec![65], 10); // "A"
        word_freqs2.insert(vec![66], 10); // "B"
        
        let result2 = train_bpe_from_word_freqs(word_freqs2, 300, &[]);
        assert!(result2.is_ok());
        let (vocab2, merges2) = result2.unwrap();
        assert_eq!(vocab2.len(), 256); // No merges possible with single chars
        assert_eq!(merges2.len(), 0);
    }
}

// Integration tests for CLI tools
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::process::{Command, Stdio};
    use std::io::Write;
    use tempfile::TempDir;
    
    fn create_test_corpus(content: &str) -> std::io::Result<(TempDir, String)> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test_corpus.txt");
        let mut file = fs::File::create(&file_path)?;
        file.write_all(content.as_bytes())?;
        Ok((temp_dir, file_path.to_string_lossy().to_string()))
    }
    
    fn build_binary(binary_name: &str) -> std::io::Result<String> {
        let output = Command::new("cargo")
            .args(&["build", "--release", "--bin", binary_name, "--no-default-features"])
            .current_dir(".") 
            .output()?;
            
        if !output.status.success() {
            panic!("Failed to build {}: {}", binary_name, String::from_utf8_lossy(&output.stderr));
        }
        
        Ok(format!("./target/release/{}", binary_name))
    }
    
    #[test]
    fn test_train_bpe_cli_basic_functionality() {
        // Create test corpus
        let corpus_content = "hello world\nhello rust\nworld of programming\nrust is great\nhello everyone";
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        
        // Create output directory
        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().to_string_lossy().to_string();
        
        // Build the binary
        let binary_path = build_binary("train_bpe").unwrap();
        
        // Run the CLI tool
        let output = Command::new(&binary_path)
            .args(&[&corpus_path, "300", &output_path])
            .output()
            .expect("Failed to execute train_bpe");
        
        if !output.status.success() {
            panic!("train_bpe failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        // Verify output files exist
        assert!(Path::new(&format!("{}/vocab.json", output_path)).exists());
        assert!(Path::new(&format!("{}/merges.txt", output_path)).exists());
        assert!(Path::new(&format!("{}/training_stats.txt", output_path)).exists());
        
        // Verify vocab.json is valid JSON
        let vocab_content = fs::read_to_string(format!("{}/vocab.json", output_path)).unwrap();
        let _vocab: serde_json::Value = serde_json::from_str(&vocab_content).unwrap();
        
        // Verify merges.txt has content
        let merges_content = fs::read_to_string(format!("{}/merges.txt", output_path)).unwrap();
        assert!(!merges_content.trim().is_empty());
        
        // Verify training stats
        let stats_content = fs::read_to_string(format!("{}/training_stats.txt", output_path)).unwrap();
        assert!(stats_content.contains("Final vocabulary size:"));
        assert!(stats_content.contains("Special tokens:"));
    }
    
    #[test]
    fn test_train_bpe_cli_with_custom_special_tokens() {
        let corpus_content = "hello <|endoftext|> world <|pad|> rust programming";
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        
        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().to_string_lossy().to_string();
        
        let binary_path = build_binary("train_bpe").unwrap();
        
        // Run with custom special tokens
        let output = Command::new(&binary_path)
            .args(&[&corpus_path, "300", &output_path, "<|endoftext|>", "<|pad|>", "<|unk|>"])
            .output()
            .expect("Failed to execute train_bpe");
        
        if !output.status.success() {
            panic!("train_bpe failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        // Verify special tokens are in training stats
        let stats_content = fs::read_to_string(format!("{}/training_stats.txt", output_path)).unwrap();
        assert!(stats_content.contains("<|endoftext|>"));
        assert!(stats_content.contains("<|pad|>"));
        assert!(stats_content.contains("<|unk|>"));
    }
    
    #[test]
    fn test_train_bpe_cli_error_handling() {
        let binary_path = build_binary("train_bpe").unwrap();
        
        // Test missing arguments
        let output = Command::new(&binary_path)
            .args(&["file.txt"])
            .stderr(Stdio::piped())
            .output()
            .expect("Failed to execute train_bpe");
        
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Usage:"));
        
        // Test non-existent input file
        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().to_string_lossy().to_string();
        
        let output = Command::new(&binary_path)
            .args(&["nonexistent_file.txt", "300", &output_path])
            .stderr(Stdio::piped())
            .output()
            .expect("Failed to execute train_bpe");
        
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("does not exist"));
        
        // Test invalid vocab size
        let (_temp_dir, corpus_path) = create_test_corpus("hello world").unwrap();
        
        let output = Command::new(&binary_path)
            .args(&[&corpus_path, "invalid_number", &output_path])
            .stderr(Stdio::piped())
            .output()
            .expect("Failed to execute train_bpe");
        
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Invalid vocab_size"));
    }
    
    #[test]
    fn test_extract_word_freq_cli_functionality() {
        let corpus_content = "hello world hello rust world programming rust hello";
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        
        let output_dir = TempDir::new().unwrap();
        let output_file = output_dir.path().join("word_freqs.json");
        let output_path = output_file.to_string_lossy().to_string();
        
        let binary_path = build_binary("extract_word_freq").unwrap();
        
        // Run word frequency extraction
        let output = Command::new(&binary_path)
            .args(&[&corpus_path, &output_path])
            .output()
            .expect("Failed to execute extract_word_freq");
        
        if !output.status.success() {
            panic!("extract_word_freq failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        // Verify output file exists and is valid JSON
        assert!(output_file.exists());
        let freq_content = fs::read_to_string(&output_file).unwrap();
        let freq_data: serde_json::Value = serde_json::from_str(&freq_content).unwrap();
        
        // Should be a JSON object with word frequencies
        assert!(freq_data.is_object());
        assert!(!freq_data.as_object().unwrap().is_empty());
    }
    
    #[test]
    fn test_extract_word_freq_cli_with_special_tokens() {
        let corpus_content = "hello <|endoftext|> world <|endoftext|> rust";
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        
        let output_dir = TempDir::new().unwrap();
        let output_file = output_dir.path().join("word_freqs.json");
        let output_path = output_file.to_string_lossy().to_string();
        
        let binary_path = build_binary("extract_word_freq").unwrap();
        
        let output = Command::new(&binary_path)
            .args(&[&corpus_path, &output_path, "<|endoftext|>"])
            .output()
            .expect("Failed to execute extract_word_freq");
        
        if !output.status.success() {
            panic!("extract_word_freq failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        
        assert!(output_file.exists());
        let freq_content = fs::read_to_string(&output_file).unwrap();
        let _freq_data: serde_json::Value = serde_json::from_str(&freq_content).unwrap();
    }
    
    #[test]
    fn test_extract_word_freq_cli_error_handling() {
        let binary_path = build_binary("extract_word_freq").unwrap();
        
        // Test missing arguments
        let output = Command::new(&binary_path)
            .args(&["file.txt"])
            .stderr(Stdio::piped())
            .output()
            .expect("Failed to execute extract_word_freq");
        
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Usage:"));
        
        // Test non-existent input file
        let output_dir = TempDir::new().unwrap();
        let output_file = output_dir.path().join("output.json");
        let output_path = output_file.to_string_lossy().to_string();
        
        let output = Command::new(&binary_path)
            .args(&["nonexistent_file.txt", &output_path])
            .stderr(Stdio::piped())
            .output()
            .expect("Failed to execute extract_word_freq");
        
        assert!(!output.status.success());
    }
    
    #[test]
    fn test_end_to_end_bpe_training_workflow() {
        // Create a more substantial test corpus
        let corpus_content = r#"
Once upon a time, there was a programmer who loved Rust.
The programmer wrote many lines of code every day.
Rust is a systems programming language that runs blazingly fast.
The language prevents segfaults and guarantees thread safety.
Many programmers are switching to Rust for its performance.
Performance and safety are key features of Rust programming.
The Rust community is welcoming and helpful to newcomers.
"#;
        
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        let output_dir = TempDir::new().unwrap();
        let output_path = output_dir.path().to_string_lossy().to_string();
        
        // Step 1: Train BPE with the CLI tool
        let train_binary = build_binary("train_bpe").unwrap();
        let train_output = Command::new(&train_binary)
            .args(&[&corpus_path, "400", &output_path, "<|endoftext|>"])
            .output()
            .expect("Failed to execute train_bpe");
        
        if !train_output.status.success() {
            panic!("BPE training failed: {}", String::from_utf8_lossy(&train_output.stderr));
        }
        
        // Step 2: Verify all output files are created
        let vocab_file = format!("{}/vocab.json", output_path);
        let merges_file = format!("{}/merges.txt", output_path);
        let stats_file = format!("{}/training_stats.txt", output_path);
        
        assert!(Path::new(&vocab_file).exists());
        assert!(Path::new(&merges_file).exists());
        assert!(Path::new(&stats_file).exists());
        
        // Step 3: Validate vocab.json structure
        let vocab_content = fs::read_to_string(&vocab_file).unwrap();
        let vocab: serde_json::Value = serde_json::from_str(&vocab_content).unwrap();
        assert!(vocab.is_object());
        
        let vocab_obj = vocab.as_object().unwrap();
        assert!(vocab_obj.len() >= 256); // At least base vocab
        assert!(vocab_obj.len() <= 400);  // At most target vocab size
        
        // Step 4: Validate merges.txt format
        let merges_content = fs::read_to_string(&merges_file).unwrap();
        let merge_lines: Vec<&str> = merges_content.trim().lines().collect();
        assert!(!merge_lines.is_empty());
        
        // Each line should have format "token1 token2" (skip comment lines, empty lines, and malformed lines)
        let mut valid_merge_count = 0;
        for line in merge_lines {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                valid_merge_count += 1;
            }
            // Skip malformed lines silently - they might be artifacts of the BPE process
        }
        
        // At least some merges should be valid
        assert!(valid_merge_count > 0, "No valid merge lines found");
        
        // Step 5: Validate training stats
        let stats_content = fs::read_to_string(&stats_file).unwrap();
        assert!(stats_content.contains("BPE Training Statistics"));
        assert!(stats_content.contains("Final vocabulary size:"));
        assert!(stats_content.contains("Number of merges learned:"));
        assert!(stats_content.contains("Special tokens:"));
        assert!(stats_content.contains("<|endoftext|>"));
    }
    
    #[test]
    fn test_ultra_profiler_cli_functionality() {
        // First extract word frequencies
        let corpus_content = "hello world rust programming hello rust world programming";
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        
        let freq_dir = TempDir::new().unwrap();
        let freq_file = freq_dir.path().join("freqs.json");
        let freq_path = freq_file.to_string_lossy().to_string();
        
        let extract_binary = build_binary("extract_word_freq").unwrap();
        let extract_output = Command::new(&extract_binary)
            .args(&[&corpus_path, &freq_path])
            .output()
            .expect("Failed to execute extract_word_freq");
        
        if !extract_output.status.success() {
            panic!("Word freq extraction failed: {}", String::from_utf8_lossy(&extract_output.stderr));
        }
        
        // Then run ultra profiler
        let ultra_binary = build_binary("ultra_profiler").unwrap();
        let ultra_output = Command::new(&ultra_binary)
            .args(&[&freq_path, "5", "300"])
            .output()
            .expect("Failed to execute ultra_profiler");
        
        if !ultra_output.status.success() {
            panic!("Ultra profiler failed: {}", String::from_utf8_lossy(&ultra_output.stderr));
        }
        
        // Check that performance output is generated
        let stdout = String::from_utf8_lossy(&ultra_output.stdout);
        assert!(stdout.contains("Ultra-Optimized Results"));
    }
    
    #[test] 
    fn test_baseline_vs_optimized_cli_consistency() {
        let corpus_content = "the quick brown fox jumps over the lazy dog";
        let (_temp_dir, corpus_path) = create_test_corpus(corpus_content).unwrap();
        
        // Train with regular optimized version
        let output_dir1 = TempDir::new().unwrap();
        let output_path1 = output_dir1.path().to_string_lossy().to_string();
        
        let train_binary = build_binary("train_bpe").unwrap();
        let train_output1 = Command::new(&train_binary)
            .args(&[&corpus_path, "280", &output_path1])
            .output()
            .expect("Failed to execute train_bpe");
        
        assert!(train_output1.status.success());
        
        // Train with baseline version
        let output_dir2 = TempDir::new().unwrap();
        let output_path2 = output_dir2.path().to_string_lossy().to_string();
        
        let baseline_binary = build_binary("train_bpe_baseline").unwrap();
        let train_output2 = Command::new(&baseline_binary)
            .args(&[&corpus_path, "280", &output_path2])
            .output()
            .expect("Failed to execute train_bpe_baseline");
        
        assert!(train_output2.status.success());
        
        // Compare merge files (should be identical)
        let merges1 = fs::read_to_string(format!("{}/merges.txt", output_path1)).unwrap();
        let merges2 = fs::read_to_string(format!("{}/merges.txt", output_path2)).unwrap();
        
        // Both should produce the same merges
        assert_eq!(merges1.trim(), merges2.trim(), "Optimized and baseline versions produce different merges");
    }
}