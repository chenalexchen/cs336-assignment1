#[cfg(feature = "python")]
use pyo3::prelude::*;
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
    #[new]
    fn new(
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
}

/// Pre-tokenize text into words using GPT-2 style regex
fn pre_tokenize(text: &str, special_tokens: &[String]) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    
    // Split on special tokens first
    let mut parts = vec![text.to_string()];
    
    for special_token in special_tokens {
        let mut new_parts = Vec::new();
        for part in parts {
            if part == *special_token {
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
    
    Ok((word_freqs, lines.len() / chunk_size))
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