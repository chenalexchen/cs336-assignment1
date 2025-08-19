use std::time::Instant;
use std::path::Path;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use rustc_hash::FxHashMap;

// Import the BPE training functions from lib.rs
use rust_bpe::{train_bpe_from_word_freqs, train_bpe_from_word_freqs_baseline};

/// Command-line arguments structure
#[derive(Debug)]
struct Args {
    word_freq_file: String,
    num_merges: usize,
    vocab_size: usize,
    special_tokens: Vec<String>,
    use_baseline: bool,
}

/// Parse command-line arguments
fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 4 {
        eprintln!("Usage: {} <word_freq_file> <num_merges> <vocab_size> [--baseline] [special_tokens...]", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  word_freq_file  Path to word frequencies JSON file");
        eprintln!("  num_merges      Number of merge iterations to perform");
        eprintln!("  vocab_size      Target vocabulary size");
        eprintln!("  --baseline      Use baseline algorithm instead of optimized");
        eprintln!("  special_tokens  Optional special tokens (default: <|endoftext|>)");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} word_freqs/owt_train_freqs.json 10 32000", args[0]);
        eprintln!("  {} word_freqs/owt_train_freqs.json 100 32000 --baseline", args[0]);
        std::process::exit(1);
    }
    
    let word_freq_file = args[1].clone();
    let num_merges = args[2].parse::<usize>()
        .map_err(|_| "Invalid num_merges: must be a positive integer")?;
    let vocab_size = args[3].parse::<usize>()
        .map_err(|_| "Invalid vocab_size: must be a positive integer")?;
    
    // Check for --baseline flag
    let mut use_baseline = false;
    let mut special_tokens = Vec::new();
    
    for arg in &args[4..] {
        if arg == "--baseline" {
            use_baseline = true;
        } else {
            special_tokens.push(arg.clone());
        }
    }
    
    // Default special tokens
    if special_tokens.is_empty() {
        special_tokens.push("<|endoftext|>".to_string());
    }
    
    // Validate input file exists
    if !Path::new(&word_freq_file).exists() {
        return Err(format!("Word frequency file does not exist: {}", word_freq_file).into());
    }
    
    Ok(Args {
        word_freq_file,
        num_merges,
        vocab_size,
        special_tokens,
        use_baseline,
    })
}

/// Load word frequencies from JSON file
fn load_word_frequencies(file_path: &str) -> Result<FxHashMap<Vec<u16>, u64>, Box<dyn std::error::Error>> {
    println!("📖 Loading word frequencies from {}...", file_path);
    let load_start = Instant::now();
    
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    let word_freqs_serialized: std::collections::BTreeMap<String, u64> = 
        serde_json::from_str(&contents)?;
    
    // Convert back to FxHashMap<Vec<u16>, u64>
    let mut word_freqs = FxHashMap::default();
    for (word_str, freq) in word_freqs_serialized {
        // Parse comma-separated token IDs back to Vec<u16>
        let word_tokens: Vec<u16> = word_str
            .split(',')
            .map(|s| s.parse::<u16>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse token IDs: {}", e))?;
        
        word_freqs.insert(word_tokens, freq);
    }
    
    let load_time = load_start.elapsed();
    println!("✓ Loaded {} unique word types in {:.3}s", word_freqs.len(), load_time.as_secs_f64());
    
    Ok(word_freqs)
}

/// Modified BPE training function that stops after N merges
fn train_bpe_limited_merges(
    word_freqs: FxHashMap<Vec<u16>, u64>,
    max_merges: usize,
    vocab_size: usize,
    special_tokens: &[String],
    use_baseline: bool,
) -> Result<(HashMap<u16, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>, f64), Box<dyn std::error::Error>> {
    
    if use_baseline {
        println!("🔧 Using BASELINE merge algorithm");
    } else {
        println!("🔧 Using OPTIMIZED merge algorithm");
    }
    
    let merge_start = Instant::now();
    
    // For fair comparison, we'll just run the full training but track time per merge
    let result = if use_baseline {
        train_bpe_from_word_freqs_baseline(word_freqs, vocab_size.min(256 + special_tokens.len() + max_merges), special_tokens)
    } else {
        train_bpe_from_word_freqs(word_freqs, vocab_size.min(256 + special_tokens.len() + max_merges), special_tokens)
    };
    
    let merge_time = merge_start.elapsed();
    let merge_time_secs = merge_time.as_secs_f64();
    
    match result {
        Ok((vocab, merges)) => {
            // Limit the merges to the requested number
            let limited_merges = merges.into_iter().take(max_merges).collect();
            Ok((vocab, limited_merges, merge_time_secs))
        }
        Err(e) => Err(e)
    }
}

fn main() {
    // Parse command-line arguments
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("🚀 BPE Merge Profiler");
    println!("======================");
    println!("Word freq file: {}", args.word_freq_file);
    println!("Number of merges: {}", args.num_merges);
    println!("Vocab size: {}", args.vocab_size);
    println!("Special tokens: {:?}", args.special_tokens);
    println!("Algorithm: {}", if args.use_baseline { "Baseline" } else { "Optimized" });
    println!("CPU cores available: {}", rayon::current_num_threads());
    println!();
    
    let total_start = Instant::now();
    
    // Load word frequencies
    let word_freqs = match load_word_frequencies(&args.word_freq_file) {
        Ok(freqs) => freqs,
        Err(e) => {
            eprintln!("❌ Failed to load word frequencies: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("📊 Starting merge profiling...");
    println!("Target merges: {}", args.num_merges);
    println!("Unique word types: {}", word_freqs.len());
    println!();
    
    // Run BPE merge training
    let (vocab, merges, merge_time) = match train_bpe_limited_merges(
        word_freqs,
        args.num_merges,
        args.vocab_size,
        &args.special_tokens,
        args.use_baseline,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ BPE merge training failed: {}", e);
            std::process::exit(1);
        }
    };
    
    let total_time = total_start.elapsed();
    
    // Performance analysis
    println!();
    println!("📈 Performance Results");
    println!("======================");
    println!("✓ Merges completed: {}", merges.len());
    println!("✓ Final vocabulary size: {}", vocab.len());
    println!("✓ Merge time: {:.3}s", merge_time);
    println!("✓ Time per merge: {:.1}ms", (merge_time * 1000.0) / merges.len() as f64);
    println!("✓ Total time: {:.3}s", total_time.as_secs_f64());
    
    if merges.len() > 0 {
        println!("✓ Merge throughput: {:.1} merges/sec", merges.len() as f64 / merge_time);
    }
    
    // Show some example merges
    println!();
    println!("🔀 Example merges:");
    for (i, (token1, token2)) in merges.iter().take(5).enumerate() {
        let token1_str = String::from_utf8_lossy(token1);
        let token2_str = String::from_utf8_lossy(token2);
        println!("  {}. {:?} + {:?}", i + 1, token1_str, token2_str);
    }
    
    if merges.len() > 5 {
        println!("  ... and {} more", merges.len() - 5);
    }
}