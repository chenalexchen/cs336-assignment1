use std::time::Instant;
use std::collections::HashMap;
use std::path::Path;

// Import the baseline BPE training functions from lib.rs
use rust_bpe::{train_bpe_from_word_freqs_baseline, extract_word_frequencies_with_stats};

/// Command-line arguments structure
#[derive(Debug)]
struct Args {
    input_file: String,
    vocab_size: usize,
    output_dir: String,
    special_tokens: Vec<String>,
}

/// Parse command-line arguments
fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 4 {
        eprintln!("Usage: {} <input_file> <vocab_size> <output_dir> [special_tokens...]", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  input_file    Path to the input text file for training");
        eprintln!("  vocab_size    Target vocabulary size (e.g., 32000)");
        eprintln!("  output_dir    Directory to save the trained tokenizer files");
        eprintln!("  special_tokens  Optional special tokens (default: <|endoftext|>)");
        std::process::exit(1);
    }
    
    let input_file = args[1].clone();
    let vocab_size = args[2].parse::<usize>()
        .map_err(|_| "Invalid vocab_size: must be a positive integer")?;
    let output_dir = args[3].clone();
    
    // Parse special tokens (default to <|endoftext|> if none provided)
    let special_tokens = if args.len() > 4 {
        args[4..].to_vec()
    } else {
        vec!["<|endoftext|>".to_string()]
    };
    
    // Validate input file exists
    if !Path::new(&input_file).exists() {
        return Err(format!("Input file does not exist: {}", input_file).into());
    }
    
    // Validate vocab_size
    if vocab_size < 256 {
        return Err("vocab_size must be at least 256 (for base byte tokens)".into());
    }
    
    Ok(Args {
        input_file,
        vocab_size,
        output_dir,
        special_tokens,
    })
}

/// Save tokenizer outputs to directory
fn save_tokenizer_outputs(
    vocab: &HashMap<i32, Vec<u8>>,
    merges: &[(Vec<u8>, Vec<u8>)],
    output_dir: &str,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;
    
    // Create output directory
    std::fs::create_dir_all(output_dir)?;
    
    // Save vocabulary as JSON
    let vocab_path = format!("{}/vocab.json", output_dir);
    let mut vocab_json = std::collections::BTreeMap::new();
    for (&id, bytes) in vocab {
        let token_str = String::from_utf8_lossy(bytes);
        vocab_json.insert(token_str.to_string(), id);
    }
    
    let vocab_json_str = serde_json::to_string_pretty(&vocab_json)?;
    let mut vocab_file = File::create(&vocab_path)?;
    vocab_file.write_all(vocab_json_str.as_bytes())?;
    println!("✓ Saved vocabulary to: {}", vocab_path);
    
    // Save merges as text file
    let merges_path = format!("{}/merges.txt", output_dir);
    let mut merges_file = File::create(&merges_path)?;
    writeln!(merges_file, "#version: 0.2")?;
    for (token1, token2) in merges {
        let token1_str = String::from_utf8_lossy(token1);
        let token2_str = String::from_utf8_lossy(token2);
        writeln!(merges_file, "{} {}", token1_str, token2_str)?;
    }
    println!("✓ Saved merges to: {}", merges_path);
    
    // Save training configuration and statistics
    let stats_path = format!("{}/training_stats.txt", output_dir);
    let mut stats_file = File::create(&stats_path)?;
    writeln!(stats_file, "BPE Training Statistics (Baseline)")?;
    writeln!(stats_file, "====================================")?;
    writeln!(stats_file)?;
    writeln!(stats_file, "Training Configuration:")?;
    writeln!(stats_file, "- Input file: {}", args.input_file)?;
    writeln!(stats_file, "- Target vocab size: {}", args.vocab_size)?;
    writeln!(stats_file, "- Special tokens: {:?}", args.special_tokens)?;
    writeln!(stats_file, "- Output directory: {}", args.output_dir)?;
    writeln!(stats_file, "- Algorithm: Baseline (no optimizations)")?;
    writeln!(stats_file)?;
    writeln!(stats_file, "Results:")?;
    writeln!(stats_file, "- Final vocabulary size: {}", vocab.len())?;
    writeln!(stats_file, "- Number of merges learned: {}", merges.len())?;
    
    println!("✓ Saved training statistics to: {}", stats_path);
    
    Ok(())
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
    
    println!("📊 BPE Tokenizer Training (BASELINE)");
    println!("=====================================");
    println!("Input file: {}", args.input_file);
    println!("Target vocab size: {}", args.vocab_size);
    println!("Special tokens: {:?}", args.special_tokens);
    println!("Output directory: {}", args.output_dir);
    println!("CPU cores available: {}", rayon::current_num_threads());
    println!();
    
    // Display file size information
    if let Ok(metadata) = std::fs::metadata(&args.input_file) {
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        let file_size_gb = file_size_mb / 1024.0;
        if file_size_gb >= 1.0 {
            println!("📁 Input file size: {:.1} GB", file_size_gb);
        } else {
            println!("📁 Input file size: {:.1} MB", file_size_mb);
        }
    }
    
    println!("⏱️  Starting BPE training...");
    let start_time = Instant::now();
    
    // Step 1: Extract word frequencies
    println!("📊 Extracting word frequencies...");
    let extract_start = Instant::now();
    
    let (word_freqs, chunk_count) = match extract_word_frequencies_with_stats(&args.input_file, &args.special_tokens) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ Failed to extract word frequencies: {}", e);
            std::process::exit(1);
        }
    };
    
    let extract_time = extract_start.elapsed();
    println!("✓ File processing completed in {:.3}s", extract_time.as_secs_f64());
    println!("  - Chunks processed: {}", chunk_count);
    println!("  - Unique word types: {}", word_freqs.len());
    println!();
    
    // Step 2: Train BPE merges using BASELINE algorithm
    println!("🔧 Training BPE merges (baseline)...");
    let bpe_start = Instant::now();
    
    let (vocab, merges) = match train_bpe_from_word_freqs_baseline(word_freqs, args.vocab_size, &args.special_tokens) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ BPE training failed: {}", e);
            std::process::exit(1);
        }
    };
    
    let bpe_time = bpe_start.elapsed();
    let total_time = start_time.elapsed();
    
    println!("✓ BPE training completed in {:.3}s", bpe_time.as_secs_f64());
    println!("  - Final vocabulary size: {}", vocab.len());
    println!("  - Merges learned: {}", merges.len());
    println!();
    
    // Step 3: Save outputs
    println!("💾 Saving tokenizer files...");
    if let Err(e) = save_tokenizer_outputs(&vocab, &merges, &args.output_dir, &args) {
        eprintln!("❌ Failed to save outputs: {}", e);
        std::process::exit(1);
    }
    
    // Final summary
    println!();
    println!("🎉 Baseline training completed successfully!");
    println!("⏱️  Total time: {:.3}s", total_time.as_secs_f64());
    
    if let Ok(metadata) = std::fs::metadata(&args.input_file) {
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("🚀 Throughput: {:.1} MB/s", file_size_mb / total_time.as_secs_f64());
    }
    
    println!("📂 Output files saved to: {}", args.output_dir);
    println!("   - vocab.json: Token to ID mapping");
    println!("   - merges.txt: BPE merge rules");
    println!("   - training_stats.txt: Training configuration and statistics");
}