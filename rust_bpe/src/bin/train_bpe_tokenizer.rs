use clap::{Parser, ValueEnum};
use rustc_hash::FxHashMap;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// Import the optimized BPE training functions from lib.rs
use rust_bpe::{extract_word_frequencies_with_stats, train_bpe_from_word_freqs};

#[derive(Clone, ValueEnum, Debug)]
enum Mode {
    Full,
    ExtractFreqs,
    TrainFromFreqs,
}

/// Train a BPE (Byte Pair Encoding) tokenizer from text data
#[derive(Parser)]
#[command(name = "train_bpe_tokenizer")]
#[command(about = "Train a BPE (Byte Pair Encoding) tokenizer from text data")]
#[command(long_about = None)]
struct Args {
    /// Path to input text file for training (not required for train-from-freqs mode)
    #[arg(short = 'i', long = "input")]
    input: Option<String>,

    /// Output directory for tokenizer files
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Target vocabulary size (includes base bytes and special tokens)
    #[arg(short = 'v', long = "vocab-size")]
    vocab_size: Option<usize>,

    /// Training mode
    #[arg(short = 'm', long = "mode", value_enum, default_value = "full")]
    mode: Mode,

    /// Path to word frequencies JSON file (required for train-from-freqs mode)
    #[arg(long = "word-freqs-file")]
    word_freqs_file: Option<String>,

    /// Special tokens to add to vocabulary
    #[arg(short = 's', long = "special-tokens", default_values = &["<|endoftext|>"])]
    special_tokens: Vec<String>,

    /// Optional test text to encode/decode after training
    #[arg(long = "test-text")]
    test_text: Option<String>,

    /// Enable verbose output during training
    #[arg(long = "verbose")]
    verbose: bool,
}

/// Save tokenizer vocabulary and merges to files
fn save_tokenizer_files(
    vocab: &HashMap<u16, Vec<u8>>,
    merges: &[(Vec<u8>, Vec<u8>)],
    output_dir: &str,
    special_tokens: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let output_path = Path::new(output_dir);
    std::fs::create_dir_all(output_path)?;

    println!("Saving tokenizer files to {}", output_dir);

    // Save vocabulary as JSON (convert bytes to strings)
    let vocab_path = output_path.join("vocab.json");
    let mut vocab_for_json = std::collections::BTreeMap::new();

    for (&k, v) in vocab {
        // Try to decode as UTF-8 first
        let token_str = match std::str::from_utf8(v) {
            Ok(s) => s.to_string(),
            Err(_) => {
                // If that fails, decode as latin-1 (preserves all byte values)
                v.iter().map(|&b| b as char).collect()
            }
        };
        vocab_for_json.insert(token_str, k);
    }

    let vocab_json = serde_json::to_string_pretty(&vocab_for_json)?;
    let mut vocab_file = File::create(&vocab_path)?;
    vocab_file.write_all(vocab_json.as_bytes())?;
    println!("✓ Saved vocabulary to {:?}", vocab_path);

    // Save merges as text file
    let merges_path = output_path.join("merges.txt");
    let mut merges_file = File::create(&merges_path)?;
    writeln!(merges_file, "#version: 0.2")?;

    for (token1, token2) in merges {
        // Convert bytes back to strings for saving
        let token1_str = match std::str::from_utf8(token1) {
            Ok(s) => s.to_string(),
            Err(_) => token1.iter().map(|&b| b as char).collect(),
        };
        let token2_str = match std::str::from_utf8(token2) {
            Ok(s) => s.to_string(),
            Err(_) => token2.iter().map(|&b| b as char).collect(),
        };
        writeln!(merges_file, "{} {}", token1_str, token2_str)?;
    }
    println!("✓ Saved merges to {:?}", merges_path);

    // Save training statistics
    let stats_path = output_path.join("training_stats.txt");
    let mut stats_file = File::create(&stats_path)?;
    writeln!(stats_file, "Vocabulary size: {}", vocab.len())?;
    writeln!(stats_file, "Number of merges: {}", merges.len())?;
    writeln!(stats_file, "Special tokens: {:?}", special_tokens)?;

    // Count token types
    let base_tokens = vocab.keys().filter(|&&k| k < 256).count();
    let special_count = special_tokens.len();
    let learned_merges = vocab.len() - base_tokens - special_count;

    writeln!(stats_file, "Base tokens (0-255): {}", base_tokens)?;
    writeln!(stats_file, "Special tokens: {}", special_count)?;
    writeln!(stats_file, "Learned merges: {}", learned_merges)?;

    println!("✓ Saved training stats to {:?}", stats_path);

    Ok(())
}

/// Save word frequencies to a JSON file, ordered by frequency (descending)
fn save_word_freqs(
    word_freqs: &FxHashMap<Vec<u16>, u64>,
    output_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = Path::new(output_file);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Convert to vector and sort by frequency (descending), then by word for determinism
    let mut sorted_freqs: Vec<(&Vec<u16>, &u64)> = word_freqs.iter().collect();
    sorted_freqs.sort_by(|a, b| {
        // First by frequency (descending), then by word bytes (ascending) for determinism
        b.1.cmp(a.1).then_with(|| a.0.cmp(b.0))
    });

    // Convert to JSON-serializable format
    let mut json_data = Vec::new();
    for (word_tokens, freq) in sorted_freqs {
        // Convert token IDs to displayable string (assuming they map to bytes)
        let word_str = word_tokens
            .iter()
            .map(|&token_id| {
                if token_id < 256 {
                    let b = token_id as u8;
                    if b < 128 {
                        (b as char).to_string()
                    } else {
                        format!("\\x{:02x}", b)
                    }
                } else {
                    format!("[{}]", token_id) // Show token ID for non-byte tokens
                }
            })
            .collect::<String>();

        json_data.push(serde_json::json!({
            "word": word_tokens,
            "word_display": word_str,
            "frequency": freq
        }));
    }

    let json_str = serde_json::to_string_pretty(&json_data)?;
    let mut file = File::create(output_file)?;
    file.write_all(json_str.as_bytes())?;

    println!(
        "✓ Saved {} word frequencies to {}",
        word_freqs.len(),
        output_file
    );
    Ok(())
}

/// Load word frequencies from a JSON file
fn load_word_freqs(
    input_file: &str,
) -> Result<FxHashMap<Vec<u16>, u64>, Box<dyn std::error::Error>> {
    let file = File::open(input_file)?;
    let json_data: Vec<serde_json::Value> = serde_json::from_reader(file)?;

    let mut word_freqs = FxHashMap::default();
    for item in json_data {
        let word_vec: Vec<u16> = item["word"]
            .as_array()
            .ok_or("Invalid word format in JSON")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or("Invalid token ID value")
                    .map(|id| id as u16)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let frequency = item["frequency"]
            .as_u64()
            .ok_or("Invalid frequency value")?;

        word_freqs.insert(word_vec, frequency);
    }

    println!(
        "✅ Loaded {} word frequencies from {}",
        word_freqs.len(),
        input_file
    );
    Ok(word_freqs)
}

/// Test the trained tokenizer with sample text
fn test_tokenizer(
    vocab: &HashMap<u16, Vec<u8>>,
    _merges: &[(Vec<u8>, Vec<u8>)],
    _special_tokens: &[String],
    test_text: &str,
) {
    println!("\n=== Testing Tokenizer ===");
    println!("Test text: \"{}\"", test_text);

    // For this demo, just show that we would create a tokenizer
    // In a full implementation, you'd create a BPETokenizer instance and test encode/decode
    println!(
        "✅ Tokenizer ready for testing with {} vocab items",
        vocab.len()
    );
    println!("   (Full encode/decode testing would be implemented here)");
}

fn main() {
    let args = Args::parse();

    println!("🚀 Starting BPE tokenizer training");
    println!("==================================================");

    // Validate arguments based on mode
    match args.mode {
        Mode::TrainFromFreqs => {
            if args.word_freqs_file.is_none() {
                eprintln!("❌ Error: --word-freqs-file is required for 'train-from-freqs' mode");
                std::process::exit(1);
            }
            if let Some(ref freqs_file) = args.word_freqs_file {
                if !Path::new(freqs_file).exists() {
                    eprintln!("❌ Error: Word frequencies file '{}' not found", freqs_file);
                    std::process::exit(1);
                }
            }
            if args.vocab_size.is_none() {
                eprintln!("❌ Error: --vocab-size is required for 'train-from-freqs' mode");
                std::process::exit(1);
            }
        }
        _ => {
            if args.input.is_none() {
                eprintln!("❌ Error: --input is required for 'full' and 'extract-freqs' modes");
                std::process::exit(1);
            }
            if let Some(ref input_file) = args.input {
                if !Path::new(input_file).exists() {
                    eprintln!("❌ Error: Input file '{}' not found", input_file);
                    std::process::exit(1);
                }
            }
        }
    }

    if let Some(vocab_size) = args.vocab_size {
        if vocab_size <= 256 {
            eprintln!(
                "❌ Error: vocab_size ({}) must be > 256 (base byte vocabulary)",
                vocab_size
            );
            std::process::exit(1);
        }
    }

    // Show configuration
    println!("🔧 Mode: {:?}", args.mode);
    if let Some(ref input_file) = args.input {
        println!("📁 Input file: {}", input_file);
        if let Ok(metadata) = std::fs::metadata(input_file) {
            let file_size = metadata.len();
            let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
            println!(
                "📏 Input file size: {} bytes ({:.1} MB)",
                file_size, file_size_mb
            );
        }
    }
    if !matches!(args.mode, Mode::ExtractFreqs) {
        println!("📁 Output directory: {}", args.output);
        if let Some(vocab_size) = args.vocab_size {
            println!("📊 Target vocabulary size: {}", vocab_size);
        }
    }
    if let Some(ref freqs_file) = args.word_freqs_file {
        println!("📋 Word frequencies file: {}", freqs_file);
    }
    println!("🏷️  Special tokens: {:?}", args.special_tokens);

    let start_time = Instant::now();

    match args.mode {
        Mode::ExtractFreqs => {
            // Only extract word frequencies
            println!("\n📊 Extracting word frequencies...");
            let input_file = args.input.as_ref().unwrap();
            let (word_freqs, _chunk_count) =
                match extract_word_frequencies_with_stats(input_file, &args.special_tokens) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("❌ Error extracting word frequencies: {}", e);
                        std::process::exit(1);
                    }
                };

            // Save word frequencies
            let freqs_output = format!("{}/word_freqs.json", args.output);
            if let Err(e) = save_word_freqs(&word_freqs, &freqs_output) {
                eprintln!("❌ Error saving word frequencies: {}", e);
                std::process::exit(1);
            }

            let elapsed = start_time.elapsed();
            println!(
                "✅ Word frequency extraction completed in {:.1}s",
                elapsed.as_secs_f64()
            );
            println!("📊 Extracted {} unique word types", word_freqs.len());
            println!("📁 Word frequencies saved to: {}", freqs_output);
        }

        Mode::TrainFromFreqs => {
            // Load word frequencies and train
            let freqs_file = args.word_freqs_file.as_ref().unwrap();
            println!("\n📋 Loading word frequencies from {}...", freqs_file);
            let word_freqs = match load_word_freqs(freqs_file) {
                Ok(freqs) => freqs,
                Err(e) => {
                    eprintln!("❌ Error loading word frequencies: {}", e);
                    std::process::exit(1);
                }
            };

            println!("\n⏳ Training BPE from loaded frequencies...");
            let vocab_size = args.vocab_size.unwrap();
            let (vocab, merges) =
                match train_bpe_from_word_freqs(word_freqs, vocab_size, &args.special_tokens) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("❌ Error during BPE training: {}", e);
                        std::process::exit(1);
                    }
                };

            let training_time = start_time.elapsed();
            println!(
                "✅ Training completed in {:.1}s",
                training_time.as_secs_f64()
            );
            show_results(&vocab, &merges, &args.special_tokens);
            save_and_test_tokenizer(&vocab, &merges, &args);
        }

        Mode::Full => {
            // Full training pipeline
            println!("\n⏳ Training BPE tokenizer...");

            if args.verbose {
                println!("Training in verbose mode...");
            }

            let input_file = args.input.as_ref().unwrap();
            let vocab_size = args.vocab_size.unwrap();

            // Step 1: Extract word frequencies
            let (word_freqs, _chunk_count) =
                match extract_word_frequencies_with_stats(input_file, &args.special_tokens) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("❌ Error extracting word frequencies: {}", e);
                        std::process::exit(1);
                    }
                };

            // Step 2: Train BPE
            let (vocab, merges) =
                match train_bpe_from_word_freqs(word_freqs, vocab_size, &args.special_tokens) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("❌ Error during BPE training: {}", e);
                        std::process::exit(1);
                    }
                };

            let training_time = start_time.elapsed();
            println!(
                "✅ Training completed in {:.1}s",
                training_time.as_secs_f64()
            );
            show_results(&vocab, &merges, &args.special_tokens);
            save_and_test_tokenizer(&vocab, &merges, &args);
        }
    }
}

fn show_results(
    vocab: &HashMap<u16, Vec<u8>>,
    merges: &[(Vec<u8>, Vec<u8>)],
    special_tokens: &[String],
) {
    println!("📈 Final vocabulary size: {}", vocab.len());
    println!("🔗 Number of merges learned: {}", merges.len());

    // Calculate effective compression
    let base_tokens = vocab.keys().filter(|&&k| k < 256).count();
    let special_count = special_tokens.len();
    let learned_merges = vocab.len() - base_tokens - special_count;

    println!("📊 Token breakdown:");
    println!("   • Base bytes (0-255): {}", base_tokens);
    println!("   • Special tokens: {}", special_count);
    println!("   • Learned merge tokens: {}", learned_merges);
}

fn save_and_test_tokenizer(
    vocab: &HashMap<u16, Vec<u8>>,
    merges: &[(Vec<u8>, Vec<u8>)],
    args: &Args,
) {
    // Save tokenizer files
    if let Err(e) = save_tokenizer_files(vocab, merges, &args.output, &args.special_tokens) {
        eprintln!("❌ Error saving files: {}", e);
        std::process::exit(1);
    }

    // Test tokenizer if test text provided
    if let Some(test_text) = &args.test_text {
        test_tokenizer(vocab, merges, &args.special_tokens, test_text);
    } else {
        // Default test with a simple sentence
        let default_test = "Hello world! This is a test of the BPE tokenizer.";
        test_tokenizer(vocab, merges, &args.special_tokens, default_test);
    }

    println!("\n🎉 BPE tokenizer training completed successfully!");
    println!("📁 Tokenizer files saved to: {}", args.output);
    println!("\nTo use this tokenizer:");
    println!(
        "  ./bpe_tokenize --vocab {}/vocab.json --merges {}/merges.txt --input your_text.txt",
        args.output, args.output
    );
}
