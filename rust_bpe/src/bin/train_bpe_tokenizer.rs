use clap::Parser;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

// Import the optimized BPE training functions from lib.rs
use rust_bpe::{extract_word_frequencies_with_stats, train_bpe_from_word_freqs};

/// Train a BPE (Byte Pair Encoding) tokenizer from text data
#[derive(Parser)]
#[command(name = "train_bpe_tokenizer")]
#[command(about = "Train a BPE (Byte Pair Encoding) tokenizer from text data")]
#[command(long_about = None)]
struct Args {
    /// Path to input text file for training
    #[arg(short = 'i', long = "input")]
    input: String,

    /// Output directory for tokenizer files
    #[arg(short = 'o', long = "output")]
    output: String,

    /// Target vocabulary size (includes base bytes and special tokens)
    #[arg(short = 'v', long = "vocab-size")]
    vocab_size: usize,

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
    println!("✅ Tokenizer ready for testing with {} vocab items", vocab.len());
    println!("   (Full encode/decode testing would be implemented here)");
}

fn main() {
    let args = Args::parse();

    println!("🚀 Starting BPE tokenizer training");
    println!("==================================================");

    // Validate arguments
    if args.vocab_size <= 256 {
        eprintln!("❌ Error: vocab_size ({}) must be > 256 (base byte vocabulary)", args.vocab_size);
        std::process::exit(1);
    }

    if !Path::new(&args.input).exists() {
        eprintln!("❌ Error: Input file '{}' not found", args.input);
        std::process::exit(1);
    }

    // Show configuration
    println!("📁 Input file: {}", args.input);
    println!("📁 Output directory: {}", args.output);
    println!("📊 Target vocabulary size: {}", args.vocab_size);
    println!("🏷️  Special tokens: {:?}", args.special_tokens);

    // Check input file size
    if let Ok(metadata) = std::fs::metadata(&args.input) {
        let file_size = metadata.len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
        println!("📏 Input file size: {} bytes ({:.1} MB)", file_size, file_size_mb);
    }

    // Start training
    println!("\n⏳ Training BPE tokenizer...");
    let start_time = Instant::now();

    if args.verbose {
        println!("Training in verbose mode...");
    }

    // Step 1: Extract word frequencies
    let (word_freqs, _chunk_count) = match extract_word_frequencies_with_stats(&args.input, &args.special_tokens) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ Error extracting word frequencies: {}", e);
            std::process::exit(1);
        }
    };

    // Step 2: Train BPE
    let (vocab, merges) = match train_bpe_from_word_freqs(word_freqs, args.vocab_size, &args.special_tokens) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ Error during BPE training: {}", e);
            std::process::exit(1);
        }
    };

    let training_time = start_time.elapsed();

    println!("✅ Training completed in {:.1}s", training_time.as_secs_f64());
    println!("📈 Final vocabulary size: {}", vocab.len());
    println!("🔗 Number of merges learned: {}", merges.len());

    // Calculate effective compression
    let base_tokens = vocab.keys().filter(|&&k| k < 256).count();
    let special_count = args.special_tokens.len();
    let learned_merges = vocab.len() - base_tokens - special_count;

    println!("📊 Token breakdown:");
    println!("   • Base bytes (0-255): {}", base_tokens);
    println!("   • Special tokens: {}", special_count);
    println!("   • Learned merge tokens: {}", learned_merges);

    // Save tokenizer files
    if let Err(e) = save_tokenizer_files(&vocab, &merges, &args.output, &args.special_tokens) {
        eprintln!("❌ Error saving files: {}", e);
        std::process::exit(1);
    }

    // Test tokenizer if test text provided
    if let Some(test_text) = &args.test_text {
        test_tokenizer(&vocab, &merges, &args.special_tokens, test_text);
    } else {
        // Default test with a simple sentence
        let default_test = "Hello world! This is a test of the BPE tokenizer.";
        test_tokenizer(&vocab, &merges, &args.special_tokens, default_test);
    }

    println!("\n🎉 BPE tokenizer training completed successfully!");
    println!("📁 Tokenizer files saved to: {}", args.output);
    println!("\nTo use this tokenizer:");
    println!("  ./bpe_tokenize --vocab {}/vocab.json --merges {}/merges.txt --input your_text.txt", 
             args.output, args.output);
}