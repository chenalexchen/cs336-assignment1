use std::time::Instant;
use std::path::Path;
use std::fs::File;
use std::io::Write;

// Import the word frequency extraction function from lib.rs
use rust_bpe::extract_word_frequencies_with_stats;

/// Command-line arguments structure
#[derive(Debug)]
struct Args {
    input_file: String,
    output_file: String,
    special_tokens: Vec<String>,
}

/// Parse command-line arguments
fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <input_file> <output_file> [special_tokens...]", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  input_file      Path to the input text file");
        eprintln!("  output_file     Path to save word frequencies (JSON format)");
        eprintln!("  special_tokens  Optional special tokens (default: <|endoftext|>)");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} data/owt_train.txt word_freqs/owt_train_freqs.json", args[0]);
        std::process::exit(1);
    }
    
    let input_file = args[1].clone();
    let output_file = args[2].clone();
    
    // Parse special tokens (default to <|endoftext|> if none provided)
    let special_tokens = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["<|endoftext|>".to_string()]
    };
    
    // Validate input file exists
    if !Path::new(&input_file).exists() {
        return Err(format!("Input file does not exist: {}", input_file).into());
    }
    
    Ok(Args {
        input_file,
        output_file,
        special_tokens,
    })
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
    
    println!("📊 Word Frequency Extractor");
    println!("============================");
    println!("Input file: {}", args.input_file);
    println!("Output file: {}", args.output_file);
    println!("Special tokens: {:?}", args.special_tokens);
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
    
    println!("⏱️  Starting word frequency extraction...");
    let start_time = Instant::now();
    
    // Extract word frequencies
    let (word_freqs, chunk_count) = match extract_word_frequencies_with_stats(&args.input_file, &args.special_tokens) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("❌ Failed to extract word frequencies: {}", e);
            std::process::exit(1);
        }
    };
    
    let extract_time = start_time.elapsed();
    println!("✓ Word frequency extraction completed in {:.3}s", extract_time.as_secs_f64());
    println!("  - Chunks processed: {}", chunk_count);
    println!("  - Unique word types: {}", word_freqs.len());
    
    if let Ok(metadata) = std::fs::metadata(&args.input_file) {
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("  - Throughput: {:.1} MB/s", file_size_mb / extract_time.as_secs_f64());
    }
    
    // Create output directory if needed
    if let Some(parent) = Path::new(&args.output_file).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("❌ Failed to create output directory: {}", e);
            std::process::exit(1);
        }
    }
    
    // Serialize word frequencies to JSON
    println!("💾 Saving word frequencies to {}...", args.output_file);
    let save_start = Instant::now();
    
    // Convert to a serializable format
    let word_freqs_serializable: std::collections::BTreeMap<String, u64> = word_freqs
        .into_iter()
        .map(|(word_tokens, freq)| {
            // Convert Vec<i32> to a string representation for JSON serialization
            let word_str = word_tokens
                .iter()
                .map(|&token| format!("{}", token))
                .collect::<Vec<String>>()
                .join(",");
            (word_str, freq)
        })
        .collect();
    
    let json_str = match serde_json::to_string(&word_freqs_serializable) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("❌ Failed to serialize word frequencies: {}", e);
            std::process::exit(1);
        }
    };
    
    let mut output_file = match File::create(&args.output_file) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("❌ Failed to create output file: {}", e);
            std::process::exit(1);
        }
    };
    
    if let Err(e) = output_file.write_all(json_str.as_bytes()) {
        eprintln!("❌ Failed to write word frequencies: {}", e);
        std::process::exit(1);
    }
    
    let save_time = save_start.elapsed();
    println!("✓ Word frequencies saved in {:.3}s", save_time.as_secs_f64());
    
    // Also save metadata
    let metadata_file = args.output_file.replace(".json", "_metadata.json");
    let metadata = serde_json::json!({
        "input_file": args.input_file,
        "special_tokens": args.special_tokens,
        "unique_word_types": word_freqs_serializable.len(),
        "extraction_time_seconds": extract_time.as_secs_f64(),
        "chunks_processed": chunk_count,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });
    
    if let Ok(mut meta_file) = File::create(&metadata_file) {
        if let Ok(meta_json) = serde_json::to_string_pretty(&metadata) {
            let _ = meta_file.write_all(meta_json.as_bytes());
            println!("✓ Metadata saved to: {}", metadata_file);
        }
    }
    
    println!();
    println!("🎉 Word frequency extraction completed successfully!");
    println!("⏱️  Total time: {:.3}s", start_time.elapsed().as_secs_f64());
    println!("📂 Output files:");
    println!("   - {}: Word frequencies", args.output_file);
    println!("   - {}: Extraction metadata", metadata_file);
}