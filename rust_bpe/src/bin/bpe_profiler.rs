use std::time::Instant;
use std::collections::HashMap;

// Import the optimized BPE training functions from lib.rs
use rust_bpe::{train_bpe_from_word_freqs, extract_word_frequencies_with_stats};





/// Save tokenizer outputs to directory
fn save_tokenizer_outputs(
    vocab: &HashMap<i32, Vec<u8>>,
    merges: &[(Vec<u8>, Vec<u8>)],
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;
    
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
    println!("Saved vocabulary to: {}", vocab_path);
    
    // Save merges as text file
    let merges_path = format!("{}/merges.txt", output_dir);
    let mut merges_file = File::create(&merges_path)?;
    writeln!(merges_file, "#version: 0.2")?;
    for (token1, token2) in merges {
        let token1_str = String::from_utf8_lossy(token1);
        let token2_str = String::from_utf8_lossy(token2);
        writeln!(merges_file, "{} {}", token1_str, token2_str)?;
    }
    println!("Saved merges to: {}", merges_path);
    
    // Save summary statistics
    let stats_path = format!("{}/training_stats.txt", output_dir);
    let mut stats_file = File::create(&stats_path)?;
    writeln!(stats_file, "OpenWebText BPE Training Statistics")?;
    writeln!(stats_file, "=====================================\n")?;
    writeln!(stats_file, "Vocabulary size: {}", vocab.len())?;
    writeln!(stats_file, "Number of merges: {}", merges.len())?;
    writeln!(stats_file, "Special tokens: [\"<|endoftext|>\"]")?;
    writeln!(stats_file, "\nBase vocabulary (0-255): 256 byte tokens")?;
    writeln!(stats_file, "Special tokens: 1")?;
    writeln!(stats_file, "Learned merges: {}", merges.len())?;
    
    // Analyze vocabulary composition
    let mut byte_tokens = 0;
    let mut special_tokens = 0;
    let mut merged_tokens = 0;
    
    for (&id, bytes) in vocab {
        if id < 256 {
            byte_tokens += 1;
        } else if bytes == b"<|endoftext|>" {
            special_tokens += 1;
        } else {
            merged_tokens += 1;
        }
    }
    
    writeln!(stats_file, "\nVocabulary composition:")?;
    writeln!(stats_file, "- Byte tokens (0-255): {}", byte_tokens)?;
    writeln!(stats_file, "- Special tokens: {}", special_tokens)?;
    writeln!(stats_file, "- Merged tokens: {}", merged_tokens)?;
    writeln!(stats_file, "- Total: {}", vocab.len())?;
    
    println!("Saved training statistics to: {}", stats_path);
    
    Ok(())
}

fn main() {
    // Train BPE tokenizer on OpenWebText dataset
    let input_path = "/home/chenchen/projects/cs336_2025/cs336-assignment1/data/owt_train.txt";
    let vocab_size = 32000;  // Vocabulary size 32,000 as requested
    let special_tokens = vec!["<|endoftext|>".to_string()];
    let output_dir = "/home/chenchen/projects/cs336_2025/cs336-assignment1/tokenizer_output/open_web_text_32000";
    
    println!("🚀 Using parallelized BPE merge implementation!");
    println!("CPU cores available: {}", rayon::current_num_threads());
    
    println!("=== OpenWebText BPE Training ===");
    println!("Input: {}", input_path);
    
    // Check file size
    if let Ok(metadata) = std::fs::metadata(input_path) {
        let file_size_gb = metadata.len() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("File size: {:.1} GB", file_size_gb);
    }
    
    println!("Vocab size: {}", vocab_size);
    println!("Special tokens: {:?}", special_tokens);
    println!("Output directory: {}", output_dir);
    println!();
    
    // Create output directory
    if let Err(e) = std::fs::create_dir_all(output_dir) {
        println!("Failed to create output directory: {}", e);
        std::process::exit(1);
    }
    
    println!("Starting BPE training...");
    let start = Instant::now();
    
    // Extract word frequencies (uses lib.rs optimized approach)
    let extract_start = Instant::now();
    match extract_word_frequencies_with_stats(input_path, &special_tokens) {
        Ok((word_freqs, chunk_count)) => {
            let file_processing_time = extract_start.elapsed();
            println!("File processing: {:.3}s", file_processing_time.as_secs_f64());
            println!("Total chunks processed: {}", chunk_count);
            println!("Unique word types: {}", word_freqs.len());
            
            // Train BPE on the word frequencies
            println!("Starting BPE merge training...");
            let bpe_start = Instant::now();
            
            match train_bpe_from_word_freqs(word_freqs, vocab_size, &special_tokens) {
                Ok((vocab, merges)) => {
                    let bpe_time = bpe_start.elapsed();
                    let total_time = start.elapsed();
                    
                    println!("BPE training: {:.3}s", bpe_time.as_secs_f64());
                    println!("TOTAL time: {:.3}s", total_time.as_secs_f64());
                    println!("Final vocab size: {}", vocab.len());
                    println!("Number of merges: {}", merges.len());
                    
                    if let Ok(metadata) = std::fs::metadata(input_path) {
                        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
                        println!("Throughput: {:.1} MB/s", file_size_mb / total_time.as_secs_f64());
                    }
                    
                    // Save outputs
                    println!("Saving tokenizer outputs...");
                    if let Err(e) = save_tokenizer_outputs(&vocab, &merges, output_dir) {
                        println!("Failed to save outputs: {}", e);
                        std::process::exit(1);
                    }
                    
                    println!("✅ OpenWebText BPE training completed successfully!");
                    println!("Outputs saved to: {}", output_dir);
                }
                Err(e) => {
                    println!("BPE training failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            println!("File processing failed: {}", e);
            std::process::exit(1);
        }
    }
}