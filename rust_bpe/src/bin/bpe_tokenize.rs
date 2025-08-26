use clap::Parser;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

/// Tokenize or detokenize text using a trained BPE tokenizer
#[derive(Parser)]
#[command(name = "bpe_tokenize")]
#[command(about = "Tokenize or detokenize text using a trained BPE tokenizer")]
#[command(long_about = None)]
struct Args {
    /// Path to vocab.json file
    #[arg(short = 'v', long = "vocab")]
    vocab: String,

    /// Path to merges.txt file
    #[arg(short = 'm', long = "merges")]
    merges: String,

    /// Input file path
    #[arg(short = 'i', long = "input")]
    input: Option<String>,

    /// Output file path
    #[arg(short = 'o', long = "output")]
    output: Option<String>,

    /// Direct text input for quick tokenization
    #[arg(long = "text")]
    text: Option<String>,

    /// Operation mode (tokenize or detokenize)
    #[arg(long = "mode", default_value = "tokenize")]
    mode: String,

    /// Interactive mode for testing tokenization
    #[arg(long = "interactive")]
    interactive: bool,

    /// Output format for tokenization (ids, json, or text)
    #[arg(long = "output-format", default_value = "ids")]
    output_format: String,

    /// Input format for detokenization (ids or json)
    #[arg(long = "input-format", default_value = "ids")]
    input_format: String,

    /// Special tokens list
    #[arg(short = 's', long = "special-tokens", default_values = &["<|endoftext|>"])]
    special_tokens: Vec<String>,

    /// Disable statistics output
    #[arg(long = "no-stats")]
    no_stats: bool,
}

/// BPE tokenizer structure
struct BPETokenizer {
    vocab: HashMap<u16, Vec<u8>>,         // token_id -> token_bytes
    bytes_to_id: HashMap<Vec<u8>, u16>,   // token_bytes -> token_id
    merges: Vec<(Vec<u8>, Vec<u8>)>,      // merge rules in order
    merge_priorities: HashMap<(Vec<u8>, Vec<u8>), usize>, // merge -> priority index
    special_tokens: Vec<String>,
    pat: Regex,                           // pre-tokenization pattern
}

impl BPETokenizer {
    fn new(
        vocab: HashMap<u16, Vec<u8>>, 
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Vec<String>
    ) -> Self {
        // Create reverse vocab lookup
        let mut bytes_to_id = HashMap::new();
        for (&token_id, token_bytes) in &vocab {
            bytes_to_id.insert(token_bytes.clone(), token_id);
        }

        // Pre-compute merge priorities (earlier merges have higher priority = lower index)
        let mut merge_priorities = HashMap::new();
        for (i, merge) in merges.iter().enumerate() {
            let merged_token = [merge.0.clone(), merge.1.clone()].concat();
            if bytes_to_id.contains_key(&merged_token) {
                merge_priorities.insert(merge.clone(), i);
            }
        }

        // Pre-compile regex pattern (GPT-2 style, adapted for Rust regex limitations)
        // Note: Rust regex doesn't support lookahead, so we simplify the whitespace handling
        let pat = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .expect("Failed to compile regex pattern");

        BPETokenizer {
            vocab,
            bytes_to_id,
            merges,
            merge_priorities,
            special_tokens,
            pat,
        }
    }

    fn encode(&self, text: &str) -> Vec<u16> {
        let mut result = Vec::new();

        if !self.special_tokens.is_empty() {
            // Handle special tokens by splitting text preserving them
            // For simplicity in this implementation, we'll just handle text without special tokens
            // A full implementation would need more complex special token handling
            result.extend(self.encode_text_part(text));
        } else {
            result.extend(self.encode_text_part(text));
        }

        result
    }

    fn encode_text_part(&self, text: &str) -> Vec<u16> {
        let mut result = Vec::new();

        // Process each pre-token separately using the compiled pattern
        for captures in self.pat.find_iter(text) {
            let pre_token = captures.as_str();
            if pre_token.is_empty() {
                continue;
            }

            // Start with byte-level tokens for this pre-token
            let mut tokens: Vec<Vec<u8>> = pre_token.bytes()
                .map(|b| vec![b])
                .collect();

            // Apply merges using priority-based approach
            while tokens.len() > 1 {
                // Find the best merge available in current token sequence
                let mut best_merge: Option<(Vec<u8>, Vec<u8>)> = None;
                let mut best_priority = self.merges.len(); // Higher than any real priority
                let mut best_pos = 0;

                for i in 0..tokens.len() - 1 {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    if let Some(&priority) = self.merge_priorities.get(&pair) {
                        if priority < best_priority {
                            best_merge = Some(pair);
                            best_priority = priority;
                            best_pos = i;
                        }
                    }
                }

                if best_merge.is_none() {
                    break;
                }

                // Apply the best merge
                let merged_token = [tokens[best_pos].clone(), tokens[best_pos + 1].clone()].concat();
                let mut new_tokens = tokens[..best_pos].to_vec();
                new_tokens.push(merged_token);
                new_tokens.extend_from_slice(&tokens[best_pos + 2..]);
                tokens = new_tokens;
            }

            // Convert to token IDs and add to result
            for token in tokens {
                if let Some(&token_id) = self.bytes_to_id.get(&token) {
                    result.push(token_id);
                }
            }
        }

        result
    }

    fn decode(&self, token_ids: &[u16]) -> String {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            if let Some(token_bytes) = self.vocab.get(&token_id) {
                bytes.extend_from_slice(token_bytes);
            } else if token_id < 256 {
                // Fallback for byte tokens
                bytes.push(token_id as u8);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

/// Load a BPE tokenizer from vocab and merges files
fn load_tokenizer(
    vocab_path: &str,
    merges_path: &str,
    special_tokens: &[String],
) -> Result<BPETokenizer, Box<dyn std::error::Error>> {
    println!("📁 Loading tokenizer from {} and {}", vocab_path, merges_path);

    // Load vocabulary
    let vocab_file = File::open(vocab_path)?;
    let vocab_data: HashMap<String, u16> = serde_json::from_reader(vocab_file)?;

    // Convert: vocab_data has token_str -> token_id, we need token_id -> token_bytes
    let mut vocab = HashMap::new();
    for (token_str, token_id) in vocab_data {
        let token_bytes = token_str.as_bytes().to_vec();
        vocab.insert(token_id, token_bytes);
    }

    // Load merges
    let mut merges = Vec::new();
    let merges_file = File::open(merges_path)?;
    let reader = BufReader::new(merges_file);
    
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        if let Some(space_pos) = line.find(' ') {
            let token1 = line[..space_pos].as_bytes().to_vec();
            let token2 = line[space_pos + 1..].as_bytes().to_vec();
            merges.push((token1, token2));
        }
    }

    println!("✅ Loaded tokenizer: {} vocab, {} merges, special tokens: {:?}", 
             vocab.len(), merges.len(), special_tokens);

    Ok(BPETokenizer::new(vocab, merges, special_tokens.to_vec()))
}

/// Tokenize an entire file
fn tokenize_file(
    tokenizer: &BPETokenizer,
    input_path: &str,
    output_path: Option<&str>,
    output_format: &str,
    show_stats: bool,
) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    println!("🔤 Tokenizing file: {}", input_path);

    // Read input file
    let text = std::fs::read_to_string(input_path)?;
    let start_time = Instant::now();

    // Use the proper BPE tokenizer
    let token_ids = tokenizer.encode(&text);

    let tokenization_time = start_time.elapsed();

    if show_stats {
        println!("📊 Tokenization stats:");
        println!("   • Input characters: {}", text.len());
        println!("   • Output tokens: {}", token_ids.len());
        println!("   • Compression ratio: {:.2}x", text.len() as f64 / token_ids.len() as f64);
        println!("   • Time: {:.3}s", tokenization_time.as_secs_f64());
        println!("   • Speed: {:.1}K chars/sec", text.len() as f64 / tokenization_time.as_secs_f64() / 1000.0);
    }

    // Save output if path provided
    if let Some(output_path) = output_path {
        match output_format {
            "ids" => {
                // Save as space-separated token IDs
                let ids_str = token_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" ");
                std::fs::write(output_path, ids_str)?;
                println!("💾 Saved token IDs to {}", output_path);
            }
            "json" => {
                // Save as JSON array
                let json_str = serde_json::to_string(&token_ids)?;
                std::fs::write(output_path, json_str)?;
                println!("💾 Saved tokens as JSON to {}", output_path);
            }
            "text" => {
                // Save as human-readable text with token boundaries
                let mut output = String::new();
                output.push_str("Token ID : Token String\n");
                output.push_str("==================================================\n");
                for (i, &token_id) in token_ids.iter().enumerate().take(100) {
                    if let Some(token_bytes) = tokenizer.vocab.get(&token_id) {
                        let token_str = String::from_utf8_lossy(token_bytes);
                        output.push_str(&format!("{}: {} -> {:?}\n", i, token_id, token_str));
                    }
                }
                if token_ids.len() > 100 {
                    output.push_str(&format!("\n... and {} more tokens\n", token_ids.len() - 100));
                }
                std::fs::write(output_path, output)?;
                println!("💾 Saved readable tokens to {}", output_path);
            }
            _ => {
                eprintln!("❌ Unknown output format: {}", output_format);
                std::process::exit(1);
            }
        }
    }

    Ok(token_ids)
}

/// Detokenize token IDs back to text
fn detokenize_file(
    tokenizer: &BPETokenizer,
    input_path: &str,
    output_path: Option<&str>,
    input_format: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("🔠 Detokenizing file: {}", input_path);

    // Read token IDs
    let token_ids: Vec<u16> = match input_format {
        "ids" => {
            let content = std::fs::read_to_string(input_path)?;
            let content = content.trim();
            if content.is_empty() {
                Vec::new()
            } else {
                content.split_whitespace()
                    .map(|s| s.parse())
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
        "json" => {
            let file = File::open(input_path)?;
            serde_json::from_reader(file)?
        }
        _ => {
            return Err(format!("Unknown input format: {}", input_format).into());
        }
    };

    let start_time = Instant::now();

    // Detokenize
    let text = tokenizer.decode(&token_ids);

    let detokenization_time = start_time.elapsed();

    println!("📊 Detokenization stats:");
    println!("   • Input tokens: {}", token_ids.len());
    println!("   • Output characters: {}", text.len());
    println!("   • Time: {:.3}s", detokenization_time.as_secs_f64());
    println!("   • Speed: {:.1}K tokens/sec", token_ids.len() as f64 / detokenization_time.as_secs_f64() / 1000.0);

    // Save output if path provided
    if let Some(output_path) = output_path {
        std::fs::write(output_path, &text)?;
        println!("💾 Saved detokenized text to {}", output_path);
    }

    Ok(text)
}

/// Interactive tokenization mode
fn interactive_mode(
    tokenizer: &BPETokenizer,
) {
    println!("\n🤖 Interactive BPE Tokenization Mode");
    println!("Commands:");
    println!("  encode <text>     - Tokenize text");
    println!("  decode <ids>      - Detokenize space-separated token IDs");
    println!("  stats             - Show tokenizer statistics");
    println!("  quit/exit         - Exit interactive mode");
    println!("{}", "-".repeat(50));

    loop {
        print!("\n>>> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if ["quit", "exit", "q"].contains(&input.to_lowercase().as_str()) {
            break;
        }

        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();

        match cmd.as_str() {
            "encode" if parts.len() > 1 => {
                let text = parts[1];
                let token_ids = tokenizer.encode(text);
                println!("Text: \"{}\"", text);
                println!("Token IDs: {:?}", token_ids);
                println!("Length: {} tokens", token_ids.len());

                // Show token breakdown for short sequences
                if token_ids.len() <= 20 {
                    println!("Token breakdown:");
                    for (i, &token_id) in token_ids.iter().enumerate() {
                        if let Some(token_bytes) = tokenizer.vocab.get(&token_id) {
                            let token_str = String::from_utf8_lossy(token_bytes);
                            println!("  {}: {} -> {:?}", i, token_id, token_str);
                        }
                    }
                }
            }
            "decode" if parts.len() > 1 => {
                match parts[1].split_whitespace()
                    .map(|s| s.parse::<u16>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(token_ids) => {
                        let text = tokenizer.decode(&token_ids);
                        println!("Token IDs: {:?}", token_ids);
                        println!("Text: \"{}\"", text);
                    }
                    Err(e) => {
                        println!("Error: Invalid token IDs - {}", e);
                    }
                }
            }
            "stats" => {
                let vocab_size = tokenizer.vocab.len();
                let merge_count = tokenizer.merges.len();
                println!("Tokenizer statistics:");
                println!("  • Vocabulary size: {}", vocab_size);
                println!("  • Merge rules: {}", merge_count);
            }
            _ => {
                println!("Unknown command. Type 'encode <text>', 'decode <ids>', 'stats', or 'quit'.");
            }
        }
    }

    println!("\nExiting interactive mode...");
}


fn main() {
    let args = Args::parse();

    println!("🔤 BPE Tokenizer");
    println!("==============================");

    // Validate arguments
    if !Path::new(&args.vocab).exists() {
        eprintln!("❌ Error: Vocab file '{}' not found", args.vocab);
        std::process::exit(1);
    }

    if !Path::new(&args.merges).exists() {
        eprintln!("❌ Error: Merges file '{}' not found", args.merges);
        std::process::exit(1);
    }

    // Load tokenizer
    let tokenizer = match load_tokenizer(&args.vocab, &args.merges, &args.special_tokens) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            eprintln!("❌ Error loading tokenizer: {}", e);
            std::process::exit(1);
        }
    };

    // Interactive mode
    if args.interactive {
        interactive_mode(&tokenizer);
        return;
    }

    // Direct text input
    if let Some(text) = &args.text {
        println!("\n🔤 Tokenizing direct input: \"{}\"", text);
        let token_ids = tokenizer.encode(text);
        println!("Token IDs: {:?}", token_ids);
        println!("Token count: {}", token_ids.len());

        // Show token breakdown
        println!("Token breakdown:");
        for (i, &token_id) in token_ids.iter().enumerate() {
            if let Some(token_bytes) = tokenizer.vocab.get(&token_id) {
                let token_str = String::from_utf8_lossy(token_bytes);
                println!("  {}: {} -> {:?}", i, token_id, token_str);
            }
        }
        return;
    }

    // File processing mode
    let input_path = match &args.input {
        Some(path) => path,
        None => {
            eprintln!("❌ Error: No input specified. Use --input, --text, or --interactive");
            std::process::exit(1);
        }
    };

    if !Path::new(input_path).exists() {
        eprintln!("❌ Error: Input file '{}' not found", input_path);
        std::process::exit(1);
    }

    let show_stats = !args.no_stats;

    match args.mode.as_str() {
        "tokenize" => {
            match tokenize_file(&tokenizer, input_path, args.output.as_deref(), &args.output_format, show_stats) {
                Ok(token_ids) => {
                    println!("✅ Tokenization completed: {} tokens", token_ids.len());
                }
                Err(e) => {
                    eprintln!("❌ Error during tokenization: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "detokenize" => {
            match detokenize_file(&tokenizer, input_path, args.output.as_deref(), &args.input_format) {
                Ok(text) => {
                    println!("✅ Detokenization completed: {} characters", text.len());
                }
                Err(e) => {
                    eprintln!("❌ Error during detokenization: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("❌ Error: Unknown mode '{}'. Use 'tokenize' or 'detokenize'", args.mode);
            std::process::exit(1);
        }
    }
}