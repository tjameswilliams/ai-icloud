fn main() {
    if let Err(err) = ai_icloud::cli::run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
