//! Compile the Swift OCR sidecar (PDFKit + Vision) so it can be embedded
//! into the binary. Requires the Xcode Command Line Tools at build time
//! only; end users never need swiftc.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=swift/ocr-helper.swift");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let dest = out_dir.join("ocr-helper");

    let status = Command::new("swiftc")
        .args(["-O", "swift/ocr-helper.swift", "-o"])
        .arg(&dest)
        .status()
        .expect(
            "could not run swiftc — install the Xcode Command Line Tools \
             (xcode-select --install) to build ai-icloud",
        );
    assert!(status.success(), "swiftc failed to build the OCR sidecar");
}
