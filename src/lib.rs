//! ai-icloud: local-first iCloud Drive document RAG index and MCP server.
//!
//! Sister project of ai-imessage; see SPEC.md for the architecture and the
//! patterns deliberately carried over.

pub mod chunk;
pub mod cli;
pub mod config;
pub mod doctor;
pub mod embed;
pub mod extract;
pub mod index;
pub mod ingest;
pub mod paths;
pub mod retrieve;
pub mod scan;
pub mod sidecar;
