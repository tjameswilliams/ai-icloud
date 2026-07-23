//! Audio/video transcription: ffmpeg demuxes and resamples to 16kHz mono
//! f32, whisper.cpp (Metal) transcribes. The ggml model downloads once
//! into the index directory. Transcripts carry [mm:ss] timestamps so
//! answers can point into the recording.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::config::Config;

const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub struct Transcriber {
    ctx: whisper_rs::WhisperContext,
}

impl Transcriber {
    /// Load the configured whisper model, downloading it on first use.
    pub fn load(config: &Config, model_dir: &Path) -> Result<Transcriber> {
        let model_path = ensure_model(&config.transcription.whisper_model, model_dir)?;
        let ctx = whisper_rs::WhisperContext::new_with_params(
            &model_path,
            whisper_rs::WhisperContextParameters::default(),
        )
        .context("could not load the whisper model")?;
        Ok(Transcriber { ctx })
    }

    /// Transcribe one media file into a single page of timestamped lines.
    pub fn transcribe(&self, path: &Path) -> Result<Vec<String>> {
        let samples = decode_to_pcm(path)?;
        if samples.is_empty() {
            bail!("no audio stream in {}", path.display());
        }
        let mut state = self
            .ctx
            .create_state()
            .context("could not create whisper state")?;
        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("auto"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        state
            .full(params, &samples)
            .context("whisper transcription failed")?;

        let mut lines = Vec::new();
        for i in 0..state.full_n_segments() {
            let Ok(segment) = state.get_segment(i).context("missing segment") else {
                continue;
            };
            let text = segment.to_str_lossy().map(|s| s.trim().to_string());
            let Ok(text) = text else { continue };
            if text.is_empty() {
                continue;
            }
            // start_timestamp is in centiseconds.
            let secs = segment.start_timestamp() / 100;
            lines.push(format!("[{:02}:{:02}] {}", secs / 60, secs % 60, text));
        }
        Ok(vec![lines.join("\n")])
    }
}

/// ffmpeg lives in Homebrew's prefix, which launchd agents don't have on
/// PATH — resolve known locations before falling back to lookup.
pub fn ffmpeg_bin() -> &'static str {
    for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"] {
        if Path::new(candidate).exists() {
            return candidate;
        }
    }
    "ffmpeg"
}

/// ffmpeg → 16kHz mono f32le PCM on stdout. ffmpeg handles every
/// container/codec we scope in (mp3, m4a, mp4, mov, wav).
fn decode_to_pcm(path: &Path) -> Result<Vec<f32>> {
    let mut child = Command::new(ffmpeg_bin())
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("could not run ffmpeg — install it with `brew install ffmpeg`")?;

    let mut bytes = Vec::new();
    child
        .stdout
        .take()
        .context("ffmpeg gave no stdout")?
        .read_to_end(&mut bytes)?;
    let mut err = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        let _ = stderr.read_to_string(&mut err);
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg failed for {}: {}", path.display(), err.trim());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The ggml model file, downloaded once. A partial download can never be
/// mistaken for a model: we write to a temp name and rename on success.
fn ensure_model(model: &str, model_dir: &Path) -> Result<PathBuf> {
    let dir = model_dir.join("whisper");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("ggml-{model}.bin"));
    if path.exists() {
        return Ok(path);
    }
    let url = format!("{MODEL_BASE_URL}/ggml-{model}.bin");
    tracing::info!("downloading whisper model {model} (one time, this is large)…");
    let resp = ureq::get(&url)
        .timeout(std::time::Duration::from_secs(3600))
        .call()
        .with_context(|| format!("could not download {url}"))?;
    let tmp = dir.join(format!(".ggml-{model}.bin.partial"));
    let mut file = std::fs::File::create(&tmp)?;
    std::io::copy(&mut resp.into_reader(), &mut file)
        .with_context(|| format!("download of {url} interrupted"))?;
    std::fs::rename(&tmp, &path)?;
    tracing::info!("whisper model stored at {}", path.display());
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rejects_missing_files() {
        assert!(decode_to_pcm(Path::new("/nonexistent.mp3")).is_err());
    }

    #[test]
    fn decode_handles_real_audio_when_ffmpeg_exists() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return; // environment without ffmpeg: nothing to verify
        }
        // Generate one second of silence via ffmpeg itself, then decode it.
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("t.wav");
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1"])
            .arg(&wav)
            .status()
            .unwrap()
            .success();
        assert!(ok);
        let samples = decode_to_pcm(&wav).unwrap();
        assert!((15000..=17000).contains(&samples.len()), "{}", samples.len());
    }
}
