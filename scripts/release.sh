#!/usr/bin/env bash
# Build, sign, optionally notarize, and publish a release binary.
#
# Usage:
#   scripts/release.sh <version> [--identity "Developer ID Application: …"]
#                      [--notary-profile NAME] [--no-upload]
#
# Defaults: the first "Developer ID Application" identity in the keychain;
# no notarization unless --notary-profile names a `xcrun notarytool
# store-credentials` profile; upload to the existing GitHub release
# gh://v<version> unless --no-upload.
#
# Why sign: macOS TCC anchors the iCloud-Drive / Full Disk Access grant to
# the binary's code-signing identity. Ad-hoc builds get a per-build
# identity, so every upgrade silently invalidates the grant (we learned
# this the hard way — an unreadable tree once looked like a mass
# deletion); Developer ID-signed builds keep it across releases.
#
# Build-time requirement: the Xcode Command Line Tools (swiftc compiles
# the embedded Vision OCR sidecar). End users of the released binary do
# not need them.
set -euo pipefail

VERSION="${1:?usage: release.sh <version> [--identity ID] [--notary-profile NAME] [--no-upload]}"
shift
IDENTITY=""
NOTARY_PROFILE=""
UPLOAD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --identity) IDENTITY="$2"; shift 2 ;;
    --notary-profile) NOTARY_PROFILE="$2"; shift 2 ;;
    --no-upload) UPLOAD=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

CARGO_VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
if [[ "$CARGO_VERSION" != "$VERSION" ]]; then
  echo "error: Cargo.toml is $CARGO_VERSION but releasing $VERSION" >&2
  exit 1
fi

if [[ -z "$IDENTITY" ]]; then
  IDENTITY=$(security find-identity -v -p codesigning \
    | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)
fi
if [[ -z "$IDENTITY" ]]; then
  echo "error: no 'Developer ID Application' identity in the keychain." >&2
  echo "Create one in Xcode → Settings → Accounts → Manage Certificates" >&2
  echo "(requires the Account Holder role), or pass --identity." >&2
  exit 1
fi
echo "signing identity: $IDENTITY"

cargo build --release --locked
BIN=target/release/ai-icloud

codesign --force --options runtime --timestamp --sign "$IDENTITY" "$BIN"
codesign --verify --strict --verbose=1 "$BIN"
echo "signed and verified"

if [[ -n "$NOTARY_PROFILE" ]]; then
  ZIP=$(mktemp -d)/ai-icloud.zip
  ditto -c -k "$BIN" "$ZIP"
  xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
  # Plain executables cannot be stapled; Gatekeeper checks the ticket online.
  echo "notarized"
fi

TARGET=$(rustc -vV | sed -n 's/^host: //p')
TARBALL="ai-icloud-v${VERSION}-${TARGET}.tar.gz"
tar -czf "$TARBALL" -C target/release ai-icloud
SHA=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)
echo "built $TARBALL"
echo "sha256 $SHA"

if [[ "$UPLOAD" -eq 1 ]]; then
  gh release upload "v${VERSION}" "$TARBALL" --clobber
  echo "uploaded to release v${VERSION}"
fi

cat <<FORMULA

Formula stanza for the tap (binary install on this architecture):

  class AiIcloud < Formula
    desc "Local-first iCloud Drive document RAG index and MCP server"
    homepage "https://github.com/tjameswilliams/ai-icloud"
    url "https://github.com/tjameswilliams/ai-icloud/releases/download/v${VERSION}/${TARBALL}"
    sha256 "${SHA}"
    version "${VERSION}"

    def install
      bin.install "ai-icloud"
    end

    def caveats
      <<~EOS
        Start with the interactive wizard:
          ai-icloud setup
        It walks through the OpenAI-compatible LLM backend (LM Studio is
        the happy path), privacy exclusions, and transcription, then:
          ai-icloud scan
          ai-icloud service install
          ai-icloud connect   # MCP JSON for any agent framework
        If the background daemon logs permission errors, grant Full Disk
        Access to #{HOMEBREW_PREFIX}/bin/ai-icloud in System Settings.
      EOS
    end
  end
FORMULA
