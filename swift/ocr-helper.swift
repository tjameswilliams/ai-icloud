// OCR sidecar: PDFKit for text layers and page rendering, Vision for OCR.
// Compiled by build.rs, embedded in the ai-icloud binary, materialized to
// the index directory at runtime. Speaks JSON on stdout; errors go to
// stderr with a non-zero exit.
//
// Commands:
//   probe
//   pdf-text <path>                      — text layer of every page
//   ocr-pdf <path> <pages-csv> <dpi>     — render + OCR selected pages
//   ocr-image <path>                     — OCR one image (png/jpg/heic)
//   render-pdf <path> <pages-csv> <dpi>  — render pages to base64 PNGs
//   render-image <path>                  — re-encode an image (any format
//                                          CGImageSource reads, incl. HEIC)
//                                          as a base64 PNG, downscaled

import AppKit
import Foundation
import PDFKit
import Vision

struct ProbeReply: Codable {
    let ok: Bool
    let version: Int
}

struct TextPage: Codable {
    let index: Int
    let text: String
}

struct TextReply: Codable {
    let pageCount: Int
    let pages: [TextPage]
}

struct OcrPage: Codable {
    let index: Int
    let text: String
    let confidence: Double
}

struct OcrReply: Codable {
    let pages: [OcrPage]
}

struct RenderPage: Codable {
    let index: Int
    let pngBase64: String
}

struct RenderReply: Codable {
    let pages: [RenderPage]
}

func pngBase64(_ image: CGImage) -> String {
    let rep = NSBitmapImageRep(cgImage: image)
    guard let data = rep.representation(using: .png, properties: [:]) else {
        die("could not encode PNG")
    }
    return data.base64EncodedString()
}

/// Downscale so the longest edge is at most `maxEdge` (vision models don't
/// need print resolution, and request bodies stay small).
func downscale(_ image: CGImage, maxEdge: Int) -> CGImage {
    let w = image.width
    let h = image.height
    let longest = max(w, h)
    if longest <= maxEdge {
        return image
    }
    let scale = Double(maxEdge) / Double(longest)
    let nw = Int(Double(w) * scale)
    let nh = Int(Double(h) * scale)
    guard
        let ctx = CGContext(
            data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
        die("could not create a drawing context")
    }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
    guard let scaled = ctx.makeImage() else {
        die("could not downscale image")
    }
    return scaled
}

func die(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func emit<T: Codable>(_ value: T) {
    let encoder = JSONEncoder()
    guard let data = try? encoder.encode(value) else {
        die("could not encode reply")
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

func openPdf(_ path: String) -> PDFDocument {
    guard let doc = PDFDocument(url: URL(fileURLWithPath: path)) else {
        die("could not open PDF \(path)")
    }
    if doc.isLocked {
        die("PDF \(path) is password-protected")
    }
    return doc
}

/// OCR a CGImage with Vision in accurate mode. Returns joined text plus
/// mean per-line confidence.
func recognize(_ image: CGImage) -> (String, Double) {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do {
        try handler.perform([request])
    } catch {
        die("vision request failed: \(error.localizedDescription)")
    }
    var lines: [String] = []
    var confidences: [Double] = []
    for observation in request.results ?? [] {
        guard let candidate = observation.topCandidates(1).first else { continue }
        lines.append(candidate.string)
        confidences.append(Double(candidate.confidence))
    }
    let mean = confidences.isEmpty ? 0 : confidences.reduce(0, +) / Double(confidences.count)
    return (lines.joined(separator: "\n"), mean)
}

func renderPage(_ page: PDFPage, dpi: Double) -> CGImage {
    let bounds = page.bounds(for: .mediaBox)
    let scale = dpi / 72.0
    let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    let thumbnail = page.thumbnail(of: size, for: .mediaBox)
    guard let cg = thumbnail.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        die("could not rasterize a PDF page")
    }
    return cg
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    die("usage: ocr-helper <probe|pdf-text|ocr-pdf|ocr-image> ...")
}

switch args[1] {
case "probe":
    emit(ProbeReply(ok: true, version: 1))

case "pdf-text":
    guard args.count == 3 else { die("usage: ocr-helper pdf-text <path>") }
    let doc = openPdf(args[2])
    var pages: [TextPage] = []
    for i in 0..<doc.pageCount {
        let text = doc.page(at: i)?.string ?? ""
        pages.append(TextPage(index: i, text: text))
    }
    emit(TextReply(pageCount: doc.pageCount, pages: pages))

case "ocr-pdf":
    guard args.count == 5 else { die("usage: ocr-helper ocr-pdf <path> <pages-csv> <dpi>") }
    let doc = openPdf(args[2])
    let indices = args[3].split(separator: ",").compactMap { Int($0) }
    guard let dpi = Double(args[4]), dpi >= 72, dpi <= 600 else {
        die("dpi must be between 72 and 600")
    }
    var pages: [OcrPage] = []
    for i in indices {
        guard i >= 0, i < doc.pageCount, let page = doc.page(at: i) else {
            die("page \(i) out of range (document has \(doc.pageCount))")
        }
        let (text, confidence) = recognize(renderPage(page, dpi: dpi))
        pages.append(OcrPage(index: i, text: text, confidence: confidence))
    }
    emit(OcrReply(pages: pages))

case "render-pdf":
    guard args.count == 5 else { die("usage: ocr-helper render-pdf <path> <pages-csv> <dpi>") }
    let doc = openPdf(args[2])
    let indices = args[3].split(separator: ",").compactMap { Int($0) }
    guard let dpi = Double(args[4]), dpi >= 36, dpi <= 600 else {
        die("dpi must be between 36 and 600")
    }
    var pages: [RenderPage] = []
    for i in indices {
        // Out-of-range renders are skipped, not fatal: callers may only
        // know an approximate page count.
        guard i >= 0, i < doc.pageCount, let page = doc.page(at: i) else { continue }
        let rendered = downscale(renderPage(page, dpi: dpi), maxEdge: 1568)
        pages.append(RenderPage(index: i, pngBase64: pngBase64(rendered)))
    }
    emit(RenderReply(pages: pages))

case "render-image":
    guard args.count == 3 else { die("usage: ocr-helper render-image <path>") }
    guard let image = NSImage(contentsOfFile: args[2]),
        let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    else {
        die("could not load image \(args[2])")
    }
    let rendered = downscale(cg, maxEdge: 1568)
    emit(RenderReply(pages: [RenderPage(index: 0, pngBase64: pngBase64(rendered))]))

case "ocr-image":
    guard args.count == 3 else { die("usage: ocr-helper ocr-image <path>") }
    guard let image = NSImage(contentsOfFile: args[2]),
        let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
    else {
        die("could not load image \(args[2])")
    }
    let (text, confidence) = recognize(cg)
    emit(OcrReply(pages: [OcrPage(index: 0, text: text, confidence: confidence)]))

default:
    die("unknown command \(args[1])")
}
