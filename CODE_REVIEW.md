# Code review

Reviewed the tracked backend, extraction code, Docker setup, and the standalone `DocuraAI.tsx` component on 2026-10-05. Findings below describe the code before the fixes on `nabiel/fix-code-review-findings`. They came from static inspection; the application and test suite were not run.

## Resolution on this branch

Findings 1–3 are addressed by required bearer authentication, owner bound upload IDs, a remote host allowlist, and immutable indexes scoped to each owner and document set; the exposed token was removed from source. Findings 4–8 are addressed by preserving batch order and chunk metadata, accurate ingestion results, PPTX system dependencies, and bounded input and archive reads. Findings 9–14 are addressed by reduced logging, a configurable frontend API origin, retained conversation documents, upload IDs in place of file URLs, unique attachment IDs, and validated search options. The service refactor added persistent document records with expiry and deletion, one extraction path, reusable bounded indexes, and one model calling endpoint. The previously committed token still needs rotation wherever it was used.

## Critical

### 1. API routes are public while accepting server side file paths and URLs

`verify_token` is defined but never attached to the router or any route (`main.py:1779-1787`). `/api/v1/hackrx/run` accepts caller supplied document paths (`main.py:2026-2047`), and `process_documents_parallel` opens an existing path or `file://` path on the server (`main.py:1420-1463`). `/api/v1/parse` also passes caller supplied URLs to the extractor (`main.py:1834-1856`), which can fetch local or internal resources (`text_extractor.py:726-765`). An unauthenticated caller can therefore make the server read accessible local files or contact internal services; retrieved text can be returned through parse, search, or Gemini answers. Require authentication and authorization, use opaque upload IDs, restrict files to the upload directory, and validate remote destinations including redirects.

### 2. A shared global index mixes users' document context

The application creates one `system` instance (`main.py:1775-1777`). Every ingestion assigns `self.chunks` and rebuilds its indices (`main.py:944-946`, `main.py:1571-1572`). Search routes and question answering read that same instance (`main.py:1909-1944`, `main.py:1609-1623`). One user's request can replace another user's searchable documents, and concurrent work can observe a mixture of chunks and indices while they are being rebuilt. Scope indexes to a document set or session, then publish a fully built index atomically.

### 3. The hardcoded bearer token is exposed in source

`main.py:92` contains a bearer token literal. Even if authentication is wired up later, that value is already exposed to anyone with repository access. Remove it from source, rotate it, and read the replacement from a secret or environment variable.

## High

### 4. Batch results can be assigned to the wrong URL

`extract_text_from_multiple_sources` appends futures in completion order (`text_extractor.py:949-956`). `/api/v1/parse/batch` pairs those results with `request.urls[i]` by input position (`main.py:1888-1898`). If the second URL completes first, its text is reported as belonging to the first URL. Keep each future's input index or URL with its result and return results in input order.

### 5. Failed local ingestion is reported as successful and can reuse stale documents

`process_documents_parallel` returns an empty chunk list for invalid paths and extraction failures (`main.py:1395-1495`), but `ingest_documents_async` marks *every* local file successful if the batch call itself returns (`main.py:1554-1565`). With no new chunks, it leaves the previous global index intact (`main.py:1567-1575`). `/hackrx/run` then passes its failure check (`main.py:2047-2056`) and may answer a new request using an earlier user's documents. Return a per-file ingestion result, clear or isolate failed request state, and require at least one newly indexed chunk before answering.

### 6. Chunk merging discards metadata and source identity

`merge_chunks` converts chunks to strings and returns new `DocumentChunk` objects with empty metadata (`main.py:262`, `main.py:324`). It is called after chunking (`main.py:749-751`), so `chunk_type`, position, and other metadata disappear. `merge_chunks_search` also creates a fresh chunk with empty metadata (`main.py:389-395`), so ensemble results lose the source added during ingestion. Preserve metadata when merging and avoid merging text from different sources.

### 7. Presentation extraction fails in the provided Docker image

PPTX extraction calls the `soffice` executable before reaching its fallback (`text_extractor.py:241-244`, `text_extractor.py:287-300`), but the Dockerfile installs no LibreOffice package (`Dockerfile:5-9`). Install the converter or use native slide text without conversion. The original review also called the Gemini `model` variable undefined; that was incorrect because it was initialized at module import.

### 8. Uploaded files and remote responses have no size limit

The upload route reads the entire file into memory (`main.py:1804-1807`). Remote extraction reads the entire response (`text_extractor.py:764-765`); web and API extraction use whole `response.content` (`text_extractor.py:803-805`, `text_extractor.py:870-875`). ZIP extraction reads every supported member (`text_extractor.py:1429-1435`). An oversized file or compressed archive can exhaust memory or CPU. Enforce request and decompressed size limits and stream bounded reads.

### 9. Sensitive document content is written to persistent logs

The HTTP middleware logs the request URL, query parameters, and body (`main.py:2180-2206`). Question processing logs the full prompt, including retrieved document text, and raw model response (`main.py:1721-1743`). This exposes uploaded content, questions, URL tokens, and answers in `logs/app.log` (`main.py:51-64`). Log request IDs, sizes, status, and timing instead; redact secrets and document text.

## Medium

### 10. The standalone chat component calls localhost from the user's browser

Both fetch calls use `http://localhost:8000` (`DocuraAI.tsx:66`, `DocuraAI.tsx:191`). On a deployed frontend, `localhost` means the visitor's own machine, so uploads and chat fail unless that visitor runs the API locally. Use a configured public API origin or a same-origin proxy.

### 11. Follow-up chat messages do not carry prior documents

`handleSend` clears `uploadedFiles` after the first message (`DocuraAI.tsx:173-177`). Its next payload sets `documents` to an empty array when no new file is attached (`DocuraAI.tsx:180-185`). The backend treats an empty ingestion result as all failed (`main.py:2047-2056`), so a follow-up question receives the empty-document response. Retain a document-set ID for the conversation and pass it on follow-up messages.

### 12. A failed parse of the upload URL uses the wrong local path

The upload route returns `file:/{absolute_path}` (`main.py:1809-1810`), producing a string such as `file://app/uploaded_docs/...`. The URL extractor removes the first seven characters (`text_extractor.py:726-729`), leaving `app/uploaded_docs/...` rather than `/app/uploaded_docs/...`. `/api/v1/parse` therefore cannot reliably parse a URL returned by `/api/v1/upload`. Use standard file URI construction/parsing, or preferably an upload ID and a dedicated file lookup.

### 13. Duplicate filenames update or remove the wrong attachment

The component matches uploaded items by name when recording an upload URL and when removing a failed upload (`DocuraAI.tsx:99-109`). Attaching two different files with the same name updates or removes both. Give each attachment a unique client ID and match by that ID.

### 14. Search behavior and request validation are inconsistent

`SearchRequest.top_k` has no bounds (`models.py:15-23`), `advanced_search` treats any unknown strategy as hybrid but reports the unknown value in the response (`main.py:1261-1304`), and semantic search retrieves up to `top_k * 3` results without slicing back to `top_k` (`main.py:1116-1138`). Validate strategy and a bounded positive `top_k`, and apply the requested limit consistently.

## Suggested fix order

1. Protect routes and restrict local and remote sources; rotate the committed token.
2. Isolate index state per document set and make failed ingestion fail closed.
3. Fix batch result ordering and preserve chunk source metadata.
4. Add upload/download/archive limits and remove sensitive logging.
5. Repair PPTX extraction, upload URL handling, and browser API configuration.
