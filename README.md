# llmman

A command-line tool for managing and serving LLM models using OCI registries.
Models are packaged as standard OCI artifacts and stored in any compatible registry (Docker Hub, GHCR, quay, self-hosted, etc.).
`llmman serve` exposes Ollama-, OpenAI-, and Anthropic-compatible HTTP APIs.

## Commands

| Command | Description |
|---------|-------------|
| `serve`   | Start an inference server (Ollama / OpenAI / Anthropic APIs) |
| `pull`    | Pull a model from a registry or HuggingFace |
| `list`    | List locally stored models |
| `build`   | Package model files into a local OCI image |
| `push`    | Push a local image to a registry |
| `transfer` | Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry) |
| `rm`      | Remove a local image |
| `tag`     | Create a new local tag pointing to an existing image |
| `inspect` | Show the manifest of a local or remote image |
| `login`   | Log in to a container registry |
| `logout`  | Log out from a container registry |

## Quick start

### Pull a model

```
llmman pull unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
```

### Transfer a model between locations

Transfer an image directly from a source to a destination without storing
it locally first — e.g. HuggingFace straight to an OCI registry:

```
llmman transfer hf.co/unsloth/Qwen3.5-0.8B-GGUF registry.example.com/owner/model:latest
```

Any source `llmman pull` understands (an OCI registry, `hf://`, `ms://`,
`ngc://`, `s3://`, `gs://`, ...) can be paired with any OCI registry
destination.

### Serve

Start the inference server. Requires `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) to be on `PATH`.

```
llmman serve
```

Optionally pass a model to pre-load on startup:

```
llmman serve unsloth/Qwen3.5-0.8B-GGUF:latest
```

The server listens on `127.0.0.1:17434` and exposes:

| API | Endpoints |
|-----|-----------|
| Ollama | `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/pull`, `/api/ps`, `/api/delete` |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/responses/input_tokens` |
| Anthropic | `/v1/messages` |

`/v1/responses` implements the OpenAI Responses API (the dialect [OpenAI
Codex](https://github.com/openai/codex) requires), including streaming SSE
and function-tool-call re-mapping. This is a plain pass-through to
`llama-server`'s own native `/v1/responses` support, so a recent enough
`llama-server` build is required for it to work.

Use it as an Ollama-compatible server:

```
OLLAMA_HOST=127.0.0.1:17434 ollama run unsloth/Qwen3.5-0.8B-GGUF
```

Or with any Ollama, Anthropic or OpenAI-compatible client.

Models are loaded on demand. Each model gets its own `llama-server` subprocess on a random loopback port; subsequent requests reuse the running process.

Short names work with all commands: `pull`, `push`, `transfer`, `rm`, `tag`, `inspect`, and `serve`.

## Store location

Default locations (override with `--store <DIR>`):

| OS | Path |
|----|------|
| Linux, macOS | `~/.local/share/llmman/store` |
| Windows | `%LOCALAPPDATA%\llmman\store` |

The store uses [OCI Image Layout](https://github.com/opencontainers/image-spec/blob/main/image-layout.md), readable by `docker` and `podman`.

## Transport backends

The registry transport is a compiled-in Go shim. Two backends are available via Cargo feature flags.

### Docker (default)

Uses [`github.com/containerd/containerd`](https://github.com/containerd/containerd) — the same OCI resolver used by Docker.

```
cargo build --release
```

### Podman

Uses [`github.com/podman-container-tools/container-libs`](https://github.com/podman-container-tools/container-libs) — the same library Podman uses internally.

```
cargo build --release --no-default-features --features podman
```

