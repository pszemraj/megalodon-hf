# Megalodon Long-Context Streaming (Conceptual Flow)

This note sketches how "unlimited context" is achieved without holding a global KV cache. The long-range signal flows through **stateful EMA + TimestepNorm**, while attention uses a **local KV** (chunk-local by default; optional sliding window).

Assume `chunk_size = 1024` and a 17,000-token sequence (17 chunks). Attention stays chunk-local by default; EMA/Norm carry the full history. A sliding KV window is an optional extension.

## 1) Chunked Attention vs. Stateful Memory

```mermaid
flowchart LR
    subgraph Chunk1["Chunk 1 (t=1-1024)"]
        A1[Tokens 1-1024] -->|TimestepNorm| TN1
        TN1 -->|CEMA| C1[CEMA output]
        C1 -->|RMSNorm| MX1[mx]
        MX1 -->|Z/Q/K| Z1
        TN1 -->|V| V1
        MX1 -->|Gate/Input| G1
        Z1 -->|"Local attn (KV: chunk 1 window)"| O1
        O1 -->|Gate+Proj| Y1
    end

    subgraph Chunk2["Chunk 2 (t=1025-2048)"]
        A2[Tokens 1025-2048] -->|"TimestepNorm (running stats)"| TN2
        TN2 -->|"CEMA (init = h_prev)"| C2[CEMA output]
        C2 -->|RMSNorm| MX2[mx]
        MX2 -->|Z/Q/K| Z2
        TN2 -->|V| V2
        MX2 -->|Gate/Input| G2
        Z2 -->|"Local attn (KV: chunk-local)"| O2
        O2 -->|Gate+Proj| Y2
    end

    subgraph ChunkN["Chunk N (t > 2048)"]
        AN[Tokens ...] -->|"TimestepNorm (running stats)"| TNN
        TNN -->|"CEMA (init = h_prev)"| CN[CEMA output]
        CN -->|RMSNorm| MXN[mx]
        MXN -->|Z/Q/K| ZN
        TNN -->|V| VN
        MXN -->|Gate/Input| GN
        ZN -->|"Local attn (KV: chunk-local or sliding window)"| ON
        ON -->|Gate+Proj| YN
    end

    C1 -.-> C2
    C2 -.-> CN
    TN1 -.-> TN2
    TN2 -.-> TNN
```

- **Long-range path:** CEMA state `h` + TimestepNorm running stats propagate across all chunks (O(1) memory). This is the "unlimited" context carrier.
- **Local path:** Attention uses only a local KV window (within chunk or sliding window). Older KV can be dropped once their effect is absorbed into EMA/Norm state.

## 2) Optional Sliding KV Window (decode)

```mermaid
sequenceDiagram
    participant Cache as KV Cache (window)
    participant State as EMA + TN state
    participant Block as Attn Block
    Note over Cache: default max_cache_len = chunk_size (chunk-local)<br/>max_cache_len = W => keep last W<br/>cache_unbounded = True => keep all

    Loop for each chunk
        Block->>State: TimestepNorm (update running mean/var)
        State-->>Block: stats for this chunk
        Block->>State: CEMA (init with h_prev)
        State-->>Block: h_new
        Block->>Cache: append K,V (after RoPE)
        Cache-->>Block: (optionally) drop oldest if |KV|>W
        Block->>Block: Local SDPA over windowed KV
        Block-->>State: store h_new (for next chunk)
    end
```

- Default behavior clamps KV to one chunk (`max_cache_len = chunk_size`); chunk-local streaming calls can span boundaries and are processed chunk-by-chunk with cache reset at each boundary.
- A finite `max_cache_len` above the chunk size enables a sliding window.
- Setting `cache_unbounded=True` keeps all KV (VRAM grows linearly).

## 3) RoPE Offsets

```mermaid
flowchart TD
    subgraph RoPE
        P0[Position counter] -->|start_index for chunk| R1["RoPE apply(Q,K)"]
        R1 --> O[Attention]
        O --> P1[Update position counter]
    end
```

- We track absolute positions so rotary phases remain continuous across chunks, even when KV is trimmed.

## 4) Training vs. Inference

- **Training:** block-diagonal attention per chunk; EMA uses FFT (no cache).
- **Inference:** sequential EMA; attention is chunk-local by default with optional sliding/unbounded KV; RoPE offset advances with absolute position.

## Defaults and Options

- **Upstream reference:** trims KV to one chunk; enforces `cache_len + seq_len <= chunk_size`.
- **Paper spirit:** "unlimited" via EMA + stateful norms; KV need not be global.
- **This repo:** default `max_cache_len = chunk_size` (faithful, chunk-local). Set `max_cache_len` above `chunk_size` for sliding-window attention; use `cache_unbounded=True` to disable clamping.
