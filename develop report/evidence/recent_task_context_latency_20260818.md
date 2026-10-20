# Recent Task Context and Latency Audit

## Scope

- Hermes session: `20260818_162840_0dc04f`
- Task: `go to find the bed in bedroom`
- User input: 16:29:10.512
- Final response: 16:30:22.312
- Final model call: 26,248 input tokens, 96 output tokens
- Total across 10 model calls: 159,941 input tokens, 2,760 output tokens

The 159,941 value is cumulative billing input across repeated calls. The terminal's 26.2k value is the final single-call context window.

## Final 26.2k Context

Approximate additive partition, aligned to the provider's actual per-call usage:

| Component | Tokens | Share |
| --- | ---: | ---: |
| System/core tool framing and user instruction | 4,597 | 17.5% |
| Begin result text plus initial three images | 8,378 | 31.9% |
| Tool discovery, schema, invalid begin retry, and setup history | 2,986 | 11.4% |
| Latest navigation decision/result and its eight-image packet | 9,736 | 37.1% |
| Finish call/result | 551 | 2.1% |

The system prompt alone is 3,838 tokens. The begin JSON is 7,333 tokens plus 136 attachment-label tokens. Its largest fields are the full route graph (2,882 tokens) and memory view (2,595 tokens). The initial LEFT/FRONT/RIGHT images add 909 tokens.

The latest navigation result contains 6,012 JSON tokens plus 629 image-label tokens. Its largest fields are memory view (2,836), trajectory memory (792), topology update (455), memory delta (324), and semantic route plan (268). The latest packet contains one topdown image, four trajectory keyframes, and three current views. At the live Qwen endpoint, each 640x480 RGB image costs 303 tokens and the 658x512 topdown image costs 339, for 2,460 tokens. Together with the three persistent initial images, all 11 images cost about 3,369 tokens, or 12.8% of the final context.

Hermes replaces the preceding navigation decision/result pair with the latest pair: model input stayed near 24.7k-26.2k rather than growing by another 8 images every step. The fixed prefix remains large. Runtime task memory is cleared by finish, but the Hermes chat history is not immediately reset. The following task initially entered at 26,335 tokens and its begin call peaked at 32,891, then the first advance pruning reduced the next call to 24,365.

## Latency

| Link | Time |
| --- | ---: |
| User instruction to first physical control frame | 19.84 s |
| Full task wall time | 71.80 s |
| Physical Habitat control spans | 34.66 s |
| Non-control user-visible overhead | 37.14 s |
| All 10 Agent model calls | 31.50 s |
| Initial observe/begin tool | 0.43 s |
| Post-control observe/package average | 0.95 s |
| Qwen anchor grounding, one call | about 1.1-1.3 s |
| Finish tool | 0.03 s |

Agent call latency sequence: 0.6, 0.4, 0.5, 1.6, 0.5, 15.2, 4.7, 4.4, 2.3, and 1.3 seconds. The 15.2-second first navigation decision is anomalous: the stream produced three unrepairable tool-call argument fragments before a valid call completed. Stable post-subtask Agent reassessment took 4.7 and 4.4 seconds (4.55-second average). Finish verification took 2.3 seconds.

Excluding physical control, subtask transition latency was:

- First execution end to second execution start: 5.71 s (post-observe/package about 0.77 s, Agent 4.7 s, dispatch about 0.24 s).
- Second execution end to anchor execution start: 7.32 s (post-observe/package about 1.49 s, Agent 4.4 s, anchor grounding about 1.1-1.3 s, dispatch remainder).
- Final execution end to user-visible completion: 4.25 s (post-observe/package about 0.58 s, finish verifier 2.3 s, finish tool 0.03 s, final response 1.3 s).

## Main Findings

1. Structured memory and route data, not historical-image accumulation, are the largest avoidable context cost. Begin repeats five relevant places and all nine graph nodes; latest advance repeats five relevant places plus memory/topology deltas.
2. Tool discovery is unnecessarily expensive. `tool_describe` alone contributes 2,019 persistent tokens, and discovery/argument correction triggers five model calls before the first valid navigation decision.
3. Initial latency is dominated by malformed tool-call generation, not observation or Habitat. The 15.2-second outlier accounts for most of the 19.84-second startup-to-motion delay.
4. Keeping initial views and four latest keyframes is affordable relative to structured JSON. Text compaction should be prioritized before reducing visual evidence.
