# LLM X-Ray — Full Project Deep Dive

## 1. What it is and why it would go viral

### The concept

LLM X-Ray is an interactive web tool that lets anyone type a prompt and **watch** a language model process it in real-time — visualizing attention patterns, token probabilities, hidden state transformations, and layer-by-layer predictions as beautiful, animated graphics.

Think of it as a medical MRI scan, but for AI. You type "The cat sat on the" and watch:

- **Attention heatmaps** light up showing which tokens are paying attention to which
- **Token probability bars** animate as the model decides the next word
- **Layer-by-layer predictions** reveal how the model's "guess" evolves from nonsense in early layers to coherent language in later ones (the "logit lens" effect)
- **A flowing attention river** connects tokens with lines whose thickness represents attention weight

The user can click any token, scrub through layers, toggle attention heads, and generate a **shareable screenshot card** of the visualization.

### The one-sentence pitch

> "Type a prompt. Watch an AI think. Share the X-ray."

### Why it would go viral

**The gap is real.** After deep research, here's what exists and what's missing:

| Existing tool | What it does | What's missing |
|---|---|---|
| bbycroft/llm-viz | 3D GPT-2 visualization, very technical | Focused on architecture education, not YOUR prompt. No shareable output. |
| Transformer Explainer (Georgia Tech) | Interactive GPT-2 in browser, 560k+ users, Svelte+D3 | Educational focus for non-experts. No beautiful shareable cards. No "wow" factor. |
| AnimatedLLM | Academic tool for non-technical audiences | Research paper project, not a consumer product. |
| BertViz / TransformerLens | Python libraries for attention analysis | Code-only, no web UI, for researchers. |
| Logit Lens / Tuned Lens | Project intermediate hidden states | Command-line tools, no visualization. |

**The gap:** Nobody has built a *beautiful, prompt-driven, shareable* LLM visualization tool. Existing tools are either too academic, too technical, or too ugly. LLM X-Ray would be the first tool that combines:

1. **YOUR prompt** — not a pre-loaded demo, your actual text
2. **Beautiful animations** — not academic charts, but genuinely stunning data art
3. **Shareable output** — a screenshot card optimized for LinkedIn/X
4. **Progressive disclosure** — simple at first glance, deep on exploration

**Viral mechanics:**

- **Ego + curiosity** — "What does the AI see when *I* type this?"
- **Visual artifact** — every session produces a beautiful image worth sharing
- **Debate fuel** — "Look how GPT attends to 'not' in this sentence" sparks conversation
- **Education angle** — teachers, students, journalists all want to explain AI
- **LinkedIn bait** — "I built a tool that lets you see inside an LLM" with a gorgeous screenshot

**Proof this category works:** Transformer Explainer hit 150,000 users in its first 3 months and 560,000+ total — and it's an academic project with no marketing. A beautifully designed version with shareable output would massively outperform it.

---

## 2. Full tech stack

### Backend (Python)

| Component | Technology | Why |
|---|---|---|
| Web framework | **FastAPI** | Async support, WebSocket native, your comfort zone |
| Model inference | **HuggingFace Transformers** | Direct access to model internals via PyTorch hooks |
| Model | **GPT-2 Small (124M params)** | Runs on CPU, well-documented internals, architecture representative of larger models |
| Hook engine | **PyTorch forward hooks** | Register hooks on every attention layer and FFN to capture intermediate outputs |
| Logit lens | **Custom implementation** | Project hidden states through unembedding matrix at each layer |
| Serialization | **orjson** | 10x faster JSON serialization than stdlib, critical for streaming tensor data |
| Task queue | **asyncio** (built-in) | Inference is fast enough on GPT-2 to run inline, no Celery needed |
| Server | **Uvicorn** | ASGI server, WebSocket support |

### Frontend

| Component | Technology | Why |
|---|---|---|
| Framework | **React** (Vite) | Component-based, great for interactive UIs |
| Attention visualization | **D3.js** | The gold standard for custom data visualization |
| 3D layer view (optional) | **Three.js** or **React Three Fiber** | For the optional 3D layer-scrubbing view |
| Animations | **Framer Motion** | Smooth transitions between states |
| Shareable cards | **html-to-image** | Convert visualization state to PNG for sharing |
| Styling | **Tailwind CSS** | Fast, utility-first styling |
| WebSocket client | **Native WebSocket API** | No library needed for this |

### Infrastructure

| Component | Technology | Why |
|---|---|---|
| Deployment (backend) | **Railway** or **Render** | Free tier, easy Python deployment |
| Deployment (frontend) | **Vercel** | Free, instant deploys, great DX |
| Domain | **llm-xray.com** or similar | Check availability |
| Analytics | **Plausible** (self-hosted) or **Umami** | Privacy-first, no cookie banner needed |

### Key Python packages

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
transformers>=4.46.0
torch>=2.5.0
orjson>=3.10.0
websockets>=14.0
numpy>=2.0.0
Pillow>=11.0.0  # For server-side card generation
```

---

## 3. Architecture

### Data flow

```
User types prompt
    │
    ▼
[React Frontend] ──WebSocket──▶ [FastAPI Backend]
                                      │
                                      ▼
                               [Tokenizer] → tokens + IDs
                                      │
                                      ▼
                               [GPT-2 with hooks registered]
                                      │
                                      ├──▶ Layer 0: attention weights, hidden state, logit lens prediction
                                      ├──▶ Layer 1: attention weights, hidden state, logit lens prediction
                                      ├──▶ ...
                                      └──▶ Layer 11: attention weights, hidden state, final logits
                                      │
                                      ▼
                               [Serializer] → JSON chunks
                                      │
                                      ▼
                               Streams back via WebSocket
                                      │
                                      ▼
                               [React] renders each layer incrementally
                                      │
                                      ▼
                               User sees animation build up layer by layer
```

### Hook engine (the core innovation)

The heart of LLM X-Ray is a PyTorch hook engine that captures model internals during inference:

```python
class XRayHookEngine:
    """Registers forward hooks on every layer of a transformer model
    to capture attention weights, hidden states, and intermediate predictions."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.captures = {}

    def register_hooks(self):
        for i, layer in enumerate(self.model.transformer.h):
            # Capture attention weights
            layer.attn.register_forward_hook(
                self._make_attention_hook(i)
            )
            # Capture hidden states after each layer
            layer.register_forward_hook(
                self._make_hidden_state_hook(i)
            )

    def _make_attention_hook(self, layer_idx):
        def hook(module, input, output):
            # output[1] contains attention weights: (batch, heads, seq, seq)
            attn_weights = output[1].detach().cpu().numpy()
            self.captures[f"layer_{layer_idx}_attention"] = attn_weights
        return hook

    def _make_hidden_state_hook(self, layer_idx):
        def hook(module, input, output):
            hidden = output[0].detach()
            # Logit lens: project hidden state through unembedding
            logits = self.model.lm_head(self.model.transformer.ln_f(hidden))
            top_tokens = logits[0, -1].topk(10)
            self.captures[f"layer_{layer_idx}_predictions"] = {
                "top_tokens": [self.tokenizer.decode(t) for t in top_tokens.indices],
                "top_probs": torch.softmax(top_tokens.values, dim=0).tolist()
            }
        return hook
```

### WebSocket streaming protocol

Each inference run streams a sequence of JSON messages:

```json
// Message 1: Tokenization result
{
  "type": "tokens",
  "data": {
    "tokens": ["The", " cat", " sat", " on", " the"],
    "token_ids": [464, 3797, 3332, 319, 262]
  }
}

// Messages 2-13: One per layer (GPT-2 small has 12 layers)
{
  "type": "layer",
  "data": {
    "layer_idx": 0,
    "attention": [[0.2, 0.1, ...], ...],  // (heads × seq × seq)
    "top_predictions": [
      {"token": " mat", "prob": 0.15},
      {"token": " floor", "prob": 0.12},
      {"token": " table", "prob": 0.09}
    ]
  }
}

// Final message: Complete output
{
  "type": "output",
  "data": {
    "predicted_token": " mat",
    "probability": 0.23,
    "all_probs": [...]
  }
}
```

### Frontend visualization components

| Component | What it shows | Interaction |
|---|---|---|
| **Token bar** | Input tokens as colored chips | Click any token to see its attention pattern |
| **Attention heatmap** | Matrix of attention weights | Hover to highlight connections, toggle heads |
| **Attention river** | Flowing lines between tokens, thickness = weight | Animate on layer change |
| **Layer slider** | Scrub through 12 layers | Watch predictions evolve from noise → coherent |
| **Logit lens panel** | Top-5 predictions at each layer | Shows "what the model is thinking" at each step |
| **Probability waterfall** | Final token probabilities as animated bars | Sorts by probability, shows top 20 |

---

## 4. Week-by-week roadmap (3 weeks)

### Week 1: Backend + core inference pipeline

**Day 1-2: Project setup + model loading**
- Initialize FastAPI project with proper structure
- Set up GPT-2 loading with HuggingFace Transformers
- Implement basic `/api/generate` REST endpoint
- Write the tokenization pipeline
- Test: send a prompt, get tokens back

**Day 3-4: Hook engine implementation**
- Build the `XRayHookEngine` class
- Register forward hooks for attention weights extraction
- Register forward hooks for hidden state extraction
- Implement logit lens (project hidden states through unembedding matrix)
- Test: run inference, verify captures contain valid attention matrices and predictions
- Optimize: ensure hooks don't significantly slow inference

**Day 5-7: WebSocket streaming**
- Implement WebSocket endpoint `/ws/xray`
- Build the streaming protocol (tokens → layers → output)
- Serialize numpy arrays efficiently with orjson
- Add connection management (handle disconnects, timeouts)
- Test with a simple WebSocket client (wscat or browser console)
- Benchmark: full inference + serialization should complete in under 2 seconds

**Week 1 deliverable:** A working FastAPI backend that accepts a prompt via WebSocket and streams back tokenization, attention weights per layer, logit lens predictions per layer, and final output probabilities.

### Week 2: Frontend + core visualizations

**Day 8-9: React project setup + WebSocket integration**
- Initialize React project with Vite
- Build the WebSocket client hook (`useXRay`)
- Create the prompt input component (clean, minimal)
- Wire up: type prompt → send via WS → receive streamed data → store in state
- Build the token bar component (colored chips showing input tokens)

**Day 10-12: Core visualizations with D3**
- **Attention heatmap**: D3 matrix visualization with color intensity = weight. Click a token to highlight its row/column. Toggle between attention heads.
- **Attention river**: D3 curved paths between token chips, stroke-width = attention weight, opacity animated on layer transitions. This is the hero visual — make it beautiful.
- **Layer slider**: Range input that scrubs through 12 layers. Each layer change triggers smooth D3 transitions on the heatmap and river.
- **Logit lens panel**: Animated bar chart showing top-5 predictions at the current layer. Watch predictions shift from random early layers to confident final layers.

**Day 13-14: Polish + responsive layout**
- Layout: split-screen (prompt + tokens on left, visualizations on right) or stacked on mobile
- Color scheme: dark mode default (visualizations pop on dark backgrounds)
- Animations: Framer Motion page transitions, D3 data transitions
- Loading states: skeleton UI while model loads, progress while inference runs
- Mobile: simplified view with tab switching between visualizations

**Week 2 deliverable:** A fully functional web app where you type a prompt, watch attention patterns animate layer by layer, scrub through layers, click tokens to explore attention, and see predictions evolve.

### Week 3: Share features + launch prep

**Day 15-16: Shareable output (THE viral feature)**
- **Screenshot card generator**: Capture current visualization state as a beautiful PNG
- Card design: dark background, your prompt as headline, attention heatmap as hero image, "Generated with LLM X-Ray" watermark, URL
- **OG meta tags**: When someone shares a link, auto-generate an OG image from their last prompt
- **Embed snippet**: Generate an `<iframe>` embed code so bloggers can embed an X-ray in their posts
- **Example gallery**: Pre-computed X-rays of interesting prompts ("The meaning of life is", "To be or not to be")

**Day 17-18: Landing page + SEO**
- Hero section: animated demo showing an X-ray in action (pre-recorded or live)
- "Try it now" CTA above the fold
- Example gallery section
- "How it works" section (3-step: type → watch → share)
- Footer: GitHub link, tech stack badges, your name + links
- SEO: title, description, OG tags, structured data

**Day 19-20: Testing + deployment**
- Cross-browser testing (Chrome, Firefox, Safari, mobile Safari)
- Performance: lazy-load D3, code-split visualizations
- Deploy backend to Railway/Render
- Deploy frontend to Vercel
- Set up domain + HTTPS
- Write README.md with architecture diagram, screenshots, GIF demo
- Record a 30-second demo GIF for the README and social posts

**Day 21: Pre-launch prep**
- Write launch posts (see Section 5 below)
- Prepare Product Hunt listing (schedule for Tuesday/Wednesday)
- Draft Hacker News "Show HN" post
- Final bug fixes

**Week 3 deliverable:** A polished, deployed web app with shareable cards, a landing page, and all launch materials ready.

---

## 5. Launch strategy for maximum visibility

### Pre-launch (1 week before)

- **Teaser posts**: Share 2-3 WIP screenshots on X and LinkedIn showing the visualization in progress. Use the "building in public" angle. Tag it #buildinpublic.
- **Teaser GIF**: Record a 10-second GIF of the attention river animating. Post with "What if you could watch an AI think? Coming soon."
- **DM outreach**: Identify 10-15 AI/ML influencers on X who post about LLM internals (Andrej Karpathy, Jay Alammar, Chip Huyen, etc.). Don't ask them to share — just show them the tool and ask for feedback. If they like it, they'll share organically.

### Launch day

**Hacker News (Show HN)** — this is your #1 channel:
- Title: "Show HN: LLM X-Ray — Watch what happens inside GPT when you type a prompt"
- Post at 8-9am ET (peak HN traffic)
- Include a direct link to the live tool
- In the text, explain what it does in 2 sentences, what tech you used, and what you learned building it. HN loves technical depth.

**Reddit** — hit multiple subreddits on the same day:
- r/MachineLearning — "I built an interactive tool that visualizes GPT-2 attention patterns, logit lens predictions, and token probabilities in real-time"
- r/Python — focus on the FastAPI + PyTorch hooks angle
- r/dataisbeautiful — post a screenshot of the attention river visualization
- r/artificial — focus on the "demystifying AI" angle

**X (Twitter)** — thread format:
- Tweet 1: "I built a tool that lets you see inside an AI's brain." + hero screenshot
- Tweet 2: "Type any prompt. Watch attention patterns light up. See what the model predicts at every layer." + GIF
- Tweet 3: "The tech: FastAPI + PyTorch hooks + D3.js. Here's how it works:" + architecture diagram
- Tweet 4: "Try it now: [link]. It's free, open source, and runs GPT-2 in real-time."
- Tag: @AnthropicAI @HuggingFace @kaborymera

**LinkedIn** — long-form post:
- Headline: "I built a tool that lets you watch an AI think. Here's what I learned."
- Structure: problem → solution → technical insight → link
- Include 3-4 screenshots showing different visualizations
- End with "Try it yourself: [link]"

**Product Hunt** — launch on a Tuesday or Wednesday:
- Title: "LLM X-Ray — See inside the AI black box"
- Tagline: "Type a prompt. Watch an AI think. Share the X-ray."
- 4 screenshots + 1 GIF
- Maker comment explaining the motivation

### Post-launch (week after)

- **Blog post**: Write a technical deep-dive on dev.to or your own blog about how PyTorch hooks work and how you built the logit lens visualization. This becomes evergreen content that drives traffic.
- **YouTube/Loom video**: 3-5 minute walkthrough showing the tool in action with interesting prompts. Submit to AI-focused YouTube channels.
- **Respond to every comment**: On HN, Reddit, and X. Engagement drives algorithmic visibility.
- **Feature requests**: Track what people ask for. Common asks will likely be: support for other models, compare two models side-by-side, save and share specific X-rays. Build the most-requested feature within a week and announce "v1.1".

### The shareable artifact strategy

The core viral loop is: **user creates an X-ray → shares the screenshot → someone sees it → visits the tool → creates their own X-ray → shares it.**

To maximize this:
- Make the share button prominent and frictionless (one click → clipboard + download)
- The card must look stunning without any context — it should make someone stop scrolling
- Include the URL on the card so people who see the screenshot know where to go
- Pre-generate X-rays of famous prompts and share them yourself to seed the loop

---

## Quick reference: competitive positioning

| | LLM X-Ray (you) | Transformer Explainer | bbycroft/llm-viz |
|---|---|---|---|
| **Target audience** | Everyone curious about AI | Students, educators | Technical learners |
| **Model** | GPT-2 (expandable) | GPT-2 | GPT-2 (minGPT) |
| **Your prompt?** | Yes | Yes | No (pre-loaded demo) |
| **Visualization style** | Beautiful, animated, shareable | Educational, clean | 3D technical |
| **Shareable output** | Yes (cards, embeds) | No | No |
| **Logit lens** | Yes | No | No |
| **Attention river** | Yes (animated) | Sankey-style | 3D matrix |
| **Stack** | Python + React | Svelte + JS | TypeScript + WebGL |
| **Wow factor** | High (designed for screenshots) | Medium | High (3D) |

---

## File structure

```
llm-xray/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app, CORS, WebSocket endpoint
│   │   ├── xray_engine.py       # Hook engine, inference, logit lens
│   │   ├── serializer.py        # Tensor → JSON streaming
│   │   ├── models.py            # Pydantic schemas
│   │   └── card_generator.py    # Server-side PNG card generation
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── hooks/
│   │   │   └── useXRay.js       # WebSocket client hook
│   │   ├── components/
│   │   │   ├── PromptInput.jsx
│   │   │   ├── TokenBar.jsx
│   │   │   ├── AttentionHeatmap.jsx
│   │   │   ├── AttentionRiver.jsx
│   │   │   ├── LayerSlider.jsx
│   │   │   ├── LogitLensPanel.jsx
│   │   │   ├── ProbabilityBars.jsx
│   │   │   └── ShareCard.jsx
│   │   └── utils/
│   │       ├── colors.js        # Attention weight → color mapping
│   │       └── export.js        # html-to-image wrapper
│   ├── package.json
│   └── vite.config.js
├── examples/                    # Pre-computed X-rays for the gallery
├── docs/
│   └── architecture.png
├── LICENSE
└── README.md
```

---

## Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| GPT-2 feels outdated to users | Medium | Frame as "same architecture as GPT-4, just smaller." Add model comparison feature in v2. |
| Backend inference too slow | Low | GPT-2 small runs in ~200ms on CPU. Use async + streaming to feel instant. |
| Too similar to Transformer Explainer | Medium | Differentiate on beauty, shareability, and logit lens. Your angle is "art," theirs is "education." |
| Frontend too complex for 3 weeks | Medium | Prioritize: attention river + heatmap + layer slider first. Logit lens and 3D view are stretch goals. |
| Low engagement after launch | Low | The shareable card mechanic creates organic growth. Double down on interesting pre-generated examples. |
