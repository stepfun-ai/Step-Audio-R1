# Step-Audio-R1
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-r1/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-R1"><img src="https://img.shields.io/static/v1?label=Step-Audio-R1&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://modelscope.cn/models/stepfun-ai/Step-Audio-R1"><img src="https://img.shields.io/static/v1?label=Step-Audio-R1&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-R1"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* Nov 19, 2025: 📦 We release the **optimized inference code** and **model weights** of **Step-Audio-R1** ([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-R1);  [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-R1)) and **Step-Audio-Tokenizer**([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer);  [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer))
* Nov 19, 2025: ✨ [Demo Page](https://stepaudiollm.github.io/step-audio-r1/) ; 🎮  [HF Space Playground](https://huggingface.co/spaces/stepfun-ai/Step-Audio-R1)
* Nov 19, 2025: 👋 We release the technical report of [Step-Audio-R1](https://arxiv.org/abs/2511.03601).

## Overview
### Introduction

Step-Audio-R1 is the **first audio language model to successfully unlock Chain-of-Thought (CoT) reasoning**. It decisively solves the "inverted scaling" problem that plagues existing models, where performance *degrades* with longer reasoning. Step-Audio-R1 is the first model to demonstrate that for audio, like text and vision, allocating more compute at test-time *predictably improves* performance.

We found the root cause of this anomaly: models were engaging in **textual surrogate reasoning** (analyzing transcripts, not audio) due to a modality mismatch. We solve this with a new training framework, **Modality-Grounded Reasoning Distillation (MGRD)**, that teaches the model "native audio think."

Key features:

* **State-of-the-Art Reasoning:** The first audio model that truly benefits from step-by-step deliberation. Step-Audio-R1 **surpasses Gemini 2.5 Pro** across comprehensive audio benchmarks.
* **Solves Inverted Scaling:** We reverse the audio anomaly. Longer, more deliberate reasoning chains now lead to *better* answers, transforming deliberation from a liability into a powerful asset.
* **Novel MGRD Framework:** We introduce **Modality-Grounded Reasoning Distillation**, an iterative self-distillation method that progressively shifts the model's reasoning from textual abstractions (e.g., "sad lyrics") to acoustic properties (e.g., "minor key progressions").
* **Powerful & Efficient Architecture:** Built on a frozen **Qwen2 audio encoder** and a **Qwen2.5 32B LLM decoder**, our architecture is optimized for deep audio reasoning.

<p align="center">
    <img src="assets/Benchmark.png" width="80%"/>
<p>
  
### Introduction
Chain-of-Thought (CoT) reasoning has transformed AI, enabling models to solve complex problems by "thinking step-by-step." Allocating more computation at inference time—longer reasoning—predictably improves performance in text and vision. In audio, existing models show inverted scaling behavior: performance systematically gets worse as reasoning chains get longer. This has led to a critical question:

**Is audio inherently resistant to deliberate reasoning?**

Our answer is **No**. Through systematic analysis, we found the root cause: models aren't reasoning about *audio*; they're reasoning about *text transcripts*.

* The Problem: Textual Surrogate Reasoning. When asked why music sounds "melancholic," models reason about "lyrics mentioning sadness" (text) instead of "minor key progressions and descending melodic contours" (audio).
* The Cause: These models inherit text-based reasoning from their initialization, creating a fundamental modality mismatch.

To solve this, we introduce **Modality-Grounded Reasoning Distillation (MGRD)**, an iterative training framework that shifts the model's reasoning from textual abstractions to acoustic properties.

This new approach allows us to create **Step-Audio-R1**, which:
* Is the **first audio reasoning model** that successfully benefits from test-time compute scaling.
* **Surpasses Gemini 2.5 Pro** across comprehensive audio benchmarks.
* Transforms extended deliberation from a liability into a **powerful asset** for audio intelligence.

<p align="center">
    <img src="assets/Benchmark.png" width="80%"/>
<p>

### Introduction
Step-Audio-R1 is the **first audio language model to successfully unlock Chain-of-Thought (CoT) reasoning**. It decisively solves the "inverted scaling" problem that plagues existing models, where performance *degrades* with longer reasoning. Step-Audio-R1 is the first model to demonstrate that for audio, like text and vision, allocating more compute at test-time *predictably improves* performance.

We found the root cause of this anomaly: models were engaging in **textual surrogate reasoning** (analyzing transcripts, not audio) due to a modality mismatch. To solve this, we introduce **Modality-Grounded Reasoning Distillation (MGRD)**, an iterative training framework that shifts the model's reasoning from textual abstractions to acoustic properties.

This new approach allows us to create **Step-Audio-R1**, which:
* Is the **first audio reasoning model** that successfully benefits from test-time compute scaling.
* **Surpasses Gemini 2.5 Pro** across comprehensive audio benchmarks.
* Transforms extended deliberation from a liability into a **powerful asset** for audio intelligence.

<p align="center">
    <img src="assets/Benchmark.png" width="80%"/>
<p>


### Model Architecture

<p align="center">
    <img src="assets/overview.png" width="80%"/>
<p>

Step-Audio-R1 builds on the architecture of our previous StepAudio 2 and consists of three main components:

1.  **Audio Encoder:** We use the pre-trained **Qwen2 audio encoder**. It operates at a 25 Hz frame rate and is frozen during training.
2.  **Audio Adaptor:** A simple adaptor (identical to Step-Audio 2) connects the encoder to the LLM and downsamples the feature frame rate to 12.5 Hz.
3.  **LLM Decoder:** We use **Qwen2.5 32B** as the core reasoning component. It directly takes the latent audio features from the adaptor to generate a purely textual output (first the reasoning, then the final reply).

The key innovation is our training method, **Modality-Grounded Reasoning Distillation (MGRD)**. This process iteratively refines the model's thoughts, progressively strengthening their connection to the underlying audio features until they evolve into **"native audio think."**

<p align="center">
    <img src="assets/MGRD.png" width="80%"/>
<p>

This ensures the model's reasoning is not merely about the transcribed text but is deeply grounded in the **acoustic nuances** of the audio itself.

### Cookbooks for Usage Cases

## QuickStart

### Model Description and Download

### Transformers Usage

#### Installation

#### Code Snippet

### vLLM Usage

### Usage Tips (Recommended Reading)

#### Minimum GPU memory requirements

#### Best Practices for the Thinking Model

## Interaction with Qwen3-Omni

### Online Demo

### Real-Time Interaction

Real-time streaming interaction with Step-Audio-R1 is available now. Please visit [Step-Audio-R1 Chat](https) and select the XXX call option in the chat box to experience it.

### Launch Local Web UI Demo

#### Installation

#### Running the Demo

## 🐳 Docker

## Evaluation

### Performance of Step-Audio-R1

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)


```BibTeX
@article{,
  title={},
  author={},
  journal={},
  year={2025}
}
```

<br>
