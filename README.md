# Step-Audio-R1
<p align="center">
  <img src="assets/logo.png"  height=100>
</p>

<div align="center">
    <a href="https://stepaudiollm.github.io/step-audio-r1/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-R1"><img src="https://img.shields.io/static/v1?label=Step-Audio-R1&message=HuggingFace&color=yellow"></a> &ensp;
    <a href="https://modelscope.cn/models/stepfun-ai/Step-Audio-R1"><img src="https://img.shields.io/static/v1?label=Step-Audio-R1&message=ModelScope&color=blue"></a> &ensp;
  <a href="https://huggingface.co/spaces/stepfun-ai/Step-Audio-R1"><img src="https://img.shields.io/static/v1?label=Space%20Playground&message=HuggingFace&color=yellow"></a> &ensp;
</div>

## 🔥🔥🔥 News!!
* Nov 19, 2025: 🎉 We release **Step-Audio-R1**, including the [Demo Page](https://stepaudiollm.github.io/step-audio-r1/) and model weights ([HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-R1)).
* Nov 19, 2025: 📄 We release the technical report of [Step-Audio-R1]().

## 📑 Open-source Plan
- [ ] Inference Code (vLLM)
- [ ] Online demo (Gradio)
- [ ] Model Checkpoints

## Overview

### Introduction
Step-Audio-R1 is the **first audio language model to successfully unlock test-time compute scaling**. It decisively solves the "inverted scaling" anomaly plaguing existing models, where performance paradoxically degrades with longer reasoning chains.

We identify the root cause of this failure as **Textual Surrogate Reasoning**: conventional models, due to text-based initialization, rely on linguistic abstractions (analyzing transcripts) rather than genuine acoustic properties. To resolve this modality mismatch, we introduce **Modality-Grounded Reasoning Distillation (MGRD)**, an iterative training framework that shifts the model's reasoning focus from textual surrogates to acoustic analysis.

This new approach allows us to create **Step-Audio-R1**, which:
* Is the **first audio reasoning model** that successfully benefits from test-time compute scaling.
* **Surpasses Gemini 2.5 Pro and is comparable to Gemini 3** across comprehensive audio benchmarks.
* Transforms extended deliberation from a liability into a **powerful asset** for audio intelligence.

<p align="center">
    <img src="assets/Benchmark.jpg" width="80%"/>
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

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

<br>
