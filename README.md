# Hebrew-LLMs

A curated collection of Hebrew language AI models available on Hugging Face as of April 17, 2025. This repository aims to provide a good starting point for anyone looking to work with Hebrew language AI models.

## About Hebrew Language and AI

Hebrew is a Semitic language with approximately 9 million native speakers worldwide, primarily in Israel. Despite its relatively small speaker base, Hebrew presents several interesting characteristics for AI research:

- **Modern vs Biblical Hebrew**: There are significant differences between Modern Hebrew and Biblical Hebrew, with specialized models developed for biblical text analysis.

- **Punctuation Challenges**: Modern written Hebrew typically lacks extensive punctuation, creating a need for specialized models that can infer and add appropriate punctuation.

- **Technological Hub**: Israel is a renowned center for technology and AI research, making Hebrew language AI models particularly interesting from an experimental and innovation perspective.

- **Rich Linguistic Structure**: Hebrew's non-Latin script, right-to-left writing system, and complex morphology present unique challenges for language models.

These factors make Hebrew language AI development both challenging and valuable, with applications ranging from biblical text analysis to modern NLP tasks.

## Table of Contents
- [Hebrew-LLMs](#hebrew-llms)
  - [About Hebrew Language and AI](#about-hebrew-language-and-ai)
  - [Table of Contents](#table-of-contents)
  - [Large Language Models (LLMs)](#large-language-models-llms)
    - [Mistral Fine-tunes](#mistral-fine-tunes)
    - [Mixtral Fine-tunes](#mixtral-fine-tunes)
    - [Gemma/Google Fine-tunes](#gemmagoogle-fine-tunes)
  - [Niche Text Models](#niche-text-models)
    - [Summarization](#summarization)
    - [Biblical](#biblical)
    - [Metaphor Detection](#metaphor-detection)
    - [Translation](#translation)
    - [Offensive Language](#offensive-language)
    - [Punctuation](#punctuation)
    - [Sentiment Analysis](#sentiment-analysis)
  - [Specialized Language Models](#specialized-language-models)
    - [Hebrew to SQL](#hebrew-to-sql)
    - [Hebrew Medical Terms (NER)](#hebrew-medical-terms-ner)
  - [ASR Models (Speech Recognition)](#asr-models-speech-recognition)
  - [TTS Models (Text-to-Speech)](#tts-models-text-to-speech)
  - [Benchmarks and Leaderboards](#benchmarks-and-leaderboards)
    - [Leaderboard Insights](#leaderboard-insights)
  - [Organizations to Follow](#organizations-to-follow)
  - [Other Interesting Projects](#other-interesting-projects)
  - [Additional Links](#additional-links)
  - [Worthy Follow](#worthy-follow)
  - [Reading](#reading)
  - [Resources](#resources)

## Large Language Models (LLMs)

### Mistral Fine-tunes

| Model | Link |
|-------|------|
| Hebrew-Mistral-7B | [![Hebrew-Mistral-7B](https://img.shields.io/badge/🤗-Hebrew--Mistral--7B-yellow)](https://huggingface.co/yam-peleg/Hebrew-Mistral-7B) |
| Hebrew-Mistral-7B-200K | [![Hebrew-Mistral-7B-200K](https://img.shields.io/badge/🤗-Hebrew--Mistral--7B--200K-yellow)](https://huggingface.co/yam-peleg/Hebrew-Mistral-7B-200K) |
| Hebrew-Mistral-7B_Chat-GGUF | [![Hebrew-Mistral-7B_Chat-GGUF](https://img.shields.io/badge/🤗-Hebrew--Mistral--7B__Chat--GGUF-yellow)](https://huggingface.co/mradermacher/Hebrew-Mistral-7B_Chat-GGUF) |
| Hebrew-Mistral-7B-Instruct-v0.1-GGUF | [![Hebrew-Mistral-7B-Instruct-v0.1-GGUF](https://img.shields.io/badge/🤗-Hebrew--Mistral--7B--Instruct--v0.1--GGUF-yellow)](https://huggingface.co/mradermacher/Hebrew-Mistral-7B-Instruct-v0.1-GGUF) |

### Mixtral Fine-tunes

| Model | Link |
|-------|------|
| Hebrew-Mixtral-8x22B | [![Hebrew-Mixtral-8x22B](https://img.shields.io/badge/🤗-Hebrew--Mixtral--8x22B-yellow)](https://huggingface.co/yam-peleg/Hebrew-Mixtral-8x22B) |

### Gemma/Google Fine-tunes

| Model | Link |
|-------|------|
| Hebrew-Gemma-11B | [![Hebrew-Gemma-11B](https://img.shields.io/badge/🤗-Hebrew--Gemma--11B-yellow)](https://huggingface.co/yam-peleg/Hebrew-Gemma-11B) |
| Hebrew-Gemma-11B-Instruct | [![Hebrew-Gemma-11B-Instruct](https://img.shields.io/badge/🤗-Hebrew--Gemma--11B--Instruct-yellow)](https://huggingface.co/yam-peleg/Hebrew-Gemma-11B-Instruct) |
| Hebrew-Gemma-11B-V2-mlx-4bit | [![Hebrew-Gemma-11B-V2-mlx-4bit](https://img.shields.io/badge/🤗-Hebrew--Gemma--11B--V2--mlx--4bit-yellow)](https://huggingface.co/itayl/Hebrew-Gemma-11B-V2-mlx-4bit) |

## Niche Text Models

### Summarization
[![hebrew-summarization-llm](https://img.shields.io/badge/🤗-hebrew--summarization--llm-blue)](https://huggingface.co/maayanorner/hebrew-summarization-llm)

### Biblical
[![hebrew_bible_ai](https://img.shields.io/badge/🤗-hebrew__bible__ai-blue)](https://huggingface.co/tombenj/hebrew_bible_ai)

### Metaphor Detection
[![hebert-finetuned-hebrew-metaphor](https://img.shields.io/badge/🤗-hebert--finetuned--hebrew--metaphor-blue)](https://huggingface.co/tdklab/hebert-finetuned-hebrew-metaphor)

### Translation
[![t5-hebrew-translation](https://img.shields.io/badge/🤗-t5--hebrew--translation-blue)](https://huggingface.co/tejagowda/t5-hebrew-translation)
[![english-hebrew-translation](https://img.shields.io/badge/🤗-english--hebrew--translation-blue)](https://huggingface.co/ashercn97/english-hebrew-translation)

### Offensive Language
[![Offensive-Hebrew](https://img.shields.io/badge/🤗-Offensive--Hebrew-blue)](https://huggingface.co/SinaLab/Offensive-Hebrew)

### Punctuation
[![hebrew_punctuation](https://img.shields.io/badge/🤗-hebrew__punctuation-blue)](https://huggingface.co/verbit/hebrew_punctuation)

### Sentiment Analysis
[![xlm-r_hebrew_sentiment](https://img.shields.io/badge/🤗-xlm--r__hebrew__sentiment-blue)](https://huggingface.co/DGurgurov/xlm-r_hebrew_sentiment)

## Specialized Language Models

### Hebrew to SQL
[![Llama-3.1-8b-Hebrew2SQL](https://img.shields.io/badge/🤗-Llama--3.1--8b--Hebrew2SQL-green)](https://huggingface.co/AryehRotberg/Llama-3.1-8b-Hebrew2SQL)

### Hebrew Medical Terms (NER)
[![hebrew_medical_ner_v5](https://img.shields.io/badge/🤗-hebrew__medical__ner__v5-green)](https://huggingface.co/cp500/hebrew_medical_ner_v5)

## ASR Models (Speech Recognition)

[![wav2vec2-large-xlsr-53-hebrew](https://img.shields.io/badge/🤗-wav2vec2--large--xlsr--53--hebrew-purple)](https://huggingface.co/imvladikon/wav2vec2-large-xlsr-53-hebrew)
[![Whisper_hebrew_medium](https://img.shields.io/badge/🤗-Whisper__hebrew__medium-purple)](https://huggingface.co/Shiry/Whisper_hebrew_medium)

## TTS Models (Text-to-Speech)

*Note: This section will be populated with Hebrew TTS models in the future.*

## Benchmarks and Leaderboards

The Hebrew LLM Leaderboard provides valuable insights into the performance of various models on Hebrew language tasks:

| Resource | Link |
|----------|------|
| Hebrew LLM Leaderboard | [![Hebrew-LLM-Leaderboard](https://img.shields.io/badge/🏆-Hebrew--LLM--Leaderboard-brightgreen)](https://huggingface.co/spaces/hebrew-llm-leaderboard/leaderboard) |
| Hebrew Question Answering Dataset | [![Hebrew-QA-Dataset](https://img.shields.io/badge/📊-Hebrew--QA--Dataset-brightgreen)](https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset) |

### Leaderboard Insights

An interesting observation from the leaderboard is that large multilingual LLMs (like Mistral and Meta-Llama models) generally outperform specialized Hebrew models due to their significantly larger parameter counts. However, specialized Hebrew models still appear on the leaderboard and perform reasonably well considering their size constraints.

The benchmark evaluates models across several categories:
- SNLI (Natural Language Inference)
- QA (Question Answering)
- TLNLS (Text Classification)
- Sentiment Analysis
- Winograd Schema Challenge
- Translation
- Israeli Trivia (a unique category testing cultural and local knowledge)

This comprehensive evaluation provides a holistic view of model capabilities in the Hebrew language context.

## Organizations to Follow

| Organization | Link |
|--------------|------|
| Dicta | [![Dicta](https://img.shields.io/badge/🏢-Dicta-orange)](https://dicta.org.il/) |
| MAFAT (National Natural Language Processing Plan Of Israel) | [![MAFAT](https://img.shields.io/badge/🏢-MAFAT--NNLP--IL-orange)](https://nnlp-il.mafat.ai/) |

## Other Interesting Projects

[![HebrewManuscriptsMNIST](https://img.shields.io/badge/🤗-HebrewManuscriptsMNIST-orange)](https://huggingface.co/bsesic/HebrewManuscriptsMNIST)

## Additional Links

[![Hebrew-Models-Collection](https://img.shields.io/badge/🤗-Hebrew--Models--Collection-red)](https://huggingface.co/collections/yam-peleg/hebrew-models-65e957875324e2b9a4b68f08)

## Worthy Follow

[![yam-peleg](https://img.shields.io/badge/🤗-yam--peleg-red)](https://huggingface.co/yam-peleg)

## Reading

| Resource | Link |
|----------|------|
| Best LLM for Hebrew Classification | [![Best-LLM-for-Hebrew-Classification](https://img.shields.io/badge/📄-Best--LLM--for--Hebrew--Classification-lightgrey)](https://medium.com/@gilinachum/whats-the-best-llm-for-hebrew-classification-58a61b8b9f10) |
| Hebrew LLM Paper | [![Hebrew-LLM-Paper](https://img.shields.io/badge/📄-Hebrew--LLM--Paper-lightgrey)](https://arxiv.org/html/2407.07080v1) |
| Hebrew Model Sentiment Analysis | [![Hebrew-Model-Sentiment-Analysis](https://img.shields.io/badge/📄-Hebrew--Model--Sentiment--Analysis-lightgrey)](https://www.reddit.com/r/LocalLLaMA/comments/1dc8gyf/new_hebrew_model_achieves_highest_sentiment/) |
| Huggingface Hebrew Leaderboard | [![Huggingface-Hebrew-Leaderboard](https://img.shields.io/badge/📄-Huggingface--Hebrew--Leaderboard-lightgrey)](https://github.com/huggingface/blog/blob/main/leaderboard-hebrew.md) |
| Hebrew GPT Neo XL | [![Hebrew-GPT-Neo-XL](https://img.shields.io/badge/📄-Hebrew--GPT--Neo--XL-lightgrey)](https://llm.extractum.io/model/Norod78%2Fhebrew-gpt_neo-xl,40TSQ7PnUPrDfT68oxaK1) |
 
## Resources

- [All Hebrew models on Hugging Face](https://huggingface.co/models?language=he&sort=trending)
- [Hebrew models search on Hugging Face](https://huggingface.co/models?language=he&sort=trending&search=hebrew)
