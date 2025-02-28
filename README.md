<div align="center">

<img src="./assets/logo.png" width="300em" ></img> 

## **Open-Source End-to-End Speech Interaction Foundation Model**

  <strong>[中文](./README_zh.md)|
  English </strong>
  
  <p align="center">
  Baichuan-Audio <a href="https://huggingface.co/baichuan-inc/Baichuan-Audio-Instruct">🤗</a> | Baichuan-Audio-Base <a href="https://huggingface.co/baichuan-inc/Baichuan-Audio-Base">🤗</a>  | Technical Report <a href="https://arxiv.org/abs/2502.17239">📖</a> 
</p>
</p>
<p align="center">
OpenAudioBench <a href="https://huggingface.co/datasets/baichuan-inc/openAudioBench">🤗</a>  | Training Data <a href="#">🤗</a> <small>(Coming Soon)</small>
</p>

  <!-- <p align="center">
    OpenMM-Medical <a href="https://huggingface.co/datasets/baichuan-inc/OpenMM_Medical">🤗</a> | OpenAudioBench <a href="https://huggingface.co/datasets/baichuan-inc/OpenAudioBench">🤗</a> 
</p> -->
</div>

## Baichuan-Audio

**Baichuan-Audio** is an open-source end-to-end speech interaction model, seamlessly integrating audio understanding and generation capabilities, supporting high-quality, controllable real-time bilingual (Chinese-English) dialogue.

- **Baichuan-Audio-Base**: To promote the development of speech models, we have open-sourced an end-to-end speech foundation model trained with high-quality, extensive data. This model has not undergone SFT instruction fine-tuning, offering strong plasticity.

- **Baichuan-Audio**: This model accepts text and audio as input, generating high-quality text and audio output, capable of **seamless high-quality speech interaction while maintaining the intelligence of pre-trained LLMs, enabling real-time voice dialogue with users**.

- Additionally, we have open-sourced an audio understanding and generation benchmark (OpenAudio-Bench) to evaluate end-to-end audio capabilities. Furthermore, pre-training data will also be open-sourced soon.


<br>

### Model Architecture

<div align="center">
<img src="./assets/audiollm.png" , width=85%>
</div>
<br>

**Baichuan-Audio** mainly consists of Baichuan-Audio Tokenizer, Audio LLM, and Flow-matching based Audio Decoder. First, speech is converted into discrete audio tokens by the Baichuan-Audio Tokenizer. Then, Audio LLM generates aligned text and audio tokens in an interleaved manner, achieving seamless modality switching between text and audio through special tokens. Audio tokens are processed by an independent audio head and reconstructed into high-quality Mel spectrograms using a flow-matching based audio decoder, which are then converted into audio waveforms via a vocoder.

- Baichuan-Audio-Tokenizer uses a 12.5hz frame rate design. It employs Whisper Large Encoder to extract high-level audio features from Mel spectrograms, then uses 8-layer RVQ to minimize information loss during quantization. To capture both semantic and acoustic information, we use Mel spectrogram reconstruction and Pre-trained LLM for acoustic and semantic supervision, respectively.
<div align="center">
<img src="./assets/vq.png" , width=30%>
</div>

- Audio LLM generates aligned text and audio tokens in an interleaved manner, achieving seamless switching between text and audio modalities through special tokens. Audio tokens are processed by an independent audio head.

- Flow-matching based Audio Decoder is used to reconstruct high-quality Mel spectrograms. The model is trained on 24 kHz audio to generate target Mel spectrograms, which are then converted into audio waveforms via a vocoder.

<div align="center">
<img src="./assets/decoder.png" , width=24%>
</div>


### Pre-training details
- #### Pre-training data
Audio training data can be broadly divided into two main types: audio understanding data and audio generation data.
<div align="center">
<img src="./assets/table.png" , width=80%>
</div>

Audio-text paired data (e.g., ASR and TTS data) improves performance on basic speech tasks. On the other hand, pure audio data enhances the ability to handle audio modalities independently. Audio-Text Interleaved data consists of alternating text and audio modalities, segmented by punctuation to facilitate cross-modal knowledge transfer. Interleaved Text-to-Speech data consists of fully aligned text and audio content, aimed at enhancing the model's ability to generate audio tokens under text supervision.

The interleaved data collection process is divided into crawling and synthesis types, resulting in a total of 142k hours of ITTS data and 393k hours of INTLV data.
<div align="center">
<img src="./assets/data.png" , width=80%>
</div>

<br>

- #### Two stage training strategy
The conflict between speech and text modalities may interfere with the pre-trained text knowledge representation in pre-trained LLMs, leading to degradation in model intelligence performance. To mitigate this, we adopt a two-stage training strategy. In the first stage, the LLM parameters remain fixed, and only the audio embedding layer and audio head parameters are updated. In the second stage, all parameters except the LM embedding layer and LM head parameters are trained.


### Local WebUI Demo

#### Preparation

##### Create a Virtual Environment
```bash
conda create -n baichuan_omni python==3.12
conda activate baichuan_omni
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install accelerate flash_attn==2.6.3 speechbrain==1.0.0 deepspeed==0.14.4
apt install llvm ffmpeg
```
##### Download the Model and Modify the Model Path
Modify MODEL_PATH in web_demo/constants.py to the local model path.
#### ASR and TTS Demo

```bash
cd web_demo
python base_asr_demo.py
python base_tts_demo.py
```
#### Speech interaction Demo

```bash
cd web_demo
python s2s_gradio_demo_cosy_multiturn.py
```

### Cases

Below is an example of audio input and output:

| **Input Type** | **Input Content**   | **Output Type** | **Output Content**                                                                                   |
|----------------|---------------------|-----------------|------------------------------------------------------------------------------------------------------|
| Audio          | "介绍下北京" | Audio           | [Audio Output](https://raw.githubusercontent.com/baichuan-inc/Baichuan-Audio/refs/heads/main/assets/audio_out.wav) |

### Open-Source Evaluation Set

**OpenAudioBench**

To more efficiently evaluate the model's "intelligence," we have constructed OpenAudioBench, which includes 5 sub-evaluation sets for end-to-end audio understanding. These include 4 public evaluation sets (llama question, WEB QA, TriviaQA, AlpacaEval) and a speech logical reasoning evaluation set built by the Baichuan team, totaling 2701 data points. This comprehensive set reflects the model's "intelligence" level.

### Model performance
<div align="center">
<img src="./assets/result.png" , width=90%>
</div>


### Acknowledgments

- Automatic Speech Recognition (ASR) Model: 【Whisper】(https://github.com/openai/whisper)
- Large Language Model (LLM): 【Qwen2.5 7B】(https://arxiv.org/abs/2412.15115)
- Partial code from: CosyVoice and Matcha-TTS: (https://github.com/FunAudioLLM/CosyVoice, https://github.com/shivammehta25/Matcha-TTS/)
- HiFi-GAN Vocoder from CosyVoice 2.0: (https://funaudiollm.github.io/cosyvoice2/)


### License
The use of Baichuan-Audio-Base/Baichuan-Audio model weights must comply with the [Apache 2.0](https://github.com/baichuan-inc/Baichuan-Audio/blob/main/LICENSE)

### Citation

If you find our model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@article{li2025baichuan,
  title={Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction},
  author={Li, Tianpeng and Liu, Jun and Zhang, Tao and Fang, Yuanbo and Pan, Da and Wang, Mingrui and Liang, Zheng and Li, Zehuan and Lin, Mingan and Dong, Guosheng and others},
  journal={arXiv preprint arXiv:2502.17239},
  year={2025}
}
```


