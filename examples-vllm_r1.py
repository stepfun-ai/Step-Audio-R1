"""
Step-Audio-R1 Examples with vLLM

This script demonstrates various audio understanding capabilities of Step-Audio-R1:
- Audio Understanding (MMAU, MMSU)
- Math Audio Question Answering
- Audio Reasoning (MMAR)
- Wild Speech Processing
- Universal Audio Caption
- Song Appreciation
- Speaker Trait Inference
"""

from stepaudior1vllm import StepAudioR1


# ============================================================================
# Audio Understanding Tasks
# ============================================================================

def mmau_test(model):
    """Test multi-modal audio understanding with multiple choice questions."""
    question = "Which of the following best describes the male vocal in the audio?"
    choices = ["Soft and melodic", "Aggressive and talking", "High-pitched and singing", "Whispering"]
    
    # 构建完整的问题文本
    question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content: \n"
    for i, choice in enumerate(choices):
        question_text += f"{chr(65+i)}. {choice}\n"
    
    messages = [
        {"role": "human", "content": [
            {"type": "text", "text": question_text},
            {"type": "audio", "audio": "assets/mmau_test.wav"}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=32000, temperature=0.7, repetition_penalty=1.0, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


def mmsu_test(model):
    """Test multi-modal sound understanding for non-verbal sounds."""
    question = "What type of non-verbal sound is in the audio?"
    choices = ["laugh", "burp", "cough", "yawn"]
    
    # 构建完整的问题文本（与 mmsu_inferencer.py 格式一致）
    question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content: \n"
    for i, choice in enumerate(choices):
        question_text += f"{chr(65+i)}. {choice}\n"
    
    messages = [
        {"role": "human", "content": [
            {"type": "text", "text": question_text},
            {"type": "audio", "audio": "assets/mmsu_test.wav"}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=32000, temperature=0.7, repetition_penalty=1.0, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


# ============================================================================
# Math Audio Question Answering
# ============================================================================

def spoken_mqa_test(model):
    """Test mathematical reasoning with spoken audio questions."""
    question = "Solve the given math question step by step."
    
    # 与 spokenmqa_inferencer.py 格式一致：先文本，后音频
    messages = [
        {"role": "human", "content": [
            {"type": "text", "text": question},
            {"type": "audio", "audio": "assets/spoken_mqa_test.wav"}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=32000, temperature=0.7, repetition_penalty=1.0, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


def big_bench_audio_test(model):
    """Test comprehensive audio understanding with BigBench Audio tasks."""
    instruction = '仅用 "valid" 或 "invalid"，"yes" 或 "no"，或者数字来直接回答上面的问题，不要添加任何其他描述。'
    
    # 与 big_bench_inferencer.py 格式一致：先音频，后文本
    messages = [
        {"role": "human", "content": [
            {"type": "audio", "audio": "assets/big_bench_audio_test.wav"},
            {"type": "text", "text": instruction}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=32000, temperature=0.7, repetition_penalty=1.0, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


# ============================================================================
# Audio Reasoning Tasks
# ============================================================================

def mmar_test(model):
    """Test multi-modal audio reasoning with contextual understanding."""
    question = "Is the first little girl sincerely praising the other for being kind?"
    choices = ["Yes", "No"]
    
    # 构建完整的问题文本（与 mmar_inferencer.py 格式一致）
    question_text = f"{question}\nPlease choose the answer from the following options, do not provide any additional explanations or content: \n"
    for i, choice in enumerate(choices):
        question_text += f"{chr(65+i)}. {choice}\n"
    
    messages = [
        {"role": "human", "content": [
            {"type": "text", "text": question_text},
            {"type": "audio", "audio": "assets/mmar_test.wav"}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=32000, temperature=0.7, repetition_penalty=1.07, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


# ============================================================================
# Wild Speech Processing
# ============================================================================

def wild_speech_test(model):
    """Test automatic speech recognition in challenging acoustic conditions."""
    messages = [
        {"role": "human", "content": [
            {"type": "audio", "audio": "assets/wild_speech_test.wav"}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=32000, temperature=0.7, repetition_penalty=1.0, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


# ============================================================================
# Universal Audio Caption
# ============================================================================

def uac_test(model):
    """Test universal audio caption generation with detailed analysis."""
    messages = [
        {"role": "system", "content": "你是一位经验丰富的音频分析专家，擅长对各种语音音频进行深入细致的分析。你的任务不仅仅是将音频内容准确转写为文字，还要对说话人的声音特征（如性别、年龄、情绪状态）、背景声音、环境信息以及可能涉及的事件进行全面描述。请以专业、客观的视角，详细、准确地完成每一次分析和转写。"},
        {"role": "human", "content": [{"type": "audio", "audio": "assets/music_playing_followed_by_a_woman_speaking.wav"}]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]
    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=1024, temperature=0.5, top_p=0.9, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    print("\n\nFull response:", full_text)


# ============================================================================
# Music and Song Analysis
# ============================================================================

def song_appreciation(model):
    """Test song appreciation and music analysis capabilities."""
    messages = [
        {"role": "system", "content": "你是一个语音助手，你有非常丰富的音频处理经验。"},
        {"role": "human", "content": [
            {"type": "text", "text": "鉴赏一下这段歌声。"},
            {"type": "audio", "audio": "assets/song.wav"},
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]
    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=2048, temperature=0.7, top_p=0.9, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    print("\n\nFull response:", full_text)


# ============================================================================
# Speaker Analysis
# ============================================================================

def Speaker_Trait_Inference(model):
    """Test speaker trait inference from voice characteristics."""
    messages = [
        {"role": "system", "content": "你是一个语音助手，你有非常丰富的音频处理经验。"},
        {"role": "human", "content": [
            {"type": "text", "text": "说话人的语气和音色如何反映他的性格和情绪特征？"},
            {"type": "audio", "audio": "assets/Speaker_Trait_Inference.wav"},
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]
    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=2048, temperature=0.7, top_p=0.9, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    print("\n\nFull response:", full_text)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    # Initialize the model with API configuration
    api_url = "http://localhost:9999/v1/chat/completions"
    model_name = "Step-Audio-R1"
    
    model = StepAudioR1(api_url, model_name)
    
    # Run all test cases
    print("=" * 80)
    print("Running Step-Audio-R1 Test Suite")
    print("=" * 80)
    
    # Music and Creative Analysis
    song_appreciation(model)
    Speaker_Trait_Inference(model)
    
    # Universal Audio Caption
    uac_test(model)
    
    # Math and Reasoning Tasks
    spoken_mqa_test(model)
    
    # Audio Understanding Tasks
    mmau_test(model)
    mmsu_test(model)
    big_bench_audio_test(model)
    
    # Audio Reasoning
    mmar_test(model)
    
    # Wild Speech Processing
    wild_speech_test(model)
    
    print("=" * 80)
    print("Test Suite Completed")
    print("=" * 80)

