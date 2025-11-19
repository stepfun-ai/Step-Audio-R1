from stepaudior1vllm import StepAudioR1
# from token2wav import Token2wav



# Audio understanding
def mmau_test(model):
    messages = [
        {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
        {"role": "human", "content": [
            {"type": "audio", "audio": "assets/mmau_test.wav"},
            {"type": "text", "text": f"Which of the following best describes the male vocal in the audio? Please choose the answer from the following options: [Soft and melodic, Aggressive and talking, High-pitched and singing, Whispering] Output the final answer in <RESPONSE> </RESPONSE>."}
        ]},
        {"role": "assistant", "content": "<think>\n", "eot": False},
    ]

    full_text = ""
    try:
        for response, text, audio in model.stream(messages, max_tokens=1024, temperature=0.7, repetition_penalty=1.07, stop_token_ids=[151665]):
            if text:
                full_text += text
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nFull response:", full_text)


# Universal audio caption
def uac_test(model):
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

def song_appreciation(model):
    messages = [
        {"role": "system", "content": "你是一个语音助手，你有非常丰富的音频处理经验。"},
        {"role": "human", "content": [
            {"type": "audio", "audio": "assets/song.wav"},
            {"type": "text", "text": "鉴赏一下这段歌声。"}
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

def Speaker_Trait_Inference(model):
    messages = [
        {"role": "system", "content": "你是一个语音助手，你有非常丰富的音频处理经验。"},
        {"role": "human", "content": [
            {"type": "audio", "audio": "assets/Speaker_Trait_Inference.wav"},
            {"type": "text", "text": "说话人的语气和音色如何反映他的性格和情绪特征？"}
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

if __name__ == '__main__':
    # 修改为新的API URL和模型名称
    api_url = "http://localhost:9999/v1/chat/completions"
    model_name = "step-audio-2-r1"
    
    model = StepAudioR1(api_url, model_name)
    
    
    mmau_test(model)
    song_appreciation(model)
    Speaker_Trait_Inference(model)
    uac_test(model)
    
   
