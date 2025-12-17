import gradio as gr
import speech_recognition as sr
import numpy as np
import os
import tempfile
import scipy.io.wavfile as wavfile
import librosa
import asyncio
import edge_tts

from groq import Groq

def init_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

client = init_groq_client()





# ========== Feature Extraction ===========
def _extract_voice_features(y, sr, transcript):
    
    duration = len(y) / sr
    words = transcript.split()

    # ===== Volume range =====
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms)
    volume_mean = np.mean(rms_db)
    volume_std = np.std(rms_db)

    volume_range = {
        'mean': volume_mean,
        'std': volume_std
    }

    # ===== Tone & pitch =====
    pitches = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    valid_pitches = pitches[~np.isnan(pitches)]

    if len(valid_pitches) > 0:
        pitch_mean = np.mean(valid_pitches)
        pitch_std = np.std(valid_pitches)
        pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
    else:
        pitch_mean = pitch_std = pitch_range = 0

    tone_pitch = {
        'mean': pitch_mean,
        'std': pitch_std,
        'range': pitch_range
    }

    # ===== Pace =====
    speech_rate = (len(words) / duration) * 60 if duration > 0 else 0

    pace = {
        'speech_rate': speech_rate,
        'word_count': len(words),
        'duration': duration
    }

    # ===== Pause detection =====
    silence_threshold = np.percentile(rms, 20)
    hop_length = 512
    frames_per_second = sr / hop_length

    is_silence = rms < silence_threshold
    pause_count = 0
    pause_durations = []
    current_pause_length = 0

    for silent in is_silence:
        if silent:
            current_pause_length += 1
        else:
            if current_pause_length > frames_per_second * 0.2:  # >200ms silence counts as pause
                pause_duration = current_pause_length / frames_per_second
                pause_durations.append(pause_duration)
                pause_count += 1
            current_pause_length = 0

    pause = {
        'count': pause_count,
        'average_duration': np.mean(pause_durations) if pause_durations else 0,
        'durations': pause_durations
    }

    # ===== Emphasis (relative volume per word) =====
    frames_per_word = len(rms) / len(words) if len(words) > 0 else 0
    word_emphasis = []

    if frames_per_word > 0 and len(words) > 0:
        for i, word in enumerate(words):
            start_frame = int(i * frames_per_word)
            end_frame = int((i + 1) * frames_per_word)
            end_frame = min(end_frame, len(rms))

            if start_frame < end_frame:
                word_volume = np.mean(rms_db[start_frame:end_frame])
                word_emphasis.append({
                    'word': word,
                    'relative_volume': word_volume - volume_mean,
                    'position': i
                })

    emphasis = word_emphasis

    # ===== Filler words detection =====
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally', 'well', 'right', 'okay']
    transcript_lower = transcript.lower()

    detected_fillers = [fw for fw in filler_words if fw in transcript_lower]

    fillers = detected_fillers

    # ===== Return all features =====
    return {
        'transcript': transcript,
        'volume_range': volume_range,
        'tone_pitch': tone_pitch,
        'pace': pace,
        'pause': pause,
        'emphasis': emphasis,
        'fillers': fillers,
        'rms': rms,
        'pitches': pitches
    }


def emphasize_transcript_words(transcript, emphasis, threshold_db=4.0):
    words = transcript.split()
    emphasized = {e['position']: e for e in emphasis if e['relative_volume'] >= threshold_db}
    
    new_words = []
    for i, word in enumerate(words):
        if i in emphasized:
            new_words.append(f"**{word}**")  
        else:
            new_words.append(word)
    
    return " ".join(new_words)






# ========== LLM Prompt Builder ===========
def build_prompt(features):

    transcript = features['transcript'].lower().strip()
    chat_history = features.get('chat_history', []) # Get chat_history safely, default to empty list

    if (transcript.startswith('follow up') or
            transcript.startswith('follow-up') or
            transcript.startswith('followup')):
        prompt = f"""
            You are a vocal delivery coach with over 10 years of experience as a TV news and weather presenter.
            A speaker has asked a follow-up question: {features['transcript']}. Please provide a concise, helpful answer.
            Your goal is to help the speaker improve their vocal delivery.

            Conversation so far: {chat_history}.

            1. If they request SPECIFIC advice/exercise (e.g., "that emphasis exercise again"): 
            - Provide what they asked for.
            2. If they ask for NEW suggestions (e.g., "another exercise"):
            - Give something different from previous sessions.
            
            Be creative in a way that helps vocal improvement.
            Keep your answer under 200 tokens. Do NOT analyze their vocal delivery for this question.
            Your response will be read out loud, so ensure your answer has **natural pauses** and **clear phrasing** for TTS recognition.
            In case the question is not clear, politely ask for clarification. (e.g., Can you please clarify your question? etc)
        """
    else:

        emphasis_summary = ", ".join(
            f"{e['word']}({e['relative_volume']:+.1f}dB)" for e in features['emphasis'][:5]
        ) + ("..." if len(features['emphasis']) > 5 else "")


        previous_bot_reply = ""
        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                previous_bot_reply = msg["content"]
                break

        if previous_bot_reply:
            prompt = f"""
                You are a vocal delivery coach with over 10 years of experience as a TV news and weather presenter.
                A speaker has shared a transcript and a set of vocal delivery features (e.g., volume range, tone, pitch, pace, pause, emphasis, filler words, etc.). 
                Your job is to give feedback on how they spoke (Note: not what they said) and goal is to help the speaker improve their vocal delivery.
                Keep your answer under 300 tokens.

                PREVIOUS FEEDBACK:
                \"\"\"{previous_bot_reply}\"\"\"
                
                TRANSCRIPT:
                \"{features['transcript']}\"

                VOICE FEATURES:
                Duration: {features['pace']['duration']:.2f} sec
                Words: {features['pace']['word_count']}
                Pace: {features['pace']['speech_rate']:.1f} words/min
                Pitch (mean / std / range): {features['tone_pitch']['mean']:.1f} / {features['tone_pitch']['std']:.1f} / {features['tone_pitch']['range']:.1f} Hz
                Volume (mean / std): {features['volume_range']['mean']:.1f} / {features['volume_range']['std']:.1f} dB
                Pauses: {features['pause']['count']} total, avg {features['pause']['average_duration']:.2f} sec
                Filler words: {', '.join(features['fillers']) if features['fillers'] else 'None'}
                Word emphasis: {emphasis_summary}
                
                
                🎯 Your task is to give feedback on the follow-up performance:
                Acknowledge the effort of the speaker based on the previous feedback.
                Provide NEW, refined insights or improvements.
                If there is still scope of improvement in the same area, point it out politely and suggest a micro-exercise with a new phrase(s) unrelated to their transcript.
                Your response will be read out loud, so ensure your answer has **natural pauses** and **clear intonation** for TTS.
                
                ✅ Keep tone: Encouraging, supportive, non-technical (no technical or statistical terms).
                ❌ Avoid: Commenting on content, language, word choice, ideas or grammar.
                """
        
        else:

            prompt = f"""
                You are a vocal delivery coach with over 10 years of experience as a TV news and weather presenter.
                A speaker has shared a transcript and a set of vocal delivery features (e.g., volume range, tone, pitch, pace, pause, emphasis, filler words, etc.). 
                Your job is to give feedback on how they spoke (Note: not what they said) and goal is to help the speaker improve their vocal delivery.
                Keep your answer under 450 tokens.

                TRANSCRIPT:
                \"{features['transcript']}\"

                VOICE FEATURES:
                Duration: {features['pace']['duration']:.2f} sec
                Words: {features['pace']['word_count']}
                Pace: {features['pace']['speech_rate']:.1f} words/min
                Pitch (mean / std / range): {features['tone_pitch']['mean']:.1f} / {features['tone_pitch']['std']:.1f} / {features['tone_pitch']['range']:.1f} Hz
                Volume (mean / std): {features['volume_range']['mean']:.1f} / {features['volume_range']['std']:.1f} dB
                Pauses: {features['pause']['count']} total, avg {features['pause']['average_duration']:.2f} sec
                Filler words: {', '.join(features['fillers']) if features['fillers'] else 'None'}
                Word emphasis: {emphasis_summary}
                
                
                🎯 Your task is to give feedback on:
                One strength in their vocal delivery (e.g., volume range, tone, pitch, pace, pause, emphasis, filler words, etc.) and in case of areas to improve give only important 1-2 areas to improve.
                Use specific examples from their speech. **Where applicable**, give them exact phrases like "Instead of saying '...' try saying '...'/"emphasize...", etc., not a generic example.
                Instead of generic advice like "record yourself and listen back," give the speaker a CREATIVE micro-exercise that involves speaking aloud with a specific phrase or applying a specific vocal technique (e.g., exaggerating pauses, varying pitch, or stretching syllables).
                For exercises, always suggest NEW phrases unrelated to their transcript.
                Your response will be read out loud, so ensure your answer has **natural pauses** and **clear phrasing** for TTS recognition.
                
                ✅ Keep tone: Encouraging, supportive, non-technical (no technical or statistical terms).
                ❌ Avoid: Commenting on content, language, word choice, ideas or grammar.
                """
    return prompt.strip()







# ========== LLM API call ===========
def call_llm(prompt):
    print("\n\nPROMPT:", prompt)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=300
    )
    print("\n\nRESPONSE", response)

    reply = response.choices[0].message.content # Accessing content correctly

    return {"role": "assistant", "content": reply}






# ========== Edge TTS Async Wrapper ===========
async def generate_tts_edge_tts(text, voice="en-US-EmmaNeural"): #en-US-JennyNeural
    communicate = edge_tts.Communicate(text, voice)
    # Use a different tempfile for TTS output that Gradio can manage
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    await communicate.save(wav_path)
    return wav_path

def run_tts_sync(text):
    return asyncio.run(generate_tts_edge_tts(text))

def tts_audio_from_wav(wav_path):
    data, sr = librosa.load(wav_path, sr=44100)  # force resample to 44100
    return data.astype(np.float32), sr # Return (data, samplerate) for Gradio gr.Audio(type="numpy")







# ========== Gradio processing function ===========
def transcribe_analyze_and_respond(audio_tuple, chat_history):
    if chat_history is None:
        chat_history = []

    MAX_MESSAGES = 6
    if len(chat_history) >= MAX_MESSAGES:
        return chat_history, chat_history, None, None, gr.update(visible=True)

    if audio_tuple is None:
        # Pass current chat_history, not the empty list, for correct display
        return chat_history, chat_history, None, None, gr.update(visible=False)

    sr_audio, audio_np = audio_tuple

    # Convert audio to int16 if it's float, as speech_recognition expects int16
    if audio_np.dtype != np.int16:
        audio_np = (audio_np * 32767).astype(np.int16)

    # Temporary file for Speech Recognition input
    tmp_wav_for_stt_path = None 
    # Variable to hold the path for TTS output (managed by Edge TTS and Gradio)
    wav_path_for_tts = None 

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_for_stt_path = tmp_wav.name

        wavfile.write(tmp_wav_for_stt_path, sr_audio, audio_np)

        r = sr.Recognizer()
        with sr.AudioFile(tmp_wav_for_stt_path) as source:
            audio_data = r.record(source)
        
        # Recognize speech
        try:
            transcript = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            chat_history.append({"role": "assistant", "content": "I couldn't understand what you said. Please try recording again with clearer audio."})
            return chat_history, chat_history, None, None, gr.update(visible=False)
        except sr.RequestError as e:
            chat_history.append({"role": "assistant", "content": f"Speech recognition failed: {e}"})
            return chat_history, chat_history, None, None, gr.update(visible=True)
                
        chat_history.append({"role": "user", "content": transcript})

        # Load audio for feature extraction (librosa prefers float)
        y, sr_loaded = librosa.load(tmp_wav_for_stt_path, sr=None)
        features = _extract_voice_features(y, sr_loaded, transcript)
        # features['transcript'] = emphasize_transcript_words(features['transcript'], features['emphasis']) # Keep this commented if not needed
        features['chat_history'] = chat_history
        prompt_text = build_prompt(features)
        
        # Call LLM and get the response dictionary (already in {"role": ..., "content": ...} format)
        llm_response_dict = call_llm(prompt_text)
        
        # Append the dictionary directly to chat_history
        chat_history.append(llm_response_dict)

        try:
            # Pass only the content string to TTS for audio generation
            wav_path_for_tts = run_tts_sync(llm_response_dict['content'])
            # Load the generated TTS audio into numpy format for Gradio output
            tts_audio_data, tts_audio_sr = tts_audio_from_wav(wav_path_for_tts)
            tts_audio = (tts_audio_sr, tts_audio_data) # Tuple for gr.Audio(type="numpy")
        except Exception as e:
            print(f"TTS error: {e}")
            tts_audio = None

        return chat_history, chat_history, None, tts_audio, gr.update(visible=False)

    except Exception as e:
        print(f"Main processing error: {e}") # Print error for debugging
        # Ensure error messages are also in the correct chat message format
        chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})
        return chat_history, chat_history, None, None, gr.update(visible=True)

    finally:
        # Only delete the temporary file used for Speech Recognition.
        # The temporary file generated by Edge TTS (wav_path_for_tts)
        # should ideally be managed by Gradio or explicitly cleaned up later
        # once the audio has been served to the frontend.
        if tmp_wav_for_stt_path and os.path.exists(tmp_wav_for_stt_path):
            os.remove(tmp_wav_for_stt_path)
        # We do NOT delete wav_path_for_tts here, as Gradio needs access to it for playback.


def clear_all():
    return [], [], None, None, gr.update(visible=False)






# ========== Gradio UI ===========
with gr.Blocks(title="SpeakConfident: Vocal Delivery AI Coach") as demo:
    
    with gr.Accordion("ℹ️ Quick Start Guide", open=False):
        gr.Markdown(
        """
**SpeakConfident: Vocal Delivery AI Coach**

1. **Record your voice**
   - *Performance Analysis*: Just record your speech normally  
   - *Follow-up Questions*: Say `"Follow-up"` or `"Follow up Question"` then ask  
     - Example: "Follow-up, what did you mean by emphasis?"

2. **Get feedback on what to improve and how!**
"""
        )

    # Latest chatbot component: "messages" format
    chatbot = gr.Chatbot(height=400, type="messages", elem_id="chatBox", label="Conversation")
    state = gr.State([]) # State to hold the chat history

    max_turns_warning = gr.Markdown(
        """<span style='color: red; font-weight: bold;'>⚠️ Maximum turns reached. Please clear chat to start over.</span>""",
        visible=False
    )    

    with gr.Row():
        # Microphone recording input
        audio_input = gr.Audio(
            sources=["microphone"],
            type="numpy", # Receive audio as (samplerate, numpy_array)
            label="Record here",
            elem_id="audioInputComponent",
            streaming=False, # Process audio after full recording
        )

        # Audio output component for TTS playback
        tts_output = gr.Audio(
            label="Bot Response",
            type="numpy", # Expects (samplerate, numpy_array)
            autoplay=True, # Automatically play the audio
            waveform_options={
                "skip_length": 0,
                "waveform_color": "#667eea",
                "waveform_progress_color": "#764ba2",
            }
        )

    clear_btn = gr.Button("Clear")

    # Event listener for audio input change (when recording stops)
    audio_input.change(
        fn=transcribe_analyze_and_respond,
        inputs=[audio_input, state],
        outputs=[chatbot, state, audio_input, tts_output, max_turns_warning],
    )

    # Event listener for clear button click
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[chatbot, state, audio_input, tts_output, max_turns_warning],
    )

demo.launch()
