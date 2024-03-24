from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["audio_samples/sc_short.mp3"]

transcriptions = model.transcribe(audio_paths)

print(transcriptions[0]['transcription'])
