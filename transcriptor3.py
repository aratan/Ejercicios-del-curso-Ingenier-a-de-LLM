import whisper
import gradio as gr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from datetime import datetime
import ollama
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

model = whisper.load_model("small")  # Cargar modelo una sola vez

# Variables globales para grabación
is_recording = False
audio_data = []
sample_rate = 16000

client = ollama.Client(host='http://localhost:11434')

def summarize_text(text):
    try:
        response = client.generate(
            model="falcon3:10b",
            prompt=f"Resume las ideas principales del siguiente texto: {text}",
        )
        return response['response'].strip()
    except Exception as e:
        return f"Error al conectar con Ollama: {e}"

def transcribir_audio(ruta_audio):
    try:
        resultado = model.transcribe(ruta_audio, language="es")
        return resultado["text"]
    except Exception as e:
        return f"Error: {str(e)}"

def grabar_audio():
    global is_recording, audio_data
    is_recording = True
    audio_data = []
    print("Grabando...")
    def callback(indata, frames, time, status):
        global audio_data
        if is_recording:
            audio_data.extend(indata.copy())

    stream = sd.InputStream(callback=callback, samplerate=sample_rate, channels=1)
    stream.start()
    return "Grabación iniciada"

def detener_grabacion():
    global is_recording, audio_data
    is_recording = False
    print("Grabación detenida")

    # Guardar archivo temporal
    if audio_data:
        filename = f"grabacion_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        audio_array = np.array(audio_data, dtype=np.float32)
        write(filename, sample_rate, audio_array)
        audio_data = [] # Limpiar audio_data
        return filename, "Grabación guardada: " + filename
    else:
       return "", "No se grabó ningún audio"

def process_transcription(transcription):
    if isinstance(transcription, str):  # Verificar que no sea un error
        if transcription.startswith("Error:"): # si es un error, no procesa el resumen
          return transcription, ""
        else:
          summary = summarize_text(transcription)
          return transcription, summary
    else:
        return "Error al procesar la transcripción", ""


with gr.Blocks(title="Transcripción de Audio", theme="soft") as demo:
    gr.Markdown("# 🎤 Transcripción de Audio con Whisper y Ollama")

    with gr.Row():
        with gr.Column():
            archivo_audio = gr.File(label="Seleccionar archivo de audio", type="filepath")
            btn_transcribir = gr.Button("Transcribir archivo")

            with gr.Row():
                btn_grabar = gr.Button("🎤 Iniciar grabación")
                btn_detener = gr.Button("⏹ Detener grabación")

            audio_guardado = gr.Textbox(label="Última grabación", interactive=False)
            mensajes = gr.Textbox(label="Mensajes del sistema", interactive=False)

        with gr.Column():
            transcripcion = gr.Textbox(label="Transcripción", lines=10, max_lines=20)
            resumen = gr.Textbox(label="Resumen", lines=5, max_lines=10)

    # Eventos
    btn_grabar.click(
        grabar_audio,
        outputs=mensajes
    )

    btn_detener.click(
        detener_grabacion,
        outputs=[audio_guardado, mensajes]
    )


    btn_transcribir.click(
        transcribir_audio,
        inputs=archivo_audio,
        outputs=transcripcion
    ).then(
        process_transcription,
        inputs=transcripcion,
        outputs=[transcripcion, resumen]
    )

if __name__ == "__main__":
    demo.launch()
