import whisper
import gradio as gr
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from datetime import datetime

model = whisper.load_model("small")  # Cargar modelo una sola vez

# Variables globales para grabación
is_recording = False
audio_data = []
sample_rate = 16000

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
        if is_recording:
            audio_data.extend(indata.copy())

    sd.InputStream(callback=callback, samplerate=sample_rate, channels=1).start()
    return "Grabación iniciada"

def detener_grabacion():
    global is_recording
    is_recording = False
    print("Grabación detenida")

    # Guardar archivo temporal
    filename = f"grabacion_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    audio_array = np.array(audio_data, dtype=np.float32)
    write(filename, sample_rate, audio_array)
    return filename, "Grabación guardada: " + filename

def reproducir_audio(audio_file):
    if audio_file:
        try:
            data, fs = sd.read(audio_file)
            sd.play(data, fs)
            sd.wait()
            return "Reproducción completada"
        except Exception as e:
            return f"Error al reproducir: {str(e)}"
    return "Seleccione un archivo primero"

with gr.Blocks(title="Transcripción de Audio", theme="soft") as demo:
    gr.Markdown("# 🎤 Transcripción de Audio con Whisper")

    with gr.Row():
        with gr.Column():
            archivo_audio = gr.File(label="Seleccionar archivo de audio", type="filepath")
            btn_transcribir = gr.Button("Transcribir archivo")

            with gr.Row():
                btn_grabar = gr.Button("🎤 Iniciar grabación")
                btn_detener = gr.Button("⏹ Detener grabación")
                btn_reproducir = gr.Button("▶ Reproducir último audio")

            audio_guardado = gr.Textbox(label="Última grabación", interactive=False)
            mensajes = gr.Textbox(label="Mensajes del sistema", interactive=False)

        with gr.Column():
            transcripcion = gr.Textbox(label="Transcripción", lines=10, max_lines=20)

    # Eventos
    btn_grabar.click(
        grabar_audio,
        outputs=mensajes
    )

    btn_detener.click(
        detener_grabacion,
        outputs=[audio_guardado, mensajes]
    )

    btn_reproducir.click(
        reproducir_audio,
        inputs=audio_guardado,
        outputs=mensajes
    )

    btn_transcribir.click(
        lambda archivo: transcribir_audio(archivo) if archivo else "Seleccione un archivo primero",
        inputs=archivo_audio,
        outputs=transcripcion
    )

if __name__ == "__main__":
    demo.launch()
