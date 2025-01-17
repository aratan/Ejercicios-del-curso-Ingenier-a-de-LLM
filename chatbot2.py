import gradio as gr
import ollama
from gtts import gTTS
import os

# Inicializa el cliente de Ollama
client = ollama.Client()

# Obtiene la lista de modelos disponibles desde Ollama
try:
    available_models_data = client.list()
    available_models_ollama = [model['name'] for model in available_models_data['models']]
except Exception as e:
    print(f"Error al obtener la lista de modelos de Ollama: {e}")
    available_models_ollama = []

# Añade manualmente los modelos específicos que deseas asegurar que aparezcan
modelos_adicionales = ['falcon3:10b', 'aratan/phi4-o1','aratan/qwen2.5-14bu','llama3.2-vision:11b']  # Asegúrate de usar los nombres exactos

# Combina las listas, eliminando duplicados
all_models = sorted(list(set(available_models_ollama + modelos_adicionales)))

def obtener_respuesta(message, history, model_name):
    """Obtiene la respuesta del modelo seleccionado usando la librería ollama."""
    try:
        messages = [{"role": "system", "content": """Tu eres un experto cocinero, habla solo de cocina y relacionado, sino sabes algo dilo y pregunta.

Aquí hay algunos ejemplos de cómo debes comportarte:

Usuario: ¿Cuál es la mejor forma de hacer una paella?
Asistente: La mejor forma de hacer paella depende de tus gustos, pero generalmente comienza con un buen sofrito, el arroz adecuado (bomba o senia), un caldo sabroso y los ingredientes principales como pollo, conejo o mariscos. ¿Qué tipo de paella te gustaría preparar?

Usuario: ¿Qué herramientas necesito para hornear un pastel?
Asistente: Para hornear un pastel necesitarás varios utensilios básicos como moldes para pastel, varillas o batidora, espátulas, tazas y cucharas medidoras, y papel de hornear. Algunos ingredientes esenciales son harina, azúcar, huevos y algún tipo de grasa como mantequilla o aceite. ¿Qué tipo de pastel tienes en mente?

Usuario: ¿Qué opinas de la física cuántica?
Asistente: Disculpa, mi especialidad es la cocina y todo lo relacionado con ella. No tengo conocimientos sobre física cuántica. ¿Te gustaría saber algo sobre técnicas de cocción o ingredientes?

Usuario: Recomiéndame un buen vino para acompañar un plato de pasta con salsa de tomate.
Asistente: Para una pasta con salsa de tomate, un vino tinto de cuerpo medio con buena acidez sería ideal. Un Chianti Classico italiano o un Sangiovese serían excelentes opciones. ¿Te gustaría saber más sobre maridaje de vinos y comidas?

Usuario: ¿Cómo se hace la mayonesa casera?
Asistente: Hacer mayonesa casera es relativamente sencillo. Necesitarás un huevo, aceite vegetal (como girasol o canola), zumo de limón o vinagre, y sal. La clave está en emulsionar lentamente el aceite con el huevo. ¿Te gustaría que te diera las proporciones exactas y los pasos detallados?
"""}]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})

        response_stream = client.chat(model=model_name, messages=messages, stream=True)
        respuesta_completa = ""
        for chunk in response_stream:
            respuesta_completa += chunk['message']['content']
        return respuesta_completa

    except Exception as e:
        return f"Ocurrió un error: {e}"

def text_to_speech(text):
    """Convierte texto a voz y guarda el audio en un archivo temporal."""
    tts = gTTS(text=text, lang='es')
    filename = "output.mp3"
    tts.save(filename)
    return filename

def process_message(message, history, model_name):
    response = obtener_respuesta(message, history, model_name)
    audio_file = text_to_speech(response)
    return (history + [(message, response)], audio_file)

with gr.Blocks() as iface:
    gr.Markdown("# Chat con Modelos LLM (Ollama) con TTS")
    model_selection = gr.Dropdown(choices=all_models, label="Selecciona un modelo", value=all_models[0] if all_models else None)
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    audio_output = gr.Audio(label="Respuesta de Voz")
    clear = gr.ClearButton([msg, chatbot, audio_output])

    def respond(message, chat_history, model):
        bot_message, audio_file = process_message(message, chat_history, model)
        chat_history.append((message, bot_message[-1][1])) # Añade solo el último mensaje del bot al historial
        return "", chat_history, audio_file

    msg.submit(respond, [msg, chatbot, model_selection], [msg, chatbot, audio_output])

if __name__ == "__main__":
    iface.launch(share=False)
