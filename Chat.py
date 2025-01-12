import gradio as gr
import ollama

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
modelos_adicionales = ['falcon3:10b', 'phi4']  # Asegúrate de usar los nombres exactos

# Combina las listas, eliminando duplicados
all_models = sorted(list(set(available_models_ollama + modelos_adicionales)))

def obtener_respuesta(message, history, model_name):
    """Obtiene la respuesta del modelo seleccionado usando la librería ollama."""
    try:
        messages = [{"role": "system", "content": "Tu eres un experto cocinero"}]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})

        response_stream = client.chat(model=model_name, messages=messages, stream=True)
        respuesta_completa = ""
        for chunk in response_stream:
            respuesta_completa += chunk['message']['content']
            yield respuesta_completa

    except Exception as e:
        yield f"Ocurrió un error: {e}"

def process_message(message, history, model_name):
    full_response = ""
    for response in obtener_respuesta(message, history, model_name):
        full_response = response
        yield (history + [(message, full_response)])

with gr.Blocks() as iface:
    gr.Markdown("# Chat con Modelos LLM (Ollama)")
    model_selection = gr.Dropdown(choices=all_models, label="Selecciona un modelo", value=all_models[0] if all_models else None)
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(process_message, [msg, chatbot, model_selection], chatbot)

if __name__ == "__main__":
    iface.launch(share=False)
