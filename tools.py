from datetime import datetime
import ollama

# Función para obtener la hora actual
def get_current_time() -> str:
    return datetime.now().strftime("%H:%M")

# Llamada al modelo con tools en properties van los parametros de la funcion si las usa.
response = ollama.chat(
    model='qwen2.5',
    messages=[{'role': 'user', 'content': '¿Es la hora de cenar? Ceno de 22:00 a 23:00.'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Obtiene la hora actual en formato HH:MM.',
            'parameters': {'type': 'object', 'properties': {}},
        },
    }],
)

# Si el modelo llama a la función, obtener la hora y generar respuesta contextualizada.
if response['message'].get('tool_calls'):
    hora_actual = get_current_time()
    respuesta = ollama.chat(
        model='qwen2.5',
        messages=[
            {'role': 'system', 'content': 'Responde siempre en español.'},
            {'role': 'assistant', 'content': f'La hora actual es {hora_actual}'},
            {'role': 'user', 'content': '¿Es la hora de cenar? Ceno de 22:00 a 23:00 '},
        ],
    )
    print('Respuesta del modelo:', respuesta['message']['content'])
else:
    print('Respuesta del modelo:', response['message']['content'])
