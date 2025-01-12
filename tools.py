from datetime import datetime
import ollama

# Definir la función que devuelve la hora actual
def get_current_time() -> str:
    """
    Obtiene la hora actual.

    Returns:
        str: La hora actual en formato 'HH:MM'.
    """
    return datetime.now().strftime("%H:%M")

# Pasar la función como herramienta a Ollama
response = ollama.chat(
    model='qwen2.5',
    messages=[
        {'role': 'system', 'content': 'Responde siempre en español.'},  # Mensaje de sistema
        {'role': 'user', 'content': '¿Es la hora de cenar?'},
    ],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Obtiene la hora actual en formato HH:MM.',
            'parameters': {
                'type': 'object',
                'properties': {},
            },
        },
    }],
)

# Verificar si el modelo llamó a la función
if response['message'].get('tool_calls'):
    for tool in response['message']['tool_calls']:
        if tool['function']['name'] == 'get_current_time':
            # Ejecutar la función para obtener la hora actual
            current_time = get_current_time()
            # print('Hora actual:', current_time)


            # Pasar el resultado de vuelta al modelo para que genere una respuesta
            follow_up_response = ollama.chat(
                model='qwen2.5',
                messages=[
                    {'role': 'system', 'content': f'{current_time}, Responde siempre en español.'},  # Mensaje de sistema
                    {'role': 'user', 'content': '¿Es la hora de cenar?, ceno de 22:00 a 23:00'},
                    {'role': 'assistant', 'content': f'La hora actual es {current_time}. '},
                ],
            )
            print('Respuesta del modelo:', follow_up_response['message']['content'])
        else:
            print('Función no encontrada:', tool['function']['name'])
else:
    # Si no se llama a ninguna función, mostrar la respuesta del modelo
    print('Respuesta del modelo:', response['message']['content'])
