import os
import pandas as pd
from collections import defaultdict

# Ruta al directorio raíz donde están las subcarpetas
root_directory = '/home/elural/EXIST-TFM/logs/T4'

# Ruta al directorio de destino donde se guardarán los archivos combinados
output_directory = os.path.join(root_directory, 'test')

# Crear la carpeta 'test/' si no existe
os.makedirs(output_directory, exist_ok=True)

# Crear un diccionario para almacenar los DataFrames combinados por nombre común
dataframes = defaultdict(list)

# Iterar sobre las subcarpetas en el directorio principal
for subdir, _, files in os.walk(root_directory):
    for filename in files:
        # Filtrar solo los archivos JSON con el formato deseado
        if filename.startswith('task') and filename.endswith(('.json')) and '_EUA_' in filename:
            # Verificar que el formato del archivo es el adecuado
            parts = filename.split('_')
            if len(parts) == 4 and parts[3].startswith(('hard.json', 'soft.json')):
                # Leer el archivo JSON utilizando pandas
                file_path = os.path.join(subdir, filename)
                try:
                    df = pd.read_json(file_path)

                    # Extraer el nombre común del archivo
                    common_name = '_'.join(parts[:4])

                    # Agregar el DataFrame a la lista correspondiente en el diccionario
                    dataframes[common_name].append(df)
                except ValueError as e:
                    print(f"Error al leer el archivo {file_path}: {e}")

# Combinar los DataFrames y guardarlos en la carpeta de salida
for common_name, dfs in dataframes.items():
    # Combinar todos los DataFrames en uno solo
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Guardar el DataFrame combinado en un archivo JSON
    output_path = os.path.join(output_directory, f"{common_name}")
    combined_df.to_json(output_path, orient='records')
    print(f"Guardado: {output_path}")

print("Proceso completado.")
