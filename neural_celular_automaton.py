# Se importan las bibliotecas necesarias
import cv2
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm

# Funcion para aplicar las reglas de 
# actualizacion al los estados iniciales
def create_animation(frame, kernel, 
                     activation_function, N, 
                     persistent=False):
    # Se crea un arreglo para almacenar 
    # todas las matrices
    frames = np.empty((N,*frame.shape))
    # Se crea la nueva matriz con los nuevos 
    # estados
    for i in tqdm(range(N), 
                  desc='Applying convolutions'):
        frames[i] = frame
        # Se aplica una convolucion de la matriz 
        # con el kernel, 
        # seguido de la funcion de activacion
        frame = convolve2d(frame, kernel, 
                           mode='same', 
                           boundary='wrap')
        frame = activation_function(frame)
        # Se suman los valores anteriores 
        # con los nuevos
        if persistent and i > 0:
            frame = frame + frames[i-1]
        # Se recortan los valores entre 0 y 1
        frame[frame<0] = 0
        frame[frame>1] = 1
    return frames

# Funcion para exportar crear un video de
# la evolucion del automata
def export_video_cv2(frames, name='output.mp4', 
                 fps = 25, skip_frames=False, 
                 skip='even'):
    # Se obtiene el tamano de cada dimension
    duration, height, weight = frames.shape
    if skip_frames:
        if skip == 'even':
            k = 0
        else:
            k = 1 
    out = cv2.VideoWriter(name, 
              cv2.VideoWriter_fourcc(*'mp4v'), 
              fps, (weight, height), False)
    # Se crea el video
    for i in tqdm(range(duration), 
                  desc='Creating video'):
        if skip_frames:
            if i%2 != k:
                data = (frames[i]*255)
                data = data.astype('uint8')
                out.write(data)
        else:
            data = (frames[i]*255)
            data = data.astype('uint8')
            out.write(data)
    out.release()
    

if __name__ == '__main__':
    # Tamano del espacio
    weight = 640
    height = 360
    init_frame = np.zeros((height, weight))
    # Numero de iteraciones
    N = 2000
    # Condiciones iniciales
    init_frame[3*height//4,weight//3] = 1
    init_frame[height//4,2*weight//3] = 1
    init_frame[5*height//6,5*weight//6] = 1
    init_frame[5*height//6+1,5*weight//6] = 1
    init_frame[5*height//6,5*weight//6+1] = 1
    # Se selecciona el kernel
    kernel = np.array(
        [[-0.84899998,0.912,-0.84899998],
         [ 0.912,0,0.912],
         [-0.84899998,0.912,-0.84899998]])
    # Se define la funcion de activacion 
    activation = lambda x: np.sin(np.abs(x/2))
    # Se computan las N iteraciones
    frames = create_animation(init_frame, kernel, 
                              activation, 
                              N, persistent=False)
    # Se crea y exporta el video
    export_video_cv2(frames, name='moho.mp4', 
                     fps=60, skip_frames=True, 
                     skip='odd')