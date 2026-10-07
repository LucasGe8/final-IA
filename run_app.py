# ==============================================================================
# CELDA 1: INSTALACIÓN DE DEPENDENCIAS
# ==============================================================================


# ==============================================================================
# CELDA 2: IMPORTS Y VERIFICACIÓN DE ENTORNO
# ==============================================================================
import os
import cv2
import torch
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# --- Verificación de dispositivo (GPU vs CPU) ---
if torch.cuda.is_available():
    DEVICE = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU detectada: {gpu_name} ({vram:.1f} GB VRAM)')
    print(f'   CUDA Version: {torch.version.cuda}')
else:
    DEVICE = 'cpu'
    print('⚠️  GPU no disponible. Usando CPU (el entrenamiento será más lento).')

print(f'\n🔧 Dispositivo seleccionado: {DEVICE}')
print(f'   PyTorch version: {torch.__version__}')
print(f'   Ultralytics YOLO instalado correctamente ✓')

# ==============================================================================
# CELDA 3: DESCARGA AUTOMÁTICA DEL DATASET DESDE KAGGLE
# ==============================================================================
import kagglehub

print("📥 Descargando dataset desde Kaggle...")
path_dataset = kagglehub.dataset_download("anushkagovindkadam/smart-helmet-detection-using-dl")

print("✅ Descarga completada.")
print("Path to dataset files:", path_dataset)

DATASET_PATH = Path(path_dataset)
print(f"Ruta del dataset configurada en: {DATASET_PATH.resolve()}")

# ==============================================================================
# CELDA 4: EXPLORACIÓN DEL DATASET — ANÁLISIS DE CALIDAD
# ==============================================================================
# Analizamos las etiquetas ANTES de limpiar para entender los problemas.

DATA_ROOT = DATASET_PATH / 'data'

# Clases objetivo (solo 2 clases: cascos)
CLASES_OBJETIVO = {0, 1}  # 0=Con_Casco, 1=Sin_Casco
NOMBRES_CLASES = {0: 'Con_Casco', 1: 'Sin_Casco'}

def analizar_labels(carpeta_labels: Path, nombre_split: str):
    """Analiza las etiquetas de un split y reporta estadísticas."""
    todas_clases = []
    archivos_corruptos = []
    archivos_ok = []
    total_archivos = 0
    
    if not carpeta_labels.exists():
        print(f'  ❌ Carpeta no encontrada: {carpeta_labels}')
        return [], []
    
    for label_file in sorted(carpeta_labels.glob('*.txt')):
        total_archivos += 1
        tiene_clases_validas = False
        tiene_clases_invalidas = False
        
        with open(label_file, 'r') as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) >= 5:
                    clase = int(partes[0])
                    todas_clases.append(clase)
                    if clase in CLASES_OBJETIVO:
                        tiene_clases_validas = True
                    else:
                        tiene_clases_invalidas = True
        
        if tiene_clases_invalidas:
            archivos_corruptos.append(label_file)
        if tiene_clases_validas:
            archivos_ok.append(label_file)
    
    # Reporte
    conteo = Counter(todas_clases)
    print(f'\n📂 Split: {nombre_split}')
    print(f'   Total archivos de labels: {total_archivos}')
    print(f'   Con clases válidas (0,1): {len(archivos_ok)}')
    print(f'   Con clases fuera de rango: {len(archivos_corruptos)}')
    print(f'   Distribución de clases:')
    for clase_id in sorted(conteo.keys()):
        estado = '✅' if clase_id in CLASES_OBJETIVO else '❌'
        nombre = NOMBRES_CLASES.get(clase_id, f'Clase_{clase_id} (ELIMINAR)')
        print(f'     {estado} Clase {clase_id} ({nombre}): {conteo[clase_id]} instancias')
    
    return archivos_ok, archivos_corruptos

print('🔍 ANÁLISIS DE CALIDAD DEL DATASET')
print('=' * 60)

# Detectar nombres de carpetas (el dataset tiene "vaid" en lugar de "val")
splits_info = [
    ('train', DATA_ROOT / 'train' / 'labels'),
    ('val', DATA_ROOT / 'vaid' / 'labels'),   # Nota: typo en el dataset original
    ('test', DATA_ROOT / 'test' / 'labels'),
]

resultados_analisis = {}
for nombre, ruta in splits_info:
    ok, corruptos = analizar_labels(ruta, nombre)
    resultados_analisis[nombre] = {'ok': ok, 'corruptos': corruptos}

# ==============================================================================
# CELDA 5: PIPELINE DE LIMPIEZA DE DATOS
# Crea un dataset limpio con solo las clases 0 (Con_Casco) y 1 (Sin_Casco)
# ==============================================================================

CLEAN_DATASET = Path('./dataset_limpio')

def limpiar_split(nombre_split: str, carpeta_imagenes: Path, carpeta_labels: Path, destino: Path):
    """
    Limpia un split del dataset:
    - Copia solo imágenes que tengan labels válidos (clase 0 o 1)
    - Reescribe los labels eliminando clases fuera de rango
    """
    dest_images = destino / nombre_split / 'images'
    dest_labels = destino / nombre_split / 'labels'
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    imagenes_copiadas = 0
    imagenes_descartadas = 0
    lineas_eliminadas = 0
    lineas_conservadas = 0
    
    if not carpeta_labels.exists():
        print(f'  ⚠️  Carpeta de labels no encontrada: {carpeta_labels}')
        return
    
    for label_file in sorted(carpeta_labels.glob('*.txt')):
        # Leer y filtrar líneas
        lineas_validas = []
        with open(label_file, 'r') as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) >= 5:
                    clase = int(partes[0])
                    if clase in CLASES_OBJETIVO:
                        lineas_validas.append(linea.strip())
                        lineas_conservadas += 1
                    else:
                        lineas_eliminadas += 1
        
        # Solo copiar si hay al menos una detección válida
        if lineas_validas:
            # Buscar la imagen correspondiente
            stem = label_file.stem
            imagen_encontrada = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                candidata = carpeta_imagenes / (stem + ext)
                if candidata.exists():
                    imagen_encontrada = candidata
                    break
            
            if imagen_encontrada:
                # Copiar imagen
                shutil.copy2(imagen_encontrada, dest_images / imagen_encontrada.name)
                # Escribir label limpio
                with open(dest_labels / label_file.name, 'w') as f:
                    f.write('\n'.join(lineas_validas) + '\n')
                imagenes_copiadas += 1
            else:
                imagenes_descartadas += 1
        else:
            imagenes_descartadas += 1
    
    print(f'  📦 {nombre_split}: {imagenes_copiadas} imágenes copiadas, '
          f'{imagenes_descartadas} descartadas')
    print(f'     Labels: {lineas_conservadas} líneas conservadas, '
          f'{lineas_eliminadas} líneas eliminadas')

# Limpiar si existe una versión anterior
if CLEAN_DATASET.exists():
    shutil.rmtree(CLEAN_DATASET)

print('🧹 LIMPIEZA Y SANITIZACIÓN DEL DATASET')
print('=' * 60)
print(f'Destino: {CLEAN_DATASET.resolve()}')
print(f'Clases objetivo: {NOMBRES_CLASES}\n')

# Procesar cada split
splits_limpieza = [
    ('train', DATA_ROOT / 'train' / 'images', DATA_ROOT / 'train' / 'labels'),
    ('val',   DATA_ROOT / 'vaid' / 'images',  DATA_ROOT / 'vaid' / 'labels'),
    ('test',  DATA_ROOT / 'test' / 'images',  DATA_ROOT / 'test' / 'labels'),
]

for nombre, imgs, lbls in splits_limpieza:
    limpiar_split(nombre, imgs, lbls, CLEAN_DATASET)

print('\n✅ Dataset limpio creado exitosamente.')

# Verificación post-limpieza
print('\n📊 Verificación post-limpieza:')
for split_name in ['train', 'val', 'test']:
    n_imgs = len(list((CLEAN_DATASET / split_name / 'images').glob('*')))
    n_lbls = len(list((CLEAN_DATASET / split_name / 'labels').glob('*.txt')))
    print(f'   {split_name}: {n_imgs} imágenes, {n_lbls} labels')

# ==============================================================================
# CELDA 6: GENERACIÓN DEL data.yaml CON DATASET LIMPIO (2 clases)
# ==============================================================================

data_config = {
    'path': str(CLEAN_DATASET.resolve()),
    'train': 'train/images',
    'val':   'val/images',
    'test':  'test/images',
    'nc': 2,
    'names': {
        0: 'Con_Casco',
        1: 'Sin_Casco'
    }
}

YAML_PATH = 'data.yaml'
with open(YAML_PATH, 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)

print(f'✅ Archivo "{YAML_PATH}" creado exitosamente.')
print('\n📄 Contenido del data.yaml:')
print('─' * 40)
with open(YAML_PATH, 'r') as f:
    print(f.read())

# ==============================================================================
# CELDA 8: VISUALIZACIÓN AUTOMÁTICA DE MÉTRICAS
# ==============================================================================

def mostrar_metricas_entrenamiento(ruta_runs='runs/detect/helmet_v2'):
    """Renderiza todas las métricas de entrenamiento generadas por YOLO."""
    ruta = Path(ruta_runs)

    graficos = {
        'confusion_matrix.png':            'Matriz de Confusión',
        'confusion_matrix_normalized.png':  'Matriz de Confusión (Normalizada)',
        'results.png':                     'Curvas de Pérdida y mAP',
        'P_curve.png':                     'Curva de Precision',
        'R_curve.png':                     'Curva de Recall',
        'PR_curve.png':                    'Curva Precision-Recall',
        'F1_curve.png':                    'Curva F1-Score',
        'labels.jpg':                      'Distribución de Labels',
    }

    archivos_encontrados = []
    for archivo, titulo in graficos.items():
        ruta_img = ruta / archivo
        if ruta_img.exists():
            archivos_encontrados.append((ruta_img, titulo))

    if not archivos_encontrados:
        print(f'❌ No se encontraron métricas en: {ruta}')
        return

    # Mostrar de a 2 en una fila
    n = len(archivos_encontrados)
    for i in range(0, n, 2):
        ncols = min(2, n - i)
        fig, axes = plt.subplots(1, ncols, figsize=(14 * ncols // 2, 8))
        if ncols == 1:
            axes = [axes]
        for j in range(ncols):
            ruta_img, titulo = archivos_encontrados[i + j]
            img = mpimg.imread(str(ruta_img))
            axes[j].imshow(img)
            axes[j].set_title(titulo, fontsize=14, fontweight='bold', pad=10)
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig('metrics_plot.png')

    print(f'\n✅ {n} gráficos de métricas renderizados correctamente.')


mostrar_metricas_entrenamiento('runs/detect/helmet_v2')

# ==============================================================================
# CELDA 9: EVALUACIÓN EN SET DE TEST
# ==============================================================================

BEST_WEIGHTS = 'runs/detect/helmet_v2/weights/best.pt'
model_eval = YOLO(BEST_WEIGHTS)

metricas = model_eval.val(
    data=YAML_PATH,
    split='test',
    device=DEVICE,
    verbose=True,
)

print('\n📊 RESUMEN DE MÉTRICAS EN TEST SET')
print('═' * 50)
print(f'   mAP@50:        {metricas.box.map50:.4f}')
print(f'   mAP@50-95:     {metricas.box.map:.4f}')
print(f'   Precision:     {metricas.box.mp:.4f}')
print(f'   Recall:        {metricas.box.mr:.4f}')
print('═' * 50)

# Calcular F1
p = metricas.box.mp
r = metricas.box.mr
f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
print(f'   F1-Score:      {f1:.4f}')

# Evaluación cualitativa
if metricas.box.map50 >= 0.7:
    print('\n   🟢 Rendimiento BUENO — modelo listo para producción')
elif metricas.box.map50 >= 0.5:
    print('\n   🟡 Rendimiento ACEPTABLE — considerar más datos o épocas')
else:
    print('\n   🔴 Rendimiento BAJO — revisar datos o cambiar modelo')

# ==============================================================================
# CELDA 10: FUNCIÓN DE INFERENCIA UNIVERSAL
# Soporta: imagen estática, video, y frame de cámara
# ==============================================================================

# Colores para cada clase (BGR para OpenCV)
COLORES_CLASE = {
    0: (0, 200, 0),     # Con_Casco → Verde
    1: (0, 0, 220),     # Sin_Casco → Rojo
}

def procesar_frame(
    frame: np.ndarray,
    modelo: YOLO,
    confianza: float = 0.3,
    iou_thresh: float = 0.45,
) -> tuple:
    """
    Procesa un frame individual con YOLO.
    
    Returns:
        tuple: (frame_anotado, lista_detecciones)
               lista_detecciones = [{'clase': str, 'confianza': float, 'bbox': list}, ...]
    """
    resultado = modelo(
        frame,
        conf=confianza,
        iou=iou_thresh,
        verbose=False
    )[0]
    
    detecciones = []
    frame_anotado = frame.copy()
    
    for box in resultado.boxes:
        clase_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        nombre_clase = resultado.names[clase_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        detecciones.append({
            'clase': nombre_clase,
            'clase_id': clase_id,
            'confianza': conf_val,
            'bbox': [x1, y1, x2, y2]
        })
        
        # Dibujar bounding box
        color = COLORES_CLASE.get(clase_id, (255, 255, 0))
        grosor = 3
        cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), color, grosor)
        
        # Etiqueta con fondo
        label = f'{nombre_clase} {conf_val:.0%}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Fondo del texto
        cv2.rectangle(frame_anotado, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_anotado, label, (x1 + 3, y1 - 5), font, font_scale, 
                    (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # Panel de estadísticas
    n_con = sum(1 for d in detecciones if d['clase_id'] == 0)
    n_sin = sum(1 for d in detecciones if d['clase_id'] == 1)
    
    panel_h = 90
    overlay = frame_anotado.copy()
    cv2.rectangle(overlay, (0, 0), (320, panel_h), (0, 0, 0), -1)
    frame_anotado = cv2.addWeighted(overlay, 0.65, frame_anotado, 0.35, 0)
    
    cv2.putText(frame_anotado, f'Detecciones: {len(detecciones)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_anotado, f'Con Casco: {n_con}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_anotado, f'Sin Casco: {n_sin}', (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2, cv2.LINE_AA)
    
    # Alerta visual si hay personas sin casco
    if n_sin > 0:
        h, w = frame_anotado.shape[:2]
        alerta = '⚠ ALERTA: PERSONA(S) SIN CASCO'
        (aw, ah), _ = cv2.getTextSize(alerta, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ax = w - aw - 20
        ay = 30
        cv2.rectangle(frame_anotado, (ax - 10, ay - ah - 10), (ax + aw + 10, ay + 10), (0, 0, 180), -1)
        cv2.putText(frame_anotado, alerta, (ax, ay),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame_anotado, detecciones


print('✅ Función procesar_frame() definida correctamente.')
print('   Soporta: frames de cámara, imágenes, y video.')

# ==============================================================================
# CELDA 11: INTERFAZ GRÁFICA GRADIO
# ==============================================================================
import gradio as gr
import tempfile
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Cargar modelo una sola vez
BEST_WEIGHTS = 'runs/detect/helmet_v2/weights/best.pt'

if not Path(BEST_WEIGHTS).exists():
    print('⚠️  No se encontraron pesos entrenados. Ejecuta las celdas de entrenamiento primero.')
    print(f'   Buscando en: {BEST_WEIGHTS}')
else:
    modelo_gradio = YOLO(BEST_WEIGHTS)
    print(f'✅ Modelo cargado desde: {BEST_WEIGHTS}')


# ──────────────────────────────────────
# Función para IMAGEN
# ──────────────────────────────────────
def detectar_imagen(imagen, confianza):
    """
    Procesa una imagen estática y retorna:
    - Imagen anotada con bounding boxes
    - Tabla de detecciones
    """
    if imagen is None:
        return None, "No se proporcionó imagen."
    
    # Gradio entrega imágenes en RGB, YOLO espera BGR
    frame_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    frame_anotado, detecciones = procesar_frame(frame_bgr, modelo_gradio, confianza)
    
    # Convertir de vuelta a RGB para Gradio
    frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
    
    # Crear resumen de texto
    n_con = sum(1 for d in detecciones if d['clase_id'] == 0)
    n_sin = sum(1 for d in detecciones if d['clase_id'] == 1)
    
    resumen = f"🔍 Total de detecciones: {len(detecciones)}\n"
    resumen += f"🟢 Con Casco: {n_con}\n"
    resumen += f"🔴 Sin Casco: {n_sin}\n\n"
    
    if n_sin > 0:
        resumen += "⚠️ ¡ALERTA! Se detectaron personas SIN casco de seguridad.\n\n"
    elif n_con > 0:
        resumen += "✅ Todas las personas detectadas llevan casco.\n\n"
    else:
        resumen += "ℹ️ No se detectaron personas en la imagen.\n\n"
    
    resumen += "─" * 40 + "\n"
    for i, d in enumerate(detecciones, 1):
        emoji = '🟢' if d['clase_id'] == 0 else '🔴'
        resumen += f"{emoji} [{i}] {d['clase']:<12} | Confianza: {d['confianza']:.1%}\n"
    
    return frame_rgb, resumen


# ──────────────────────────────────────
# Función para VIDEO
# ──────────────────────────────────────
def detectar_video(video_path, confianza, progreso=gr.Progress()):
    """
    Procesa un video completo frame a frame.
    Retorna la ruta al video anotado.
    """
    if video_path is None:
        return None, "No se proporcionó video."
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ No se pudo abrir el video."
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Archivo temporal para el video de salida
    output_dir = Path('runs/detect/helmet_v2/videos')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = str(output_dir / f'resultado_{Path(video_path).stem}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    total_detecciones = []
    alertas_sin_casco = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_anotado, detecciones = procesar_frame(frame, modelo_gradio, confianza)
            out.write(frame_anotado)
            
            total_detecciones.extend(detecciones)
            if any(d['clase_id'] == 1 for d in detecciones):
                alertas_sin_casco += 1
            
            frame_num += 1
            if total_frames > 0:
                progreso(frame_num / total_frames, 
                        desc=f"Procesando frame {frame_num}/{total_frames}")
    finally:
        cap.release()
        out.release()
    
    # Resumen del video
    n_con_total = sum(1 for d in total_detecciones if d['clase_id'] == 0)
    n_sin_total = sum(1 for d in total_detecciones if d['clase_id'] == 1)
    
    resumen = f"🎬 Video procesado exitosamente\n"
    resumen += f"═" * 40 + "\n"
    resumen += f"📹 Frames procesados: {frame_num}\n"
    resumen += f"⏱️ Duración: {frame_num / fps:.1f} segundos\n"
    resumen += f"🔍 Detecciones totales: {len(total_detecciones)}\n"
    resumen += f"🟢 Con Casco (acumulado): {n_con_total}\n"
    resumen += f"🔴 Sin Casco (acumulado): {n_sin_total}\n"
    resumen += f"⚠️ Frames con alertas: {alertas_sin_casco}/{frame_num}\n"
    
    if alertas_sin_casco > 0:
        pct = alertas_sin_casco / frame_num * 100
        resumen += f"\n🚨 Se detectaron personas sin casco en {pct:.1f}% del video"
    else:
        resumen += f"\n✅ No se detectaron infracciones en todo el video"
    
    return output_path, resumen


# ──────────────────────────────────────
# Función para CÁMARA EN VIVO
# ──────────────────────────────────────
def detectar_camara(frame, confianza):
    """
    Procesa un frame de la cámara en tiempo real.
    """
    if frame is None:
        return None
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_anotado, _ = procesar_frame(frame_bgr, modelo_gradio, confianza)
    frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
    return frame_rgb


# ──────────────────────────────────────
# CONSTRUCCIÓN DE LA INTERFAZ GRADIO
# ──────────────────────────────────────

css = """
.gradio-container {
    max-width: 1200px !important;
}
h1 {
    text-align: center;
    color: #2563eb;
}
.alert-box {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    border-radius: 12px;
    padding: 16px;
    color: white;
}
"""

with gr.Blocks(title="🏗️ Detección de Cascos v2.0") as demo:
    
    gr.Markdown(
        """
        # 🏗️ Sistema de Detección de Cascos de Seguridad v2.0
        **Motor de IA local** — YOLOv8m entrenado para detectar el uso y no uso de cascos de seguridad.
        
        Selecciona una pestaña para elegir el modo de entrada:
        """
    )
    
    with gr.Tabs():
        
        # ── TAB 1: IMAGEN ──
        with gr.TabItem("🖼️ Imagen", id="tab_imagen"):
            gr.Markdown("### Sube una imagen para detectar cascos de seguridad")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Imagen de entrada",
                        type="numpy",
                        sources=["upload", "clipboard"],
                    )
                    img_conf = gr.Slider(
                        minimum=0.1, maximum=0.95, value=0.3, step=0.05,
                        label="Umbral de Confianza",
                        info="Ajusta la sensibilidad de detección"
                    )
                    img_btn = gr.Button("🔍 Detectar", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    img_output = gr.Image(label="Resultado", type="numpy")
                    img_info = gr.Textbox(
                        label="Detecciones", 
                        lines=12, 
                        interactive=False
                    )
            
            img_btn.click(
                fn=detectar_imagen,
                inputs=[img_input, img_conf],
                outputs=[img_output, img_info]
            )
        
        # ── TAB 2: VIDEO ──
        with gr.TabItem("🎬 Video", id="tab_video"):
            gr.Markdown("### Sube un video para análisis completo de cascos")
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Video de entrada")
                    vid_conf = gr.Slider(
                        minimum=0.1, maximum=0.95, value=0.3, step=0.05,
                        label="Umbral de Confianza"
                    )
                    vid_btn = gr.Button("🎬 Procesar Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Video con detecciones")
                    vid_info = gr.Textbox(
                        label="Resumen del análisis",
                        lines=12,
                        interactive=False
                    )
            
            vid_btn.click(
                fn=detectar_video,
                inputs=[vid_input, vid_conf],
                outputs=[vid_output, vid_info]
            )
        
        # ── TAB 3: CÁMARA EN VIVO ──
        with gr.TabItem("📷 Cámara en Vivo", id="tab_camara"):
            gr.Markdown(
                """
                ### Detección en tiempo real con tu cámara
                Haz clic en **"Iniciar cámara"** para comenzar la detección en vivo.
                """
            )
            cam_conf = gr.Slider(
                minimum=0.1, maximum=0.95, value=0.3, step=0.05,
                label="Umbral de Confianza"
            )
            cam_input = gr.Image(
                label="Cámara",
                sources=["webcam"],
                type="numpy",
                streaming=True,
            )
            cam_output = gr.Image(label="Detección en vivo", type="numpy")
            
            cam_input.stream(
                fn=detectar_camara,
                inputs=[cam_input, cam_conf],
                outputs=[cam_output],
            )
    
    gr.Markdown(
        """
        ---
        **🏗️ Sistema de Detección de Cascos v2.0** — YOLOv8m · Ultralytics · Gradio  
        *Clases: Con_Casco (🟢) · Sin_Casco (🔴)*
        """
    )

print('✅ Interfaz Gradio construida correctamente.')
print('   Ejecuta la siguiente celda para lanzarla.')

# ==============================================================================
# CELDA 12: LANZAR INTERFAZ GRADIO
# ==============================================================================
# La interfaz se abrirá en una pestaña del navegador automáticamente.
# También genera un link público si estás en un entorno remoto (Colab, etc.)

demo.launch(
    share=False,           # Cambiar a True para generar link público
    inbrowser=True,        # Abrir automáticamente en el navegador
    server_name='0.0.0.0', # Accesible desde la red local
    server_port=7860,      # Puerto por defecto de Gradio
    css=css,               # CSS personalizado (Gradio 6.0+)
    theme=gr.themes.Soft(),# Tema visual (Gradio 6.0+)
)


