import cv2
from deepface import DeepFace
import customtkinter as ctk
import tkinter as tk
import random
from datetime import datetime
import numpy as np
from collections import Counter
import time

# Diccionario de emociones con preguntas
preguntas_emociones = {
    'happy': [
        "¿Qué te hizo sentir feliz hoy?",
        "¿Con quién compartiste ese momento especial?",
        "¿Quieres guardar este recuerdo?",
        "¿Qué fue lo más divertido del día?",
    ],
    'sad': [
        "¿Qué te hizo sentir triste?",
        "¿Hay algo que puedas hacer para mejorar tu día?",
        "¿Te gustaría hablar con alguien sobre eso?",
        "¿Qué podrías hacer para sentirte un poco mejor?",
    ],
    'angry': [
        "¿Qué provocó tu enojo?",
        "¿Lo pudiste expresar de forma saludable?",
        "¿Qué te ayudaría a calmarte ahora?",
        "¿Vale la pena seguir enojado?",
    ],
    'surprise': [
        "¿Qué te sorprendió?",
        "¿Fue una sorpresa positiva o negativa?",
        "¿Cómo reaccionaste ante eso?",
        "¿Te gustaría compartirlo con alguien?",
    ],
    'fear': [
        "¿Qué te causó miedo?",
        "¿Era un miedo justificado?",
        "¿Cómo puedes enfrentar esa situación?",
        "¿Te sientes más seguro ahora?",
    ],
    'disgust': [
        "¿Qué te causó esta sensación?",
        "¿Fue algo que viste o experimentaste?",
        "¿Cómo puedes evitar esa situación?",
        "¿Te sientes mejor ahora?",
    ],
    'neutral': [
        "¿Cómo te sientes realmente en este momento?",
        "¿Hay algo que te gustaría cambiar de tu día?",
        "¿Qué podrías hacer para sentirte más motivado?",
        "¿Te gustaría reflexionar sobre algo?",
    ]
}

class DetectorEmociones:
    def __init__(self):
        self.emotion_history = []
        self.confidence_threshold = 60  # Umbral de confianza mínimo
        self.stabilization_frames = 10  # Frames para estabilizar la emoción
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def preprocess_frame(self, frame):
        """Preprocesa el frame para mejorar la detección"""
        # Mejora el contraste y brillo
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def detect_face_regions(self, frame):
        """Detecta múltiples regiones faciales para mayor precisión"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    
    def analyze_emotion_multiple_methods(self, frame):
        """Analiza emociones usando múltiples métodos para mayor precisión"""
        results = []
        
        # Método 1: Análisis del frame completo
        try:
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv'
            )
            if isinstance(result, list):
                result = result[0]
            results.append(result)
        except:
            pass
        
        # Método 2: Análisis de regiones faciales detectadas
        faces = self.detect_face_regions(frame)
        for (x, y, w, h) in faces:
            if w > 80 and h > 80:  # Solo rostros de tamaño razonable
                face_roi = frame[y:y+h, x:x+w]
                try:
                    result = DeepFace.analyze(
                        face_roi, 
                        actions=['emotion'], 
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    if isinstance(result, list):
                        result = result[0]
                    # Ajustar las coordenadas de la región
                    result['region'] = {'x': x, 'y': y, 'w': w, 'h': h}
                    results.append(result)
                except:
                    pass
        
        return results
    
    def get_most_confident_emotion(self, results):
        """Obtiene la emoción con mayor confianza de múltiples análisis"""
        if not results:
            return None
        
        best_result = None
        best_confidence = 0
        
        for result in results:
            dominant_emotion = result['dominant_emotion']
            confidence = result['emotion'][dominant_emotion]
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_result = result
        
        return best_result
    
    def stabilize_emotion(self, emotion, confidence):
        """Estabiliza la emoción detectada usando un historial"""
        if confidence > self.confidence_threshold:
            self.emotion_history.append(emotion)
        
        # Mantener solo los últimos N frames
        if len(self.emotion_history) > self.stabilization_frames:
            self.emotion_history = self.emotion_history[-self.stabilization_frames:]
        
        # Si tenemos suficientes muestras, usar la más frecuente
        if len(self.emotion_history) >= 5:
            emotion_counts = Counter(self.emotion_history)
            most_common = emotion_counts.most_common(1)[0]
            if most_common[1] >= 3:  # Al menos 3 detecciones iguales
                return most_common[0]
        
        return emotion if confidence > self.confidence_threshold else None

def detectar_emocion_en_camara():
    detector = DetectorEmociones()
    cap = cv2.VideoCapture(0)
    
    # Configurar la cámara para mejor calidad
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    emocion_detectada = None
    confianza = 0
    frames_sin_deteccion = 0
    max_frames_sin_deteccion = 30
    
    print("[INFO] Cámara activada. Asegúrate de tener buena iluminación.")
    print("[INFO] Mantén el rostro centrado y mira a la cámara.")
    print("[INFO] Presiona 'q' para continuar al cuestionario...")
    print("[INFO] Presiona 'r' para reiniciar el historial de emociones...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Voltear el frame horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Preprocesar el frame
        enhanced_frame = detector.preprocess_frame(frame)
        
        try:
            # Analizar emociones con múltiples métodos
            results = detector.analyze_emotion_multiple_methods(enhanced_frame)
            best_result = detector.get_most_confident_emotion(results)
            
            if best_result:
                current_emotion = best_result['dominant_emotion']
                current_confidence = best_result['emotion'][current_emotion]
                
                # Estabilizar la emoción
                stable_emotion = detector.stabilize_emotion(current_emotion, current_confidence)
                
                if stable_emotion:
                    emocion_detectada = stable_emotion
                    confianza = current_confidence
                    frames_sin_deteccion = 0
                
                # Obtener región facial
                if 'region' in best_result:
                    area = best_result['region']
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                else:
                    # Si no hay región específica, usar detección de rostro
                    faces = detector.detect_face_regions(frame)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                    else:
                        x, y, w, h = 0, 0, 0, 0
                
                # Dibujar cuadro y información
                if w > 0 and h > 0:
                    # Cuadro principal
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Texto con emoción y confianza
                    texto_principal = f"{current_emotion.upper()} ({current_confidence:.1f}%)"
                    cv2.putText(frame, texto_principal, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Texto con emoción estabilizada
                    if stable_emotion:
                        texto_estable = f"Estable: {stable_emotion.upper()}"
                        cv2.putText(frame, texto_estable, (x, y + h + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Mostrar historial de emociones
                if len(detector.emotion_history) > 0:
                    historial_texto = f"Historial: {len(detector.emotion_history)} muestras"
                    cv2.putText(frame, historial_texto, (10, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            else:
                frames_sin_deteccion += 1
                if frames_sin_deteccion < max_frames_sin_deteccion:
                    cv2.putText(frame, "Detectando rostro...", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No se detecta rostro - Mejora la iluminacion", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        except Exception as e:
            frames_sin_deteccion += 1
            error_msg = f"Error en deteccion: {str(e)[:30]}"
            cv2.putText(frame, error_msg, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Mostrar estado actual
        if emocion_detectada:
            status_text = f"Emocion final: {emocion_detectada.upper()} - Presiona 'q' para continuar"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mostrar controles
        cv2.putText(frame, "Controles: 'q' = continuar, 'r' = reiniciar", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar video
        cv2.imshow("Detector de Emociones Mejorado", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.emotion_history = []
            emocion_detectada = None
            print("[INFO] Historial de emociones reiniciado")
    
    cap.release()
    cv2.destroyAllWindows()
    return emocion_detectada

def mostrar_gui(emocion_detectada):
    if emocion_detectada not in preguntas_emociones:
        emocion_detectada = 'neutral'
    
    pregunta = random.choice(preguntas_emociones[emocion_detectada])
    
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    app = ctk.CTk()
    app.geometry("1000x700")
    app.title("Detector Emocional - Diario Personal")
    
    # Título principal
    title_label = ctk.CTkLabel(
        app, 
        text=f"EMOCIÓN DETECTADA: {emocion_detectada.upper()}", 
        font=("Arial", 28, "bold"),
        text_color="#00ff88"
    )
    title_label.pack(pady=20)
    
    # Descripción de la emoción
    descripciones = {
        'happy': "😊 Te ves feliz y radiante",
        'sad': "😢 Pareces un poco triste",
        'angry': "😠 Detecté algo de enojo",
        'surprise': "😲 Te ves sorprendido",
        'fear': "😨 Pareces preocupado",
        'disgust': "🤢 Algo te disgusta",
        'neutral': "😐 Te ves tranquilo"
    }
    
    desc_label = ctk.CTkLabel(
        app, 
        text=descripciones.get(emocion_detectada, "Estado emocional detectado"),
        font=("Arial", 18),
        text_color="#cccccc"
    )
    desc_label.pack(pady=10)
    
    # Pregunta personalizada
    pregunta_label = ctk.CTkLabel(
        app, 
        text=pregunta, 
        font=("Arial", 22), 
        wraplength=900,
        text_color="#ffffff"
    )
    pregunta_label.pack(pady=40)
    
    # Área de texto para respuesta más grande
    respuesta_text = ctk.CTkTextbox(
        app, 
        height=150,
        width=800,
        font=("Arial", 16)
    )
    respuesta_text.pack(pady=20)
    
    # Función para guardar respuesta
    def guardar_respuesta():
        respuesta = respuesta_text.get("1.0", tk.END).strip()
        if respuesta != "":
            with open("diario_emocional.txt", "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n{'='*50}\n")
                f.write(f"FECHA: {timestamp}\n")
                f.write(f"EMOCIÓN: {emocion_detectada.upper()}\n")
                f.write(f"PREGUNTA: {pregunta}\n")
                f.write(f"RESPUESTA: {respuesta}\n")
                f.write(f"{'='*50}\n")
            
            respuesta_text.delete("1.0", tk.END)
            pregunta_label.configure(text="¡Respuesta guardada en tu diario emocional! 📝")
            guardar_btn.configure(text="Respuesta Guardada ✓", state="disabled")
            
            # Reactivar el botón después de 2 segundos
            app.after(2000, lambda: [
                guardar_btn.configure(text="Guardar otra respuesta", state="normal"),
                pregunta_label.configure(text=random.choice(preguntas_emociones[emocion_detectada]))
            ])
    
    # Función para nueva pregunta
    def nueva_pregunta():
        nueva_preg = random.choice(preguntas_emociones[emocion_detectada])
        pregunta_label.configure(text=nueva_preg)
    
    # Botones
    button_frame = ctk.CTkFrame(app, fg_color="transparent")
    button_frame.pack(pady=20)
    
    guardar_btn = ctk.CTkButton(
        button_frame, 
        text="Guardar respuesta", 
        command=guardar_respuesta,
        font=("Arial", 18),
        height=50,
        width=200
    )
    guardar_btn.pack(side="left", padx=10)
    
    nueva_pregunta_btn = ctk.CTkButton(
        button_frame, 
        text="Nueva pregunta", 
        command=nueva_pregunta,
        font=("Arial", 18),
        height=50,
        width=200,
        fg_color="#ff6b6b"
    )
    nueva_pregunta_btn.pack(side="left", padx=10)
    
    # Información adicional
    info_label = ctk.CTkLabel(
        app, 
        text="💡 Tip: Tómate tu tiempo para reflexionar. Tus pensamientos son importantes.",
        font=("Arial", 14),
        text_color="#888888"
    )
    info_label.pack(pady=20)
    
    app.mainloop()

# Ejecutar flujo completo
if __name__ == "__main__":
    print("🎭 DETECTOR DE EMOCIONES MEJORADO 🎭")
    print("=====================================")
    print("Iniciando detección de emociones...")
    
    emocion = detectar_emocion_en_camara()
    
    if emocion:
        print(f"\n✅ Emoción detectada: {emocion.upper()}")
        print("Abriendo diario emocional...")
        mostrar_gui(emocion)
    else:
        print("\n❌ No se detectó una emoción válida.")
        print("Consejos para mejorar la detección:")
        print("- Asegúrate de tener buena iluminación")
        print("- Mantén el rostro centrado en la cámara")
        print("- Evita movimientos bruscos")
        print("- Prueba con diferentes expresiones")
        print("\nVuelve a intentarlo.")