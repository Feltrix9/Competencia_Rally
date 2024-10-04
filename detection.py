import cv2
import pytesseract

# Configura la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Cambia esto según tu instalación

def detectar_y_reconocer_placa(imagen):
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Usar un filtro Gaussian para reducir ruido
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Detección de bordes
    bordes = cv2.Canny(gris, 100, 200)
    
    # Detección de contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contorno in contornos:
        # Obtener un rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Filtrar contornos que no sean del tamaño esperado para una placa
        if 200 < w < 600 and 50 < h < 150:
            placa_region = imagen[y:y+h, x:x+w]
            # Convertir la región de la placa a escala de grises
            placa_gray = cv2.cvtColor(placa_region, cv2.COLOR_BGR2GRAY)
            # Umbralización
            _, placa_thresh = cv2.threshold(placa_gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Reconocer caracteres
            texto = pytesseract.image_to_string(placa_thresh, config='--psm 8')
            return texto.strip(), placa_region, (x, y, w, h)

    return None, None, None

# Captura de video
cap = cv2.VideoCapture(0)  # Cambia 0 por la ruta del video si no es de la cámara

while True:
    ret, frame = cap.read()
    if not ret:
        break

    texto, placa_region, rectangulo = detectar_y_reconocer_placa(frame)

    if texto:
        # Dibuja un rectángulo alrededor de la placa detectada
        x, y, w, h = rectangulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Muestra el texto de la placa
        cv2.putText(frame, f'Placa: {texto}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Guarda la imagen en lugar de mostrar
    cv2.imwrite("captura.jpg", frame)
    print(texto)

    # Muestra la imagen (puedes comentarlo si solo quieres guardar)
    cv2.imshow("Reconocimiento de Placas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
