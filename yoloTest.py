import cv2
import numpy as np

# Carregando as configurações do modelo
config = "yolov3_training.cfg"
weights = "yolov3_training_final.weights"
names = open("obj.names").read().strip().split("\n")

net = cv2.dnn.readNet(config, weights)

# Definindo cores para os rótulos
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(names), 3), dtype="uint8")

# Obtendo os nomes das camadas
ln = net.getLayerNames()
out_layer_indices = net.getUnconnectedOutLayers()
if out_layer_indices.ndim == 1:
    ln = [ln[i - 1] for i in out_layer_indices]
else:
    ln = [ln[i[0] - 1] for i in out_layer_indices]

# Inicializando a captura de vídeo
cap = cv2.VideoCapture(2)  # 0 para a primeira webcam conectada


while True:
    # Captura frame a frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame,(100,100))
    if not ret:
        break

    # Preparando a entrada para a rede
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Inicializando listas para caixas delimitadoras, confianças e IDs de classe
    boxes = []
    confidences = []
    classIDs = []

    # Loop sobre cada saída das camadasq
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Aplicando non-maxima suppression para evitar caixas delimitadoras redundantes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Verifica se há algum objeto detectado
    if len(idxs) > 0:
        print("Number of detections:", len(idxs))
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(names[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print("Detected:", text)
    else:
        print("No objects detected.")

    # Mostrando o frame resultante
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla 'q' for pressionada, sai do loop
    if key == ord('q'):
        break

# Limpeza
cap.release()
cv2.destroyAllWindows()
