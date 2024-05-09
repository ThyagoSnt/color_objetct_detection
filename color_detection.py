import cv2
import numpy as np
import argparse

kernel_e = np.ones((7, 7), np.uint8)
kernel_d = np.ones((27, 27), np.uint8)

def color_choose(color):
    if (color == 'yellow'):
        # Define o intervalo de valores de matiz para a cor amarela
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    elif color == 'blue':
        # Define o intervalo de valores de matiz para a cor azul
        lower = np.array([90, 100, 100])
        upper = np.array([120, 255, 255])
    elif color == 'green':
        # Define o intervalo de valores de matiz para a cor verde
        lower = np.array([50, 100, 100])
        upper = np.array([70, 255, 255])
    elif color == 'red':
        # Define o intervalo de valores de matiz para a cor vermelha
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])
        lower, upper = (lower1, lower2), (upper1, upper2)
    return lower, upper

def filtrar_cor(frame, lower, upper):
    # Aplica o filtro gaussiano
    guassian = cv2.GaussianBlur(frame, (15, 15), 0)

    # Converte o frame de RGB para o espaço de cores HSV
    hsv = cv2.cvtColor(guassian, cv2.COLOR_BGR2HSV)

    if isinstance(lower, tuple):
        # Filtra os pixels dentro do intervalo de valores de matiz
        mask1 = cv2.inRange(hsv, lower[0], upper[0])
        mask2 = cv2.inRange(hsv, lower[1], upper[1])
        # Aplica um ou bit a bit para gera a mascara final:
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Filtra os pixels dentro do intervalo de valores de matiz
        mask = cv2.inRange(hsv, lower, upper)

    # Aplica limiarização para binarizar a máscara
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Aplica erosão seguida de dilatação
    erode_img = cv2.erode(binary, kernel_e, iterations=1)
    dilate_img = cv2.dilate(erode_img, kernel_d, iterations=1)

    return dilate_img


def main():
    # Criação do parser
    parser = argparse.ArgumentParser(description="...")

    # Adiciona argumentos posicionais
    parser.add_argument("--cor", type=str, help="Cor que vamos filtrar nas imagens")

    # Faz o parsing dos argumentos da linha de comando
    args = parser.parse_args()

    lower, upper = color_choose(args.cor)

    # Captura de vídeo da câmera
    cap = cv2.VideoCapture(2)

    # Verifica se a câmera está aberta
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return

    # Defina a resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Loop para capturar e exibir o vídeo da câmera
    while True:
        # Captura frame a frame
        ret, frame = cap.read()
        
        # Verifica se a captura de vídeo foi bem-sucedida
        if not ret:
            print("Erro ao capturar o frame.")
            break

        frame_filtered = filtrar_cor(frame, lower, upper)

        contornos,_ = cv2.findContours(frame_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Calcula o centro da imagem
        height, width, _ = frame.shape
        center_image = (width // 2, height // 2)

        # Desenha um ponto no centro da imagem
        cv2.circle(frame, center_image, 5, (0, 255, 0), -1)

        if contornos:
            for contorno in contornos:
                x, y, w, h = cv2.boundingRect(contorno)

                # Calcula o centro do objeto
                center_objeto = (x + w // 2, y + h // 2)

                # Desenha um ponto no centro do objeto
                cv2.circle(frame, center_objeto, 5, (0, 0, 255), -1)

                # Desenha uma linha do centro da imagem ao centro do objeto
                cv2.arrowedLine(frame, center_image, center_objeto, (0, 255, 0), 3)

                # Calculando os eixos do vetor:
                dist_x, disty = center_objeto[0] - center_image[0], center_image[1] - center_objeto[1]
                
                # Desenha a posição do pixel
                cv2.putText(frame, f'({dist_x}, {disty})', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        
        # Mostra o frame capturado
        cv2.imshow('Camera', frame)
        
        # Verifica se o usuário pressionou a tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a captura da câmera e fecha a janela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
