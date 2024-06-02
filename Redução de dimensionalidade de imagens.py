import cv2
import numpy as np

# Carregar a imagem colorida
imagem_colorida = cv2.imread('6m6r1jyy3kp91.webp')

# Converter a imagem para tons de cinza
imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)

# Aplicar a binarização
_, imagem_binarizada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)

# Mostrar a imagem original, em tons de cinza e binarizada
cv2.imshow('Imagem Original', imagem_colorida)
cv2.imshow('Imagem em Tons de Cinza', imagem_cinza)
cv2.imshow('Imagem Binarizada', imagem_binarizada)

# Aguardar até uma tecla ser pressionada
cv2.waitKey(0)
cv2.destroyAllWindows()

