import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np

#CARREGANDO O MNIST
(x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

#VISUALIZAÇÃO DOS DADOS
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.tight_layout()
    plt.imshow(x_treino[i].reshape(28,28), cmap='gray')
    plt.title('Rótulo: {}'.format(y_treino[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#SEPARANDO OS CONJUNTOS DE DADOS
quantidade_dados_treino = 55000

x_validacao = x_treino[quantidade_dados_treino:, ..., np.newaxis]
y_validacao = y_treino[quantidade_dados_treino:]
x_treino = x_treino[:quantidade_dados_treino, ..., np.newaxis]
y_treino = y_treino[:quantidade_dados_treino]

x_teste = x_teste[..., np.newaxis]

print('Formato da Imagem: {}'.format(x_treino[0].shape), end='\n')
print('Conjunto de Treinamento {} registros'.format(len(x_treino)))
print('Conjunto de Validação: {} registros'.format(len(x_validacao)))
print('Conjunto de Testes: {} registros'.format(len(x_teste)))

#FAZENDO O PREENCHIMENTO, ALTERANDO OS DADOS DE ENTRADA
x_treino = np.pad(x_treino, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_validacao = np.pad(x_validacao, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_teste = np.pad(x_teste, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print('Informações das alterações dos dados de entrada: ', end='\n\n')
print('Conjunto de treinamento: {}'.format(x_treino.shape))
print('Conjunto de validação: {}'.format(x_validacao.shape))
print('Conjunto de testes: {}'.format(x_teste.shape))

#NORMALIZA OS DADOS PARA ESTAREM EM INTERVALO DE 0 A 1
normalizar_dados = lambda t: t/255
x_treino = normalizar_dados(x_treino)
x_validacao = normalizar_dados(x_validacao)
x_teste = normalizar_dados(x_teste)

#CRIANDO A ARQUITETURA
#CONV2D = CONVULAÇÃO // AVERAGE = POOLING // FLATTEN = // DENSE = CAMADAS TOTALMENTE CONECTADAS
def arquitetura_lenet_5(fun_ativacao):

    modelo = Sequential()
    modelo.add(Conv2D(6, kernel_size=(5,5), 
                        strides=(1,1),
                        activation = fun_ativacao,
                        input_shape = (32,32,1),
                        padding='valid'))

    modelo.add(AveragePooling2D(pool_size=(2,2),
                                strides=(2,2),
                                padding='valid'))

    modelo.add(Conv2D(16, kernel_size=(5,5),
                      strides=(2,2),
                      padding='valid'))

    modelo.add(AveragePooling2D(pool_size=(2,2),
                                strides=(2,2),
                                padding='valid'))

    modelo.add(Conv2D(120, kernel_size=(1,1),
                      strides=(1,1),
                      activation=fun_ativacao,
                      padding='valid'))

    modelo.add(Flatten())

    modelo.add(Dense(84, activation=fun_ativacao))
    modelo.add(Dense(10, activation='softmax'))
    return modelo

modelo = arquitetura_lenet_5('relu')
modelo.summary()

#FASE DE TREINAMENTO
modelo.compile(loss='sparse_categorical_crossentropy',
               optimizer='sgd', metrics=['accuracy'])

quantidade_de_epocas = 10

historico_treinamento = modelo.fit(x_treino, y_treino,
                                   validation_data=(x_validacao,y_validacao),
                                   batch_size=64,
                                   epochs=quantidade_de_epocas)
modelo.save('modelo_lenet5')

#CALCULANDO A FUNÇÃO DE PERDA E A PRECISÃO
loss, accuracy = modelo.evaluate(x_teste, y_teste, batch_size=64)
print('loss:{}'.format(loss))
print('accuracy:{}'.format(accuracy))

#FAZENDO A PREDIÇÃO
indice_imagem = 1976

predicao = modelo.predict(x_teste[indice_imagem].reshape(1,32,32,1))

print('Predição: {}'.format(predicao[0]), end='\n\n')

print('Meu modelo CNN prevê que o dígito na imagem é:', predicao.argmax())
plt.imshow(x_teste[indice_imagem].reshape(32,32), cmap='Greys')

#RECUPERANDO O MODELO QUE SALVAMOS ANTES
modelo_recuperado = tf.keras.models.load_model('modelo_lenet5')

plt.title('Calculo do Erro ao longo do treinamento')
plt.ylabel('Erro')
plt.xlabel('Época')

plt.plot(historico_treinamento.history['loss'])
plt.plot(historico_treinamento.history['val_loss'])
plt.legend(['loss (treinamento)', 'val_loss (validação)'], loc='upper right')
plt.show()