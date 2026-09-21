LeNet-5 com MNIST

Exercício prático da disciplina de [nome da disciplina], em que implementei a arquitetura LeNet-5 com TensorFlow/Keras para reconhecer dígitos escritos à mão (dataset MNIST).

Este código foi feito acompanhando as aulas do professor. O objetivo aqui é estudo: entender o que cada etapa da arquitetura faz.

O que o código faz
Carrega o MNIST (60.000 imagens de treino e 10.000 de teste, 28x28 pixels em tons de cinza).
Separa os dados: 55.000 para treino, 5.000 para validação e 10.000 para teste.
Prepara as imagens:
adiciona a dimensão do canal, (28, 28) → (28, 28, 1);
aplica padding de 2 pixels nas bordas para chegar em (32, 32, 1), o tamanho de entrada da LeNet-5;
normaliza os pixels de 0–255 para 0–1.
Monta a arquitetura LeNet-5:
Camada	Detalhes	Saída
Conv2D	6 filtros 5x5	(28, 28, 6)
AveragePooling2D	2x2	(14, 14, 6)
Conv2D	16 filtros 5x5	(10, 10, 16)
AveragePooling2D	2x2	(5, 5, 16)
Conv2D	120 filtros 1x1	(5, 5, 120)
Flatten		(3000)
Dense	84 neurônios	(84)
Dense	10 neurônios, softmax	(10)
Treina por 10 épocas (SGD, batch de 64) e acompanha a perda de treino e validação.
Avalia no conjunto de teste e faz a predição de uma imagem.
Resultado
Acurácia no teste: XX% (preencha com o valor que apareceu no seu terminal)
Perda no teste: X.XX
O que aprendi
(escreva aqui, com suas palavras, 2 ou 3 coisas. Exemplos: por que o padding foi necessário, para que serve o Flatten, o que a normalização muda no treino.)
Como rodar
bash
# criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows

# instalar as dependências
pip install -r requirements.txt

# executar
python lenet5_mnist.py

Na primeira execução, o MNIST é baixado automaticamente.

Tecnologias

Python · TensorFlow/Keras · NumPy · Matplotlib