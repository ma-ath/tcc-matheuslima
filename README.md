# Método de Estimação da Pressão Sonora em Câmeras de Segurança Usando Redes Neurais Artificiais
# 画像処理、深層学習を用いた都市騒音推測方式

  O nível de pressão sonora em uma certa região da cidade é uma informação importante para as autoridades públicas, e que deve ser levada em conta em diversas atividades de tomadas de decisão relacionados ao georeferenciamento de empreendimentos públicos e privados. Em contrapartida, medir ativamente ponto a ponto da cidade para o levantamento desses dados é uma tarefa arduosa e cara. Câmeras de segurança já oferencem um banco de dados que contêm informações visuais de diversos pontos da cidades, sem entretanto, por motivos de legislação, possuírem qualquer informação de áudio. Esse trabalho tem por objetivo então de, utilizando-se de uma infraestrutura já implementada de câmeras de segurança, fazer a estimação do nível de pressão sonora em vídeos mudos a partir do treinamento de um modelo computacional feito sobre vídeo não-mudos, de forma a leventar um mapa de ruído sonoro em cidades.

# Como rodar o código

1 - Ponha os vídeos "raw" na pasta dataset/raw

2 - Rode o arquivo dataset_rawVideoProcess.py
       Esse arquivo irá extrair todos os frames e audio dos vídeos na pasta raw, além de calcular sua presão sonora.

3 - Rode o arquivo dataset_build.py
      Esse arquivo irá pegar todas as informações de audio e vídeo dos arquivos anteriormente processados, e montar um único dataset, no qual a rede irá ser treinada

4 - Agora você já pode treinar a rede, com o arquivo network_train.py

5 - Por fim, o arquivo network_evaluate.py irá avaliar o treinamento.

Todo o processo pode ser monitorado remotamente por meio de um bot de telegram, que envia logs definidos pelo programador a respeito de como anda o andamento do processo.

# Bibliotecas e Dependências

Principais Dependências:
* **Python 3.6.9**
* **Tensorflow 2.0**
* **Keras**
* **ffmpeg** e **ffprobe** para a geração do banco de dados
* **OpenCV**

# Bibliografia & Links Externos
[1] SOUND PRESSURE LEVEL PREDICTION FROM VIDEO FRAMES USING DEEP CONVOLUTIONAL NEURAL NETWORKS - Leonardo Oliveira Mazza (Mestrado) Orientador: José Gabriel Rodriguez Carneiro Gomes
