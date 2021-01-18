# Método de Estimação da Pressão Sonora em Câmeras de Segurança Usando Redes Neurais Artificiais
[English](https://github.com/ma-ath/tcc-matheuslima/blob/master/readme/README.en.md)
[日本語](https://github.com/ma-ath/tcc-matheuslima/blob/master/readme/README.jp.md)

O nível de pressão sonora em ruas é uma informação importante para o planejamento urbano. Medir esta informação através de uma rede de microfones é uma solução que requer a implementação de uma nova infraestrutura, o que pode demandar um investimento monetário considerável. Cidades costumam já possuir uma infraestrutura de câmeras de tráfego. Tais câmeras entretanto não costumam possuir microfones. Propomos uma solução para avaliação de pressão sonora em cidades utilizando a infraestrutura já existente de câmeras de tráfego, estimando a pressão sonora a partir de imagens das mesmas.

Testamos modelos de redes neurais convolucionais com arquitetura não-temporal e propomos modelos com arquitetura temporal (LSTM, do inglês _"long short-term memory"_). Utilizando uma base de dados de 38 vídeos de tráfego com áudio e um total de 995 minutos, treinamos 130 variações de redes convolucionais para fazer a predição de valores médios do sinal de áudio, em escala logarítmica, a partir de imagens do vídeo. Avaliamos o desempenho das redes neurais em termos do erro médio quadrático entre as suas saídas e os seus alvos, e também em termos da correlação entre esses sinais, fazendo uma validação cruzada entre 10 diferentes _"folds"_.

Neste trabalho observamos que as redes neurais temporais não-causais baseadas em LSTM obtêm consistentemente resultados melhores que aquelas que não possuem arquitetura temporal. As redes propostas obtiveram um erro de medição abaixo das estudadas em trabalhos anteriores, demonstrando uma correlação entre o sinal predito e o real de 71,3%. As redes LSTM também apresentam um sinal de saída menos ruidoso que aquele apresentado pelas redes não-temporais. O uso de técnicas regularizadores como o _"dropout”_ se mostra decisivo para o treinamento. A rede convolucional testada que apresenta o melhor resultado é a VGG16.

Concluímos que a predição do nível sonoro de ruas a partir de imagens de câmeras é possível dentro de uma margem de erro. Constatamos que o uso de redes classificadoras auxiliares (como a _Faster R-CNN_ ou _YoloV4_) têm potencial para melhorar as predições.

Palavras-Chave: Pressão sonora, aprendizado de máquina, redes convolucionais, processamento audiovisual, cidades inteligentes.

[Texto completo pode ser encontrado aqui](https://drive.google.com/file/d/1H2Wuc7mlNF-sxCVDyYWWtQqwYYK3zNZe/view?usp=sharing)

# Como rodar o código

1. Ponha os vídeos _"raw"_ na pasta dataset/raw
2. Execute os arquivos na seguinte ordem
   - _dataset_process.py_ - Esse script irá extrair todos os frames e audio dos vídeos da pasta raw, além de calcular sua presão sonora.
   - _dataset_extract.py_ - Esse script passará todos as imagens extraídas através das redes convolucionais, com pesos congelados da _imagenet_.
   - _dataset_build.py_   - Esse script montará todos os folds especificados, gerando os arquivos de entrada e saída para cada fold. No caso, os arquivos de saída são as pressões sonoras calculadas, e as entradas são as _features_ das imagens das câmeras extraídas pelas redes convolucionais.   

3. Especifique as redes a serem treinadas no arquivo _networks.py_. Para rodas as redes, basta usar o script _network_train.py_

4. Os resultados de cada análise são organizados automaticamente na pasta gerada _"results"_, divididos por _folds_.

Todo o processo pode ser monitorado remotamente por meio de um bot de telegram, que envia logs definidos pelo programador a respeito de como anda o andamento do processo. Basta inserir o código do bot e seu código de contato no arquivo _include/telegram_credentials.json_.
Configurações de quais GPUs são utilizadas durante o treinamento, entre outras coisas, podem ser encontrados no arquivo _include/globals_and_functions.py_.

# Bibliotecas e Dependências

Principais Dependências:
* **Python 3.6.9**
* **Tensorflow 2.0**
* **Keras**
* **ffmpeg** e **ffprobe** para a geração do banco de dados
* **OpenCV**

# Bibliografia & Links Externos
Toda a bibliografia utilizada durante esse trabalho está disponível no texto final apresentado.
