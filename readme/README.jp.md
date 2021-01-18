# 画像処理、深層学習を用いた都市騒音推測
[Português](https://github.com/ma-ath/tcc-matheuslima/blob/master/README.md)
[English](https://github.com/ma-ath/tcc-matheuslima/blob/master/readme/README.en.md)

都市における騒音分布図は都市計画に関する重要な情報である。Measurement of this information using microphones requires the implementation of a new infrastructure, what may require a considerable monetary investment. Cities usually have a CCTV (closed-circuit television) infrastructure to monitor traffic, but such cameras do not have microphones. We propose a solution for sound pressure inference using CCTV cameras, estimating the sound pressure on streets from traffic images.

In this work, we test previously proposed non-temporal convolutional neural networks architectures, and compare them with our proposed temporal convolutional neural network based on LSTM (long-short therm memory) architecture. The database used in this research is composed of 38 videos showing a traffic intersection over different days, hours and weather conditions, with overall length of 995 minutes. Using the images and sound signal from such videos, we train 130 model variations to predict mean sound pressure on unseen data, using only the video images as input. We evaluate model performance using the mean squared error and Pearson correlation of the predicted and targeted output signals, and applying cross validating with 10 different folds.

We observe that LSTM based neural networks consistently yield better results than non-temporal based architectures. The proposed neural networks have estimation errors below previously proposed networks, with a correlation of 71.3% between predicted and target signals. LSTM networks also have a filtered output signal. We observe that regularization methods such as dropout are essential during training. The convolutional neural network that yields the best results is the VGG16 (Oxford Visual Geometry Group, 16 convolutional layers). Classification architectures such as the Faster R-CNN (region-based convolutional neural network) have significant potential for improving prediction results.

We conclude that traffic sound pressure prediction from CCTV camera images is possible within an error limit. For future works, we propose improvements such as creating a new database with different places and better audio capture, exploring 3D convolutions, convolutional-LSTM layers, and investigating how classification networks may further improve results.

Key-words: Sound pressure, machine learning, convolutional neural networks, audiovisual signal processing, smart cities.

[Read full text](https://drive.google.com/file/d/1H2Wuc7mlNF-sxCVDyYWWtQqwYYK3zNZe/view?usp=sharing)
How to run
    Place all videos in dataset/raw
    Run the following scripts
        dataset_process.py - This script extracts all video frames and audio from the videos, and calculates their sound pressure.
        dataset_extract.py - This script inputs all extracted images through the convolutional networks, with frozen imagenet weights.
        dataset_build.py - This script build all folds, geranting the input and output files for each fold. Output files are the sound pressures, and input files are the extracted features from the CCTV images.
    Specify the networks to be trained on networks.py. To train, simply run the script network_train.py.
    All results are automatically organized into the generated "results" folder, dividided by folds.

All process can be monitored through a telegram bot. Simply insert your credentials into the include/telegram_credentials.json file. Configurations about GPU and other things can be adjusted in the include/globals_and_functions.py file.
Dependences
    Python 3.6.9
    Tensorflow 2.0
    Keras
    ffmpeg e ffprobe para a geração do banco de dados
    OpenCV

Bibliografy & External Links

All bibliografy can be find in the final text of this work.
