# Sound Pressure Estimation Method Using Traffic Cameras and Convolutional Neural Networks
[Português](https://github.com/ma-ath/tcc-matheuslima/blob/master/README.md)
[]:[日本語](https://github.com/ma-ath/tcc-matheuslima/blob/master/readme/README.jp.md)

Sound pressure level on city streets is an important information for urban planning. Measurement of this information using microphones requires the implementation of a new infrastructure, what may require a considerable monetary investment. Cities usually have a CCTV (closed-circuit television) infrastructure to monitor traffic, but such cameras do not have microphones. We propose a solution for sound pressure inference using CCTV cameras, estimating the sound pressure on streets from traffic images.

In this work, we test previously proposed non-temporal convolutional neural networks architectures, and compare them with our proposed temporal convolutional neural network based on LSTM (long-short therm memory) architecture. The database used in this research is composed of 38 videos showing a traffic intersection over different days, hours and weather conditions, with overall length of 995 minutes. Using the images and sound signal from such videos, we train 130 model variations to predict mean sound pressure on unseen data, using only the video images as input. We evaluate model performance using the mean squared error and Pearson correlation of the predicted and targeted output signals, and applying cross validating with 10 different folds.

We observe that LSTM based neural networks consistently yield better results than non-temporal based architectures. The proposed neural networks have estimation errors below previously proposed networks, with a correlation of 71.3% between predicted and target signals. LSTM networks also have a filtered output signal. We observe that regularization methods such as dropout are essential during training. The convolutional neural network that yields the best results is the VGG16 (Oxford Visual Geometry Group, 16 convolutional layers). Classification architectures such as the Faster R-CNN (region-based convolutional neural network) have significant potential for improving prediction results.

We conclude that traffic sound pressure prediction from CCTV camera images is possible within an error limit. For future works, we propose improvements such as creating a new database with different places and better audio capture, exploring 3D convolutions, convolutional-LSTM layers, and investigating how classification networks may further improve results.

Key-words: Sound pressure, machine learning, convolutional neural networks, audiovisual signal processing, smart cities.


[Read full text](https://www.monografias.poli.ufrj.br/monografias/monopoli10032736.pdf)

# How to run

1. Place all videos in dataset/raw
2. Run the following scripts
   - _dataset_process.py_ - This script extracts all video frames and audio from the videos, and calculates their sound pressure.
   - _dataset_extract.py_ - This script inputs all extracted images through the convolutional networks, with frozen _imagenet_ weights.
   - _dataset_build.py_   - This script build all folds, geranting the input and output files for each fold. Output files are the sound pressures, and input files are the extracted _features_ from the CCTV images.   

3. Specify the networks to be trained on _networks.py_. To train, simply run the script _network_train.py_.

4. All results are automatically organized into the generated _"results"_ folder, dividided by _folds_.

All process can be monitored through a telegram bot. Simply insert your credentials into the _include/telegram_credentials.json_ file.
Configurations about GPU and other things can be adjusted in the _include/globals_and_functions.py_ file.

# Dependences

* **Python 3.6.9**
* **Tensorflow 2.0**
* **Keras**
* **ffmpeg** e **ffprobe** para a geração do banco de dados
* **OpenCV**

# Bibliografy & External Links
All bibliografy can be find in the final text of this work.
