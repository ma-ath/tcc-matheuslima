from include.globals_and_functions import *

"""
    This file contains 
    Python dictionaries with all networks to be tested 

    Initialize a python dictionary with the networks structures to be tested
   
    The basic network structure that is being tested is based on network a) of Carreira, J., & Zisserman, A. (2017). Quo Vadis, action recognition? A new model and the kinetics dataset.

    timeDistributed(VGG16) -> 
    timeDistributed(Pooling) -> 
    rcnn -> 
    Hidden FC -> 
    FC (output)

    We only change some parameters
"""

networks = [dict() for i in range(1)]

#   Network LSTM 37 -------------------------------------------- #
networks[0]['model_name'] = 'model_lstm_37'
networks[0]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[0]['batch_size'] = 32
networks[0]['number_of_epochs'] = 128
networks[0]['time_steps'] = 9
networks[0]['features_input'] = True
networks[0]['pooling_input'] = 'GAP'

networks[0]['optimizer'] = 'adam'
networks[0]['rcnn_type'] = 'lstm'
#networks[0]['rcnn_filters'] = 1
#networks[0]['rcnn_kernel'] = (3,3)
#networks[0]['rcnn_activation'] = 'relu'

networks[0]['overlaping_window'] = True
networks[0]['lstm_units'] = 128
networks[0]['lstm_dropout'] = 0.2

networks[0]['pooling'] = 'GAP'
networks[0]['hidden_fc'] = True
networks[0]['fc_nlinear_activation'] = 'tanh'
networks[0]['fc_nlinear_size'] = 128
networks[0]['fc_nlinear_activity_regularizer'] = 'l2'
networks[0]['loss_function'] = 'mse'