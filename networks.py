from include.globals_and_functions import *

"""
    This file contains 
    Python dictionaries with all networks to be tested 

    Initialize a python dictionary with the networks structures to be tested
   
    The basic network structure that is being tested is based on network a) of Carreira, J., & Zisserman, A. (2017). Quo Vadis, action recognition? A new model and the kinetics dataset.

    timeDistributed(VGG16) -> 
    rcnn -> 
    timeDistributed(Pooling) -> 
    timeDistributed(FC) -> 
    timeDistributed(output)

    We only change some parameters
"""

networks = [dict() for i in range(4)]

#   Network LEO 01 -------------------------------------------- #
networks[0]['model_name'] = 'model_leo_01'
networks[0]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[0]['batch_size'] = 1
networks[0]['number_of_epochs'] = 64
networks[0]['time_steps'] = 9
networks[0]['features_input'] = True

networks[0]['optimizer'] = 'adam'
networks[0]['rcnn_type'] = 'no_rnn'
#networks[0]['rcnn_filters'] = 1
#networks[0]['rcnn_kernel'] = (3,3)
#networks[0]['rcnn_activation'] = 'relu'

#networks[0]['overlaping_window'] = True
#networks[0]['lstm_units'] = 512

networks[0]['pooling'] = 'GAP'
networks[0]['hidden_fc'] = True
networks[0]['fc_nlinear_activation'] = 'tanh'
networks[0]['fc_nlinear_size'] = 128
networks[0]['loss_function'] = 'mse'

#   Network LEO 02 -------------------------------------------- #
networks[1]['model_name'] = 'model_leo_02'
networks[1]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[1]['batch_size'] = 1
networks[1]['number_of_epochs'] = 64
networks[1]['time_steps'] = 9
networks[1]['features_input'] = True

networks[1]['optimizer'] = 'adam'
networks[1]['rcnn_type'] = 'no_rnn'
#networks[0]['rcnn_filters'] = 1
#networks[0]['rcnn_kernel'] = (3,3)
#networks[0]['rcnn_activation'] = 'relu'

#networks[0]['overlaping_window'] = True
#networks[0]['lstm_units'] = 512

networks[1]['pooling'] = 'GAP'
networks[1]['hidden_fc'] = False
networks[1]['fc_nlinear_activation'] = 'tanh'
networks[1]['fc_nlinear_size'] = 128
networks[1]['loss_function'] = 'mse'

#   Network LEO 03 -------------------------------------------- #
networks[2]['model_name'] = 'model_leo_03'
networks[2]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[2]['batch_size'] = 1
networks[2]['number_of_epochs'] = 64
networks[2]['time_steps'] = 9
networks[2]['features_input'] = True

networks[2]['optimizer'] = 'adam'
networks[2]['rcnn_type'] = 'no_rnn'
#networks[0]['rcnn_filters'] = 1
#networks[0]['rcnn_kernel'] = (3,3)
#networks[0]['rcnn_activation'] = 'relu'

#networks[0]['overlaping_window'] = True
#networks[0]['lstm_units'] = 512

networks[2]['pooling'] = 'GAP'
networks[2]['hidden_fc'] = True
networks[2]['fc_nlinear_activation'] = 'relu'
networks[2]['fc_nlinear_size'] = 128
networks[2]['loss_function'] = 'mse'

#   Network LEO 04 -------------------------------------------- #
networks[3]['model_name'] = 'model_leo_04'
networks[3]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[3]['batch_size'] = 1
networks[3]['number_of_epochs'] = 64
networks[3]['time_steps'] = 9
networks[3]['features_input'] = True

networks[3]['optimizer'] = 'adam'
networks[3]['rcnn_type'] = 'no_rnn'
#networks[0]['rcnn_filters'] = 1
#networks[0]['rcnn_kernel'] = (3,3)
#networks[0]['rcnn_activation'] = 'relu'

#networks[0]['overlaping_window'] = True
#networks[0]['lstm_units'] = 512

networks[3]['pooling'] = 'GAP'
networks[3]['hidden_fc'] = False
networks[3]['fc_nlinear_activation'] = 'relu'
networks[3]['fc_nlinear_size'] = 128
networks[3]['loss_function'] = 'mse'