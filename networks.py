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

networks = [dict() for i in range(8)]

#   Network LSTM 38 -------------------------------------------- #
networks[0]['model_name'] = 'model_lstm_38'
networks[0]['dataset_causal_prediction'] = True
networks[0]['dataset_overlap_windows'] = True
networks[0]['features_input'] = True
networks[0]['pooling_input'] = 'GMP'
networks[0]['time_steps'] = 9

networks[0]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[0]['batch_size'] = 32
networks[0]['number_of_epochs'] = 90

networks[0]['optimizer'] = 'adam'
networks[0]['rcnn_type'] = 'lstm'
#networks[0]['rcnn_filters'] = 1
#networks[0]['rcnn_kernel'] = (3,3)
#networks[0]['rcnn_activation'] = 'relu'

networks[0]['overlaping_window'] = True
networks[0]['lstm_units'] = 128
networks[0]['lstm_dropout'] = 0.2

networks[0]['pooling'] = 'GMP'
networks[0]['hidden_fc'] = True
networks[0]['fc_nlinear_activation'] = 'tanh'
networks[0]['fc_nlinear_size'] = 128
networks[0]['fc_nlinear_activity_regularizer'] = 'l2'
networks[0]['loss_function'] = 'mse'

#   Network LSTM 39 -------------------------------------------- #
networks[1]['model_name'] = 'model_lstm_39'
networks[1]['dataset_causal_prediction'] = True
networks[1]['dataset_overlap_windows'] = True
networks[1]['features_input'] = True
networks[1]['pooling_input'] = 'GAP'
networks[1]['time_steps'] = 9

networks[1]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[1]['batch_size'] = 32
networks[1]['number_of_epochs'] = 90

networks[1]['optimizer'] = 'adam'
networks[1]['rcnn_type'] = 'lstm'
#networks[1]['rcnn_filters'] = 1
#networks[1]['rcnn_kernel'] = (3,3)
#networks[1]['rcnn_activation'] = 'relu'

networks[1]['overlaping_window'] = True
networks[1]['lstm_units'] = 128
networks[1]['lstm_dropout'] = 0.2

networks[1]['pooling'] = 'GAP'
networks[1]['hidden_fc'] = True
networks[1]['fc_nlinear_activation'] = 'tanh'
networks[1]['fc_nlinear_size'] = 128
networks[1]['fc_nlinear_activity_regularizer'] = None
networks[1]['loss_function'] = 'mse'

#   Network LSTM 40 -------------------------------------------- #
networks[2]['model_name'] = 'model_lstm_40'
networks[2]['dataset_causal_prediction'] = True
networks[2]['dataset_overlap_windows'] = True
networks[2]['features_input'] = True
networks[2]['pooling_input'] = 'GMP'
networks[2]['time_steps'] = 9

networks[2]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[2]['batch_size'] = 32
networks[2]['number_of_epochs'] = 90

networks[2]['optimizer'] = 'adam'
networks[2]['rcnn_type'] = 'lstm'
#networks[2]['rcnn_filters'] = 1
#networks[2]['rcnn_kernel'] = (3,3)
#networks[2]['rcnn_activation'] = 'relu'

networks[2]['overlaping_window'] = True
networks[2]['lstm_units'] = 128
networks[2]['lstm_dropout'] = 0.2

networks[2]['pooling'] = 'GMP'
networks[2]['hidden_fc'] = True
networks[2]['fc_nlinear_activation'] = 'tanh'
networks[2]['fc_nlinear_size'] = 128
networks[2]['fc_nlinear_activity_regularizer'] = None
networks[2]['loss_function'] = 'mse'

#   Network LSTM 41 -------------------------------------------- #
networks[3]['model_name'] = 'model_lstm_41'
networks[3]['dataset_causal_prediction'] = False
networks[3]['dataset_overlap_windows'] = True
networks[3]['features_input'] = True
networks[3]['pooling_input'] = 'GAP'
networks[3]['time_steps'] = 9

networks[3]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[3]['batch_size'] = 32
networks[3]['number_of_epochs'] = 90

networks[3]['optimizer'] = 'adam'
networks[3]['rcnn_type'] = 'lstm'
#networks[3]['rcnn_filters'] = 1
#networks[3]['rcnn_kernel'] = (3,3)
#networks[3]['rcnn_activation'] = 'relu'

networks[3]['overlaping_window'] = True
networks[3]['lstm_units'] = 128
networks[3]['lstm_dropout'] = 0.2

networks[3]['pooling'] = 'GAP'
networks[3]['hidden_fc'] = True
networks[3]['fc_nlinear_activation'] = 'tanh'
networks[3]['fc_nlinear_size'] = 128
networks[3]['fc_nlinear_activity_regularizer'] = 'l2'
networks[3]['loss_function'] = 'mse'

#   Network LSTM 42 -------------------------------------------- #
networks[4]['model_name'] = 'model_lstm_42'
networks[4]['dataset_causal_prediction'] = False
networks[4]['dataset_overlap_windows'] = True
networks[4]['features_input'] = True
networks[4]['pooling_input'] = 'GMP'
networks[4]['time_steps'] = 9

networks[4]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[4]['batch_size'] = 32
networks[4]['number_of_epochs'] = 90

networks[4]['optimizer'] = 'adam'
networks[4]['rcnn_type'] = 'lstm'
#networks[4]['rcnn_filters'] = 1
#networks[4]['rcnn_kernel'] = (3,3)
#networks[4]['rcnn_activation'] = 'relu'

networks[4]['overlaping_window'] = True
networks[4]['lstm_units'] = 128
networks[4]['lstm_dropout'] = 0.2

networks[4]['pooling'] = 'GMP'
networks[4]['hidden_fc'] = True
networks[4]['fc_nlinear_activation'] = 'tanh'
networks[4]['fc_nlinear_size'] = 128
networks[4]['fc_nlinear_activity_regularizer'] = 'l2'
networks[4]['loss_function'] = 'mse'

#   Network LSTM 43 -------------------------------------------- #
networks[5]['model_name'] = 'model_lstm_43'
networks[5]['dataset_causal_prediction'] = False
networks[5]['dataset_overlap_windows'] = True
networks[5]['features_input'] = True
networks[5]['pooling_input'] = 'GAP'
networks[5]['time_steps'] = 9

networks[5]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[5]['batch_size'] = 32
networks[5]['number_of_epochs'] = 90

networks[5]['optimizer'] = 'adam'
networks[5]['rcnn_type'] = 'lstm'
#networks[5]['rcnn_filters'] = 1
#networks[5]['rcnn_kernel'] = (3,3)
#networks[5]['rcnn_activation'] = 'relu'

networks[5]['overlaping_window'] = True
networks[5]['lstm_units'] = 128
networks[5]['lstm_dropout'] = 0.2

networks[5]['pooling'] = 'GAP'
networks[5]['hidden_fc'] = True
networks[5]['fc_nlinear_activation'] = 'tanh'
networks[5]['fc_nlinear_size'] = 128
networks[5]['fc_nlinear_activity_regularizer'] = None
networks[5]['loss_function'] = 'mse'

#   Network LSTM 44 -------------------------------------------- #
networks[6]['model_name'] = 'model_lstm_44'
networks[6]['dataset_causal_prediction'] = False
networks[6]['dataset_overlap_windows'] = True
networks[6]['features_input'] = True
networks[6]['pooling_input'] = 'GMP'
networks[6]['time_steps'] = 9

networks[6]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[6]['batch_size'] = 32
networks[6]['number_of_epochs'] = 90

networks[6]['optimizer'] = 'adam'
networks[6]['rcnn_type'] = 'lstm'
#networks[6]['rcnn_filters'] = 1
#networks[6]['rcnn_kernel'] = (3,3)
#networks[6]['rcnn_activation'] = 'relu'

networks[6]['overlaping_window'] = True
networks[6]['lstm_units'] = 128
networks[6]['lstm_dropout'] = 0.2

networks[6]['pooling'] = 'GMP'
networks[6]['hidden_fc'] = True
networks[6]['fc_nlinear_activation'] = 'tanh'
networks[6]['fc_nlinear_size'] = 128
networks[6]['fc_nlinear_activity_regularizer'] = None
networks[6]['loss_function'] = 'mse'

#   Network LSTM 44 -------------------------------------------- #
networks[6]['model_name'] = 'model_lstm_44'
networks[6]['dataset_causal_prediction'] = False
networks[6]['dataset_overlap_windows'] = True
networks[6]['features_input'] = True
networks[6]['pooling_input'] = 'GMP'
networks[6]['time_steps'] = 9

networks[6]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[6]['batch_size'] = 32
networks[6]['number_of_epochs'] = 90

networks[6]['optimizer'] = 'adam'
networks[6]['rcnn_type'] = 'lstm'
#networks[6]['rcnn_filters'] = 1
#networks[6]['rcnn_kernel'] = (3,3)
#networks[6]['rcnn_activation'] = 'relu'

networks[6]['overlaping_window'] = True
networks[6]['lstm_units'] = 128
networks[6]['lstm_dropout'] = 0.2

networks[6]['pooling'] = 'GMP'
networks[6]['hidden_fc'] = True
networks[6]['fc_nlinear_activation'] = 'tanh'
networks[6]['fc_nlinear_size'] = 128
networks[6]['fc_nlinear_activity_regularizer'] = None
networks[6]['loss_function'] = 'mse'

#   Network LSTM 45 -------------------------------------------- #
networks[7]['model_name'] = 'model_lstm_45'
networks[7]['dataset_causal_prediction'] = False
networks[7]['dataset_overlap_windows'] = True
networks[7]['features_input'] = True
networks[7]['pooling_input'] = 'GAP'
networks[7]['time_steps'] = 9

networks[7]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
networks[7]['batch_size'] = 32
networks[7]['number_of_epochs'] = 90

networks[7]['optimizer'] = 'adam'
networks[7]['rcnn_type'] = 'lstm'
#networks[7]['rcnn_filters'] = 1
#networks[7]['rcnn_kernel'] = (3,3)
#networks[7]['rcnn_activation'] = 'relu'

networks[7]['overlaping_window'] = True
networks[7]['lstm_units'] = 128
networks[7]['lstm_dropout'] = 0

networks[7]['pooling'] = 'GAP'
networks[7]['hidden_fc'] = True
networks[7]['fc_nlinear_activation'] = 'tanh'
networks[7]['fc_nlinear_size'] = 128
networks[7]['fc_nlinear_activity_regularizer'] = 'l2'
networks[7]['loss_function'] = 'mse'