from include.globals_and_functions import *

net_number = 4
networks = [dict() for i in range(net_number)]

#   Diferent parameters generation  --------------------------------- #

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_newlstm_'+str(i)
    # --- loadDataset --- #
    networks[i]['cnn'] = 'vgg16'
    networks[i]['pooling'] = 'GAP'
    networks[i]['lstm'] = True
    networks[i]['time_steps'] = 3
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2

    networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    networks[i]['hiddenfc_activity_regularizer'] = None
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 1
    # --- training --- #

networks[1]['hiddenfc'] = False
networks[2]['lstm'] = False
networks[3]['lstm'] = False
networks[3]['hiddenfc'] = False
