from include.globals_and_functions import *

net_number = 4
networks = [dict() for i in range(net_number)]

for i in range(net_number):
    networks[i]['model_name'] = 'model_foldtraining_'+str(i+60)
    # --- loadDataset --- #
    networks[i]['cnn'] = 'vgg16'
    networks[i]['pooling'] = 'GAP'
    #networks[i]['lstm'] = True
    networks[i]['time_steps'] = 32
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2
    networks[i]['lstm_stateful'] = False
    networks[i]['hiddenfc_before_lstm'] = False
    networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    #networks[i]['hiddenfc_activity_regularizer'] = None
    #networks[i]['hiddenfc_dropout'] = 0
    networks[i]['fasterRCNN_support'] = False
    networks[i]['fasterRCNN_type'] = 'dense'    #'dense' or 'sparse'
    networks[i]['fasterRCNN_dense_size'] = 64
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 300
    # --- training --- #
#   0
networks[0]['lstm'] = True
networks[0]['hiddenfc_activity_regularizer'] = 'l2'
networks[0]['hiddenfc_dropout'] = 0
#   1
networks[1]['lstm'] = True
networks[1]['hiddenfc_activity_regularizer'] = None
networks[1]['hiddenfc_dropout'] = 0.5
#   2
networks[2]['lstm'] = False
networks[2]['hiddenfc_activity_regularizer'] = 'l2'
networks[2]['hiddenfc_dropout'] = 0
#   3
networks[3]['lstm'] = False
networks[3]['hiddenfc_activity_regularizer'] = None
networks[3]['hiddenfc_dropout'] = 0.5

"""
    I still have to run these networks. I paused this training to do some other more-important training before

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_foldtraining_'+str(i+54)
    # --- loadDataset --- #
    #networks[i]['cnn'] = 'inceptionV3'
    networks[i]['pooling'] = 'GAP'
    #networks[i]['lstm'] = True
    networks[i]['time_steps'] = 32
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2
    networks[i]['lstm_stateful'] = False
    networks[i]['hiddenfc_before_lstm'] = False
    networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    networks[i]['hiddenfc_activity_regularizer'] = None
    networks[i]['hiddenfc_dropout'] = 0
    networks[i]['fasterRCNN_support'] = True
    #networks[i]['fasterRCNN_type'] = 'dense'    #'dense' or 'sparse'
    networks[i]['fasterRCNN_dense_size'] = 64
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 50
    # --- training --- #
#   0
networks[0]['cnn'] = 'vgg16'
networks[0]['lstm'] = True
networks[0]['fasterRCNN_type'] = 'dense'
#   1
networks[1]['cnn'] = 'resnet50'
networks[1]['lstm'] = True
networks[1]['fasterRCNN_type'] = 'dense'
#   2
networks[2]['cnn'] = 'inceptionV3'
networks[2]['lstm'] = True
networks[2]['fasterRCNN_type'] = 'dense'
#   3
networks[3]['cnn'] = 'vgg16'
networks[3]['lstm'] = True
networks[3]['fasterRCNN_type'] = 'sparse'
#   4
networks[4]['cnn'] = 'resnet50'
networks[4]['lstm'] = True
networks[4]['fasterRCNN_type'] = 'sparse'
#   5
networks[5]['cnn'] = 'inceptionV3'
networks[5]['lstm'] = True
networks[5]['fasterRCNN_type'] = 'sparse'
"""
#   Diferent parameters generation  --------------------------------- #
"""
    Networks trained between november 10 and november 17
#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_foldtraining_'+str(i+42)
    # --- loadDataset --- #
    #networks[i]['cnn'] = 'inceptionV3'
    networks[i]['pooling'] = 'GAP'
    #networks[i]['lstm'] = True
    networks[i]['time_steps'] = 32
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2
    networks[i]['lstm_stateful'] = False
    networks[i]['hiddenfc_before_lstm'] = False
    networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    networks[i]['hiddenfc_activity_regularizer'] = None
    networks[i]['hiddenfc_dropout'] = 0
    networks[i]['fasterRCNN_support'] = True
    #networks[i]['fasterRCNN_type'] = 'dense'    #'dense' or 'sparse'
    networks[i]['fasterRCNN_dense_size'] = 64
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 50
    # --- training --- #
#   0
networks[0]['cnn'] = 'vgg16'
networks[0]['lstm'] = True
networks[0]['fasterRCNN_type'] = 'dense'
#   1
networks[1]['cnn'] = 'resnet50'
networks[1]['lstm'] = True
networks[1]['fasterRCNN_type'] = 'dense'
#   2
networks[2]['cnn'] = 'inceptionV3'
networks[2]['lstm'] = True
networks[2]['fasterRCNN_type'] = 'dense'
#   3
networks[3]['cnn'] = 'vgg16'
networks[3]['lstm'] = True
networks[3]['fasterRCNN_type'] = 'sparse'
#   4
networks[4]['cnn'] = 'resnet50'
networks[4]['lstm'] = True
networks[4]['fasterRCNN_type'] = 'sparse'
#   5
networks[5]['cnn'] = 'inceptionV3'
networks[5]['lstm'] = True
networks[5]['fasterRCNN_type'] = 'sparse'
#   6
networks[6]['cnn'] = 'vgg16'
networks[6]['lstm'] = False
networks[6]['time_steps'] = 9   #Irrelevant
networks[6]['fasterRCNN_type'] = 'dense'
#   7
networks[7]['cnn'] = 'resnet50'
networks[7]['lstm'] = False
networks[7]['time_steps'] = 9   #Irrelevant
networks[7]['fasterRCNN_type'] = 'dense'
#   8
networks[8]['cnn'] = 'inceptionV3'
networks[8]['lstm'] = False
networks[8]['time_steps'] = 9   #Irrelevant
networks[8]['fasterRCNN_type'] = 'dense'
#   9
networks[9]['cnn'] = 'vgg16'
networks[9]['lstm'] = False
networks[9]['time_steps'] = 9   #Irrelevant
networks[9]['fasterRCNN_type'] = 'sparse'
#   10
networks[10]['cnn'] = 'resnet50'
networks[10]['lstm'] = False
networks[10]['time_steps'] = 9   #Irrelevant
networks[10]['fasterRCNN_type'] = 'sparse'
#   11
networks[11]['cnn'] = 'inceptionV3'
networks[11]['lstm'] = False
networks[11]['time_steps'] = 9   #Irrelevant
networks[11]['fasterRCNN_type'] = 'sparse'
"""
#   Diferent parameters generation  --------------------------------- #

"""
The following networks are the ones trained between october 25 and november 6

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_foldtraining_'+str(i+30)
    # --- loadDataset --- #
    #networks[i]['cnn'] = 'vgg16'
    networks[i]['pooling'] = 'GAP'
    networks[i]['lstm'] = True
    #networks[i]['time_steps'] = 9
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2
    networks[i]['lstm_batchnormalization'] = False
    networks[i]['lstm_stateful'] = True
    networks[i]['hiddenfc_before_lstm'] = False
    #networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    networks[i]['hiddenfc_activity_regularizer'] = None
    networks[i]['hiddenfc_dropout'] = 0
    networks[i]['fasterRCNN_support'] = False
    networks[i]['fasterRCNN_type'] = 'dense'    #'dense' or 'sparse'
    networks[i]['fasterRCNN_dense_size'] = 128
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 50
    # --- training --- #

#   0
networks[0]['cnn'] = 'vgg16'
networks[0]['time_steps'] = 9
networks[0]['hiddenfc'] = True
#   1
networks[1]['cnn'] = 'vgg16'
networks[1]['time_steps'] = 32
networks[1]['hiddenfc'] = True
#   2
networks[2]['cnn'] = 'resnet50'
networks[2]['time_steps'] = 9
networks[2]['hiddenfc'] = True
#   3
networks[3]['cnn'] = 'resnet50'
networks[3]['time_steps'] = 32
networks[3]['hiddenfc'] = True
#   4
networks[4]['cnn'] = 'inceptionV3'
networks[4]['time_steps'] = 9
networks[4]['hiddenfc'] = True
#   5
networks[5]['cnn'] = 'inceptionV3'
networks[5]['time_steps'] = 32
networks[5]['hiddenfc'] = True
#   6
networks[6]['cnn'] = 'vgg16'
networks[6]['time_steps'] = 9
networks[6]['hiddenfc'] = False
#   7
networks[7]['cnn'] = 'vgg16'
networks[7]['time_steps'] = 32
networks[7]['hiddenfc'] = False
#   8
networks[8]['cnn'] = 'resnet50'
networks[8]['time_steps'] = 9
networks[8]['hiddenfc'] = False
#   9
networks[9]['cnn'] = 'resnet50'
networks[9]['time_steps'] = 32
networks[9]['hiddenfc'] = False
#   10
networks[10]['cnn'] = 'inceptionV3'
networks[10]['time_steps'] = 9
networks[10]['hiddenfc'] = False
#   11
networks[11]['cnn'] = 'inceptionV3'
networks[11]['time_steps'] = 32
networks[11]['hiddenfc'] = False
"""
"""
The following networks are the ones trained between october 4 and october 13

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_foldtraining_'+str(i+18)
    # --- loadDataset --- #
    #networks[i]['cnn'] = 'inceptionV3'
    networks[i]['pooling'] = 'GAP'
    networks[i]['lstm'] = True
    #networks[i]['time_steps'] = 20
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2
    networks[i]['lstm_batchnormalization'] = False
    networks[i]['lstm_stateful'] = False
    networks[i]['hiddenfc_before_lstm'] = True
    #networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    networks[i]['hiddenfc_activity_regularizer'] = None
    networks[i]['hiddenfc_dropout'] = 0
    networks[i]['fasterRCNN_support'] = False
    networks[i]['fasterRCNN_type'] = 'dense'    #'dense' or 'sparse'
    networks[i]['fasterRCNN_dense_size'] = 128
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 50
    # --- training --- #
#   0
networks[0]['cnn'] = 'vgg16'
networks[0]['time_steps'] = 9
networks[0]['hiddenfc'] = True
#   1
networks[1]['cnn'] = 'vgg16'
networks[1]['time_steps'] = 32
networks[1]['hiddenfc'] = True
#   2
networks[2]['cnn'] = 'resnet50'
networks[2]['time_steps'] = 9
networks[2]['hiddenfc'] = True
#   3
networks[3]['cnn'] = 'resnet50'
networks[3]['time_steps'] = 32
networks[3]['hiddenfc'] = True
#   4
networks[4]['cnn'] = 'inceptionV3'
networks[4]['time_steps'] = 9
networks[4]['hiddenfc'] = True
#   5
networks[5]['cnn'] = 'inceptionV3'
networks[5]['time_steps'] = 32
networks[5]['hiddenfc'] = True
#   6
networks[6]['cnn'] = 'vgg16'
networks[6]['time_steps'] = 9
networks[6]['hiddenfc'] = False
#   7
networks[7]['cnn'] = 'vgg16'
networks[7]['time_steps'] = 32
networks[7]['hiddenfc'] = False
#   8
networks[8]['cnn'] = 'resnet50'
networks[8]['time_steps'] = 9
networks[8]['hiddenfc'] = False
#   9
networks[9]['cnn'] = 'resnet50'
networks[9]['time_steps'] = 32
networks[9]['hiddenfc'] = False
#   10
networks[10]['cnn'] = 'inceptionV3'
networks[10]['time_steps'] = 9
networks[10]['hiddenfc'] = False
#   11
networks[11]['cnn'] = 'inceptionV3'
networks[11]['time_steps'] = 32
networks[11]['hiddenfc'] = False
"""

"""
The following networks are the ones trained between september 21 and october 4

net_number = 18
networks = [dict() for i in range(net_number)]

#   Diferent parameters generation  --------------------------------- #

#   FIXED PARAMETERS FOR ALL NETWORKS
for i in range(net_number):
    networks[i]['model_name'] = 'model_foldtraining_'+str(i)
    # --- loadDataset --- #
    #networks[i]['cnn'] = 'inceptionV3'
    networks[i]['pooling'] = 'GAP'
    #networks[i]['lstm'] = True
    #networks[i]['time_steps'] = 20
    networks[i]['overlap_windows'] = True
    networks[i]['causal_prediction'] = False
    # --- loadDataset --- #
    # --- networkModel --- #
    networks[i]['lstm_outputsize'] = 128
    networks[i]['lstm_dropout'] = 0.2
    networks[i]['lstm_stateful'] = False
    networks[i]['hiddenfc_before_lstm'] = False
    #networks[i]['hiddenfc'] = True
    networks[i]['hiddenfc_activation'] = 'tanh'
    networks[i]['hiddenfc_size'] = 128
    networks[i]['hiddenfc_activity_regularizer'] = None
    networks[i]['hiddenfc_dropout'] = 0
    networks[i]['fasterRCNN_support'] = False
    networks[i]['fasterRCNN_type'] = 'dense'    #'dense' or 'sparse'
    networks[i]['fasterRCNN_dense_size'] = 128
    # --- networkModel --- #
    # --- training --- #
    networks[i]['learning_schedule'] = [0.0001, 5e-05, 1e-05]
    networks[i]['optimizer'] = 'adam'
    networks[i]['loss_function'] = 'mse'
    networks[i]['batch_size'] = 32
    networks[i]['epochs'] = 50
    # --- training --- #
#   0
networks[0]['cnn'] = 'vgg16'
networks[0]['lstm'] = True
networks[0]['time_steps'] = 9
networks[0]['hiddenfc'] = True
#   1
networks[1]['cnn'] = 'vgg16'
networks[1]['lstm'] = True
networks[1]['time_steps'] = 32
networks[1]['hiddenfc'] = True
#   2
networks[2]['cnn'] = 'resnet50'
networks[2]['lstm'] = True
networks[2]['time_steps'] = 9
networks[2]['hiddenfc'] = True
#   3
networks[3]['cnn'] = 'resnet50'
networks[3]['lstm'] = True
networks[3]['time_steps'] = 32
networks[3]['hiddenfc'] = True
#   4
networks[4]['cnn'] = 'inceptionV3'
networks[4]['lstm'] = True
networks[4]['time_steps'] = 9
networks[4]['hiddenfc'] = True
#   5
networks[5]['cnn'] = 'inceptionV3'
networks[5]['lstm'] = True
networks[5]['time_steps'] = 32
networks[5]['hiddenfc'] = True
#   6
networks[6]['cnn'] = 'vgg16'
networks[6]['lstm'] = True
networks[6]['time_steps'] = 9
networks[6]['hiddenfc'] = False
#   7
networks[7]['cnn'] = 'vgg16'
networks[7]['lstm'] = True
networks[7]['time_steps'] = 32
networks[7]['hiddenfc'] = False
#   8
networks[8]['cnn'] = 'resnet50'
networks[8]['lstm'] = True
networks[8]['time_steps'] = 9
networks[8]['hiddenfc'] = False
#   9
networks[9]['cnn'] = 'resnet50'
networks[9]['lstm'] = True
networks[9]['time_steps'] = 32
networks[9]['hiddenfc'] = False
#   10
networks[10]['cnn'] = 'inceptionV3'
networks[10]['lstm'] = True
networks[10]['time_steps'] = 9
networks[10]['hiddenfc'] = False
#   11
networks[11]['cnn'] = 'inceptionV3'
networks[11]['lstm'] = True
networks[11]['time_steps'] = 32
networks[11]['hiddenfc'] = False
#   12
networks[12]['cnn'] = 'vgg16'
networks[12]['lstm'] = False
networks[12]['time_steps'] = 9   #Irrelevant
networks[12]['hiddenfc'] = True
#   13
networks[13]['cnn'] = 'resnet50'
networks[13]['lstm'] = False
networks[13]['time_steps'] = 9   #Irrelevant
networks[13]['hiddenfc'] = True
#   14
networks[14]['cnn'] = 'inceptionV3'
networks[14]['lstm'] = False
networks[14]['time_steps'] = 9   #Irrelevant
networks[14]['hiddenfc'] = True
#   15
networks[15]['cnn'] = 'vgg16'
networks[15]['lstm'] = False
networks[15]['time_steps'] = 9   #Irrelevant
networks[15]['hiddenfc'] = False
#   16
networks[16]['cnn'] = 'resnet50'
networks[16]['lstm'] = False
networks[16]['time_steps'] = 9   #Irrelevant
networks[16]['hiddenfc'] = False
#   17
networks[17]['cnn'] = 'inceptionV3'
networks[17]['lstm'] = False
networks[17]['time_steps'] = 9   #Irrelevant
networks[17]['hiddenfc'] = False
"""