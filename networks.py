from include.global_constants import *

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

networks = [dict() for i in range(8)]

#   Network 1 -------------------------------------------- #
networks[0]['model_name'] = 'model_1'
networks[0]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[0]['batch_size'] = 4
networks[0]['number_of_epochs'] = 32

networks[0]['optimizer'] = 'adam'
networks[0]['rcnn_type'] = 'convlstm'
networks[0]['rcnn_filters'] = 10
networks[0]['rcnn_kernel'] = (3,3)
networks[0]['rcnn_activation'] = 'relu'

networks[0]['pooling'] = 'GAP'
networks[0]['fc_nlinear_activation'] = 'relu'
networks[0]['fc_nlinear_size'] = 128
networks[0]['loss_function'] = 'mse'

#   Network 2 -------------------------------------------- #
networks[1]['model_name'] = 'model_2'
networks[1]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[1]['batch_size'] = 4
networks[1]['number_of_epochs'] = 32

networks[1]['optimizer'] = 'adam'
networks[1]['rcnn_type'] = 'convlstm'
networks[1]['rcnn_filters'] = 10
networks[1]['rcnn_kernel'] = (3,3)
networks[1]['rcnn_activation'] = 'relu'

networks[1]['pooling'] = 'GAP'
networks[1]['fc_nlinear_activation'] = 'tanh'
networks[1]['fc_nlinear_size'] = 128
networks[1]['loss_function'] = 'mse'

#   Network 3 -------------------------------------------- #
networks[2]['model_name'] = 'model_3'
networks[2]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[2]['batch_size'] = 4
networks[2]['number_of_epochs'] = 32

networks[2]['optimizer'] = 'adam'
networks[2]['rcnn_type'] = 'convlstm'
networks[2]['rcnn_filters'] = 15
networks[2]['rcnn_kernel'] = (3,3)
networks[2]['rcnn_activation'] = 'relu'

networks[2]['pooling'] = 'GAP'
networks[2]['fc_nlinear_activation'] = 'relu'
networks[2]['fc_nlinear_size'] = 128
networks[2]['loss_function'] = 'mse'

#   Network 4 -------------------------------------------- #
networks[3]['model_name'] = 'model_4'
networks[3]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[3]['batch_size'] = 4
networks[3]['number_of_epochs'] = 32

networks[3]['optimizer'] = 'adam'
networks[3]['rcnn_type'] = 'convlstm'
networks[3]['rcnn_filters'] = 15
networks[3]['rcnn_kernel'] = (3,3)
networks[3]['rcnn_activation'] = 'relu'

networks[3]['pooling'] = 'GAP'
networks[3]['fc_nlinear_activation'] = 'tanh'
networks[3]['fc_nlinear_size'] = 128
networks[3]['loss_function'] = 'mse'

#   Network 5 -------------------------------------------- #
networks[4]['model_name'] = 'model_5'
networks[4]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[4]['batch_size'] = 4
networks[4]['number_of_epochs'] = 32

networks[4]['optimizer'] = 'adam'
networks[4]['rcnn_type'] = 'convlstm'
networks[4]['rcnn_filters'] = 25
networks[4]['rcnn_kernel'] = (3,3)
networks[4]['rcnn_activation'] = 'relu'

networks[4]['pooling'] = 'GAP'
networks[4]['fc_nlinear_activation'] = 'relu'
networks[4]['fc_nlinear_size'] = 128
networks[4]['loss_function'] = 'mse'

#   Network 6 -------------------------------------------- #
networks[5]['model_name'] = 'model_6'
networks[5]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[5]['batch_size'] = 4
networks[5]['number_of_epochs'] = 32

networks[5]['optimizer'] = 'adam'
networks[5]['rcnn_type'] = 'convlstm'
networks[5]['rcnn_filters'] = 25
networks[5]['rcnn_kernel'] = (3,3)
networks[5]['rcnn_activation'] = 'relu'

networks[5]['pooling'] = 'GAP'
networks[5]['fc_nlinear_activation'] = 'tanh'
networks[5]['fc_nlinear_size'] = 128
networks[5]['loss_function'] = 'mse'

#   Network 7 -------------------------------------------- #
networks[6]['model_name'] = 'model_7'
networks[6]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[6]['batch_size'] = 4
networks[6]['number_of_epochs'] = 32

networks[6]['optimizer'] = 'adam'
networks[6]['rcnn_type'] = 'convlstm'
networks[6]['rcnn_filters'] = 30
networks[6]['rcnn_kernel'] = (3,3)
networks[6]['rcnn_activation'] = 'relu'

networks[6]['pooling'] = 'GAP'
networks[6]['fc_nlinear_activation'] = 'relu'
networks[6]['fc_nlinear_size'] = 128
networks[6]['loss_function'] = 'mse'

#   Network 8 -------------------------------------------- #
networks[7]['model_name'] = 'model_8'
networks[7]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[7]['batch_size'] = 4
networks[7]['number_of_epochs'] = 32

networks[7]['optimizer'] = 'adam'
networks[7]['rcnn_type'] = 'convlstm'
networks[7]['rcnn_filters'] = 30
networks[7]['rcnn_kernel'] = (3,3)
networks[7]['rcnn_activation'] = 'relu'

networks[7]['pooling'] = 'GAP'
networks[7]['fc_nlinear_activation'] = 'tanh'
networks[7]['fc_nlinear_size'] = 128
networks[7]['loss_function'] = 'mse'