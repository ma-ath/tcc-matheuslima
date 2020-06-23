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

networks = [dict() for i in range(2)]

#   Network 1 -------------------------------------------- #
networks[0]['model_name'] = 'model_1'
networks[0]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[0]['batch_size'] = 1
networks[0]['number_of_epochs'] = 1

networks[0]['optimizer'] = 'adam'
networks[0]['rcnn_type'] = 'convlstm'
networks[0]['rcnn_filters'] = 2
networks[0]['rcnn_kernel'] = (3,3)
networks[0]['rcnn_activation'] = 'relu'

networks[0]['pooling'] = 'GAP'
networks[0]['fc_nlinear_activation'] = 'relu'
networks[0]['fc_nlinear_size'] = 10
networks[0]['loss_function'] = 'mse'

#   Network 2 -------------------------------------------- #
networks[1]['model_name'] = 'model_2'
networks[1]['learning_schedule'] = [0.001, 0.0003, 9e-05]
networks[1]['batch_size'] = 1
networks[1]['number_of_epochs'] = 1

networks[1]['optimizer'] = 'adam'
networks[1]['rcnn_type'] = 'convlstm'
networks[1]['rcnn_filters'] = 4
networks[1]['rcnn_kernel'] = (3,3)
networks[1]['rcnn_activation'] = 'relu'

networks[1]['pooling'] = 'GAP'
networks[1]['fc_nlinear_activation'] = 'tanh'
networks[1]['fc_nlinear_size'] = 12
networks[1]['loss_function'] = 'mse'