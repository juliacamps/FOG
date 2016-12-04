

import numpy as np
from os.path import isfile
# from FOG.definitions import get_data_structure
from collections import Counter
from collections import OrderedDict
from FOG.utils import to_string
from FOG.definitions import get_delimiter
from FOG.definitions import _DELIMITER
from FOG.utils import from_string

problem = 'fog'
# data = get_data_structure(problem, raw_data=False)

y = [[[1,1,5,4,6,6,3,0,1,3,-3,2,-1,55,0,94,23,4,7,-19],[2,3]],[2],
     2,[9,3],5,5,5]
a = np.array([])
# for i in a:
#     print(i)
c = ([np.asarray([{'a': np.asarray([[[[4,7],4],3],[2,3],[1,1],[7,4],
                                  [[[4,7]]]])},
     {'b': 4},]), {'c': 'hola', 'd': {3:1, 9:0, 'i':['adeu',90,
                                                     -2]}}])
# print(to_string(c))

conf_mat = np.asarray([[34, 67],[23, 443]])
print(to_string(conf_mat))
print(0*'a')
dic = OrderedDict([('TP', int(conf_mat[0, 0])),
                   ('FN',
                                                 int(conf_mat[0, 1])),
             ('FP', int(conf_mat[1, 0])), ('TN', int(conf_mat[1,
                                                              1]))])

print(to_string(({'model_name': 'model_9', 'n_epoch': 100, 'epoch':
    0}, [['Epoch', 0, dic], dic, dic])))

# print(None)
# print(str(None))
# print(to_string(None))


# print(isfile('/home/juli/PycharmProjects/Keras_Projects/fog_info'))
# print(listdir)
for i in range(3):
    a = 'c'
print(i)


data_str = 'epoch_0 acc 3 spe 45 sce -4.32'
print(from_string(data_str))


    