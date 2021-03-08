# Note: 张量维度测试
# Date: 2021/3/5

import numpy as np

print(np.array(12).ndim)
print(np.array([1,2,3]).ndim)
print(np.array([[1,2,3],
               [1,2,3],
               [1,2,3]]).ndim)
print(np.array([[[1,2,3],
               [1,2,3],
               [1,2,3]],
                [[1,2,3],
               [1,2,3],
               [1,2,3]],
                [[1,2,3],
               [1,2,3],
               [1,2,3]]]).ndim)
