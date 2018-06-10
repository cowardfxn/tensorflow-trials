# Tensorflow Notes

[Documentation](https://tensorflow.google.cn/api_docs/python/tf/unstack)

## Glossary
uniform distribution  均匀分布, size/amount
normal distribution  正态分布, sigma & miu
axis indexs  first dimension is row, second dimension is column

## Methods

Method | Description
:------ | :------
`tf.graph` | concatenate parameters, axis=0 vertical, axis=1 horizontal
`tf.reshape` | reshape paramter tensor, if one dimension parameter is -1, then the size of that dimension is computed so that the total size on that dimension remains constant
`tf.concat` | concatenate tensor in one dimension, recursively concatenate until element isn't array
`tf.unstack` | unstack tensor to param n tensors, axis specified
`tf.random_uniform` | generate values follow a uniform distribution in the range [minval, maxval), parameter shape specifies result shape, all data in tensor is uniformly distributed
`tf.where(condition, x=None, y=None, name=None)` | If both x and y are not None, then the return tensor would choose row or element from x if condition is True, or from y if condition is False. If both x and y are None, then a 2-D trensor will be returned, the first dimension represents the number of True elements, the second dimension represents the index of the True elements
`tf.reduce_mean` | compute the mean of elements across dimensions of a tensor, reduce input_tensor along the given axis
