# X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.
import tensorflow as tf

# x_data = [[1., 2., 3.],\
#           [4., 5., 6.]]

x_data = [[[1., 2., 3., 4.],[1., 2., 3., 4.], [1., 2., 3., 4.]], \
          [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]]

rm = tf.reduce_mean(x_data, axis=2)


# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rm_val = sess.run([rm])
    print("rmval=", rm_val)
