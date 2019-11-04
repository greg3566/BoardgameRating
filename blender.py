import numpy as np
import tensorflow as tf

def blend(R,I,It,Rps,initial=None):
    _IL = np.sum(I)
    _TL = np.sum(It)
    _PL = len(Rps)
    zero=np.zeros_like(R)
    
    print(_PL)
    for Rp in Rps:
        print( np.sum(It*(R-Rp)**2)/_TL, end=' ')
    print("")
    
    if initial==None:
        ratios=tf.Variable( tf.random.normal((_PL,), mean=1.0/_PL, stddev=0.1/_PL) )
    else:
        ratios=tf.Variable( initial, dtype=tf.float32 )
    
    x=Rps*ratios[:,tf.newaxis,tf.newaxis]
    Rp=tf.math.reduce_sum(x,axis=0)
    
    sqd = tf.squared_difference(R,Rp)
    print("...")
    loss = tf.math.reduce_sum( tf.where(I, sqd, zero) )/_IL
    print("...")
    test_loss= tf.math.reduce_sum( tf.where(It, sqd, zero) )/_TL
    print("...")
    train = tf.train.AdamOptimizer(learning_rate=0.1/_PL).minimize(loss)

    print("...")
    
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    print("initiated")
    for _ in range(64):
        sess.run(train)
        print(sess.run(loss),sess.run(test_loss))

    return sess.run(ratios)