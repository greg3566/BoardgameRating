import numpy as np
import tensorflow as tf

def baseline(R,I,It):
    _GL, _UL = I.shape
    _IL = np.sum(I)
    _TL = np.sum(It)
    zero=np.zeros_like(R)
    
    bm = tf.Variable( np.sum(I*R)/_IL , dtype=tf.float32 )
    bg = tf.Variable( tf.random.normal((_GL,)) )
    bu = tf.Variable( tf.random.normal((_UL,)) )
    bias= bm+tf.expand_dims(bg,1)+bu
    
    print("...")
    
    cost = tf.math.reduce_variance(bg)+tf.math.reduce_variance(bu)
    print("...")
    sqd = tf.squared_difference(R,bias)
    print("...")
    loss = tf.math.reduce_sum( tf.where(I, sqd, zero) )/_IL
    print("...")
    test_loss= tf.math.reduce_sum( tf.where(It, sqd, zero) )/_TL
    print("...")
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss+0.001*cost)

    print("...")
    
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    print("initiated")
    for _ in range(64):
        sess.run(train)
        print(sess.run(loss),sess.run(test_loss),sess.run(cost))

    return sess.run(bm), sess.run(bg), sess.run(bu)

def weightbase(R,I,It,bm,bg,bu,weightlist):
    _GL, _UL = I.shape
    _IL = np.sum(I)
    _TL = np.sum(It)
    zero=np.zeros_like(R)
    Iu=np.sum(I,axis=0,keepdims=True)
    
    w = np.expand_dims( np.array(weightlist, dtype="float32"), 1)
    
    bias = bm+np.expand_dims(bg,1)+bu
    cw = tf.Variable( tf.random.normal((_UL,), mean=2.0) )
    bw = tf.Variable( tf.random.normal((_UL,), mean=0.1, stddev=0.01), constraint=lambda x: tf.where(x>0.01,x,0.01*tf.ones_like(x)) )
    wd = tf.abs(w-cw)
    wsum = tf.math.reduce_sum(tf.where(I,wd,zero), axis=0, keepdims=True)
    Rp = bias+bw*(wsum/Iu-wd)
    
    print("...")
    
    sqd = tf.squared_difference(R,Rp)
    print("...")
    loss = tf.math.reduce_sum( tf.where(I, sqd, zero) )/_IL
    print("...")
    test_loss= tf.math.reduce_sum( tf.where(It, sqd, zero) )/_TL
    print("...")
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    print("...")
    
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    print("initiated")
    for _ in range(64):
        sess.run(train)
        print(sess.run(loss),sess.run(test_loss))

    return sess.run(cw), sess.run(bw)