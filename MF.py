import numpy as np
import tensorflow as tf

def WALS(R,I,It,bm,bg,bu,_N):
    _GL, _UL = I.shape
    _IL = np.sum(I)
    _TL = np.sum(It)
    zero=np.zeros_like(R)
    
    bias = bm+np.expand_dims(bg,1)+bu
    
    idx=tf.where(I)
    print("idx ready")
    
    input_tensor = tf.SparseTensor(indices=idx,
                                values=tf.gather_nd(R-bias,idx),
                                dense_shape=(_GL,_UL)
                              )
    
    print("sparse ready")
    
    model = tf.contrib.factorization.WALSModel(input_rows=_GL, input_cols=_UL,
                                               n_components=_N,
                                               unobserved_weight=0,
                                               regularization=0.5*_GL*_UL/_IL,
                                               row_init='random',
                                               col_init='random',
                                               row_weights=1,
                                               col_weights=1 )

    row_factor = model.row_factors[0]
    col_factor = model.col_factors[0]

    row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
    col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

    sess=tf.Session()
    
    sess.run(model.initialize_op)
    sess.run(model.worker_init)
    
    Rp = row_factor@tf.transpose(col_factor)+bias
    
    sqd = tf.squared_difference(R,Rp)
    print("...")
    loss = tf.math.reduce_sum( tf.where(I, sqd, zero) )/_IL
    print("...")
    test_loss= tf.math.reduce_sum( tf.where(It, sqd, zero) )/_TL
    print("...")
    #reg_cost = ( tf.math.reduce_mean(row_factor**2) + tf.math.reduce_mean(col_factor**2) )*_GL*_UL/_IL
    
    print("initiated")
    
    for i in range(16):
        sess.run(model.row_update_prep_gramian_op)
        print("",end="-")
        sess.run(model.initialize_row_update_op)
        print("",end="-")
        sess.run(row_update_op)
        print("",end="-")
        sess.run(model.col_update_prep_gramian_op)
        print("",end="-")
        sess.run(model.initialize_col_update_op)
        print("",end="-")
        sess.run(col_update_op)
        print()
        print(sess.run(loss),sess.run(test_loss))
    return row_factor.eval(session=sess), col_factor.eval(session=sess).transpose()