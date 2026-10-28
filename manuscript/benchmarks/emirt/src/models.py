"""Benchmark heads. Vendor differs only by a package-relative import.

All inputs use [A, user, B], with label one meaning A. Historical constructors
are copied from the 2026-09-22 release, not recreated from equations.
"""
import tensorflow as tf
from .vendor.model import PoliticalCoords
from .vendor.choice_models import build_choice_model

FLOOR = 1e-6

class Baseline(tf.keras.Model):
    def __init__(self, kind, n_users, n_items, dimension, seed):
        super().__init__(name=kind)
        self.kind = kind
        if kind == 'B1':
            self.coords = PoliticalCoords(n_items, n_users, dimension, seed)
            self.raw_beta = self.add_weight(name='raw_beta', shape=(),
                initializer=tf.keras.initializers.Constant(tf.math.log(tf.math.expm1(1.-FLOOR)).numpy()))
        else:
            self.item_bias = self.add_weight(name='item_bias', shape=(n_items,), initializer='zeros')
            if kind == 'B3':
                self.user_factors = self.add_weight(name='users', shape=(n_users,dimension),
                    initializer=tf.keras.initializers.RandomNormal(stddev=.05,seed=seed))
                self.item_factors = self.add_weight(name='items', shape=(n_items,dimension),
                    initializer=tf.keras.initializers.RandomNormal(stddev=.05,seed=seed+1))
    def logits(self, x):
        x = tf.cast(x,tf.int32)
        if self.kind == 'B1':
            a,u,b = tf.split(self.coords(x),3,axis=-1)
            delta = tf.reduce_sum(tf.square(b-u)-tf.square(a-u),axis=-1,keepdims=True)
            return delta*(tf.nn.softplus(self.raw_beta)+FLOOR)
        result = tf.gather(self.item_bias,x[:,0])-tf.gather(self.item_bias,x[:,2])
        if self.kind == 'B3':
            u = tf.gather(self.user_factors,x[:,1])
            v = tf.gather(self.item_factors,x[:,0])-tf.gather(self.item_factors,x[:,2])
            result = result+tf.reduce_sum(u*v,axis=-1)
        return result[:,None]
    def call(self,x):
        return tf.math.sigmoid(self.logits(x))

def build_model(kind,n_users,n_items,dimension=2,seed=101):
    if kind not in ('B0','B1','B2','B3','B4'):
        raise ValueError(kind)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    if kind in ('B2','B4'):
        model = build_choice_model(n_items,n_users,dimension,seed,
            variant='intermediate' if kind=='B2' else 'original')
    else:
        model = Baseline(kind,n_users,n_items,dimension,seed)
        model(tf.zeros((1,3),tf.int32))
    return model

def get_logits(model,kind,x):
    if kind in ('B0','B1','B3'):
        return model.logits(x)
    coords = model.layers[0](x)
    if kind == 'B2':
        a,u,b=tf.split(coords,3,axis=-1)
        sa=tf.reduce_sum(tf.square(a-u),axis=-1,keepdims=True)
        sb=tf.reduce_sum(tf.square(b-u),axis=-1,keepdims=True)
        return (sb-sa)*model.layers[1].distance_scale(sa,sb)
    if kind == 'B4':
        h=model.layers[2](model.layers[1](coords))
        return tf.matmul(h,model.layers[3].kernel)+model.layers[3].bias
    raise ValueError(kind)

def get_coordinates(model,kind):
    if kind=='B0':return None,model.item_bias[:,None]
    if kind=='B3':return model.user_factors,model.item_factors
    layer=model.coords if kind=='B1' else model.layers[0]
    return layer.returnUsersEmbedding(),layer.returnOptionsEmbedding()

def regularization_loss(model,kind):
    if kind=='B0':return tf.reduce_mean(tf.square(model.item_bias))
    users,items=get_coordinates(model,kind)
    if kind=='B3':
        return (tf.reduce_mean(tf.square(users))+tf.reduce_mean(tf.square(items))
                +tf.reduce_mean(tf.square(model.item_bias)))
    # Fixed cube center, matching the frozen protocol.
    joined=tf.concat([users,items],axis=0)
    centered=joined-tf.cast(.5,joined.dtype)
    return tf.reduce_mean(tf.square(centered))
