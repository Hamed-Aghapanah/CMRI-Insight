import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors = tf.reshape(tf.squeeze(feature_vectors), [-1,128*128*4])
        labels = tf.reshape(tf.squeeze(labels), [-1,128*128*4])
        
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=-1)
        print("feature_vectors_normalized : ", feature_vectors_normalized.shape)
        
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        
        print("logits : ", logits.shape)
        print("labels : ", logits.shape)
        
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
