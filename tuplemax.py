import tensorflow as tf
tf.enable_eager_execution()
def tuplemax_loss(y_true, y_pred):
    """
    works only for batch_size = 1 
    unofficial implementation of pairwise tuplemax loss.
    TUPLEMAX LOSS FOR LANGUAGE IDENTIFICATION
    https://arxiv.org/pdf/1811.12290.pdf
    Eq. (2)
    """
    false_class_logits = tf.gather(params=y_pred,indices=tf.where(tf.math.equal(y_true,0)))
    true_class_logits = tf.ones_like(false_class_logits) * tf.gather(params=y_pred,indices=tf.where(tf.math.equal(y_true,1)))
    loss_array = tf.math.log(tf.math.exp(true_class_logits)/(tf.math.exp(true_class_logits)+tf.math.exp(false_class_logits)))
    return -tf.math.reduce_mean(loss_array)
  
if __name__ == "__main__":
    """
    try out the example given in the paper
    """
    y_true = tf.constant([1,0,0,0])
    y_pred = tf.constant([0.3,0.4,0.2,0.1])
    tuplemax_loss(y_true, y_pred)  #0.6624, higher because of pairwise comparison, while softmax loss will be same for both cases
    y_pred = tf.constant([0.3,0.25,0.25,0.2])
    tuplemax_loss_batched(y_true, y_pred)  #0.6605
