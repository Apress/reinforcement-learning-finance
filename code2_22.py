import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, thresholds=0.5, name="F1Score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.recall = tf.keras.metrics.Recall(thresholds=thresholds)
        self.precision = tf.keras.metrics.Precision(thresholds=thresholds)
        self.f1score = self.add_weight(name="f1", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weights=None):
        self.recall.update_state(y_true, y_pred, sample_weights)
        self.precision.update_state(y_true, y_pred, sample_weights)
        self.f1score.assign_add(1.0/(1.0/self.recall.result() + 1.0/self.precision.result()))

    def result(self):
        return self.f1score

f1 = F1Score()
f1.update_state([[0], [1], [1]], [[1], [0], [1]])
print(f1.result().numpy())