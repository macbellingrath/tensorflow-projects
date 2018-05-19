#!/usr/bin/env python3
import tensorflow as tf

feature_columns = [1, 2, 3, 4, 5]


def train_input_fn():
    pass


def predict_input_fn():
    pass


# set up a linear classifier
classifier = tf.estimator.LinearClassifier(feature_columns)

# train model on examples
classifier.train(input_fn=train_input_fn, steps=2000)

# predict
predictions = classifier.predict(input_fn=predict_input_fn)
