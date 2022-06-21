# from tensorflow.keras.layers import concatenate, Reshape, Dropout, Dense, Conv2D, MaxPool2D, Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras import initializers, regularizers

from tensorflow.python.keras.layers import concatenate, Reshape, Dropout, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import initializers, regularizers

class TextCNN(Model):

    def __init__(self,
                max_len=10,
                embed_dim=200,
                output_dim=15,
                filter_sizes=[2,3,4],
                num_filters=2,
                dropout_rate=0.5,
                regularizers_lambda=0.01
                ):
        super().__init__()
        self.reshape = Reshape(target_shape=(max_len, embed_dim, 1))
        self.dropout_1 = Dropout(dropout_rate)
        self.conv_layers = [
            Conv2D(num_filters, kernel_size=(filter_size, embed_dim), padding='valid',
                    data_format='channels_last',
                    kernel_initializer='glorot_normal', strides=(1,1), activation='relu',
                    bias_initializer=initializers.constant(0.1))
            for filter_size in filter_sizes
        ]
        self.max_pool_layers = [
            MaxPool2D(pool_size=(max_len - filter_size + 1, 1), padding='valid', strides=(1,1))
            for filter_size in filter_sizes
        ]
        self.flatten = Flatten()
        self.dropout_2 = Dropout(dropout_rate)
        self.dense = Dense(output_dim, activation='softmax',
                            kernel_initializer='glorot_normal',
                            bias_initializer=initializers.constant(0.1),
                            kernel_regularizer=regularizers.l2(regularizers_lambda),
                            bias_regularizer=regularizers.l2(regularizers_lambda))


    def call(self, input):
        input = self.reshape(input)
        input = self.dropout_1(input)
        pools = []
        for i in range(len(self.conv_layers)):
            conv = self.conv_layers[i](input)
            pool = self.max_pool_layers[i](conv)
            pools.append(pool)        
        pools = concatenate(pools, axis=-1)
        pools = self.flatten(pools)
        pools = self.dropout_2(pools)
        output = self.dense(pools)
        # output = self.dense(output)
        return output

