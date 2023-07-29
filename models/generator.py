import tensorflow as tf

class EFFeatureExtroctor:
    def extract_features(input_shape):
      # Adjusting the channel dimension to 3 for EfficientNet
      adjusted_shape = (input_shape[0], input_shape[1], 3)

      base_model = tf.keras.applications.EfficientNetB0(input_shape=adjusted_shape, include_top=False, weights='imagenet')
      base_model.trainable = False

      inputs = tf.keras.layers.Input(shape=input_shape)

      # Replicate the L channel three times
      replicated = tf.keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])

      x = base_model(replicated, training=False)
      return tf.keras.Model(inputs=inputs, outputs=x)

    def combine_features(input_shape, feature_extractor):
      inputs = tf.keras.layers.Input(shape=input_shape)
      features = feature_extractor(inputs)

      # Reduce the channels of the feature extractor's output using a 1x1 convolution
      reduced_features = tf.keras.layers.Conv2D(256, (1, 1), activation='relu')(features) # 256 is just an example. Adjust as needed.

      upsampled_features = tf.keras.layers.UpSampling2D(size=(32, 32))(reduced_features)
      combined = tf.keras.layers.Concatenate()([inputs, upsampled_features])
      return tf.keras.Model(inputs=inputs, outputs=combined)




class ImprovedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=1, name_prefix='', **kwargs):
        super(ImprovedSelfAttention, self).__init__(name=f"{name_prefix}self_attention", **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.depth = dim // num_heads

        self.query = tf.keras.layers.Conv2D(self.dim, kernel_size=1, name=f"{name_prefix}query_conv")
        self.key = tf.keras.layers.Conv2D(self.dim, kernel_size=1, name=f"{name_prefix}key_conv")
        self.value = tf.keras.layers.Conv2D(self.dim, kernel_size=1, name=f"{name_prefix}value_conv")
        self.gamma = tf.Variable(initial_value=0., trainable=True, name=f"{name_prefix}gamma")

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height, width, _ = inputs.shape[1:]
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, height * width, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scale q before matmul
        q = q / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        # Compute attention scores
        s = tf.matmul(q, k, transpose_b=True)
        attention_map = tf.nn.softmax(s, axis=-1)

        # Compute output
        o = tf.matmul(attention_map, v)
        o = tf.transpose(o, perm=[0, 2, 1, 3])  # (batch_size, height * width, num_heads, depth)
        o = tf.reshape(o, shape=tf.shape(inputs))

        return self.gamma * o + inputs


class UpsampleBlock:
    @staticmethod
    def get_instance(filters, size, apply_dropout=False, name_prefix=''):
        initializer = tf.random_normal_initializer(0., 0.02)
        block = tf.keras.Sequential(name=f"{name_prefix}upsample_block")
        block.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False, name=f"{name_prefix}convT"))
        block.add(tf.keras.layers.BatchNormalization(name=f"{name_prefix}batchnorm"))
        if apply_dropout:
            block.add(tf.keras.layers.Dropout(0.5, name=f"{name_prefix}dropout"))
        block.add(tf.keras.layers.ReLU(name=f"{name_prefix}relu"))
        return block

class DownsampleBlock:
    @staticmethod
    def get_instance(filters, size, apply_batchnorm=True, name_prefix=''):
        initializer = tf.random_normal_initializer(0., 0.02)
        block = tf.keras.Sequential(name=f"{name_prefix}downsample_block")
        block.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False, name=f"{name_prefix}conv"))
        if apply_batchnorm:
            block.add(tf.keras.layers.BatchNormalization(name=f"{name_prefix}batchnorm"))
        block.add(tf.keras.layers.LeakyReLU(name=f"{name_prefix}leaky_relu"))
        return block

class Generator:
    def __init__(self, input_size, model_path=None):
        feature_extractor = EFFeatureExtroctor.extract_features(input_size)
        combined_model = EFFeatureExtroctor.combine_features(input_size, feature_extractor)
        combined_input = combined_model.output

        down_stack = [
            DownsampleBlock.get_instance(64, 4, apply_batchnorm=False, name_prefix="down1_"),
            DownsampleBlock.get_instance(128, 4, name_prefix="down2_"),
            DownsampleBlock.get_instance(256, 4, name_prefix="down3_"),
            DownsampleBlock.get_instance(512, 4, name_prefix="down4_"),
            ImprovedSelfAttention(512, num_heads=4, name_prefix="down5_sa_"),
            DownsampleBlock.get_instance(512, 4, name_prefix="down6_"),
            DownsampleBlock.get_instance(512, 4, name_prefix="down7_"),
            DownsampleBlock.get_instance(512, 4, name_prefix="down8_"),
            DownsampleBlock.get_instance(512, 4, name_prefix="down9_")
        ]

        up_stack = [
            UpsampleBlock.get_instance(512, 4, apply_dropout=True, name_prefix="up1_"),
            UpsampleBlock.get_instance(512, 4, apply_dropout=True, name_prefix="up2_"),
            UpsampleBlock.get_instance(512, 4, apply_dropout=True, name_prefix="up3_"),
            ImprovedSelfAttention(1024, num_heads=4, name_prefix="up4_sa_"),
            UpsampleBlock.get_instance(512, 4, name_prefix="up5_"),
            UpsampleBlock.get_instance(256, 4, name_prefix="up6_"),
            UpsampleBlock.get_instance(128, 4, name_prefix="up7_"),
            UpsampleBlock.get_instance(64, 4, name_prefix="up8_"),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(2, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

        x = combined_input

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        ind = 0
        for up, skip in zip(up_stack, skips):
            x = up(x)
            if ind != 3:
              x = tf.keras.layers.Concatenate()([x, skip])
            ind += 1
        x = last(x)

        self.model = tf.keras.Model(inputs=combined_model.input, outputs=x)
        if model_path is not None:
            self.model.load_weights(model_path)
