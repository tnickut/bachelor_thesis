import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import compute_theta_ray, preprocess_image, parse_label, get_P2

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# Parameters
EPOCHS = 10
STEPS_PER_EPOCH = 3000
BIN_SIZE = 6  # Number of orientation bins
INPUT_SHAPE = (224, 224, 3)
TRAINED_CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram']
DIMS_AVG = {
    'Car': np.array([1.52131309, 1.64441358, 3.85728004]),
    'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
    'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
    'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
    'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
    'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
    'Tram': np.array([3.56020305,  2.40172589, 18.60659898])
}
TRAIN_IMAGES = './train_images'
TRAIN_LABELS = './train_labels'

# Generate bin centers (n_bins equally spaced around the circle)
bin_centers = np.linspace(-np.pi, np.pi, num=BIN_SIZE, endpoint=False)
bin_width = 2 * np.pi / BIN_SIZE  # Width of each bin


def build_model(input_shape, n_bins):
    """Build the VGG-based model with MultiBin outputs."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers
    
    x = Flatten()(base_model.output)
    
    # Orientation branches
    conf = Dense(256, activation='relu')(x)
    conf = Dense(n_bins, name='conf')(conf)
    
    delta = Dense(256, activation='relu')(x)
    delta = Dense(n_bins * 2, name='delta')(delta)
    delta = tf.reshape(delta, (-1, n_bins, 2))
    delta = tf.math.l2_normalize(delta, axis=-1)  # Normalize to unit vectors
    cos = delta[:, :, 0]
    sin = delta[:, :, 1]
    
    # Dimensions branch
    dims = Dense(512, activation='relu')(x)
    dims = Dense(3, name='dims')(dims)
    
    outputs = tf.concat([conf, cos, sin, dims], axis=1)
    model = Model(inputs=base_model.input, outputs=outputs)

    return model


class MultiBinLoss(tf.keras.losses.Loss):
    """Custom loss for MultiBin orientation and dimensions with overlapping bins."""
    def __init__(self, bin_centers, bin_width, w=1, alpha=1, overlap_ratio=0.2, **kwargs):
        super().__init__(**kwargs)
        self.bin_centers = tf.constant(bin_centers, dtype=tf.float32)
        self.bin_width = bin_width
        self.w = w
        self.alpha = alpha
        self.overlap = overlap_ratio * bin_width

    def confidence_loss(self, y_true, y_pred):
        """Computes the confidence loss using soft labels over overlapping bins."""
        n_bins = tf.shape(self.bin_centers)[0]
        # Confidence branch predictions:
        conf_logits = y_pred[:, :n_bins]
        theta = y_true[:, 0]

        # Compute angular difference and wrap to [-pi, pi]
        diff = theta[:, None] - self.bin_centers  # shape: (batch_size, n_bins)
        diff = tf.math.floormod(diff + np.pi, 2 * np.pi) - np.pi

        # Create soft labels: mark bins as active if within the extended threshold.
        threshold = (self.bin_width / 2) + self.overlap 
        active_bins = tf.cast(tf.abs(diff) < threshold, tf.float32) # shape: (batch_size, n_bins) -> e.g. [0, 0, 1, 0, 0, 0]. No one-hot encoding because of the overlap.
        # Normalize to create a probability distribution over the bins.
        conf_labels = active_bins / (tf.reduce_sum(active_bins, axis=1, keepdims=True) + 1e-6) # 1e-6 to avoid division by zero.
        
        # Compute categorical crossentropy loss using the soft labels.
        # Softmax is applied to the logits to get probabilities.
        conf_loss = tf.keras.losses.categorical_crossentropy(conf_labels, tf.nn.softmax(conf_logits))
        return tf.reduce_mean(conf_loss)


    def localization_loss(self, y_true, y_pred):
        """Computes the localization loss for orientation regression within the bin:
        L_loc = - (1/n_theta*) * sum_i[ cos(theta* - c_i - Delta_theta_i) ]
        """

        n_bins = tf.shape(self.bin_centers)[0]
        # Localization branch predictions: predicted cosine and sine for the residual angle.
        cos_deltas = y_pred[:, n_bins:2*n_bins]
        sin_deltas = y_pred[:, 2*n_bins:3*n_bins]
        theta = y_true[:, 0]

        # Compute angular differences between the ground truth angle and all bin centers,
        # wrapping the result to [-pi, pi].
        diff = theta[:, None] - self.bin_centers
        diff = tf.math.floormod(diff + np.pi, 2 * np.pi) - np.pi

        # Compute the predicted residual angle for each bin.
        delta_theta = tf.atan2(sin_deltas, cos_deltas)

        # Compute the cosine of the difference between the ground truth angle and the predicted angle (bin center + residual).
        cos_diff = tf.cos(theta[:, None] - self.bin_centers - delta_theta)
        
        # Update the mask to consider bins within half the bin width plus the overlap.
        in_bin_mask = tf.abs(diff) < ((self.bin_width / 2) + self.overlap) # shape: (batch_size, n_bins), e.g. [0, 0, 1, 0, 0, 0]

        # Only consider bins where the ground truth falls within the extended threshold.
        cos_diff = cos_diff * tf.cast(in_bin_mask, tf.float32)  # Zero out inactive bins.
        
        # Average the cosine values over the active bins and take the negative.
        n_active = tf.reduce_sum(tf.cast(in_bin_mask, tf.float32), axis=1) + 1e-6
        avg_cos = tf.reduce_sum(cos_diff, axis=1) / n_active
        loc_loss = -avg_cos

        return tf.reduce_mean(loc_loss)


    def dimension_loss(self, y_true, y_pred):
        """Computes the dimensions loss for regression as a mean squared error:
        L_dim = ||d_pred - d_true||^2.
        """
        n_bins = tf.shape(self.bin_centers)[0]
        dims_residuals = y_pred[:, 3*n_bins:]
        dims_true = y_true[:, 1:4]
        return tf.reduce_mean(tf.square(dims_residuals - dims_true))


    def compute_loss_components(self, y_true, y_pred):
        conf_loss = self.confidence_loss(y_true, y_pred)
        loc_loss = self.localization_loss(y_true, y_pred)
        dims_loss = self.dimension_loss(y_true, y_pred)
        orientation_loss = (conf_loss + self.w * loc_loss)
        total_loss = self.alpha * dims_loss + orientation_loss
        return {
            'total_loss': total_loss, 
            'orientation_loss': orientation_loss,
            'conf_loss': conf_loss, 
            'loc_loss': loc_loss, 
            'dims_loss': dims_loss
        }

    def call(self, y_true, y_pred):
        conf_loss = self.confidence_loss(y_true, y_pred)
        loc_loss = self.localization_loss(y_true, y_pred)
        dims_loss = self.dimension_loss(y_true, y_pred)
        orientation_loss = (conf_loss + self.w * loc_loss)
        total_loss = self.alpha * dims_loss + orientation_loss
        return total_loss



class LossHistoryCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to compute and record the different loss values
    at the end of each epoch.
    """
    def __init__(self, eval_generator, steps_per_epoch):
        super().__init__()
        self.eval_generator = eval_generator
        self.steps_per_epoch = steps_per_epoch
        self.history = {'total_loss': [], 'orientation_loss': [], 'conf_loss': [], 'loc_loss': [], 'dims_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        total_loss_sum = 0.0
        orientation_loss_sum = 0.0
        conf_loss_sum = 0.0
        loc_loss_sum = 0.0
        dims_loss_sum = 0.0
        # Evaluate on a fixed number of steps from the evaluation generator
        multi_bin_loss = MultiBinLoss(bin_centers, bin_width, overlap_ratio=0.2)

        for _ in range(self.steps_per_epoch):
            x_batch, y_batch = next(self.eval_generator)
            y_pred = self.model.predict(x_batch, verbose=0)
            losses = multi_bin_loss.compute_loss_components(
                tf.convert_to_tensor(y_batch, dtype=tf.float32),
                tf.convert_to_tensor(y_pred, dtype=tf.float32)
            )
            total_loss_sum += losses['total_loss'].numpy()
            orientation_loss_sum += losses['orientation_loss'].numpy()
            conf_loss_sum += losses['conf_loss'].numpy()
            loc_loss_sum += losses['loc_loss'].numpy()
            dims_loss_sum += losses['dims_loss'].numpy()
        avg_total_loss = total_loss_sum / self.steps_per_epoch
        avg_orientation_loss = orientation_loss_sum / self.steps_per_epoch
        avg_conf_loss = conf_loss_sum / self.steps_per_epoch
        avg_loc_loss = loc_loss_sum / self.steps_per_epoch
        avg_dims_loss = dims_loss_sum / self.steps_per_epoch

        self.history['total_loss'].append(avg_total_loss)
        self.history['orientation_loss'].append(avg_orientation_loss)
        self.history['conf_loss'].append(avg_conf_loss)
        self.history['loc_loss'].append(avg_loc_loss)
        self.history['dims_loss'].append(avg_dims_loss)

        print(f"Epoch {epoch+1} losses: total: {avg_total_loss:.4f}, orientation: {avg_orientation_loss:.4f}, conf: {avg_conf_loss:.4f}, loc: {avg_loc_loss:.4f}, dims: {avg_dims_loss:.4f}")


def data_generator(image_dir, label_dir, batch_size=8):
    """Generate batches of preprocessed data."""
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    while True:
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))
            objects = parse_label(label_path)
            if not objects:
                continue
            P2 = get_P2(os.path.splitext(img_file)[0])
            X_batch, y_batch = [], []
            for obj in objects:
                # Crop and resize image
                img_crop = preprocess_image(img_path, obj['box'])
                X_batch.append(img_crop)
                # Compute theta and dimensions residual
                center_x = (obj['box'][0] + obj['box'][2]) / 2
                center_y = (obj['box'][1] + obj['box'][3]) / 2
                theta_ray = compute_theta_ray(center_x, center_y, P2)
                theta = obj['rot_y'] - theta_ray
                dims_residual = np.array(obj['dims']) - DIMS_AVG[obj['type']]
                y_batch.append(np.concatenate([[theta], dims_residual]))
            yield np.array(X_batch), np.array(y_batch)


# Initialize model and data generators
model = build_model(INPUT_SHAPE, BIN_SIZE)
loss_instance = MultiBinLoss(bin_centers, bin_width, overlap_ratio=0.2)
model.compile(optimizer='adam', loss=loss_instance)

train_gen = data_generator(TRAIN_IMAGES, TRAIN_LABELS)
eval_gen = data_generator(TRAIN_IMAGES, TRAIN_LABELS) # Use a separate generator instance for evaluating losses in the callback

# Train the model with the callback attached.
loss_history_callback = LossHistoryCallback(eval_gen, STEPS_PER_EPOCH)
model.fit(train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[loss_history_callback])
model.save('orientation_model.h5')

# After training, plot the loss components over epochs.
epochs = range(1, len(loss_history_callback.history['total_loss']) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_history_callback.history['total_loss'], label='Total Loss')
plt.plot(epochs, loss_history_callback.history['orientation_loss'], label='Orientation Loss')
plt.plot(epochs, loss_history_callback.history['conf_loss'], label='Conf Loss')
plt.plot(epochs, loss_history_callback.history['loc_loss'], label='Loc Loss')
plt.plot(epochs, loss_history_callback.history['dims_loss'], label='Dims Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Components over Epochs")
plt.legend()
plt.show()
