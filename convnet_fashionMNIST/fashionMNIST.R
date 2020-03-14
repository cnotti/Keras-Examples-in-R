library(keras)     
library(tensorflow)

# update GPU options
gpu = tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)


# Data Preparation ---------------------------------------------------
fashion_mnist = dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test
train_images = train_images / 255
test_images = test_images / 255
dim(train_images) = c(nrow(train_images), 28, 28, 1)
dim(test_images) = c(nrow(test_images), 28, 28, 1)
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')


# Hyperparameter flags -----------------------------------------------
FLAGS = flags(flag_numeric("filters1", 32),
              flag_numeric("filters2", 64),
              flag_numeric("dropout1", 0.2),
              flag_numeric("dropout2", 0.2),
              flag_numeric('lr', 0.01))


# Define Model arch --------------------------------------------------
model = keras_model_sequential()
model %>% 
  layer_conv_2d(filters = FLAGS$filters1, 
                kernel_size = 3, 
                activation = 'relu', 
                input_shape = c(28, 28, 1),
                kernel_initializer = initializer_he_normal()) %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_conv_2d(filters = FLAGS$filters2, 
                kernel_size = 3, 
                activation = 'relu',
                kernel_initializer = initializer_he_normal()) %>%
  layer_dropout(FLAGS$dropout2) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# compile model
model %>% compile(optimizer = optimizer_adam(lr = FLAGS$lr),
                  loss = 'sparse_categorical_crossentropy',
                  metrics = 'accuracy')


# Training & Evaluation -----------------------------------------------
history = model %>% fit(train_images,
                        train_labels, 
                        epochs = 5, 
                        batch_size = 28, 
                        validation_data = list(test_images, test_labels))

plot(history, smooth = TRUE)

score = model %>% evaluate(test_images, 
                           test_labels,
                           verbose = 0)

cat('Test loss:', score$loss, '\n')
cat('Test accuracy:', score$acc, '\n')
