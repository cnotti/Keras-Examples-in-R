# example: fashion MNIST ---------------------------------------------
library(tensorflow)
source("resnet.R")

# update GPU options ------------------------------------------------- 
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

model = resnet(
  input_shape = c(28, 28, 1), 
  classes = length(class_names)
)

model %>% compile(
  optimizer = optimizer_adam(lr = 0.0001), 
  loss='sparse_categorical_crossentropy', 
  metrics = 'accuracy'
)

# Training & Evaluation -----------------------------------------------
history = model %>% fit(
  train_images,
  train_labels, 
  epochs = 50, 
  batch_size = 32, 
  validation_data = list(test_images, test_labels)
)

score = model %>% evaluate(
  test_images, 
  test_labels,
  verbose = 0
)

plot(history, smooth = TRUE)
cat('Test loss:', score$loss, '\n')
cat('Test accuracy:', score$acc, '\n')






if(FALSE){ # run with data augmentation
  
  # Data augmentation ---------------------------------------------------
  batch_size = 32
  n_samples = length(train_labels)
  steps_per_epoch = n_samples / batch_size
  
  train_datagen = image_data_generator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = TRUE
  )
  
  train_generator = flow_images_from_data(
    x = train_images, 
    y = train_labels, 
    generator = train_datagen,
    batch_size = 32
  )
  
  # Training & Evaluation -----------------------------------------------
  model = resnet(
    input_shape = c(28, 28, 1), 
    classes = length(class_names)
  )
  
  model %>% compile(
    optimizer = optimizer_adam(lr = 0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics = 'accuracy'
  )
  
  history <- model %>% fit_generator(
    train_generator, 
    validation_data = list(test_images, test_labels),
    steps_per_epoch = steps_per_epoch,
    epochs = 50,
    verbose = 1
  )
}