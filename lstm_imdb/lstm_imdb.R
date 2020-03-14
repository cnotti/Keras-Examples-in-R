library(keras)
library(dplyr)
library(ggplot2)

# update GPU options to avoid runtime error
gpu = tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)


# Data Preparation ---------------------------------------------------
max_features = 20000
imdb = dataset_imdb(num_words = max_features)

batch_size = 32 # Cut texts after this number of words
maxlen = 80  
x_train = imdb$train$x
y_train = imdb$train$y
x_test = imdb$test$x
y_test = imdb$test$y
x_train = pad_sequences(x_train, maxlen = maxlen)
x_test = pad_sequences(x_test, maxlen = maxlen)


# Define Model arch --------------------------------------------------
model = keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = 'accuracy')


# Training & Evaluation -----------------------------------------------
history = model %>% fit(x_train, 
                        y_train,
                        batch_size = batch_size,
                        epochs = 15,
                        validation_data = list(x_test, y_test))

plot(history, smooth = TRUE)

scores = model %>% evaluate(x_test, 
                            y_test,  
                            batch_size = batch_size)

cat('Test score:', scores[[1]])
cat('Test accuracy', scores[[2]])
