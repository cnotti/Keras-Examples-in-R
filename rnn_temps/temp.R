setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tibble)
library(readr)
library(keras)
library(tensorflow)

# update GPU options
gpu = tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)

# Disable GPU                                                                   
#Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)  


temps = read_csv("jena_climate_2009_2016.csv")

glimpse(temps)

# plot overall with thining
with(temps, plot(x = 1:nrow(temps),
                 y = `T (degC)`, type = "l"))


# plot temps for 1 month
seq_month = 1:(30*144)
with(temps[seq_month,], plot(x = seq_month,
                 y = `T (degC)`, type = "l"))


# normalize data

## lookback = 1440 - Observations will go back 10 days.
## steps = 6 - Observations will be sampled at one data point per hour.
## delay = 144 - Targets will be 24 hours in the future.

data = data.matrix(temps[,-1])
train_data = data[1:200000,]
mean = apply(train_data, 2, mean)
std = apply(train_data, 2, sd)
data = scale(data, 
             center = mean, 
             scale = std)


# generator function:
## steps: thins data from hourerly resolution to length
generator = function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index = nrow(data) - delay - 1
  i = min_index + lookback
  function() {
    if (shuffle) {
      rows = sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <= min_index + lookback
      rows = c(i:min(i+batch_size-1, max_index))
      i <= i + length(rows)
    }
    
    samples = array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets = array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices = seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] = data[indices,]
      targets[[j]] = data[rows[[j]] + delay,2]
    }           
    list(samples, targets)
  }
}


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)


# How many steps to draw from val_gen in order to see the entire validation set
val_steps = (300000 - 200001 - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps = (nrow(data) - 300001 - lookback) / batch_size


evaluate_naive_method = function() {
  batch_maes = c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds = samples[,dim(samples)[[2]],2]
    mae = mean(abs(preds - targets))
    batch_maes = c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()
celsius_mae = 0.29 * std[[2]]





# -------------------------------------------------------
model = keras_model_sequential() %>% 
  layer_gru(units = 8, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 16, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history = model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 2,
  validation_data = val_gen,
  validation_steps = val_steps
)


# -------------------------------------------------------
model = keras_model_sequential() %>% 
  layer_lstm(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_lstm(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history = model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)
