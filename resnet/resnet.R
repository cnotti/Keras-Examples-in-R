library(keras)

# residual block where input dimensions are unchanged
res_block_identity = function(X, f, filters) {
  
  # store input value
  X_skip = X
  
  X %>% 
    # first component of main path
    layer_conv_2d(filters = filters[1], 
                  kernel_size = 1,
                  strides = 1,
                  padding = "valid") %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    # second component of main path
    layer_conv_2d(filters = filters[2], 
                  kernel_size = f,
                  strides = 1,
                  padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    # third component of main path
    layer_conv_2d(filters = filters[3], 
                  kernel_size = 1,
                  strides = 1,
                  padding = "valid") %>%
    layer_batch_normalization() %>%
    layer_activation('relu')
  
  # add shortcut to main path
  X_output = layer_add(list(X, X_skip)) %>%
    layer_activation('relu')
  
  X_output
  
}

# residual block where input dimensions are altered
res_block = function(X, f, filters, s = 2) {
  
  # store input value
  X_skip = X
  
  # main path
  X %>% 
    # first component of main path
    layer_conv_2d(filters = filters[1], 
                  kernel_size = 1,
                  strides = s,
                  padding = "valid") %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    # second component of main path
    layer_conv_2d(filters = filters[2], 
                  kernel_size = f,
                  strides = 1,
                  padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>%
    # third component of main path
    layer_conv_2d(filters = filters[3], 
                  kernel_size = 1,
                  strides = 1,
                  padding = "valid") %>%
    layer_batch_normalization() %>%
    layer_activation('relu')
  
  # skip connection
  X_skip %>%
    layer_conv_2d(filters = filters[3], 
                  kernel_size = 1,
                  strides = s,
                  padding = "valid") %>%
    layer_batch_normalization()
    
  # add shortcut to main path
  X_ouput = layer_add(list(X, X_skip)) %>%
    layer_activation('relu')
  
  X_ouput
  
}

# implement the 50 layer resnet from He et al 2015
resnet = function(input_shape = c(28, 28, 1), classes = 10) {
  
  # Define the input as a tensor with shape input_shape
  X_input = layer_input(input_shape) 
  
  # Zero-Padding
  X = X_input %>% 
    layer_zero_padding_2d(padding = 3) %>% 
    
    # Stage 1
    layer_conv_2d(filters = 64, 
                  kernel_size = 7,
                  strides = 2,
                  padding = "valid") %>%
    layer_batch_normalization() %>%
    layer_activation('relu') %>% 
    layer_max_pooling_2d(pool_size = 3, strides = 2) %>% 
    
    # Stage 2
    res_block(f = 3, filters = c(64, 64, 256), s = 1) %>% 
    res_block_identity(f = 3, filters = c(64, 64, 256)) %>% 
    res_block_identity(f = 3, filters = c(64, 64, 256)) %>% 
    
    # Stage 3
    res_block(f = 3, filters = c(128, 128, 512), s=2) %>% 
    res_block_identity(f = 3, filters = c(128, 128, 512)) %>% 
    res_block_identity(f = 3, filters = c(128, 128, 512)) %>% 
    res_block_identity(f = 3, filters = c(128, 128, 512)) %>% 
    
    # Stage 4
    res_block(f = 3, filters = c(256, 256, 1024), s=2) %>% 
    res_block_identity(f = 3, filters = c(256, 256, 1024)) %>% 
    res_block_identity(f = 3, filters = c(256, 256, 1024)) %>% 
    res_block_identity(f = 3, filters = c(256, 256, 1024)) %>% 
    res_block_identity(f = 3, filters = c(256, 256, 1024)) %>% 
    res_block_identity(f = 3, filters = c(256, 256, 1024)) %>% 
    
    # Stage 5
    res_block(f = 3, filters = c(512, 512, 2048), s=2) %>% 
    res_block_identity(f = 3, filters = c(512, 512, 2048)) %>% 
    res_block_identity(f = 3, filters = c(512, 512, 2048)) %>% 
    
    # Average-pooling
    layer_average_pooling_2d(pool_size = 2, padding = 'same') %>% 
    
    # Output layer
    layer_flatten() %>%
    layer_dense(units = classes, activation = "softmax")
    
  
  # Create model obj
  model = keras_model(inputs = X_input, outputs = X)
  
  model
}
