library(tfruns)

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

FLAGS = list(filters1 = c(32, 64),
             filters2 = c(32, 64, 128),
             dropout1 = c(0.2, 0.5),
             dropout2 = c(0.2, 0.5),
             lr = c(0.01, 0.001, 0.0001))

runs = tuning_run("fashionMNIST.R", flags = FLAGS)

runs[order(runs$eval_accuracy, decreasing =TRUE),]