library(brms)
library(tidyverse)

start = proc.time()

df_splits = read.csv("../python/useful_csvs/crossv_splits.csv")
df_aspects = read.csv("../python/useful_csvs/aspect_dataframe.csv")

df_splits = df_splits %>%
  mutate(train = strsplit(gsub("\\[|\\]", "", as.character(train)), "\\s+"),
         train = map(.x = train, .f = as.numeric),
         train = map(.x = train, .f = ~.[!is.na(.)])) %>% 
  mutate(test = strsplit(gsub("\\[|\\]", "", as.character(test)), "\\s+"),
         test = map(.x = test, .f = as.numeric),
         test = map(.x = train, .f = ~.[!is.na(.)]))

list_train = df_splits$train
list_test = df_splits$test
unoise_range = c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
# For testing
# unoise_range = c(0.5, 0.6, 0.7)

split_num = as.numeric(commandArgs(trailingOnly = TRUE)[[1]])
# split_num = 1

train_trials = list_train[[split_num]]
test_trials = list_test[[split_num]]

model_list = list()

# colnames(model_comp) = c("trial", "response", "n", "data_y", "model_y", "half", "sq_err", "uncertainty_noise")

for (i in seq(1,length(unoise_range))) {
  unoise = unoise_range[[i]]


  if (unoise == 1) {
    unoise = "1.0"
  }

  data = read.csv(paste("aspect_matched_data/noise_", unoise, ".csv", sep = "")) %>% 
    mutate(response = factor(response,
                             levels = c("no_difference", "affect", "enable", "cause"),
                             ordered = TRUE))

  train_data = data %>% filter(trial %in% train_trials)

  filename = paste("splits/split", str_pad(split_num, 3, side = "left", pad = "0"), "/ord_reg_noise_", unoise, ".rds", sep = "")

  if (i == 1) {

    reg = brm(formula = response ~ whether + how + sufficient + moving,
              data = train_data,
              family = "cumulative",
              file = filename,
              seed = 1)
    # saveRDS(reg, file = filename)

  } else {
    if (file.exists(filename)) {
	reg = readRDS(filename)
    } else {
 	reg = update(reg, newdata = data)
 	saveRDS(reg, file = filename)
    }
  }
  
  # Predict with the model
  df_test_input = df_aspects %>% 
    filter(uncertainty_noise == as.numeric(unoise))
  
  set.seed(1)
  
  reg_pred = as.data.frame(predict(reg, newdata = df_test_input)) %>% 
    rename(no_difference = 1,
           affect = 2,
           enable = 3,
           cause = 4) %>% 
    mutate(trial = seq(0,29)) %>% 
    pivot_longer(cols = c("no_difference", "affect", "enable", "cause"), names_to = "response", values_to = "model_y") %>% 
    mutate(response = factor(response, levels = c("no_difference", "affect", "enable", "cause"), ordered = TRUE),
           half = ifelse(trial %in% train_trials, "train", "test"))
  
  df_test_data_agg = data %>%
    group_by(trial, response) %>% 
    tally() %>%
    mutate(data_y = n/62)

  # print(df_test_data_agg)
  
  df_model_v_data = df_test_data_agg %>% 
    right_join(reg_pred, by = c("trial", "response")) %>% 
    replace(is.na(.), 0) %>% 
    mutate(sq_err = (model_y - data_y)^2,
           uncertainty_noise = as.numeric(unoise))
  
  model_list[[i]] = df_model_v_data

}

model_comp = bind_rows(model_list)

test_models = filter(model_comp, half == "test")

best_model = test_models %>% 
  group_by(uncertainty_noise) %>% 
  summarise(sum_sq_err = sum(sq_err))

print(best_model)

best_noise = filter(best_model, sum_sq_err == min(sum_sq_err))$uncertainty_noise[[1]]

print(best_noise)

best_model = model_comp %>% filter(uncertainty_noise == best_noise)

filename = paste("splits/split", str_pad(split_num, 3, side = "left", pad = "0"), "/best_model.csv", sep = "")

write.csv(best_model, filename)

print("Time elapsed:")
print(proc.time() - start)
