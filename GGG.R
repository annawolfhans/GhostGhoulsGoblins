library(tidyverse)
library(tidymodels)
library(vroom)
library(rpart)
library(stacks)
library(embed)
library(ranger)
library(discrim)
library(naivebayes)
library(bonsai)
library(lightgbm)
library(dbarts)

g_train <- vroom::vroom("./trainGhosts.csv", col_names = TRUE, show_col_types = F)
g_test <- vroom::vroom("./testGhosts.csv", col_names = TRUE, show_col_types = F)

# remember type = class not prob


g_train <- g_train %>%
  mutate(type = as.factor(type), color = as.factor(color))

#####################
#### NAIVE BAYES ####
#####################

# my_recipe <- recipe(type~., data=g_train) %>%
#   step_zv(all_numeric_predictors()) 
# step_other(all_nominal_predictors(), threshold=0.001) %>%
# step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

my_recipe <- recipe(type ~ ., data = g_train) %>%
  step_lencode_glm(color, outcome=vars(type))

bake(prep(my_recipe))

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels=3)
#
folds <- vfold_cv(g_train, v=3, repeats=1)
#
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))
# #
# # # # do any or call of these
# # #   # metric_set(roc_auc, f_meas, sens, recall, spec,
# # #     # precision, accuracy)
# #
bestTune <- CV_results %>%
  select_best("accuracy")
# #
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=g_train)
#
# final_wf %>%
#   predict(new_data=amazonTest, type="prob")
#
ghost_predictions <- predict(final_wf,
                             new_data=g_test,
                             type="class") %>%
  bind_cols(., g_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)
#
vroom_write(x=ghost_predictions, file="./GGGBayesPreds.csv", delim=",")

###############
##### KNN #####
###############

knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)
#
tuning_grid <- grid_regular(neighbors(),
                            levels=7)

folds <- vfold_cv(g_train, v=5, repeats=1)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))
#
# # # do any or call of these
# #   # metric_set(roc_auc, f_meas, sens, recall, spec,
# #     # precision, accuracy)
# #
bestTune <- CV_results %>%
  select_best("accuracy")
# #
final_wf <- knn_wf %>%
  finalize_workflow(bestTune)%>%
  fit(data=g_train)
#
# final_wf %>%
#   predict(new_data=amazonTest, type="prob")
#

ghost_predictions <- predict(final_wf,
                             new_data=g_test,
                             type="class") %>%
  bind_cols(., g_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=ghost_predictions, file="./KNNGGGPreds.csv", delim=",")

#####################
## NEURAL NETWORKS ##
#####################

g_train <- vroom::vroom("./trainGhosts.csv", col_names = TRUE, show_col_types = F)
g_test <- vroom::vroom("./testGhosts.csv", col_names = TRUE, show_col_types = F)

g_train <- g_train %>%
  mutate(type = as.factor(type), color = as.factor(color))

nn_recipe <- recipe(type~., data=g_train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>%
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(type))
  # step_other(all_nominal_predictors(), threshold=0.001) %>%
  step_range(all_numeric_predictors(), min=0,max=1)

# PYTHON INSTALLATION
# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50, 
#                 activation="relu") %>%
#   set_engine("keras", verbose=0) %>%
#   set_mode("classification")

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1,10)),
                            levels=5)
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

folds <- vfold_cv(g_train, v=5, repeats=1)



tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

# tuned_nn %>% collect_metrics() %>%
#   filter(.metric=="accuracy") %>%
#   ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- tuned_nn %>%
  select_best("accuracy")
# #
final_wf <- nn_wf %>%
  finalize_workflow(bestTune)%>%
  fit(data=g_train)
#
# final_wf %>%
#   predict(new_data=amazonTest, type="prob")
#

ghost_predictions <- predict(final_wf,
                             new_data=g_test,
                             type="class") %>%
  bind_cols(., g_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=ghost_predictions, file="./NNPreds.csv", delim=",")


###################
##### BART ########
###################


g_train <- vroom::vroom("./trainGhosts.csv", col_names = TRUE, show_col_types = F)
g_test <- vroom::vroom("./testGhosts.csv", col_names = TRUE, show_col_types = F)

g_train <- g_train %>%
  mutate(type = as.factor(type), color = as.factor(color))

nn_recipe <- recipe(type~., data=g_train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) 
#step_lencode_mixed(all_nominal_predictors(), outcome = vars(type))
# step_other(all_nominal_predictors(), threshold=0.001) 

# PYTHON INSTALLATION
# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50, 
#                 activation="relu") %>%
#   set_engine("keras", verbose=0) %>%
#   set_mode("classification")

bart_model <- parsnip::bart(trees=tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")


nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(bart_model)

folds <- vfold_cv(g_train, v=5, repeats=1)

nn_tuneGrid <- grid_regular(trees(),
                            levels=10)

tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

# tuned_nn %>% collect_metrics() %>%
#   filter(.metric=="accuracy") %>%
#   ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- tuned_nn %>%
  select_best("accuracy")
# #
final_wf <- nn_wf %>%
  finalize_workflow(bestTune)%>%
  fit(data=g_train)
#
# final_wf %>%
#   predict(new_data=amazonTest, type="prob")
#

ghost_predictions <- predict(final_wf,
                             new_data=g_test,
                             type="class") %>%
  bind_cols(., g_test) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=ghost_predictions, file="./BARTPreds.csv", delim=",")

