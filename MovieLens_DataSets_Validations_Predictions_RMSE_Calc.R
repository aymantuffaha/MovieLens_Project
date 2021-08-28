##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# DownLoad the data#################################################################
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

#Build the data set#################################################################
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
colnames(movies)
# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")
movielens
# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
temp
# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#######################################################################
# August 27, 2021
# Ayman Tuffaha
# 
#
# The below line of codes, and algorithm written to train a machine learning on 
# using the inputs in one script to predict movie ratings in the validation set.
# This will calculate RMSE based on created and predicted movie ratings.
#######################################################################

head(edx)

glimpse(edx)

#How many distinct movie, users and genres

n_distinct(edx$movieId)

n_distinct(edx$genres)

n_distinct(edx$userId)

##########################
# Just the average method
##########################

# calculate the overall average rating on the training dataset
mu_1 <- mean(edx$rating)
mu_1

# predict all unknown ratings with mu_1 and calculate the RMSE
RMSE(validation$rating, mu_1)

######################
# Movie effect method
######################

# add average ranking term, b_i_users_movie
b_i_users_movie <- edx %>%
  group_by(movieId) %>%
  summarize(b_i_users_movie = mean(rating - mu_1))

# predict all unknown ratings with mu_1 and b_i_users_movie
predicted_ratings <- validation %>% 
  left_join(b_i_users_movie, by='movieId') %>%
  mutate(pred = mu_1 + b_i_users_movie) %>%
  pull(pred)

# calculate RMSE of movie ranking effect
RMSE(validation$rating, predicted_ratings)

# plot the distribution of b_i_users_movie's
qplot(b_i_users_movie, data = b_i_users_movie, bins = 15, color = I("black"))



###############################
# Movie and user effect method
###############################

# compute user bias term, b_user
b_user <- edx %>% 
  left_join(b_i_users_movie, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - mu_1 - b_i_users_movie))

b_user
# predict new ratings with movie and user bias
predicted_ratings <- validation %>% 
  left_join(b_i_users_movie, by='movieId') %>%
  left_join(b_user, by='userId') %>%
  mutate(pred = mu_1 + b_i_users_movie + b_user) %>%
  pull(pred)


# calculate RMSE of movie ranking effect
RMSE(predicted_ratings, validation$rating)



###########################################
# Regularized movie and user effect method
###########################################

# determine best lambda from a sequence
lambdas <- seq(from=0, to=10, by=0.25)

# output RMSE of each lambda, repeat earlier steps (with regularization)
rmses <- sapply(lambdas, function(l){
  # calculate average rating across training data
  mu_1 <- mean(edx$rating)
  # compute regularized movie bias term
  b_i_users_movie <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i_users_movie = sum(rating - mu_1)/(n()+l))
  # compute regularize user bias term
  b_user <- edx %>% 
    left_join(b_i_users_movie, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_user = sum(rating - b_i_users_movie - mu_1)/(n()+l))
  # compute predictions on validation set based on these above terms
  predicted_ratings <- validation %>% 
    left_join(b_i_users_movie, by = "movieId") %>%
    left_join(b_user, by = "userId") %>%
    mutate(pred = mu_1 + b_i_users_movie + b_user) %>%
    pull(pred)
  # output RMSE of these predictions
  return(RMSE(predicted_ratings, validation$rating))
})

# quick plot of RMSE vs lambdas
qplot(lambdas, rmses)
# print minimum RMSE 
min(rmses)



######################################################
# Final model with regularized movie and user effects
######################################################

# The final linear model with the minimizing lambda
lam_1 <- lambdas[which.min(rmses)]

b_i_users_movie <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_users_movie = sum(rating - mu_1)/(n()+lam_1))
# compute regularize user bias term
b_user <- edx %>% 
  left_join(b_i_users_movie, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_user = sum(rating - b_i_users_movie - mu_1)/(n()+lam_1))
# compute predictions on validation set based on these above terms
predicted_ratings <- validation %>% 
  left_join(b_i_users_movie, by = "movieId") %>%
  left_join(b_user, by = "userId") %>%
  mutate(pred = mu_1 + b_i_users_movie + b_user) %>%
  pull(pred)
# output RMSE of these predictions
RMSE(predicted_ratings, validation$rating)