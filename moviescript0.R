##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# installs tinytex needed to Knit pdf files
tinytex::install_tinytex()

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
 
# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove temporary datasets to release memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Validate edx and validation set have expected 90% and 10% data (approx) 
NROW(edx)
NROW(validation)



# Create further partition of edx dataset into train_set and test_set that would be used for cross validation
# 10% rows of edx are set aside to test the RMSE of various models
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_ind <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set<-edx[-test_ind]
test_set_temp<-edx[test_ind]

# Make sure userId and movieId in test_set are also in train_set
test_set<- test_set_temp %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
removed_rw<- anti_join(test_set_temp,test_set)

# Add rows removed from test_set back into train_set
train_set<- rbind(train_set,removed_rw)


# Validate the no of rows in dataset are as per expectations 90% in train_set and 10% in test_set (approx)
NROW(removed_rw)
NROW(train_set)
NROW(test_set_temp)
NROW(test_set)


# A sneak peek into first few rows of train_set
head(train_set)


# Simple Average of mean across all records in training dataset - train_set
mu_hat <- mean(train_set$rating)
mu_hat


# RMSE function - to calculate Root Mean Square Error
# we will be using this function a lot to test various models accuracy
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#Validating the RMSE value for simple mean, matching with ratings in test set
naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# A RMSE results table/dataframe created to store RMSEs of various models as we analyze it
rmse_results <- data_frame(method = "Simple average", RMSE = naive_rmse)

# Print rmse results for simple average - A baseline RMSE
rmse_results


# Calculate the Average rating across the train_set and storing in mu
mu <- mean(train_set$rating) 


# Calculating the Average impact of movieId on rating
# For this we first substract the mu (average rating across train_set) from actual rating 
# and then calculate the impact of movieId on this and store this in new data frame - movie_avgs
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
    summarize(b_i = mean(rating - mu))


# Join the movie_avgs dataframe created above with test_set on movieId and add the movie Impact (b_i)
# to the overall mean (mu) to get predicted ratings (store in predicted_ratings dataset)
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  .$b_i


# Test out the Root mean square error of predicted ratings calculated above 
# to actual ratings in test_set using RMSE function
# append/add the RMSE value results to rmse_results dataframe
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

# View the RMSE values just added
rmse_results %>% knitr::kable()



# Calculate the impact that userId has on the movie ratings. Create dataset user_avgs for this
# grouping on userId and substracting previous calculated impacts (mu, b_i)
# filter out outliers
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n()>3) %>%
  summarize(b_u = mean(rating - mu - b_i))


# Generate predicted ratings using movie avgs and user avgs and mu
predicted_ratings_mu <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + ifelse(is.na(b_i),0,b_i) + ifelse(is.na(b_u),0,b_u)) %>%  
  .$pred


# Test out the RMSE with the latest predicted ratings and append these to existing rmse_results dataframe
model_2_rmse <- RMSE(predicted_ratings_mu, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()



# DATA PREPRATION :
# Further data preparation
# we will further use these enrichments and try to predict the ratings using them as well

# Add column to all datasets "nt" that holds the length of the title column
train_set<- train_set %>% mutate(nt=str_length(title))
test_set <- test_set %>% mutate(nt=str_length(title))
validation <- validation %>% mutate(nt=str_length(title))


# Add column to all datasets "s" that holds the count of spaces in the title
train_set <- train_set %>% mutate(s=str_count(title,"\\ "))
test_set <- train_set %>% mutate(s=str_count(title,"\\ "))
validation <- validation %>% mutate(s=str_count(title,"\\ "))

# Visualize to see impact of gs
#train_set %>% mutate(gs=str_count(genres,"\\|")+1) %>% group_by(gs) %>% summarise(n=n(), m=mean(rating)) %>% ggplot(aes(n,m)) + geom_point()

# Add column to all datasets "gs" that holds the count of pipe symbol in genres (similar to space for title)
train_set <- train_set %>% mutate(gs=str_count(genres,"\\|"))
test_set <- test_set %>% mutate(gs=str_count(genres,"\\|"))
validation <- validation %>% mutate(gs=str_count(genres,"\\|"))


# Create column "yr" and add that to all datasets, this holds the year part from title (last 4 characters from title)
train_set<-train_set %>% mutate(yr=substr(title,str_length(title)-4,str_length(title)-1))
test_set <- test_set %>% mutate(yr=substr(title,str_length(title)-4,str_length(title)-1))
validation <- validation %>% mutate(yr=substr(title,str_length(title)-4,str_length(title)-1))


# sneak peek into enriched train_set now
head(train_set)



# graphs to see how these added columns depict variability of ratings across these columns

train_set %>% group_by(yr) %>% summarize(n=n(),r=mean(rating)) %>% ggplot(aes(yr,r)) + geom_point() + geom_smooth()
train_set %>% group_by(s) %>% summarize(n=n(), m=mean(rating)) %>% ggplot(aes(s,m)) + geom_point()
train_set %>% group_by(nt) %>% summarize(n=n(),m=mean(rating)) %>% ggplot(aes(nt,m)) + geom_point()




# Train further using these additional columns

# Use train_set to calculate raing averages across year(substring of tiles)
# Year
yr_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(yr) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u))

# Calcuate predicted values using the year averages as well
# Predicted Rating = Overall mean + movie impact + user impact + year impact
predicted_ratings_muy <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  mutate(pred = mu + b_i + b_u + b_y) %>%
  .$pred

predicted_ratings_muy

model_3_rmse <- RMSE(predicted_ratings_muy, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + year Effects Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()



#0.8655043




################################################################################

# Use train_set to calculate rating averages across spaces in title "s" column
# No of Spaces in title impact
s_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  group_by(s) %>%
  summarize(b_s = mean(rating - mu - b_i - b_u - b_y))

# View variability across no of spaces
s_avgs %>% plot()

# Use No of Spaces "s" column to predict the ratings
# Pred Ratings =  Overall Mean +  Movie Impact + UserImpact + Year Impact + NoofSpaces Impact
predicted_ratings_muys <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_s) %>%
  .$pred

# Calcuate the RMSE value of predicted ratings and add these to the rmse_results dataframe
model_4_rmse <- RMSE(predicted_ratings_muys, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + year + NoOfSpaces(title) Effects Model",  
                                     RMSE = model_4_rmse ))
rmse_results %>% knitr::kable()

#0.8654911


# Use train_set to calculate rating averages across no of pipe "|" in genres "gs" column
# No of pipe symbol in genres impact
gs_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
    group_by(gs) %>%
  summarize(b_gs = mean(rating - mu - b_i - b_u - b_y - b_s))


# Plot no of pipe in genres vs variability in ratings
gs_avgs %>% plot()


# Use No of pipes "gs" column to predict the ratings
# Pred Ratings =  Overall Mean +  Movie Impact + UserImpact + Year Impact + NoofSpaces(title) Impact + NoOfPipes(Genres)
predicted_ratings_muysgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  left_join(gs_avgs, by='gs') %>%
    mutate(pred = mu + b_i + b_u + b_y + b_s + b_gs) %>%
  .$pred


# Calcuate the RMSE value of predicted ratings and add these to the rmse_results dataframe
model_5_rmse <- RMSE(predicted_ratings_muysgs, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + year + nSpaces(title) + nPipes(Genres) Effects Model",  
                                     RMSE = model_5_rmse ))
rmse_results %>% knitr::kable()

#0.8654515



# Use train_set to calculate rating averages across length of title "nt" column
# Length of tile "nt" impact
nt_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  left_join(gs_avgs, by='gs') %>%
  group_by(nt) %>%
  summarize(b_nt = mean(rating - mu - b_i - b_u - b_y - b_s - b_gs))



# Use Length of title "nt" column to predict the ratings
# Pred Ratings =  Overall Mean +  Movie Impact + UserImpact + Year Impact + NoofSpaces(title) Impact +
#                  NoOfPipes(Genres) + LengthOfTitle
predicted_ratings_muysgsnt <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  left_join(gs_avgs, by='gs') %>%
  left_join(nt_avgs, by='nt') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_s + b_gs + b_nt) %>%
  .$pred


# Calcuate the RMSE value of predicted ratings and add these to the rmse_results dataframe
model_6_rmse <- RMSE(predicted_ratings_muysgsnt, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + year + nSpaces(title) + nPipes(Genres) + Length(title) Effects Model",  
                                     RMSE = model_6_rmse ))
rmse_results %>% knitr::kable()

#0.8654271




##################################

# Calculate rating averages across genres column
# Genres impact
genres2_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  left_join(gs_avgs, by='gs') %>%
  left_join(nt_avgs, by='nt') %>%
  group_by(genres) %>%
  filter(n()>0) %>%
  summarize(b_g3 = mean(rating - mu - b_i - b_u - b_y - b_s - b_gs - b_nt ))


# Use genres column to predict the ratings 
# Pred Ratings =  Overall Mean +  Movie Impact + UserImpact + Year Impact + NoofSpaces(title) Impact +
#                  NoOfPipes(Genres) + LengthOfTitle + genres 
# The Predicted rating has been corrected to 5 if it exceeds 5 and set to 0.5 if its less than 0.5 or negative
# The Predicted rating is then "adjusted" to 4.75 if its more than 4.75 and less than 4.95 - overshooting
# The predicted rating is also "adjusted" to 0.75 if its more than 0.55 and less than 0.75 - undershooting
# This is done as to normalize the rating 
# as the model is clearly overshooting (as there were so many predicted ratings more than 5)
# and undershooting as there were so many predicted ratings in negative as well
predicted_ratings_final <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  left_join(gs_avgs, by='gs') %>%
  left_join(nt_avgs, by='nt') %>%
  left_join(genres2_avgs, by='genres') %>%
  mutate(pred = mu + ifelse(is.na(b_i),0,b_i) + ifelse(is.na(b_u),0,b_u) +
           ifelse(is.na(b_y),0,b_y) + 
           ifelse(is.na(b_s),0,b_s) + 
           ifelse(is.na(b_nt),0,b_nt) + 
           ifelse(is.na(b_gs),0,b_gs) +
           ifelse(str_detect(genres,"\\|"),ifelse(is.na(b_g3),0,b_g3),0) 
  ) %>% mutate(pred=ifelse(pred>5,5,ifelse(pred<0.5,0.5,pred))) %>%
  mutate(pred=ifelse(pred>4.75 & pred<4.95,4.75,pred)) %>%
  mutate(pred=ifelse(pred>0.55 & pred<0.75,0.75, pred)) %>%  
  .$pred


# Calcuate the RMSE value of predicted ratings and add these to the rmse_results dataframe
# THIS IS THE FINAL RMSE OF THE MODEL
model_7_rmse <- RMSE(predicted_ratings_final, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + year + nSpaces(title) + nPipes(Genres) + Length(title) + genres Effects Model",  
                                     RMSE = model_7_rmse ))

# Display the final RMSE of the Model
rmse_results %>% knitr::kable()


###################################3











###########################
# Final Model #

# Predicting Ratings for Validation data set #

predicted_ratings_final <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(yr_avgs, by='yr') %>%
  left_join(s_avgs, by='s') %>%
  left_join(gs_avgs, by='gs') %>%
  left_join(nt_avgs, by='nt') %>%
  left_join(genres2_avgs, by='genres') %>%
  mutate(pred = mu + ifelse(is.na(b_i),0,b_i) + ifelse(is.na(b_u),0,b_u) +
           ifelse(is.na(b_y),0,b_y) + 
           ifelse(is.na(b_s),0,b_s) + 
           ifelse(is.na(b_nt),0,b_nt) + 
           ifelse(is.na(b_gs),0,b_gs) +
           ifelse(str_detect(genres,"\\|"),ifelse(is.na(b_g3),0,b_g3),0) 
  ) %>% mutate(pred=ifelse(pred>5,5,ifelse(pred<0.5,0.5,pred))) %>%
  mutate(pred=ifelse(pred>4.75 & pred<4.95,4.75,pred)) %>%
  mutate(pred=ifelse(pred>0.55 & pred<0.75,0.75, pred)) %>%  
  .$pred



# THIS IS THE FINAL RMSE OF THE MODEL
Final_Validation_rmse <- RMSE(predicted_ratings_final, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Validation RMSE - Movie,User,year,nSpaces(title),nPipes(Genres),Length(title),genres",  
                                     RMSE = Final_Validation_rmse ))

# Display the final RMSE of the Model
rmse_results %>% knitr::kable()


Final_Validation_rmse

