############################################################
# Airbnb Price Prediction Project (Kaggle NYC Dataset)
# Author: Pragathi Sharma
# File: airbnb_price_prediction.R
#
# Description:
#   This script loads the NYC Airbnb Open Data from Kaggle,
#   performs basic cleaning and exploratory analysis, then
#   fits three models (Linear Regression, Random Forest,
#   Gradient Boosting) to predict listing prices.
#
#   Models are compared using RMSE on a held-out test set.
############################################################


###############################
# 0. Setup --------------------
###############################

# Clear workspace (optional but useful for reproducibility)
rm(list = ls())

# Set a seed for reproducibility
set.seed(123)


###############################
# 1. Load Required Packages ---
###############################

# Helper function to install/load packages
load_or_install <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# List of packages used in the project
packages <- c("tidyverse", "caret", "randomForest", "gbm")

# Load or install each package
invisible(lapply(packages, load_or_install))


###############################
# 2. Data Loading -------------
###############################

# NOTE: Update this path if the CSV is stored elsewhere.
data_path <- "AB_NYC_2019.csv" 

# Read the Airbnb data
listings_raw <- readr::read_csv(data_path)

# Take a quick look at the data structure
str(listings_raw)


###############################
# 3. Data Preparation ---------
###############################

# For this project, we keep a subset of variables:
# - price: target variable
# - neighbourhood_group: borough of NYC (e.g., Manhattan, Brooklyn)
# - room_type: entire home/apt, private room, etc.
# - minimum_nights: minimum nights per booking
# - number_of_reviews: total number of reviews
# - reviews_per_month: average reviews per month
# - calculated_host_listings_count: number of listings per host
# - availability_365: number of days the listing is available per year

listings <- listings_raw %>%
  dplyr::select(
    price,
    neighbourhood_group,
    room_type,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    calculated_host_listings_count,
    availability_365
  )

# Remove rows with any missing values
listings <- na.omit(listings)

# Filter out listings with non-positive or extreme prices (> 1000)
# This keeps the focus on typical listings and reduces the impact of outliers.
listings <- listings %>%
  dplyr::filter(price > 0, price < 1000)

# Convert categorical variables to factors
listings <- listings %>%
  dplyr::mutate(
    dplyr::across(c(neighbourhood_group, room_type), as.factor)
  )

# Check summary after cleaning
summary(listings)


###############################
# 4. Exploratory Data Analysis
###############################

dev.new(width = 8, height = 6)
# 4.1 Distribution of Prices ---------------------------------

# Basic histogram of price

hist(
  listings$price,
  breaks = 50,
  main   = "Distribution of Airbnb Prices (NYC)",
  xlab   = "Price (USD)"
)

# 4.2 Count of Room Types -------------------------------------

room_type_counts <- listings %>%
  dplyr::count(room_type) %>%
  dplyr::arrange(desc(n))

print(room_type_counts)

# 4.3 Count of Neighbourhood Groups --------------------------

neighbourhood_counts <- listings %>%
  dplyr::count(neighbourhood_group) %>%
  dplyr::arrange(desc(n))

print(neighbourhood_counts)


###############################
# 5. Train/Test Split ---------
###############################

# We create an 80/20 split to evaluate models on a held-out test set.

set.seed(123)  # ensure reproducible split
train_index <- caret::createDataPartition(listings$price, p = 0.8, list = FALSE)

train_set <- listings[train_index, ]
test_set  <- listings[-train_index, ]

# Check dimensions of train and test sets
cat("Training set rows:", nrow(train_set), "\n")
cat("Test set rows    :", nrow(test_set), "\n")


###############################
# 6. Evaluation Metric --------
###############################

# Define RMSE (Root Mean Square Error) function.
# This is the primary metric used in the project.

RMSE <- function(true, predicted) {
  sqrt(mean((true - predicted)^2))
}


###############################
# 7. Model Training -----------
###############################

# We train three models:
#   1. Linear Regression
#   2. Random Forest
#   3. Gradient Boosting (GBM)

# NOTE: This may take some time depending on machine and data size.

## 7.1 Linear Regression Model -----------------------------

cat("\nFitting Linear Regression model...\n")

lm_model <- caret::train(
  price ~ .,
  data  = train_set,
  method = "lm"
)

# Predict on test set
lm_preds <- predict(lm_model, newdata = test_set)

# Compute RMSE
rmse_lm <- RMSE(test_set$price, lm_preds)

cat("Linear Regression RMSE:", rmse_lm, "\n")


## 7.2 Random Forest Model ---------------------------------

cat("\nFitting Random Forest model...\n")

# Random Forest with 100 trees (can be increased for potentially better performance)
rf_model <- randomForest::randomForest(
  price ~ .,
  data  = train_set,
  ntree = 100
)

# Predict on test set
rf_preds <- predict(rf_model, newdata = test_set)

# Compute RMSE
rmse_rf <- RMSE(test_set$price, rf_preds)

cat("Random Forest RMSE    :", rmse_rf, "\n")


## 7.3 Gradient Boosting Model (GBM) -----------------------

cat("\nFitting Gradient Boosting (GBM) model...\n")

gbm_model <- gbm::gbm(
  formula           = price ~ .,
  data              = train_set,
  distribution      = "gaussian",
  n.trees           = 100,   # number of boosting iterations
  interaction.depth = 3,     # tree depth
  shrinkage         = 0.1,   # learning rate
  verbose           = FALSE
)

# Predict on test set (must specify number of trees)
gbm_preds <- predict(gbm_model, newdata = test_set, n.trees = 100)

# Compute RMSE
rmse_gbm <- RMSE(test_set$price, gbm_preds)

cat("GBM RMSE              :", rmse_gbm, "\n")


###############################
# 8. Model Comparison ---------
###############################

# Gather the RMSE results into a data frame for easy comparison
model_results <- tibble::tibble(
  Model = c("Linear Regression", "Random Forest", "Gradient Boosting"),
  RMSE  = c(rmse_lm, rmse_rf, rmse_gbm)
)

cat("\nModel comparison (lower RMSE is better):\n")
print(model_results)

# Identify the best model based on RMSE
best_model_index <- which.min(model_results$RMSE)
best_model_name  <- model_results$Model[best_model_index]
best_model_rmse  <- model_results$RMSE[best_model_index]

cat("\nBest model:", best_model_name, "with RMSE =", best_model_rmse, "\n")


###############################
# 9. Residual Analysis --------
###############################

# For the best model (Random Forest in earlier runs),
# creating a simple residual plot.

cat("\nCreating residual plot for Random Forest predictions...\n")

rf_residuals <- test_set$price - rf_preds

plot(
  rf_residuals,
  main = "Residual Plot: Random Forest",
  xlab = "Observation Index",
  ylab = "Residual (Actual - Predicted)"
)
abline(h = 0, lty = 2, col = "gray")


###############################
# 10. Save Results
###############################

write.csv(model_results, "airbnb_model_results.csv", row.names = FALSE)

cat("\nScript completed successfully.\n")
