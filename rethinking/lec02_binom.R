library(rethinking)

NUM_PTS <- 50000
SAMPLE_SZ <- 20000

p_grid <- seq(from = 0, to = 1, length.out = NUM_PTS)
prob_p <- rep(x = 1, times = NUM_PTS)
prob_data <- dbinom(x = 6, size = 9, prob = p_grid)

posterior <- prob_data * prob_p / sum(prob_data * prob_p)

samples <- sample(p_grid , prob = posterior , size = SAMPLE_SZ, replace = TRUE)

plot(samples)

dens(samples)