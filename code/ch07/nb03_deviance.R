library(rethinking)

# Create the data
sppnames <- c("afarensis",
              "africanus",
              "habilis",
              "boisei",
              "rudolfensis",
              "ergaster",
              "sapiens")
brainvolcc <- c(438 , 452 , 612, 521, 752, 871, 1350)
masskg <- c(37.0 , 35.5 , 34.5 , 41.5 , 55.5 , 61.0 , 53.5)

d <- data.frame(species = sppnames ,
                brain = brainvolcc ,
                mass = masskg)

# Normalize
d$mass_std <- (d$mass - mean(d$mass)) / sd(d$mass)
d$brain_std <- d$brain / max(d$brain)

# utility functions
R2_is_bad <- function(quap_fit) {
  s <- sim(quap_fit , refresh = 0)
  r <- apply(s, 2, mean) - d$brain_std
  1 - var2(r) / var2(d$brain_std)
}

# Models
m7.1 <- quap(
  alist(
    brain_std ~ dnorm(mu , exp(log_sigma)),
    mu <- a + b * mass_std,
    a ~ dnorm(0.5 , 1),
    b ~ dnorm(0 , 10),
    log_sigma ~ dnorm(0 , 1)
  ),
  data = d
)


m7.2 <- quap(
  alist(
    brain_std ~ dnorm(mu , exp(log_sigma)),
    mu <- a + b[1] * mass_std + b[2] * mass_std ^ 2,
    a ~ dnorm(0.5 , 1),
    b ~ dnorm(0 , 10),
    log_sigma ~ dnorm(0 , 1)
  ),
  data = d ,
  start = list(b = rep(0, 2))
)


m7.3 <- quap(
  alist(
    brain_std ~ dnorm( mu , exp(log_sigma) ),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 +
      b[3]*mass_std^3,
    a ~ dnorm( 0.5 , 1 ),
    b ~ dnorm( 0 , 10 ),
    log_sigma ~ dnorm( 0 , 1 )
  ), data=d , start=list(b=rep(0,3)) )



m7.4 <- quap(
  alist(
    brain_std ~ dnorm( mu , exp(log_sigma) ),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 +
      b[3]*mass_std^3 + b[4]*mass_std^4,
    a ~ dnorm( 0.5 , 1 ),
    b ~ dnorm( 0 , 10 ),
    log_sigma ~ dnorm( 0 , 1 )
  ), data=d , start=list(b=rep(0,4)) )


m7.5 <- quap(
  alist(
    brain_std ~ dnorm( mu , exp(log_sigma) ),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 +
      b[3]*mass_std^3 + b[4]*mass_std^4 +
      b[5]*mass_std^5,
    a ~ dnorm( 0.5 , 1 ),
    b ~ dnorm( 0 , 10 ),
    log_sigma ~ dnorm( 0 , 1 )
  ), data=d , start=list(b=rep(0,5)) )


m7.6 <- quap(
  alist(
    brain_std ~ dnorm( mu , 0.001 ),
    mu <- a + b[1]*mass_std + b[2]*mass_std^2 +
      b[3]*mass_std^3 + b[4]*mass_std^4 +
      b[5]*mass_std^5 + b[6]*mass_std^6,
    a ~ dnorm( 0.5 , 1 ),
    b ~ dnorm( 0 , 10 )
  ), data=d , start=list(b=rep(0,6)) )

# LPPD calculations
set.seed(1)
lppd( m7.1 , n=1e4 )

set.seed(1)
sapply( list(m7.1,m7.2,m7.3,m7.4,m7.5,m7.6) , function(m) {sum(lppd(m))} )

