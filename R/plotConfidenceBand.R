#library(ggplot2)
#library(cowplot)
#library(dplyr)
#library(tidyr)

plot_confidence_band <- function(t, res) {
  Qp <- fdata(res$Qpred, argvals = t)
  Ql <- fdata(res$Q_lx, argvals = t)
  Qu <- fdata(res$Q_ux, argvals = t)
  plot(Qp)
  lines(Qu, lty=2, col="red")
  lines(Ql, lty=2, col="red")
}
