## scripts.R: Rcpp/Wass glue
##
## Copyright (C) 2019 - 2020  Juan Carlos Vidal and Marcos Matabuena
##
## This file is part of RcppWass.
##
## RcppWass is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 2 of the License, or
## (at your option) any later version.
##
## RcppWass is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with RcppWass.  If not, see <http://www.gnu.org/licenses/>.

#library(ggplot2)
#library(cowplot)
#library(dplyr)
#library(tidyr)

plot_confidence_band <- function(t, res) {
  Qp <- fdata(res$Qpred, argvals=t)
  Ql <- fdata(res$Q_lx, argvals=t)
  Qu <- fdata(res$Q_ux, argvals=t)
  plot(Qp)
  lines(Qu, lty=2, col="red")
  lines(Ql, lty=2, col="red")
}


crossvalidationprednuevo <- function(fdata, respuesta, hs=seq(0.5,100,length=200)) {
  X <- fdata$data
  n <- dim(X)[1]
  p <- dim(X)[2]
  t <- fdata$argvals
  Y <- respuesta
  conjunto <- data.frame(X,Y)
  aux <- complete.cases(conjunto)
  conjunto <- conjunto[complete.cases(conjunto),]
  X <- conjunto[,1:p]
  Y <- conjunto[,p+1]
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  t <- as.matrix(t)
  hs <- as.matrix(hs)

  n <- dim(X)[1]
  p <- dim(X)[2]

  cogerdatosentrenamiento <- n-1
  cogerdatostest <- 1
  cross <- n

  indices <- 0:(n-1)
  indices1 <- matrix(0, nrow=cogerdatosentrenamiento, ncol=cross)
  indices2 <- matrix(0, nrow=cogerdatostest, ncol=cross)
  indicesaux <- 1:n

  for(j in 1:cross){


    generar <- indicesaux[-c(j)]



    indices1[,j] <- indices[generar]
    indices2[,j] <- indices[c(j)]
  }

  indices1 <- as.matrix(indices1)

  indices2 <- as.matrix(indices2)


  res <- nadayara_reg(X,t,Y,hs,indices1, indices2)


  predictivo <- apply(res$r2_global,1,function(x){sqrt(mean(x))})
  ventanas <- hs
  cuantos <- sum(is.nan(predictivo))
  ventanas2 <- ventanas[!is.nan(predictivo)]
  predictivo2 <- predictivo[!is.nan(predictivo)]
  R2 <- res$r2[!is.nan(predictivo)]


  rango1 <- c(min(predictivo2), max(predictivo2))
  rango2 <- c(min(R2), max(R2))

  time <- ventanas2
  cell.density <- predictivo2
  betagal.abs<- R2
  betagal.abs
  ## add extra space to right margin of plot within frame
  par(mar=c(5, 4, 4, 6) + 0.1)

  ## Plot first set of data and draw its axis
  plot(time, betagal.abs, pch=16, axes=FALSE, ylim=c(0,1), xlab="", ylab="",
       type="b",col="black", main="Performance model vs. smoothing-parameter")
  # abline(h=0.8)
  # abline(h=0.9)
  # abline(h=0.85)
  # abline(h=0.75)


  axis(2, ylim=c(0,1),col="black",las=1)  ## las=1 makes horizontal labels
  mtext("R-square",side=2,line=2.5)
  box()

  ## Allow a second plot on the same graph
  par(new=TRUE)

  ## Plot the second plot and put axis scale on right
  plot(time, cell.density, pch=15,  xlab="", ylab="", ylim=rango1,
       axes=FALSE, type="b", col="red")
  ## a little farther out (line=4) to make room for labels
  mtext("Standart deviation, Cross-Validation",side=4,col="red",line=4)
  axis(4, ylim=c(0,200), col="red",col.axis="red",las=1)

  ## Draw the time axis
  axis(1,pretty(range(time),10))
  mtext("Smoothing-parameter",side=1,col="black",line=2.5)

  ## Add Legend
  legend("topright",legend=c("R-square
","Error, Leave-one-out Cross-validation"),
         text.col=c("black","red"),pch=c(16,15),col=c("black","red"))





  # rango <- c(min(min(res$error[!is.nan(predictivo)],na.rm = TRUE),min(predictivo2[!is.nan(predictivo)]),na.rm = TRUE), max(max(res$error[!is.nan(predictivo)],na.rm = TRUE),max(predictivo2[!is.nan(predictivo)],na.rm = TRUE)))
  # print(rango)
  # plot(time,res$error[!is.nan(predictivo)], ylim= c(rango))
  # lines(time, predictivo2[!is.nan(predictivo)])

  return(res)
}




prednuevo <- function(fdata, respuesta, hs=5){
  X <- fdata$data
  n <- dim(X)[1]
  p <- dim(X)[2]
  t <- fdata$argvals
  Y <- respuesta
  conjunto <- data.frame(X,Y)
  aux <- complete.cases(conjunto)
  conjunto <- conjunto[complete.cases(conjunto),]
  X <- conjunto[,1:p]
  Y <- conjunto[,p+1]
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  t <- as.matrix(t)
  hs <- as.matrix(hs)

  n <- dim(X)[1]
  p <- dim(X)[2]

  dista <- eucdistance1(X,t)

  sal <- nadayara_pred(dista,X,t,Y,hs)
  # rango <- c(min(min(res$error[!is.nan(predictivo)],na.rm = TRUE),min(predictivo2[!is.nan(predictivo)]),na.rm = TRUE), max(max(res$error[!is.nan(predictivo)],na.rm = TRUE),max(predictivo2[!is.nan(predictivo)],na.rm = TRUE)))
  # print(rango)
  # plot(time,res$error[!is.nan(predictivo)], ylim= c(rango))
  # lines(time, predictivo2[!is.nan(predictivo)])

  plot(Y, sal$coefficients, xlab="Real values", ylab="Estimated values", main= paste("R-square", round(sal$r2,3),sep=" " ))

  return(sal)
}





