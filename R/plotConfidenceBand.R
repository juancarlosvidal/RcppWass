// plotConfidenceBand.R: Rcpp/Wass glue
//
// Copyright (C) 2019 - 2020  Juan Carlos Vidal and Marcos Matabuena 
//
// This file is part of RcppWass.
//
// RcppWass is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// RcppWass is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with RcppWass.  If not, see <http://www.gnu.org/licenses/>.

plot_confidence_band <- function(t, res) {
  Qp <- fdata(res$Qpred, argvals = t)
  Ql <- fdata(res$Q_lx, argvals = t)
  Qu <- fdata(res$Q_ux, argvals = t)
  plot(Qp)
  lines(Qu, lty=2, col="red")
  lines(Ql, lty=2, col="red")
}
