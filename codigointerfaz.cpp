#include <RcppArmadillo.h>
#include <math.h>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline arma::mat cumtrapz(arma::mat X, arma::mat Y) {
  if (X.n_rows != Y.n_rows || X.n_cols != Y.n_cols)
    throw std::invalid_argument("Arguments 'x' and 'y' must be matrices of the same dimension");
  
  bool transposed = false;
  if (X.n_rows == 1) {
    // row vector
    X = X.t();
    Y = Y.t();
    transposed = true; 
  }
  
  uint n = X.n_rows;
  uint m = X.n_cols;
  arma::mat C = arma::zeros(n,m);
  arma::mat tmp = (X.rows(1,n-1) - X.rows(0,n-2)) % (Y.rows(1,n-1) + Y.rows(0,n-2));
  
  if (tmp.n_rows == 1)
    C.row(1) = 0.5 * tmp;
  else
    C.rows(1,n-1) = 0.5 * cumsum(tmp);
  
  if (transposed) 
    return C.t();
  else 
    return C;
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline arma::mat trapecio(arma::mat X, arma::mat Y) {
  if (X.n_rows != Y.n_rows || X.n_cols != Y.n_cols)
    throw std::invalid_argument("Arguments 'x' and 'y' must be matrices of the same dimension");
  
   bool transposed = false;
   
   uint m = 1;
   
   
  if (X.n_cols == 1) {
     // row vector
     // X = X.t();
     // Y = Y.t();
     
    
     transposed = true; 
  }else{
    X = X.t();
    Y = Y.t();
    
    uint m = X.n_cols;
    
    
    
  }
  
  
  
  
  // 
  
  // 
  // X = X.t();
  // Y = Y.t();
  // 
  
  uint n = X.n_rows;
  arma::mat C = arma::zeros(n,m);
  arma::mat tmp = (X.rows(1,n-1) - X.rows(0,n-2)) % (Y.rows(1,n-1) + Y.rows(0,n-2));
  arma::mat result;
  
  
  
  if (tmp.n_rows == 1){
    C.row(1) = 0.5 * tmp;
  result= C.row(n-1);
  }else{
    C.rows(1,n-1) = 0.5 * cumsum(tmp);
  result= C.row(n-1);
  }
  
  
  
  
  
  return result;  

}




// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline arma::mat eucdistance1(arma::mat X, arma::mat t){
  
  uint n = X.n_rows;
  uint m = X.n_cols;

  arma::mat distancias(n,n);   
  

  
  
  double distanciaaux=0;
  
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      //arma::mat aux= (X.row(i)-X.row(j))%(X.row(i)-X.row(j));
      //aux= aux.t();                   
      distancias(i,j)=  sqrt(trapecio(t,((X.row(i)-X.row(j))%(X.row(i)-X.row(j))).t())(0));
    
    
    }  
  }
  
  
  
 return distancias; 
 
}




// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline arma::mat eucdistance2(arma::mat X, arma::mat t, arma::mat x){
  
  uint n = X.n_rows;
  uint m = X.n_cols;
  uint nx= x.n_rows;
  
  
  arma::mat distancias(nx,n);   
  
  
  
  
  double distanciaaux=0;
  
  for(int i=0;i<nx;i++){
    for(int j=0;j<n;j++){
      //arma::mat aux= (X.row(i)-X.row(j))%(X.row(i)-X.row(j));
      //aux= aux.t();                   
      distancias(i,j)=  sqrt(trapecio(t,((x.row(i)-X.row(j))%(x.row(i)-X.row(j))).t())(0));
      
      
    }  
  }
  
  
  
  return distancias; 
  
}








// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

inline arma::vec gaussiankernelinicial(arma::vec x,double h){
  arma::vec salida;
  salida= double(2/sqrt(double(2)*M_PI))*exp(-(x/h)%(x/h)*0.5);
  return salida; 
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

inline arma::vec triangular(arma::vec x,double h){
//  salida= double(2/sqrt(double(2)*M_PI))*exp(-(x/h)%(x/h)*0.5);
 // salida= double(1.35/16)*(1-(x/h))%(1-(x/h))%(1-(x/h));
  
  int l= x.size();
  arma::vec salida(l);
  
  
  for(int i=0; i<l;i++){
    
    
    if(double(x(i)/h)<0|double(x(i)/h)>1){
      salida(i)=0;
    }else{
    //  salida(i)=double(double(35)/double(16))*pow(double(double(1)-pow(double(x(i)/double(h)),2)),3);
      
      salida(i)=double(double(35)/double(16))*pow(double(1-pow(x(i)/h,2)),3);
      
      
      
    }
    
  }
  
  
  
  
  return salida; 
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List nadayara1pred(arma::mat distancias, arma::mat X, arma::mat t, arma::mat Y, arma::mat hs){
  
  uint n = X.n_rows;
  uint m = X.n_cols;
 // arma::mat distancias(n,n); 
  
  arma::vec media= mean(Y);
  arma::vec mediavectorial(n);
  mediavectorial.fill(media(0));
  arma::vec residuos=Y-mediavectorial;
  
  
  uint nh = hs.n_rows;
  
  float h=1;
  
  //distancias= eucdistance1(X, t);
  
  
  arma::mat prediciones(n,nh);  
  arma::mat residuosglobal(n,nh);  
  arma::mat R2(nh,1);  
  
  
  
  arma::vec aux(n);
  arma::vec aux2(n);
  arma::vec aux3(n);
  arma::vec residuosaux(n);
  
  
  
  
  for(int j=0; j<nh;j++){
    
    for(int i=0; i<n; i++){
      
      aux2= distancias.row(i).t();    
      
      aux= gaussiankernelinicial(aux2,hs(j));
      
      
      
      
      
      aux3(i)=  sum((aux%Y))/sum(aux);
      
    }
    prediciones.col(j)= aux3;
    
    residuosaux= Y-aux3;  
    residuosaux= 1-double(sum(residuosaux%residuosaux)/sum(residuos%residuos)); 
    residuosglobal.col(j)= Y-aux3; 
    R2.row(j)=residuosaux(0);
  }
  
  
  
  
  
  return Rcpp::List::create(Rcpp::Named("predicciones")=prediciones,Rcpp::Named("Residuos")=residuosglobal,Rcpp::Named("R2")=R2);  
  
  
  
}





// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List nadayaracrossregnuevo(arma::mat X, arma::mat t, arma::mat Y, arma::mat hs,arma::umat  indices1, arma::umat  indices2){
  
  uint n = X.n_rows;
  uint m = X.n_cols;
  arma::mat distancias(n,n); 
  arma::vec media= mean(Y);
  arma::vec mediavectorial(n);
  mediavectorial.fill(media(0));
  arma::vec residuos=Y-mediavectorial;
  uint nh = hs.n_rows;
  
  // salida rendimiento modelos
  
  arma::mat prediciones(n,nh);  
  arma::mat residuosglobal(n,nh);  
  arma::mat R2(nh,1); 
  arma::mat error(nh,1); 
  
  
  
  
  float h=1;
  uint n1x= indices1.n_rows;
  uint n1p= indices1.n_cols;
  
  
  uint n2x= indices2.n_rows;
  uint n2p= indices2.n_cols;
  
  
  
  
  
  
  
  
  
  // arma::mat distancias2(nx,n);
  // 
  // distancias2= eucdistance2(nx,t,x);
  // 
  distancias= eucdistance1(X, t);
  // 
  // 
  // 
  // 
  // arma::mat prediciones2(nx,nh);  
  // 
  
  
 arma::mat Y1(n1x,1);  
 arma::mat Y2(n2x,1);  
 
 arma::mat distancias2(n2x,n1x);  
 
 arma::umat indices1aux(n1x,1);  
 arma::umat indices2aux(n2x,1);  
  
  
  arma::mat R2validacion(nh,n1p); 
  
  arma::mat R2validacionaux(nh,1); 
  
  
  for(int l=0; l<n1p;l++){

    indices1aux= indices1.col(l);
    indices2aux= indices2.col(l);
    
    
  
  Y1= Y.rows(indices1aux);
  Y2= Y.rows(indices2aux);
  
  
  arma::vec auxx(n1x);
  arma::vec aux2x(n1x);
  arma::vec aux3x(n2x);
  arma::mat prediciones2(n2x,nh);
  
  //arma::vec residuosauxx(nx);
  
  distancias2= distancias.rows(indices2aux);
  distancias2= distancias2.cols(indices1aux);
  
  arma::vec residuosauxx(n1x);
  
  arma::vec mediax= mean(Y2);
  arma::vec mediavectorialx(n2x);
  mediavectorialx.fill(mediax(0));
  
  
  arma::vec residuosx=Y2-mediavectorialx;
  
  
  for(int j=0; j<nh;j++){
    
    for(int i=0; i<n2x; i++){
      aux2x= distancias2.row(i).t();    
      auxx= gaussiankernelinicial(aux2x,hs(j));
      aux3x(i)=  sum((auxx%Y1))/sum(auxx);
      
    }
     prediciones2.col(j)= aux3x;
    // 
    residuosauxx= Y2-aux3x;  
    
    
    residuosauxx= sum(residuosauxx%residuosauxx);
    
    //residuosauxx= 1-double(sum(residuosauxx%residuosauxx)/sum(residuosx%residuosx)); 
    
    
    
      //residuosglobal.col(j)= Y2-aux3x; 
      R2validacionaux.row(j)=residuosauxx(0);
  }
  
   R2validacion.col(l)=R2validacionaux;
  
  }
  
  
  
  
  
  
  
  arma::vec aux(n);
  arma::vec aux2(n);
  arma::vec aux3(n);
  arma::vec residuosaux(n);
  
  
  
  for(int j=0; j<nh;j++){
    
    for(int i=0; i<n; i++){
      aux2= distancias.row(i).t();    
      aux= gaussiankernelinicial(aux2,hs(j));
      aux3(i)=  sum((aux%Y))/sum(aux);
      
    }
    prediciones.col(j)= aux3;
    
    residuosaux= Y-aux3; 
    error.row(j)= sum(residuosaux%residuosaux);
    
    residuosaux= 1-double(sum(residuosaux%residuosaux)/sum(residuos%residuos)); 
    residuosglobal.col(j)= Y-aux3; 
    R2.row(j)=residuosaux(0);
    
  }
  
  
  
  
  return Rcpp::List::create(Rcpp::Named("predicciones")=prediciones,Rcpp::Named("Residuos")=residuosglobal,  Rcpp::Named("R2")=R2,Rcpp::Named("error")=error,Rcpp::Named("R2global")=R2validacion);
  
  
  
}











// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R


crossvalidationprednuevo= function(fdata, respuesta, hs=seq(0.5,100,length=200)){
  X= fdata$data
  n= dim(X)[1]
  p= dim(X)[2]
  t= fdata$argvals
  Y= respuesta
  conjunto= data.frame(X,Y)
  aux= complete.cases(conjunto)
  conjunto= conjunto[complete.cases(conjunto),]
  X= conjunto[,1:p]  
  Y= conjunto[,p+1]
  X= as.matrix(X)
  Y= as.matrix(Y)
  t= as.matrix(t)
  hs= as.matrix(hs)
  
  n= dim(X)[1]
  p= dim(X)[2]
  
  cogerdatosentrenamiento= n-1
  cogerdatostest= 1
  cross= n
  
  indices= 0:(n-1)
  indices1= matrix(0, nrow=cogerdatosentrenamiento, ncol=cross)
  indices2= matrix(0, nrow=cogerdatostest, ncol=cross)
  indicesaux=1:n
  
  for(j in 1:cross){
    

    generar= indicesaux[-c(j)]
    
    
    
    indices1[,j]= indices[generar] 
    indices2[,j]= indices[c(j)] 
  }
  
  indices1= as.matrix(indices1)
  
  indices2= as.matrix(indices2)
  
  
  res=nadayaracrossregnuevo(X,t,Y,hs,indices1, indices2)
  
  predictivo= apply(res$R2global,1,function(x){sqrt(mean(x))}) 
  ventanas= hs
  cuantos= sum(is.nan(predictivo)) 
  ventanas2= ventanas[!is.nan(predictivo)]
  predictivo2= predictivo[!is.nan(predictivo)]
  R2= res$R2[!is.nan(predictivo)]

  
  rango1= c(min(predictivo2), max(predictivo2))
  rango2= c(min(R2), max(R2))

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
  plot(time, cell.density, pch=15,  xlab="", ylab="", ylim= rango1, 
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
  
  

  

  # rango= c(min(min(res$error[!is.nan(predictivo)],na.rm = TRUE),min(predictivo2[!is.nan(predictivo)]),na.rm = TRUE), max(max(res$error[!is.nan(predictivo)],na.rm = TRUE),max(predictivo2[!is.nan(predictivo)],na.rm = TRUE)))
  # print(rango)
  # plot(time,res$error[!is.nan(predictivo)], ylim= c(rango))
  # lines(time, predictivo2[!is.nan(predictivo)])
  
  return(res)
}





prednuevo= function(fdata, respuesta, hs=5){
  X= fdata$data
  n= dim(X)[1]
  p= dim(X)[2]
  t= fdata$argvals
  Y= respuesta
  conjunto= data.frame(X,Y)
  aux= complete.cases(conjunto)
  conjunto= conjunto[complete.cases(conjunto),]
  X= conjunto[,1:p]  
  Y= conjunto[,p+1]
  X= as.matrix(X)
  Y= as.matrix(Y)
  t= as.matrix(t)
  hs= as.matrix(hs)
  
  n= dim(X)[1]
  p= dim(X)[2]
  
  dista= eucdistance1(X,t)

  sal= nadayara1pred(dista,X,t,Y,hs)
  # rango= c(min(min(res$error[!is.nan(predictivo)],na.rm = TRUE),min(predictivo2[!is.nan(predictivo)]),na.rm = TRUE), max(max(res$error[!is.nan(predictivo)],na.rm = TRUE),max(predictivo2[!is.nan(predictivo)],na.rm = TRUE)))
  # print(rango)
  # plot(time,res$error[!is.nan(predictivo)], ylim= c(rango))
  # lines(time, predictivo2[!is.nan(predictivo)])
  
  plot(Y, sal$predicciones, xlab="Real values", ylab="Estimated values", main= paste("R-square", round(sal$R2,3),sep=" " ))
  
  return(sal)
}





n=50
p=100

X= matrix(abs(rnorm(n*p)), ncol=p, nrow= n)
x= matrix(abs(rnorm(n*p)), ncol=p, nrow= n/2)


t= seq(0,1,length=p)
t= as.matrix(t)
Y= apply(X,1, mean)+rnorm(n,0,1)
#Y= apply(X,1,mean)
Y= as.matrix(Y)
hs= seq(0.5,50,length=100)

hs= seq(0.5,10,length=500)

hs= as.matrix(hs)


library("fda.usc")

fdata= fdata(X, argvals = t)
respuesta= Y
hs= hs




# fdata objeto fda usc
# respuesta, vector con la respuesta
# hs vector de ventanas, en el primer caso tiene el rango predefinido   hs= seq(0.5,9, length=350)


# en el segundo caso, el usuario deberia poder elegir entre un valor de 1 y 9, con un boton desplage que tome ese rango de valores 


crossvalidationprednuevo(fdata,respuesta,hs)

prednuevo(fdata, respuesta, hs=4)










 # fdaaux= fdata(X, argvals= t)
 # ts= proc.time()
 # 
 # m=fregre.np.cv(fdaaux,as.numeric(Y),h=as.numeric(hs))
 # proc.time()-ts
 # 
 # 
 # data("tecator")
 # X= tecator$absorp.fdata$data
 # t= tecator$absorp.fdata$argvals
 # X= as.matrix(X)
 # t= as.matrix(t)
 # Y= as.matrix(tecator$y$Fat)
 # hs= seq(0.5,50,length=100)
 # 
 # hs= as.matrix(hs)
 # res=nadayara(X,t,Y,hs)
 # 
 # library("fda.usc")
 # fdaaux= fdata(X, argvals= t)
 # 
 # fdaaux= fdata(X, argvals= t)
 # ts= proc.time()
 # 
 # m=fregre.np.cv(fdaaux,as.numeric(Y),h=as.numeric(hs))
 # 
 # proc.time()-ts
 # 





# Y= as.matrix(Y)
# hs= 1
# hs= as.matrix(hs)
# nadayara(X,t,Y,hs)
# 
# library("fda.usc")
# 
# fdaaux= fdata(X, argvals= t)
# 
# m=fregre.np(fdaaux,as.numeric(Y),h=as.numeric(hs))
# 
# 
# max(abs(m$fitted.values-nadayara(X,t,Y,hs)))
# 
# 
# data("tecator")
# X= tecator$absorp.fdata$data
# t= tecator$absorp.fdata$argvals
# X= as.matrix(X)
# t= as.matrix(t)
# Y= as.matrix(tecator$y$Fat)
# hs= seq(0.5,50,length=100)
# 
# hs= as.matrix(hs)
# res=nadayara(X,t,Y,hs)
# 
# library("fda.usc")
# 
# fdaaux= fdata(X, argvals= t)
# 
# m=fregre.np(fdaaux,as.numeric(Y),h=as.numeric(hs))


# max(abs(m$fitted.values-nadayara(X,t,Y,hs)))


# 
# t= as.matrix(t)
# 
# tmatriz= matrix(0,nrow=n, ncol=p ) 
# 
# for(i in 1:dim(tmatriz)[1]){
#   
#   tmatriz[i,]= t
#   
# }
# 
# t2=proc.time()
# distancia1=eucdistance1(X,t)
# proc.time()-t2
# 
# fdaaux= fdata(X, argvals= t)
# t2=proc.time()
# 
# distancia2=fda.usc::metric.lp(fdaaux)
# proc.time()-t2


# hist(distancia1-distancia2)
# 
# trapz(tmatriz,X)
# 
# 
# trapz(as.matrix(tmatriz[1,]),as.matrix(X[1,]))
# 
# fdaaux= fdata(X[1,],argvals = t) 
# 
# fda.usc::int.simpson(fdaaux) 



*/
