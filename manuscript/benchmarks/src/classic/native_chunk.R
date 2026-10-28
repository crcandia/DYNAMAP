# Official pscl transition, unchanged target/priors; no data-dependent tuning.
sha<-function(p){tool<-Sys.which('shasum');args<-c('-a','256',shQuote(p));if(!nzchar(tool)){tool<-Sys.which('sha256sum');args<-shQuote(p)};stopifnot(nzchar(tool));strsplit(system2(tool,args,stdout=TRUE),' ')[[1]][1]}
atomic_rds<-function(x,path){tmp<-paste0(path,'.tmp');saveRDS(x,tmp,compress=FALSE);stopifnot(file.rename(tmp,path))}
advance_native<-function(prep,x,b,rng,iterations=500L,thin=10L){
 assign('.Random.seed',rng,envir=.GlobalEnv)
 fit<-pscl::ideal(prep$rc,d=2,maxiter=iterations,burnin=0,thin=thin,impute=FALSE,store.item=TRUE,normalize=FALSE,priors=prep$prior,startvals=list(x=x,b=b),dropList=list(codes='notInLegis',lop=0),verbose=FALSE)
 rng_end<-.Random.seed
 stopifnot(all(is.finite(fit$x)),all(is.finite(fit$beta)))
 xx<-fit$x[-1,,,drop=FALSE];bb<-fit$beta[-1,,,drop=FALSE]
 last<-dim(xx)[1];xend<-matrix(xx[last,,],nrow(prep$rc$votes),2);bend<-matrix(bb[last,,],ncol(prep$rc$votes),3)
 list(x=xx,beta=bb,last_x=xend,last_b=bend,rng=rng_end)
}
