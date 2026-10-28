#!/usr/bin/env Rscript
# Native, unmodified emIRT. No block restarts or test/self-report access.
suppressPackageStartupMessages(library(emIRT));suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('emIRT'))=='0.0.15')
a<-commandArgs(TRUE);stopifnot(length(a)==1);job<-fromJSON(a[1]);input<-job$input;out<-job$out
dir.create(out,recursive=TRUE,showWarnings=FALSE)
sha<-function(p)digest::digest(file=p,algo='sha256')
atomic<-function(x,p){write_json(x,paste0(p,'.tmp'),auto_unbox=TRUE,pretty=TRUE,digits=16,null='null');stopifnot(file.rename(paste0(p,'.tmp'),p))}
m<-fromJSON(file.path(input,'manifest.json'));stopifnot(!m$test_read)
for(n in names(m$outputs))stopifnot(sha(file.path(input,n))==m$outputs[[n]])
v<-matrix(readBin(file.path(input,'train_matrix.int8'),integer(),n=m$n_people*m$n_pairs,size=1,signed=TRUE),nrow=m$n_people,byrow=TRUE)
stopifnot(all(v%in%c(-1L,0L,1L)),sum(v>=0)==m$n_train)
votes<-matrix(0,nrow(v),ncol(v));votes[v==0]<--1;votes[v==1]<-1
stopifnot(sum(votes!=0)==m$n_train,all(votes[v==-1]==0))
rm(v);gc()
set.seed(job$seed)
starts<-emIRT:::getStarts(.N=m$n_people,.J=m$n_pairs,.D=1,.type='zeros')
priors<-emIRT:::makePriors(.N=m$n_people,.J=m$n_pairs,.D=1)
priors$beta$sigma<-diag(job$item_variance,2)
control<-list(threads=2L,verbose=TRUE,maxit=as.integer(job$maxit),convtype=1L,thresh=job$tolerance,checkfreq=50L,withLB=FALSE,withProbs=FALSE,asEM=TRUE)
atomic(list(job=job,data=m$audit,emIRT_version=as.character(packageVersion('emIRT')),controls=control,test_read=FALSE),file.path(out,'binding.json'))
t0<-proc.time();fit<-binIRT(.rc=list(votes=votes),.starts=starts,.priors=priors,.D=1,.control=control);fit_seconds<-unname((proc.time()-t0)[['elapsed']])
# Save native object immediately. It is recoverable even if evaluation fails.
saveRDS(fit,file.path(out,'native_fit.rds.tmp'),compress=FALSE);stopifnot(file.rename(file.path(out,'native_fit.rds.tmp'),file.path(out,'native_fit.rds')))
stopifnot(all(is.finite(fit$means$x)),all(is.finite(fit$means$beta)))
stopifnot(as.integer(fit$runtime$threads)==2L)
b<-fit$means$beta;theta<-as.numeric(fit$means$x)
# Verify the sign/orientation against the native training geometric mean.
# Evaluate only observed cells; block by rows to avoid another dense allocation.
logpsum<-0;stocklogpsum<-0;count<-0
for(first in seq(1,nrow(votes),by=256)){
 rr<-first:min(first+255,nrow(votes));ij<-which(votes[rr,,drop=FALSE]!=0,arr.ind=TRUE)
 eta<-b[ij[,2],1]+b[ij[,2],2]*theta[rr[ij[,1]]]
 yy<-votes[cbind(rr[ij[,1]],ij[,2])]
 lp<-pnorm(eta,lower.tail=TRUE,log.p=TRUE);lp[yy==-1]<-pnorm(eta[yy==-1],lower.tail=FALSE,log.p=TRUE)
 nativep<-pnorm(eta);nativep[yy==-1]<-1-nativep[yy==-1]
 stocklogpsum<-stocklogpsum+sum(log(nativep))
 logpsum<-logpsum+sum(lp);count<-count+length(lp)
}
stable_train_log_loss<--logpsum/count;gmp<-exp(stocklogpsum/count);gmp_error<-abs(gmp-as.numeric(fit$fit$gmp));stopifnot(count==m$n_train,is.finite(gmp_error),gmp_error<1e-8)
val<-read.csv(file.path(input,'validation.csv'));eta<-b[val$local_pair+1,1]+b[val$local_pair+1,2]*theta[val$local_user+1]
pr<-pnorm(eta);stopifnot(all(is.finite(pr)))
pp<-pmin(1-1e-7,pmax(1e-7,pr));ll<-mean(-val$vote*log(pp)-(1-val$vote)*log1p(-pp))
acc<-mean(ifelse(pr==.5,.5,as.numeric((pr>.5)==val$vote)))
val$probability<-pr;write.csv(val,file.path(out,'validation_predictions.csv'),row.names=FALSE)
people<-read.csv(file.path(input,'people.csv'));people$coordinate<-theta;write.csv(people,file.path(out,'coordinates.csv'),row.names=FALSE)
write.csv(data.frame(iteration=2:fit$runtime$iters,fit$convstats),file.path(out,'convergence.csv'),row.names=FALSE)
res<-list(status='COMPLETE',kind='EMIRT',dimension=1,seed=job$seed,data=m$audit,validation_log_loss=ll,validation_accuracy=acc,validation_brier=mean((pr-val$vote)^2),fit_call_seconds=fit_seconds,native_converged=isTRUE(fit$runtime$conv),iterations=fit$runtime$iters,stop_reason=if(isTRUE(fit$runtime$conv))'native_correlation' else 'iteration_budget',parameter_count=m$n_people+2*m$n_pairs,train_gmp_native=fit$fit$gmp,train_gmp_check=gmp,gmp_parity_error=gmp_error,stable_train_log_loss=stable_train_log_loss,effective_openmp_threads=fit$runtime$threads,settings=job,outputs=list(native_fit=sha(file.path(out,'native_fit.rds')),predictions=sha(file.path(out,'validation_predictions.csv'))),test_read=FALSE)
atomic(res,file.path(out,'fit.json'));cat(toJSON(res,auto_unbox=TRUE,digits=8),'\n')
