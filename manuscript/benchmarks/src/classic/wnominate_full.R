#!/usr/bin/env Rscript
# Native wnominate1.5 full-cohort point fit. No test/ideology inputs.
# A completed native call is saved atomically BEFORE any extraction/prediction.
# Fortran cannot resume an interrupted call. Postprocessing can resume from RDS.
suppressPackageStartupMessages(library(pscl))
suppressPackageStartupMessages(library(wnominate))
suppressPackageStartupMessages(library(jsonlite))

sha256 <- function(path) {
  tool <- Sys.which('sha256sum')
  args <- shQuote(path)
  if(!nzchar(tool)) { tool <- Sys.which('shasum'); args <- c('-a','256',args) }
  if(!nzchar(tool)) stop('A SHA256 utility is required')
  output <- system2(tool,args,stdout=TRUE)
  if(!is.null(attr(output,'status'))) stop('SHA256 command failed')
  strsplit(output[1],'[[:space:]]+')[[1]][1]
}
atomic_rds <- function(value,path) {
  temp <- paste0(path,'.tmp.',Sys.getpid())
  if(file.exists(path)||file.exists(temp)) stop('Refusing overwrite: ',path)
  saveRDS(value,temp,compress=FALSE)
  if(!file.rename(temp,path)) stop('Atomic RDS rename failed')
}
atomic_json <- function(value,path) {
  temp <- paste0(path,'.tmp.',Sys.getpid())
  if(file.exists(path)||file.exists(temp)) stop('Refusing overwrite: ',path)
  write_json(value,temp,auto_unbox=TRUE,pretty=TRUE,null='null',digits=NA)
  if(!file.rename(temp,path)) stop('Atomic JSON rename failed')
}
verify_bindings <- function(bound) {
  for(name in names(bound)) if(!identical(sha256(name),as.character(bound[[name]]))) stop('Changed input: ',name)
}
native_probability <- function(users,mid,spread,beta,weights,uu,jj) {
  uy<-users[uu,,drop=FALSE]-(mid[jj,,drop=FALSE]-spread[jj,,drop=FALSE])
  un<-users[uu,,drop=FALSE]-(mid[jj,,drop=FALSE]+spread[jj,,drop=FALSE])
  eta<-as.numeric(beta)*(exp(-rowSums(sweep(uy^2,2,as.numeric(weights)^2/2,'*')))-exp(-rowSums(sweep(un^2,2,as.numeric(weights)^2/2,'*'))))
  plogis(eta)
}

run_wnominate <- function(input,prepared,out,seed=101L,source_file) {
  input<-normalizePath(input);prepared<-normalizePath(prepared)
  source_file<-normalizePath(source_file)
  stopifnot(as.character(packageVersion('wnominate'))=='1.5',as.character(packageVersion('pscl'))=='1.5.9')
  stopifnot(seed==101L) # one deterministic point fit, not artificial five-seed replication
  start<-proc.time()[3]
  info<-fromJSON(file.path(input,'manifest.json'),simplifyVector=FALSE)
  prep<-fromJSON(file.path(prepared,'manifest.json'),simplifyVector=FALSE)
  stopifnot(identical(info$test_exported,FALSE),identical(prep$test_read,FALSE))
  stopifnot(sha256(file.path(prepared,'prepared.rds'))==prep$prepared_sha256)
  stopifnot(sha256(file.path(input,'manifest.json'))==prep$input_manifest_sha256)
  names_in<-c(file.path(input,'manifest.json'),file.path(prepared,'manifest.json'),file.path(prepared,'prepared.rds'),source_file)
  bound<-setNames(lapply(names_in,sha256),names_in)
  for(name in names(info$outputs)) {
    path<-file.path(input,name)
    stopifnot(sha256(path)==info$outputs[[name]])
    bound[[path]]<-info$outputs[[name]]
  }
  for(name in c('people_coverage.csv','pair_coverage.csv','pre_fit_design.json')) {
    path<-file.path(prepared,name);bound[[path]]<-sha256(path)
  }
  object<-readRDS(file.path(prepared,'prepared.rds'))
  stopifnot(identical(object$input_manifest_sha256,prep$input_manifest_sha256))
  stopifnot(identical(unlist(object$input_hashes),unlist(info$outputs)))
  rc<-object$rc;ui<-object$ui;ji<-object$ji
  stopifnot(inherits(rc,'rollcall'),length(ui)==nrow(rc$votes),length(ji)==ncol(rc$votes))
  stopifnot(identical(rownames(rc$votes),paste0('person_',ui)),identical(colnames(rc$votes),paste0('pair_',ji)))
  stopifnot(length(ui)==prep$n_people,length(ji)==prep$n_pairs,!anyDuplicated(ui),!anyDuplicated(ji))
  stopifnot(length(object$uuid)==length(ui),all(rc$votes %in% c(0,1,9)))
  # Native .Fortran API does not accept long vectors. Do not silently subset.
  if(as.double(length(ui))^2>.Machine$integer.max || as.double(length(ui))*length(ji)*2>.Machine$integer.max) stop('Native .Fortran work-array index limit exceeded')
  anchors<-rownames(rc$votes)[object$anchor_local]
  stopifnot(length(anchors)==3,!anyDuplicated(anchors))
  design<-list(kind='wnominate',seed=seed,dims=2L,minvotes=20L,lop=.025,trials=3L,
    polarity=anchors[c(2,3)],ubeta=15,uweights=.5,input_bindings=bound,
    R=R.version.string,packages=list(pscl=as.character(packageVersion('pscl')),wnominate=as.character(packageVersion('wnominate'))),
    rng_scope='trials3 maps to one deterministic point pass; no bootstrap uncertainty',
    native_checkpoint_supported=FALSE,test_read=FALSE)
  if(file.exists(file.path(out,'manifest.json'))) {
    old<-fromJSON(file.path(out,'manifest.json'),simplifyVector=FALSE)
    stopifnot(identical(old$design$input_bindings,bound),old$design$seed==seed)
    for(name in names(old$output_hashes)) stopifnot(sha256(file.path(out,name))==old$output_hashes[[name]])
    verify_bindings(bound);return(invisible(old))
  }
  rawpath<-file.path(out,'raw_fit_with_input_mapping.rds')
  recovered<-file.exists(rawpath)
  if(dir.exists(out)&&length(list.files(out,all.files=FALSE))&&!recovered) stop('Interrupted native call has no checkpoint. Archive partial directory before a new attempt.')
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  lock<-file.path(out,'.writer_lock')
  if(!dir.create(lock,showWarnings=FALSE)) stop('Existing writer lock, inspect PID before recovery')
  on.exit(unlink(lock,recursive=TRUE),add=TRUE)
  writeLines(as.character(Sys.getpid()),file.path(lock,'pid'))
  if(!recovered) {
    atomic_json(design,file.path(out,'pre_fit_design.json'))
    set.seed(seed);rng_before<-.Random.seed
    fit_start<-proc.time()[3]
    fit<-wnominate(rc,dims=2,minvotes=20,lop=.025,trials=3,polarity=anchors[c(2,3)],verbose=TRUE)
    # First post-return action: preserve native result and exact ID mapping.
    atomic_rds(list(fit=fit,person_index=ui,pair_index=ji,
      rc_person_names=rownames(rc$votes),rc_pair_names=colnames(rc$votes),
      design=design,fit_seconds=unname(proc.time()[3]-fit_start),rng_before=rng_before,rng_after=.Random.seed),rawpath)
    rm(fit);gc()
  }
  raw<-readRDS(rawpath)
  stopifnot(identical(raw$design$input_bindings,bound),raw$design$seed==seed)
  stopifnot(identical(raw$person_index,ui),identical(raw$pair_index,ji),
            identical(raw$rc_person_names,rownames(rc$votes)),identical(raw$rc_pair_names,colnames(rc$votes)))
  fit<-raw$fit
  users<-as.matrix(fit$legislators[,c('coord1D','coord2D')])
  mid<-as.matrix(fit$rollcalls[,c('midpoint1D','midpoint2D')])
  spread<-as.matrix(fit$rollcalls[,c('spread1D','spread2D')])
  stopifnot(identical(rownames(users),rownames(rc$votes)),nrow(users)==length(ui),nrow(mid)==length(ji),
            identical(rownames(fit$rollcalls),as.character(seq_len(length(ji)))))
  good_u<-apply(users,1,function(x)all(is.finite(x)))
  good_j<-apply(mid,1,function(x)all(is.finite(x)))&apply(spread,1,function(x)all(is.finite(x)))
  us<-head(which(good_u),20);js<-head(which(good_j),20)
  stopifnot(length(us)>0,length(js)>0)
  official<-nomprob(mid[js,,drop=FALSE]-spread[js,,drop=FALSE],mid[js,,drop=FALSE]+spread[js,,drop=FALSE],users[us,,drop=FALSE],as.numeric(fit$beta),as.numeric(fit$weights)^2/2,normal=0)
  rr<-expand.grid(u=us,j=js)
  parity<-max(abs(as.vector(official)-native_probability(users,mid,spread,fit$beta,fit$weights,rr$u,rr$j)))
  stopifnot(is.finite(parity),parity<1e-12)
  write.csv(data.frame(person_index=ui,finite_coordinates=good_u,GMP=fit$legislators$GMP,correct_classification=fit$legislators$CC),file.path(out,'person_fit_diagnostics.csv'),row.names=FALSE)
  write.csv(data.frame(pair_index=ji,finite_endpoints=good_j,GMP=fit$rollcalls$GMP,PRE=fit$rollcalls$PRE),file.path(out,'pair_fit_diagnostics.csv'),row.names=FALSE)
  write.csv(data.frame(person_index=ui[good_u],uuid=object$uuid[good_u],x1=users[good_u,1],x2=users[good_u,2]),file.path(out,'coordinates.csv'),row.names=FALSE)
  write.csv(data.frame(pair_index=ji,midpoint1=mid[,1],midpoint2=mid[,2],spread1=spread[,1],spread2=spread[,2]),file.path(out,'item_parameters.csv'),row.names=FALSE)
  for(name in c('people_coverage.csv','pair_coverage.csv')) stopifnot(file.copy(file.path(prepared,name),file.path(out,name),overwrite=TRUE))
  # Validation is read only after the native result has been preserved.
  val<-read.csv(file.path(input,'validation.csv'))
  uu<-match(val$person_index,ui);jj<-match(val$pair_index,ji);ok<-!is.na(uu)&!is.na(jj)
  ok[ok]<-good_u[uu[ok]]&good_j[jj[ok]]
  val$probability<-NA_real_;val$probability[ok]<-native_probability(users,mid,spread,fit$beta,fit$weights,uu[ok],jj[ok]);val$covered<-ok
  write.csv(val,file.path(out,'validation_predictions.csv'),row.names=FALSE)
  writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
  verify_bindings(bound)
  outputs<-list.files(out,full.names=TRUE)
  outputs<-outputs[file.info(outputs)$isdir %in% FALSE]
  manifest<-list(status='COMPLETE',kind='wnominate',seed=seed,design=design,
    n_people_input=info$n_people,n_pairs_input=info$n_rollcalls,n_people_prefiltered=length(ui),n_pairs_prefiltered=length(ji),
    n_people_retained=sum(good_u),n_pairs_retained=sum(good_j),n_validation=nrow(val),n_validation_covered=sum(ok),
    fit_seconds=raw$fit_seconds,current_process_seconds=unname(proc.time()[3]-start),recovered_postprocessing=recovered,
    raw_fit_sha256=sha256(rawpath),rng_unchanged=identical(raw$rng_before,raw$rng_after),
    diagnostics=list(scope='native point estimator, not MCMC',finite_global_parameters=all(is.finite(c(fit$beta,fit$weights))),optimization_convergence_certified=FALSE,mean_person_GMP=mean(fit$legislators$GMP[good_u],na.rm=TRUE),mean_pair_GMP=mean(fit$rollcalls$GMP[good_j],na.rm=TRUE),native_fit_statistics=as.list(fit$fits)),
    settings=list(beta=as.numeric(fit$beta),weights=as.numeric(fit$weights),official_nomprob_maxerror=parity,
      prediction='native logistic exponential utilities, yea=mid-spread, weights squared/2'),
    output_hashes=setNames(lapply(outputs,sha256),basename(outputs)),test_read=FALSE)
  atomic_json(manifest,file.path(out,'manifest.json'))
  invisible(manifest)
}

if(sys.nframe()==0) {
  args<-commandArgs(TRUE)
  if(length(args)<3||length(args)>4) stop('usage: wnominate_full.R INPUT_DIR PREPARED_DIR OUTPUT_DIR [SEED101]')
  script<-sub('^--file=','',commandArgs()[grep('^--file=',commandArgs())])
  run_wnominate(args[1],args[2],args[3],if(length(args)==4)as.integer(args[4])else 101L,script)
}
