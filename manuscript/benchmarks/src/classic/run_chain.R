#!/usr/bin/env Rscript
# Usage PREPARED.rds OUTPUT_DIR SEED TARGET_ITERATIONS (10000/15000/.../30000).
# Raising target is an explicit supervisor action after four-chain diagnostics.
a<-commandArgs(TRUE);stopifnot(length(a)==4)
script<-normalizePath(sub('^--file=','',commandArgs()[grep('^--file=',commandArgs())]))
source(file.path(dirname(script),'native_chunk.R'))
suppressPackageStartupMessages(library(pscl));suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('pscl'))=='1.5.9')
prepared<-normalizePath(a[1]);prepared_manifest<-fromJSON(file.path(dirname(prepared),'manifest.json'));stopifnot(sha(prepared)==prepared_manifest$prepared_sha256)
out<-a[2];seed<-as.integer(a[3]);target<-as.integer(a[4])
stopifnot(seed%in%c(101L,201L,301L,401L),target>=10000L,target<=30000L,target%%5000L==0L)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
lock<-file.path(out,'RUNNING');if(!dir.create(lock))stop('Chain already locked; investigate stale PID before removing RUNNING')
# on.exit is unavailable at top level: main guarantees lock cleanup on R errors.
main<-function(){
 on.exit(unlink(lock,recursive=TRUE),add=TRUE)
 writeLines(as.character(Sys.getpid()),file.path(lock,'pid'))
 fingerprint<-list(prepared_sha256=sha(prepared),runner_sha256=sha(script),kernel_sha256=sha(file.path(dirname(script),'native_chunk.R')),R=R.version.string,pscl=as.character(packageVersion('pscl')),seed=seed,chunk=500L,thin=10L,burnin=5000L)
 fp<-file.path(out,'fingerprint.rds')
 if(file.exists(fp))stopifnot(identical(readRDS(fp),fingerprint))else atomic_rds(fingerprint,fp)
 prep<-readRDS(prepared);done<-0L;previous<-NA_character_
 chunks<-sort(list.files(out,pattern='^chunk_[0-9]{5}\\.rds$',full.names=TRUE))
 for(path in chunks){z<-readRDS(path);stopifnot(z$start==done,z$end==done+500L,identical(z$fingerprint,fingerprint),identical(z$previous_sha256,previous));done<-z$end;previous<-sha(path)}
 initial_path<-file.path(out,'initial_state.rds')
 if(file.exists(initial_path)){
 initial<-readRDS(initial_path);stopifnot(identical(initial$fingerprint,fingerprint))
 }else{
 stopifnot(!length(chunks));set.seed(seed);x<-matrix(rnorm(nrow(prep$rc$votes)*2),nrow(prep$rc$votes),2);x[prep$anchor_local,]<-prep$prior$xp[prep$anchor_local,];b<-matrix(rnorm(ncol(prep$rc$votes)*3,sd=.1),ncol(prep$rc$votes),3)
 initial<-list(x=x,b=b,rng=.Random.seed,fingerprint=fingerprint,iteration=0L);atomic_rds(initial,initial_path)
 }
 jsonlite::write_json(list(iteration=0L,checkpoint='initial_state.rds',sha256=sha(initial_path)),file.path(out,'initialprogress.json.tmp'),auto_unbox=TRUE,pretty=TRUE)
 stopifnot(file.rename(file.path(out,'initialprogress.json.tmp'),file.path(out,'initialprogress.json')))
 if(length(chunks)){x<-z$last_x;b<-z$last_b;rng<-z$rng;rm(z)}else{x<-initial$x;b<-initial$b;rng<-initial$rng}
 rm(initial)
 while(done<target){
 started<-proc.time()[3];z<-advance_native(prep,x,b,rng)
 z$start<-done;z$end<-done+500L;z$iterations<-seq(done+10L,done+500L,10L);z$fingerprint<-fingerprint;z$previous_sha256<-previous;z$seconds<-unname(proc.time()[3]-started)
 path<-file.path(out,sprintf('chunk_%05d.rds',z$end));stopifnot(!file.exists(path));atomic_rds(z,path)
 previous<-sha(path);done<-z$end;x<-z$last_x;b<-z$last_b;rng<-z$rng
 jsonlite::write_json(list(iteration=done,target=target,checkpoint=basename(path),sha256=previous,seconds=z$seconds),file.path(out,'progress.json.tmp'),auto_unbox=TRUE,pretty=TRUE)
 stopifnot(file.rename(file.path(out,'progress.json.tmp'),file.path(out,'progress.json')))
 cat(format(Sys.time()),'COMMITTED',done,'sha256',previous,'\n');flush.console();rm(z);gc()
 }
 stopifnot(sha(prepared)==fingerprint$prepared_sha256)
}
main()
