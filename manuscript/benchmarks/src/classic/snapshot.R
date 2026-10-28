a<-commandArgs(TRUE);stopifnot(length(a)==4)
script<-normalizePath(sub('^--file=','',commandArgs()[grep('^--file=',commandArgs())]));source(file.path(dirname(script),'native_chunk.R'));library(jsonlite)
prepared<-normalizePath(a[1]);chain<-normalizePath(a[2]);target<-as.integer(a[3]);out<-a[4]
stopifnot(target%in%seq(10000L,30000L,5000L),!dir.exists(out))
p<-readRDS(prepared);fp<-readRDS(file.path(chain,'fingerprint.rds'))
stopifnot(fp$prepared_sha256==sha(prepared),fp$thin==10L,fp$burnin==5000L,fp$chunk==500L)
paths<-file.path(chain,sprintf('chunk_%05d.rds',seq(500L,target,500L)));stopifnot(all(file.exists(paths)))
dir.create(out,recursive=TRUE);n<-length(p$ui);m<-length(p$ji);count<-0L;sx<-matrix(0,n,2);sb<-matrix(0,m,3)
con<-setNames(lapply(c('probabilities','coordinates','item_parameters'),function(k)file(file.path(out,paste0(k,'.f64')),'wb')),c('probabilities','coordinates','item_parameters'))
prev<-NA_character_;done<-0L;hashes<-list();seconds<-0
for(path in paths){
 z<-readRDS(path);h<-sha(path);stopifnot(z$start==done,z$end==done+500L,identical(z$fingerprint,fp),identical(z$previous_sha256,prev),identical(z$iterations,seq(done+10L,done+500L,10L)))
 hashes[[path]]<-h;prev<-h;done<-z$end;seconds<-seconds+z$seconds
 for(k in which(z$iterations>5000L)){
  x<-matrix(z$x[k,,],n,2);b<-matrix(z$beta[k,,],m,3);stopifnot(all(is.finite(x)),all(is.finite(b)))
  sx<-sx+x;sb<-sb+b;count<-count+1L
  pr<-pnorm(rowSums(x[p$ij[,1],,drop=FALSE]*b[p$ij[,2],1:2,drop=FALSE])-b[p$ij[,2],3])
  writeBin(as.double(pr),con$probabilities,size=8,endian='little')
  writeBin(as.double(x[p$probe_u,,drop=FALSE]),con$coordinates,size=8,endian='little')
  writeBin(as.double(b),con$item_parameters,size=8,endian='little')
 };rm(z);gc()
}
for(cn in con)close(cn);stopifnot(count==(target-5000L)/10L)
saveRDS(list(xbar=sx/count,betabar=sb/count,person_index=p$ui,pair_index=p$ji),file.path(out,'posterior_means.rds'),compress=FALSE)
write.csv(data.frame(person_index=p$ui,uuid=p$uuid,x1=sx[,1]/count,x2=sx[,2]/count),file.path(out,'coordinates.csv'),row.names=FALSE)
probes<-list(probabilities=paste0('person_',p$ui[p$ij[,1]],'_pair_',p$ji[p$ij[,2]]),coordinates=c(paste0('person_',p$ui[p$probe_u],'_D1'),paste0('person_',p$ui[p$probe_u],'_D2')),item_parameters=c(paste0('pair_',p$ji,'_disc1'),paste0('pair_',p$ji,'_disc2'),paste0('pair_',p$ji,'_difficulty')))
write_json(probes,file.path(out,'probes.json'),auto_unbox=FALSE)
for(path in names(hashes))stopifnot(sha(path)==hashes[[path]]);stopifnot(sha(prepared)==fp$prepared_sha256)
files<-list.files(out,full.names=TRUE)
write_json(list(status='COMPLETE',seed=fp$seed,target=target,burnin=5000L,thin=10L,draws=count,n_people=n,n_pairs=m,prepared_path=prepared,prepared_sha256=fp$prepared_sha256,input_manifest_sha256=p$input_manifest_sha256,prior_variant=if(is.null(p$prior_variant))'base'else p$prior_variant,chain_path=chain,fingerprint=fp,fingerprint_sha256=sha(file.path(chain,'fingerprint.rds')),checkpoint_hashes=hashes,fit_seconds=seconds,binary_format='little-endian f64 draw-major rows, probes.json column order',source_sha256=sha(script),output_hashes=setNames(lapply(files,sha),basename(files)),test_read=FALSE),file.path(out,'manifest.json.tmp'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
stopifnot(file.rename(file.path(out,'manifest.json.tmp'),file.path(out,'manifest.json')))
