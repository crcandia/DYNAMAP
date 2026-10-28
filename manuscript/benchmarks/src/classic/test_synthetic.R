# Synthetic independent draws, not empirical model fitting.
a<-commandArgs(TRUE);base<-normalizePath(a[1]);out<-a[2];stopifnot(!dir.exists(out));dir.create(out,recursive=TRUE)
source(file.path(base,'native_chunk.R'));library(jsonlite)
input<-file.path(out,'input');dir.create(input)
write.csv(data.frame(person_index=c(4L,5L,6L,7L,8L),pair_index=1:5,vote=c(0,1,0,1,1),source_id=1:5),file.path(input,'validation.csv'),row.names=FALSE)
h<-list(validation.csv=sha(file.path(input,'validation.csv')));write_json(list(outputs=h,test_exported=FALSE),file.path(input,'manifest.json'),auto_unbox=TRUE)
p<-list(ui=1:8,ji=1:6,uuid=paste0('synthetic',1:8),probe_u=4:8,ij=cbind(4:8,1:5),anchor_local=1:3,input_hashes=h,input_manifest_sha256=sha(file.path(input,'manifest.json')),prior_variant='base')
prep<-file.path(out,'prepared');dir.create(prep);saveRDS(p,file.path(prep,'prepared.rds'),compress=FALSE)
write_json(list(prepared_sha256=sha(file.path(prep,'prepared.rds'))),file.path(prep,'manifest.json'),auto_unbox=TRUE)
expected<-numeric(5);ct<-0L;meanx<-matrix(0,8,2);meanb<-matrix(0,6,3)
for(seed in c(101L,201L,301L,401L)){
 chain<-file.path(out,paste0('chain',seed));dir.create(chain);set.seed(seed)
 fp<-list(prepared_sha256=sha(file.path(prep,'prepared.rds')),thin=10L,burnin=5000L,chunk=500L,seed=seed)
 saveRDS(fp,file.path(chain,'fingerprint.rds'));prev<-NA_character_
 for(start in seq(0L,9500L,500L)){
  x<-array(rnorm(50*8*2,sd=.7),c(50,8,2));b<-array(rnorm(50*6*3,sd=.6),c(50,6,3));b[,,3]<-b[,,3]+.8
  z<-list(start=start,end=start+500L,iterations=seq(start+10L,start+500L,10L),x=x,beta=b,fingerprint=fp,previous_sha256=prev,seconds=0)
  path<-file.path(chain,sprintf('chunk_%05d.rds',z$end));saveRDS(z,path,compress=FALSE);prev<-sha(path)
  if(start>=5000)for(k in 1:50){xx<-matrix(x[k,,],8,2);bb<-matrix(b[k,,],6,3);expected<-expected+pnorm(rowSums(xx[4:8,,drop=FALSE]*bb[1:5,1:2,drop=FALSE])-bb[1:5,3]);meanx<-meanx+xx;meanb<-meanb+bb;ct<-ct+1L}
 }
 code<-system2(file.path(R.home('bin'),'Rscript'),c('--vanilla',shQuote(file.path(base,'snapshot.R')),shQuote(file.path(prep,'prepared.rds')),shQuote(chain),'10000',shQuote(file.path(out,paste0('snapshot',seed)))))
 stopifnot(code==0L)
}
plugin<-pnorm(rowSums((meanx/ct)[4:8,,drop=FALSE]*(meanb/ct)[1:5,1:2,drop=FALSE])-(meanb/ct)[1:5,3])
write.csv(data.frame(source_id=1:5,expected=expected/ct,plugin=plugin),file.path(out,'expected.csv'),row.names=FALSE)
stopifnot(max(abs(expected/ct-plugin))>.01)
cat('Synthetic snapshots generated, direct integrated expectation differs from plugin\n')
