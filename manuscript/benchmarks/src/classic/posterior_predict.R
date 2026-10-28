# PREPARED INPUT QUERY_CSV ROLE(validation|test) DIAG_AUDIT OUT [TEST_SEAL]
a<-commandArgs(TRUE);stopifnot(length(a)%in%c(6L,7L))
script<-normalizePath(sub('^--file=','',commandArgs()[grep('^--file=',commandArgs())]));source(file.path(dirname(script),'native_chunk.R'));library(jsonlite)
prepared<-normalizePath(a[1]);input<-normalizePath(a[2]);query<-normalizePath(a[3]);role<-a[4];auditpath<-normalizePath(a[5]);out<-a[6]
stopifnot(role%in%c('validation','test'),!dir.exists(out))
p<-readRDS(prepared);audit<-fromJSON(auditpath,simplifyVector=FALSE)
stopifnot(identical(audit$all_converged,TRUE),audit$prepared_sha256==sha(prepared),sha(file.path(input,'manifest.json'))==p$input_manifest_sha256)
for(path in names(audit$input_hashes))stopifnot(sha(path)==audit$input_hashes[[path]])
for(name in names(p$input_hashes))stopifnot(sha(file.path(input,name))==p$input_hashes[[name]])
if(role=='validation')stopifnot(identical(query,normalizePath(file.path(input,'validation.csv'))))
if(role=='test'){
 stopifnot(length(a)==7);seal<-fromJSON(a[7],simplifyVector=FALSE)
 # Root's handoff explicitly binds this query AND this converged posterior.
 stopifnot(identical(seal$test_authorized,TRUE),seal$query_sha256==sha(query),seal$diagnostic_audit_sha256==sha(auditpath),seal$prepared_sha256==sha(prepared))
}
qh<-sha(query);ah<-sha(auditpath);rows<-read.csv(query)
stopifnot(all(c('person_index','pair_index','source_id','vote')%in%names(rows)),!anyDuplicated(rows$source_id),all(rows$vote%in%c(0,1)))
u<-match(rows$person_index,p$ui);j<-match(rows$pair_index,p$ji);ok<-!is.na(u)&!is.na(j);take<-which(ok)
stopifnot(length(take)>0);total<-numeric(nrow(rows));chain_sums<-list();draws<-0L
snapshots<-unlist(audit$snapshot_paths);stopifnot(length(snapshots)==4)
allhashes<-list();coordinates<-matrix(0,length(p$ui),2)
for(index in seq_along(snapshots)){
 folder<-snapshots[index];mp<-file.path(folder,'manifest.json');stopifnot(sha(mp)==audit$snapshot_manifest_hashes[[index]])
 m<-fromJSON(mp,simplifyVector=FALSE);stopifnot(m$prepared_sha256==sha(prepared))
 pm<-readRDS(file.path(folder,'posterior_means.rds'));coordinates<-coordinates+pm$xbar/4
 current<-numeric(nrow(rows));number<-0L
 paths<-names(m$checkpoint_hashes);ends<-as.integer(sub('chunk_([0-9]+).rds','\\1',basename(paths)));paths<-paths[order(ends)]
 for(path in paths){
  stopifnot(sha(path)==m$checkpoint_hashes[[path]]);allhashes[[path]]<-m$checkpoint_hashes[[path]]
  z<-readRDS(path)
  for(k in which(z$iterations>5000L)){
   x<-matrix(z$x[k,,],length(p$ui),2);b<-matrix(z$beta[k,,],length(p$ji),3)
   for(start in seq(1L,length(take),8192L)){
    ids<-take[start:min(length(take),start+8191L)]
    pr<-pnorm(rowSums(x[u[ids],,drop=FALSE]*b[j[ids],1:2,drop=FALSE])-b[j[ids],3])
    stopifnot(all(is.finite(pr)));current[ids]<-current[ids]+pr
   };number<-number+1L
  };rm(z)
 }
 stopifnot(number==m$draws);total<-total+current;draws<-draws+number;chain_sums[[as.character(m$seed)]]<-current/number
}
rows$covered<-ok;rows$probability<-NA_real_;rows$probability[ok]<-total[ok]/draws
stopifnot(all(is.finite(rows$probability[ok])))
dir.create(out,recursive=TRUE)
write.csv(rows,file.path(out,'predictions.csv'),row.names=FALSE)
write.csv(data.frame(person_index=p$ui,uuid=p$uuid,x1=coordinates[,1],x2=coordinates[,2]),file.path(out,'coordinates.csv'),row.names=FALSE)
saveRDS(chain_sums,file.path(out,'per_chain_probabilities.rds'),compress=FALSE)
for(path in names(allhashes))stopifnot(sha(path)==allhashes[[path]])
stopifnot(sha(query)==qh,sha(auditpath)==ah)
for(path in names(audit$input_hashes))stopifnot(sha(path)==audit$input_hashes[[path]])
files<-list.files(out,full.names=TRUE)
write_json(list(status='CONVERGED',converged=TRUE,kind='bayesian',role=role,prediction='mean of probit probabilities over all four chains and every retained post-burnin draw; NOT probit of parameter means',draws_total=draws,n_rows=nrow(rows),n_covered=sum(ok),n_people_covered=length(unique(rows$person_index[ok])),query_sha256=qh,diagnostic_audit_sha256=ah,prepared_sha256=sha(prepared),draw_hashes=allhashes,source_sha256=sha(script),output_hashes=setNames(lapply(files,sha),basename(files)),test_read=role=='test'),file.path(out,'manifest.json.tmp'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
stopifnot(file.rename(file.path(out,'manifest.json.tmp'),file.path(out,'manifest.json')))
