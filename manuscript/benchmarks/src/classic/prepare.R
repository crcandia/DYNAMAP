#!/usr/bin/env Rscript
# Shared training-only preparation. Uses the frozen pilot filtering/anchors.
a<-commandArgs(TRUE);stopifnot(length(a)%in%c(2L,3L))
variant<-if(length(a)==3)a[3]else "base";stopifnot(variant%in%c("base","item_shrink4","alternate_anchors"))
input<-normalizePath(a[1]);out<-a[2]
if(dir.exists(out)&&length(list.files(out)))stop('Existing preparation: verify and reuse; never overwrite')
dir.create(out,recursive=TRUE)
suppressPackageStartupMessages(library(pscl)); suppressPackageStartupMessages(library(jsonlite))
sha<-function(p){tool<-Sys.which('shasum');args<-c('-a','256',shQuote(p));if(!nzchar(tool)){tool<-Sys.which('sha256sum');args<-shQuote(p)};stopifnot(nzchar(tool));strsplit(system2(tool,args,stdout=TRUE),' ')[[1]][1]}
info <- fromJSON(file.path(input,'manifest.json')); stopifnot(!info$test_exported)
for(name in names(info$outputs)) stopifnot(sha(file.path(input,name))==info$outputs[[name]])
people <- read.csv(file.path(input,'people.csv')); pairs <- read.csv(file.path(input,'pairs.csv'))
stopifnot(identical(people$person_index,seq_len(info$n_people)),identical(pairs$pair_index,seq_len(info$n_rollcalls)))
v <- matrix(readBin(file.path(input,'train_matrix.int8'),integer(),n=info$n_people*info$n_rollcalls,size=1,signed=TRUE),nrow=info$n_people,byrow=TRUE)
stopifnot(all(v %in% c(-1L,0L,1L)),sum(v>=0)==info$n_train)
# Same prefilter for both methods: >=20 training choices, and columns with both
# outcomes and >=2.5% minority after row filtering. No validation values used.
keep_u <- rowSums(v>=0)>=20
cv <- v[keep_u,,drop=FALSE]; yes <- colSums(cv==1); no <- colSums(cv==0)
keep_j <- yes>0 & no>0 & pmin(yes,no)/pmax(1,yes+no)>=.025
# Resolve support to a fixed point using training observations only.
repeat {
 ku <- keep_u & rowSums(v[,keep_j,drop=FALSE]>=0)>=20
 cv <- v[ku,,drop=FALSE]; yes <- colSums(cv==1); no <- colSums(cv==0)
 kj <- keep_j & yes>0 & no>0 & pmin(yes,no)/pmax(1,yes+no)>=.025
 if(identical(ku,keep_u)&&identical(kj,keep_j)) break
 keep_u<-ku;keep_j<-kj
}
stopifnot(sum(keep_u)>=4,sum(keep_j)>=4)
ui<-which(keep_u);ji<-which(keep_j);vv<-v[ui,ji,drop=FALSE];rm(v,cv);gc()
# Anchor selection is a deterministic function of TRAINING responses only.
# Missing entries contribute zero to column-centered initial structure.
coverage<-rowSums(vv>=0)
anchor_pool<-head(order(-coverage,ui),512)
z<-vv[anchor_pool,,drop=FALSE];has<-colSums(z>=0)>0;z<-z[,has,drop=FALSE]
z[z<0]<-NA;cm<-colMeans(z,na.rm=TRUE);z<-sweep(z,2,cm);z[is.na(z)]<-0
init<-svd(z,nu=2,nv=0)$u
a1<-which.min(init[,1]);a2<-which.max(init[,1])
axis<-init[a2,]-init[a1,];cross<-abs((init[,1]-init[a1,1])*axis[2]-(init[,2]-init[a1,2])*axis[1]);cross[c(a1,a2)]<--Inf
a3<-which.max(cross);stopifnot(length(unique(c(a1,a2,a3)))==3,cross[a3]>1e-10)
if(variant=="alternate_anchors") {
 available<-setdiff(seq_len(nrow(init)),c(a1,a2,a3));stopifnot(length(available)>=3)
 b1<-available[which.min(init[available,1])];b2<-available[which.max(init[available,1])]
 axis2<-init[b2,]-init[b1,];cross2<-abs((init[,1]-init[b1,1])*axis2[2]-(init[,2]-init[b1,2])*axis2[1])
 cross2[setdiff(seq_len(nrow(init)),setdiff(available,c(b1,b2)))]<--Inf
 b3<-which.max(cross2);stopifnot(cross2[b3]>1e-10);a1<-b1;a2<-b2;a3<-b3
}
anchor_local<-anchor_pool[c(a1,a2,a3)]
anchor_names<-paste0('person_',ui[anchor_local]);rm(z,init)
votes<-vv;votes[votes<0]<-9L
rc<-rollcall(votes,yea=1,nay=0,missing=9,notInLegis=8,legis.names=paste0('person_',ui),vote.names=paste0('pair_',ji))
write.csv(data.frame(person_index=people$person_index,uuid=people$uuid,retained=keep_u),file.path(out,'people_coverage.csv'),row.names=FALSE)
write.csv(data.frame(pair_index=pairs$pair_index,retained=keep_j),file.path(out,'pair_coverage.csv'),row.names=FALSE)
jsonlite::write_json(list(anchor_people=ui[anchor_local],anchor_selection='training-only SVD, maximum512people by descendingcoverage/tiescanonicalindex',constraints=list(c(-1,0),c(1,0),c(0,1)),minvotes=20,lop=.025),file.path(out,'pre_fit_design.json'),auto_unbox=TRUE,pretty=TRUE)

prior<-list(xp=matrix(0,nrow(vv),2),xpv=matrix(1,nrow(vv),2),bp=matrix(0,ncol(vv),3),bpv=matrix(.04,ncol(vv),3))
if(variant=="item_shrink4")prior$bpv[,]<-.16
prior$xp[anchor_local,]<-rbind(c(-1,0),c(1,0),c(0,1));prior$xpv[anchor_local,]<-1e12
ij<-which(vv>=0,arr.ind=TRUE);set.seed(20261013);ij<-ij[sort(sample.int(nrow(ij),min(1000,nrow(ij)))),,drop=FALSE]
nonanchors<-setdiff(seq_len(nrow(vv)),anchor_local);probe_u<-sort(nonanchors[sample.int(length(nonanchors),min(100,length(nonanchors)))])
object<-list(prior_variant=variant,rc=rc,prior=prior,ui=ui,ji=ji,anchor_local=anchor_local,ij=ij,probe_u=probe_u,uuid=people$uuid[ui],input_manifest_sha256=sha(file.path(input,'manifest.json')),input_hashes=info$outputs)
saveRDS(object,file.path(out,'prepared.rds.tmp'),compress=FALSE)
stopifnot(file.rename(file.path(out,'prepared.rds.tmp'),file.path(out,'prepared.rds')))
for(name in names(info$outputs))stopifnot(sha(file.path(input,name))==info$outputs[[name]])
write_json(list(prepared_sha256=sha(file.path(out,'prepared.rds')),input_manifest_sha256=object$input_manifest_sha256,n_people=length(ui),n_pairs=length(ji),prior_variant=variant,coverage_hashes=setNames(lapply(c("people_coverage.csv","pair_coverage.csv"),function(n)sha(file.path(out,n))),c("people_coverage.csv","pair_coverage.csv")),test_read=FALSE),file.path(out,'manifest.json'),auto_unbox=TRUE,pretty=TRUE)
