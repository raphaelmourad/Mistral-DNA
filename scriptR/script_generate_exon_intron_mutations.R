# Raphael Mourad
# 18/01/2024
# MIAT


# use this R script to generate the DNA sequences to feed Mistral-DNA.

#### SETUP PROJECT FOLDER
setwd("")

#### LOAD LIBRARIES



library(GenomicRanges)
library(ensembldb)
library(AnnotationHub)
library(BSgenome.Hsapiens.UCSC.hg38)
library(seqinr)


# Function to get ref and alt allele sequences
getSeqRefAlt<-function(SNP.GR,window=201){

 region.GRr=resize(SNP.GR,window,fix="center")
 region.seq=as.character(getSeq(BSgenome.Hsapiens.UCSC.hg38, names=seqnames(region.GRr), 
 start=start(region.GRr), end=end(region.GRr)))

 SNPposrel=ceiling((window+1)/2)
 region.seqRef=region.seq
 substring(region.seqRef,SNPposrel,SNPposrel)=as.character(region.GRr$ref)
 region.seqAlt=region.seq
 substring(region.seqAlt,SNPposrel,SNPposrel)=as.character(region.GRr$alt)

 return(list(ref=DNAStringSet(region.seqRef),alt=DNAStringSet(region.seqAlt)))
}

Chr.V=c(paste0("chr",1:22),"chrX")


# Get annotations from hg38
hub = AnnotationHub()

query(hub, c("EnsDb", "Homo sapiens", "97"))

# Get exons
exons.GR=exons(hub[["AH73881"]])
exons.GR=reduce(exons.GR)

# Get gene bodies
genes.GR=genes(hub[["AH73881"]])
genes.GR=reduce(genes.GR)

# Get introns
introns.GR=unlist(subtract(genes.GR,exons.GR))


# Make SNPs in exons and introns
SNPexon.GR=resize(exons.GR,width=1,fix="center")
SNPexon.GR=GRanges(paste0("chr",seqnames(SNPexon.GR)),ranges(SNPexon.GR))
SNPexon.GR=SNPexon.GR[seqnames(SNPexon.GR)%in%Chr.V]
SNPexon.GR=sample(SNPexon.GR,10000)
SNPexon.seq=getSeq(BSgenome.Hsapiens.UCSC.hg38,SNPexon.GR)
SNPexon.GR$ref=as.character(SNPexon.seq)
SNPexon.GR$alt=sapply(1:length(SNPexon.GR),function(x){sample(setdiff(c("A","T","G","C"),SNPexon.GR$ref[x]),1)})

SNPintron.GR=resize(introns.GR,width=1,fix="center")
SNPintron.GR=GRanges(paste0("chr",seqnames(SNPintron.GR)),ranges(SNPintron.GR))
SNPintron.GR=SNPintron.GR[seqnames(SNPintron.GR)%in%Chr.V]
SNPintron.GR=sample(SNPintron.GR,10000)
SNPintron.seq=getSeq(BSgenome.Hsapiens.UCSC.hg38,SNPintron.GR)
SNPintron.GR$ref=as.character(SNPintron.seq)
SNPintron.GR$alt=sapply(1:length(SNPintron.GR),function(x){sample(setdiff(c("A","T","G","C"),SNPintron.GR$ref[x]),1)})

# Get sequences
win=201

SNPexon.seqRef=getSeqRefAlt(SNPexon.GR,window=win)[["ref"]]
SNPexon.seqAlt=getSeqRefAlt(SNPexon.GR,window=win)[["alt"]]

SNPintron.seqRef=getSeqRefAlt(SNPintron.GR,window=win)[["ref"]]
SNPintron.seqAlt=getSeqRefAlt(SNPintron.GR,window=win)[["alt"]]

# SAVE REF AND ALT SEQUENCES
writeXStringSet(x=SNPexon.seqRef,filepath=paste0("data/SNP/SNPexon_ref_",win,"b.fasta.gz"),compress=T)
writeXStringSet(x=SNPexon.seqAlt,filepath=paste0("data/SNP/SNPexon_alt_",win,"b.fasta.gz"),compress=T)

writeXStringSet(x=SNPintron.seqRef,filepath=paste0("data/SNP/SNPintron_ref_",win,"b.fasta.gz"),compress=T)
writeXStringSet(x=SNPintron.seqAlt,filepath=paste0("data/SNP/SNPintron_alt_",win,"b.fasta.gz"),compress=T)






