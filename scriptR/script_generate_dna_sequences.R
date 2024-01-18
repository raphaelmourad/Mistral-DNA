# Raphael Mourad
# 18/01/2024
# MIAT


# use this R script to generate the DNA sequences to feed Mistral-DNA.

#### SETUP PROJECT FOLDER
setwd("../")

#### LOAD LIBRARIES

# Load R packages
library(GenomicRanges)
library(Biostrings)
library(BSgenome.Hsapiens.UCSC.hg38)


#### LOAD FUNCTIONS

source("scriptR/functions.R")

# Target
target="hg38" # the target genome is the genome to predict
GenomeTarget=genomes[[target]]
binsize=200
contextsize=binsize

bin.GR=splitGenomeBins(GenomeTarget, binsize, contextsize)

bin.seq=getSeq(GenomeTarget,bin.GR)

binlf=letterFrequency(bin.seq,letters=c("A","T","G","C","N"))
#bin.GR=bin.GR[binlf[,5]==0]
bin.seq=bin.seq[binlf[,5]==0]
bin.seq=as.character(bin.seq)
bin.seq=c("text",bin.seq)

matBin=matrix(bin.seq)
fwrite(matBin, file='data/genome_sequences/hg38/sequences_hg38_200b.csv.gz',
	sep='\t',quote=F,row.names=F,col.names=F)



