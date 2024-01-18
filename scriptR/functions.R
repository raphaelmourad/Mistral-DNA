


#######################################
# Split the genome in bins for hg38
splitGenomeBins<-function(Genome, binsize, contextsize){
 SeqInfo=seqinfo(Genome)
 Chr.V=seqnames(SeqInfo)[-grep("_|chrM|chrY",seqnames(SeqInfo))]
 SeqInfo=seqinfo(Genome)[Chr.V]
 bin.GR=NULL
 for(i in 1:length(Chr.V)){
  bin.GRi=GRanges(Chr.V[i],IRanges(breakInChunks(totalsize=seqlengths(SeqInfo)[i],chunksize=binsize)))
  bin.GRi=bin.GRi[start(resize(bin.GRi,contextsize,fix='center'))>0]
  bin.GRi=bin.GRi[-((length(bin.GRi)-(contextsize/binsize)):length(bin.GRi))]
  if(i==1){
    bin.GR=bin.GRi
  }else{
    bin.GR=c(bin.GR,bin.GRi)
  }
 }
 seqinfo(bin.GR)=SeqInfo[Chr.V]
 return(bin.GR)
}


