


#### LOAD LIBRARIES

# Load genomes downloaded from Bioconductor
library(BSgenome.Hsapiens.UCSC.hg38)
library(BSgenome.Mmusculus.UCSC.mm10)
library(BSgenome.Btaurus.UCSC.bosTau9)
library(BSgenome.Cjacchus.UCSC.calJac3)
library(BSgenome.Cfamiliaris.UCSC.canFam3)
library(BSgenome.Mdomestica.UCSC.monDom5)
library(BSgenome.Mfuro.UCSC.musFur1)
library(BSgenome.Ppaniscus.UCSC.panPan2)
library(BSgenome.Ptroglodytes.UCSC.panTro6)
library(BSgenome.Mmulatta.UCSC.rheMac10)
library(BSgenome.Rnorvegicus.UCSC.rn7)
library(BSgenome.Sscrofa.UCSC.susScr11)

# Load genomes forged using the R script "script_forge_genome_assembly.R"
library(BSgenome.cavPor.UCSC.cavPor3)
library(BSgenome.criGri.UCSC.criGri1)
library(BSgenome.equCab.UCSC.equCab3)  
library(BSgenome.felCat.UCSC.felCat9) 
library(BSgenome.gorGor.UCSC.gorGor6) 
library(BSgenome.macFas.UCSC.macFas5)
library(BSgenome.micMur.UCSC.micMur2) 
library(BSgenome.nomLeu.UCSC.nomLeu3)
library(BSgenome.oryCun.UCSC.oryCun2)
library(BSgenome.otoGar.UCSC.otoGar3) 
library(BSgenome.papAnu.UCSC.papAnu4) 
library(BSgenome.ponAbe.UCSC.ponAbe3) 
library(BSgenome.saiBol.UCSC.saiBol1)
library(BSgenome.tarSyr.UCSC.tarSyr2) 

# Load other libraries
library(GenomicRanges)
library(Biostrings)
library(data.table)
library(liftOver)
library(pbmcapply)
library(BSgenome.Hsapiens.UCSC.hg19)




# Assemblies
assembs=c("hg38","rheMac10","calJac3","panTro6","panPan2","ponAbe3",
	"gorGor6","papAnu4","macFas5","saiBol1","nomLeu3",
	"micMur2","otoGar3",
	"mm10","rn7",
	"musFur1","oryCun2","susScr11",
	"felCat9","canFam3","equCab3","bosTau9","monDom5")
assembFiles=c("human","macaque","marmoset","chimpanzee","pygmy_chimpanzee",
	"gorilla","crab-eating_macaque","squirrel_monkey",
	"mouse_lemur","galago",
	"mouse","rat",
	"ferret","rabbit","pig",
	"cat","dog","horse","cow","opossum")
genomes=list("hg38"=BSgenome.Hsapiens.UCSC.hg38,
		"rheMac10"=BSgenome.Mmulatta.UCSC.rheMac10,
		"calJac3"=BSgenome.Cjacchus.UCSC.calJac3,
		"panTro6"=BSgenome.Ptroglodytes.UCSC.panTro6,
		"panPan2"=BSgenome.Ppaniscus.UCSC.panPan2,
		"ponAbe3"=BSgenome.ponAbe.UCSC.ponAbe3,
		"gorGor6"=BSgenome.gorGor.UCSC.gorGor6,
		"papAnu4"=BSgenome.papAnu.UCSC.papAnu4,
		"macFas5"=BSgenome.macFas.UCSC.macFas5,
		"saiBol1"=BSgenome.saiBol.UCSC.saiBol1,
		"nomLeu3"=BSgenome.nomLeu.UCSC.nomLeu3,
		"micMur2"=BSgenome.micMur.UCSC.micMur2,
		"otoGar3"=BSgenome.otoGar.UCSC.otoGar3,
		"mm10"=BSgenome.Mmusculus.UCSC.mm10,
		"rn7"=BSgenome.Rnorvegicus.UCSC.rn7,
		"musFur1"=BSgenome.Mfuro.UCSC.musFur1,
		"oryCun2"=BSgenome.oryCun.UCSC.oryCun2,
		"susScr11"=BSgenome.Sscrofa.UCSC.susScr11,
		"felCat9"=BSgenome.felCat.UCSC.felCat9,
		"canFam3"=BSgenome.Cfamiliaris.UCSC.canFam3,
		"equCab3"=BSgenome.equCab.UCSC.equCab3,
		"bosTau9"=BSgenome.Btaurus.UCSC.bosTau9,
		"monDom5"=BSgenome.Mdomestica.UCSC.monDom5)


#######################################
# Function to remap granges from an assembly to another using liftover
mapBinToOtherAssembly<-function(bin.GR, assemb1, assemb2, chain=NULL, gap=50, option=1){

 # Load chain file
 if(is.null(chain)){
  Assemb2=paste(toupper(substr(assemb2, 1, 1)), substr(assemb2, 2, nchar(assemb2)), sep="")
  chain = import.chain(paste0("data/liftover/",assemb1,"/",assemb1,"To",Assemb2,".over.chain"))
  #print("read chain file")
 }

 # Seqinfo
 SeqInfo2=seqinfo(genomes[[assemb2]])

 # Liftover
 lo2.GRL=liftOver(bin.GR, chain)

 if(option==1){
 # Reconstruct bins by removing GAPs
 win=max(width(bin.GR))
 lo2r.GRL=trim(resize(reduce(resize(lo2.GRL,width=width(lo2.GRL)+gap,fix="center")),width=win,fix="center"))
 lo2s.GR=unlist(lo2r.GRL)
 lo2s.GR=GRanges(seqnames(lo2s.GR),ranges(lo2s.GR),seqinfo=SeqInfo2)
 lo2s.GR=lo2s.GR[seqnames(lo2s.GR)!="chrM"]

 # Remove bins mapping to different places (this is a simplication, this should be improved latter!!!)
 if(length(lo2s.GR)>0){
  bin2.GR=lo2s.GR[!duplicated(names(lo2s.GR))]
  idx2=as.numeric(substring(names(bin2.GR),10))
  bin2.GR$idx=idx2
  bin2.GR$match=1
  idxToRemove=GenomicRanges:::get_out_of_bound_index(bin2.GR) 
  if(length(idxToRemove)>0){
   bin2.GR=bin2.GR[-idxToRemove]
  } 
 }
 }else if(option==2){
 lo2s.GR=unlist(lo2.GRL)
 lo2s.GR=GRanges(seqnames(lo2s.GR),ranges(lo2s.GR),seqinfo=SeqInfo2)
 bin2.GR=lo2s.GR[seqnames(lo2s.GR)!="chrM"]
 idx2=as.numeric(substring(names(bin2.GR),10))
 bin2.GR$idx=idx2
 bin2.GR$match=1 
 }

 # Draw remaining bins from the assemb2 genome
 if(length(lo2s.GR)>0){
  k=length(bin.GR)-length(bin2.GR)
 }else{
  k=length(bin.GR)
 } 
  
 if(k>0){
  if(length(lo2s.GR)>0){
   idx2bis=setdiff(as.numeric(substring(names(bin.GR),10)),bin2.GR$idx)
   bin2bis.GR=rep(GRanges(seqnames(bin2.GR)[1],IRanges(1,width(bin.GR)[1])),k)     
  }else if(length(lo2s.GR)==0){
   idx2bis=1:k
   bin2bis.GR=rep(GRanges(names(chain[1]),IRanges(1,width(bin.GR)[1])),k)  
  }
  bin2bis.GR$idx=idx2bis
  bin2bis.GR$match=0
  names(bin2bis.GR)=paste0("binTarget",idx2bis)

  # Merge with remaining bins
  if(length(lo2s.GR)>0){
   bin2m.GR=c(bin2.GR,bin2bis.GR)
  }else if(length(lo2s.GR)==0){
   bin2m.GR=bin2bis.GR
  }
 }else{bin2m.GR=bin2.GR}
 bin2m.GR=bin2m.GR[order(bin2m.GR$idx)]

 return(bin2m.GR)
}


#######################################
# Bin labeling given ChIP-seq peak data
labelingBin<-function(bin.GR,file_bed,minoverlap){

 SeqInfo=seqinfo(bin.GR)
 assemb=as.vector(genome(seqinfo(bin.GR))[1])
 dataPeaks=read.table(file_bed,sep="\t",header=F)
 dataPeaks=dataPeaks[dataPeaks[,1]%in%seqnames(SeqInfo),]
 peaks.GR=GRanges(as.character(dataPeaks[,1]),IRanges(dataPeaks[,2],dataPeaks[,3]),seqinfo=SeqInfo)
 label=countOverlaps(bin.GR,peaks.GR,minoverlap=minoverlap)
 label[label>1]=1

 return(label)
}


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


#######################################
# Split the genome in bins for other genomes
splitOtherGenomeBins<-function(Genome, binsize, contextsize){
 SeqInfo=seqinfo(Genome)
 #Chr.V=seqnames(SeqInfo)[-grep("_|chrM|chrY",seqnames(SeqInfo))]
 Chr.V=seqnames(SeqInfo)[seqlengths(SeqInfo)>1e6]
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




#######################################
# Get SNP allele sequences
getAllelesSeq=function(region.GR,window=201,genome=BSgenome.Hsapiens.UCSC.hg19){
 
 region.GRr=resize(region.GR,fix="center",width=window)
 region.seq=as.character(getSeq(genome, names=seqnames(region.GRr), 
 	start=start(region.GRr), end=end(region.GRr)))

 SNPposrel=ceiling((window+1)/2)
 region.seqRef=region.seq
 substring(region.seqRef,SNPposrel,SNPposrel)=as.character(region.GRr$REF)
 region.seqAlt=region.seq
 substring(region.seqAlt,SNPposrel,SNPposrel)=as.character(region.GRr$ALT)

 return(list(as.character(region.seqRef),as.character(region.seqAlt)))
}

#######################################
# Make vcf file from GRanges of SNPs
makeVcfData=function(x.GR){
 x.vcf=data.frame(CHR=seqnames(x.GR),POS=start(x.GR),ID=x.GR$ID,REF=x.GR$REF,ALT=x.GR$ALT,QUAL=30)
 return(x.vcf)	
}

#######################################
# Write vcf file
writeVcfFile=function(x.vcf,file){
 fwrite(x.vcf,file,sep='\t',quote=F,row.names=F)
}

