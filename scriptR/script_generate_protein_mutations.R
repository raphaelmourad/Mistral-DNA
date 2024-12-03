# Raphael Mourad
# 18/01/2024
# MIAT


# use this R script to generate the DNA sequences to feed Mistral-DNA.

#### SETUP PROJECT FOLDER
setwd("")

#### LOAD LIBRARIES

library(biomaRt)
library(Biostrings)

# Read hg38 proteins
proteins=readAAStringSet("data/genome_sequences/hg38/Homo_sapiens.GRCh38.pep.all.fa.gz")
proteins=proteins[width(proteins)>200 & width(proteins)<500]

# Select 100 random protein sequences
selected_proteins_wt <- sample(proteins,100)
selected_proteins_wt=as.character(selected_proteins_wt)
selected_proteins_wt=as.vector(selected_proteins_wt)

# Function to mutate a leucine to isoleucine at a random position
mutate_protein <- function(protein_seq, mut="leucine->isoleucine") {
  # Find positions of leucine (L)
  if(mut=="leucine->isoleucine"){
   leucine_positions <- which(unlist(strsplit(protein_seq, "")) == "L")
   
   # Proceed only if there are leucines in the sequence
   if (length(leucine_positions) > 0) {
    # Select a random leucine position
    random_position <- sample(leucine_positions, 1)
    
    # Mutate to isoleucine (I)
    protein_chars <- unlist(strsplit(protein_seq, ""))
    protein_chars[random_position] <- "I"
    
    # Return the mutated sequence
    return(paste(protein_chars, collapse = ""))
   } else {
    # If no leucines, return the original sequence
    return(protein_seq)
   }

  }
  if(mut=="arginine->lysine"){
   arginine_positions <- which(unlist(strsplit(protein_seq, "")) == "R")
   
      # Proceed only if there are arginines in the sequence
   if (length(arginine_positions) > 0) {
    # Select a random leucine position
    random_position <- sample(arginine_positions, 1)
    
    # Mutate to isoleucine (I)
    protein_chars <- unlist(strsplit(protein_seq, ""))
    protein_chars[random_position] <- "K"
    
    # Return the mutated sequence
    return(paste(protein_chars, collapse = ""))
   } else {
    # If no leucines, return the original sequence
    return(protein_seq)
   }
  }  
}

selected_proteins_mut_LI <- sapply(selected_proteins_wt, mutate_protein, mut="leucine->isoleucine")
selected_proteins_mut_LI=as.vector(selected_proteins_mut_LI)

selected_proteins_mut_RK <- sapply(selected_proteins_wt, mutate_protein, mut="arginine->lysine")
selected_proteins_mut_RK=as.vector(selected_proteins_mut_RK)



# SAVE REF AND ALT SEQUENCES
writeXStringSet(x=AAStringSet(selected_proteins_wt),filepath=paste0("data/SNP/SNPprot_ref.fasta.gz"),compress=T)
writeXStringSet(x=AAStringSet(selected_proteins_mut_LI),filepath=paste0("data/SNP/SNPprot_I->L.fasta.gz"),compress=T)

writeXStringSet(x=AAStringSet(selected_proteins_mut_RK),filepath=paste0("data/SNP/SNPprot_R->K.fasta.gz"),compress=T)






