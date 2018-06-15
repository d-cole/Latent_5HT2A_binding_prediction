setwd("Desktop/Molecular_autoencoder/")

kiDB.raw <- read.csv("Data/KiDatabase.csv")
#Top 10 receptors by count
sort(table(kiDB.raw$Name), decreasing = T)[1:10]

#Subset to 5-HT2A - ligand pairings
HT2A.subset <- kiDB.raw[kiDB.raw$Name == "5-HT2A",]

HT2A.subset <- HT2A.subset[HT2A.subset$SMILES != "", c("Ligand.ID",
                                                       "Ligand.Name",
                                                        "SMILES",
                                                        "Hotligand",
                                                        "species",
                                                        "source",
                                                        "ki.Note",
                                                        "ki.Val")]
head(HT2A.subset)
SMILES <- unique(HT2A.subset[,"SMILES"])

write.csv(SMILES, file='Focal_SMILES.csv')
write.csv(HT2A.subset, file="HT2A_kiDB.csv")

