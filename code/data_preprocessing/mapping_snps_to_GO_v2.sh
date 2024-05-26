#!/bin/bash

#######################################################################################
# Creates classification of SNPs according to the GO term of their corresponding gene #
#######################################################################################

# Input and output directories
data_dir="../../data/"

# Load snps IDs order file
IFS=$'\t' read -r -a snps_vector < "${data_dir}features/subsetting/refIDs_row.txt"

# Load SNP lookup table
lookuptab_file="${data_dir}biomart/revised_filtered_snp_gene_lookup_tab.txt"

# Load core_genes table (with GO classes)
GOtab_file="${data_dir}genes/core_genes_tabseparated.txt"

# Create final table
printf "SNP\tGene\tGO_term\n" > "${data_dir}SNPs/snps_gene_GO_m3.txt"

# Process each SNP
for snp in "${snps_vector[@]}"; do
    
    # Find gene for the SNP
    gene=$(grep -m 1 "$snp" "$lookuptab_file" | cut -f 5 | tr -d '[:space:]')
    
    # Look up GO class for the gene
    GO_term=$(grep -m 1 "$gene" "$GOtab_file" | cut -f 4)
    
    if [[ -z $GO_term ]]; then
        # If GO term not found, mark as extra_genes
        printf "%s\t%s\t%s\n" "${snp}" "${gene}" "extra_genes" >> "${data_dir}SNPs/snps_gene_GO_m3.txt"
    else
        # Otherwise, write to final table
        printf "%s\t%s\t%s\n" "${snp}" "${gene}" "${GO_term}" >> "${data_dir}SNPs/snps_gene_GO_m3.txt"
    fi

done
