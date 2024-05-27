# Get gene names for the core proteins table uploaded in /data

# Import packages
import pandas as pd
from Bio import ExPASy
from Bio import SwissProt
import numpy as np

def get_gene_name(uniprot_id):
    '''
    Get gene names for UniProt handles.
    
    INPUT
    uniprot_id: str

    OUTPUT
    str of corresponding gene name
    '''
    try:
        handle = ExPASy.get_sprot_raw(uniprot_id)
        record = SwissProt.read(handle)
        # take only the name, leave the synonyms
        record = record.gene_name[0]['Name']
        record = str(record).split(' ')[0]
        return record
    except ValueError:
        return np.nan  # assign NaN value if no SwissProt record found

# Get the core genes list from the core protein initial list curated and
# uplodaded in /data

concatenated_ann = pd.read_csv("../../../data/concatenated_annotations_core_prots.csv")

# Iterate over each protein name
concatenated_ann['Gene'] = concatenated_ann['Gene(UniProtKB)'].apply(get_gene_name)

# Save updated concatenated annotations into a csv
concatenated_ann.to_csv('../../../data/core_genes.csv', index=False)

# !! 3 genes where manually labelled after the execution of this script, to 
# replace the NaN placed values.