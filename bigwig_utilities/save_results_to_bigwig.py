import pyBigWig
import pandas as pd

def save_to_bigwig(bw_df, config_name, cell_type):
    # setup header
    epAllmer_bw = pyBigWig.open(f"./bigwig_utilities/example_files/{cell_type}_epAllmerPredZeta.bw")
    chrom_lengths = epAllmer_bw.chroms()
    chrom_lengths = list(chrom_lengths.items())

    bw = pyBigWig.open(f"./results/{config_name}/{cell_type}_epAllmer{config_name}Zeta.bw", "w")
    bw.addHeader(chrom_lengths)

    bw.addEntries(bw_df['Chr'].tolist(), bw_df['Start'].tolist(), ends=bw_df['End'].tolist(), values=bw_df['Value'].tolist())

    bw.close()
