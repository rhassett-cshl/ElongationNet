import pyBigWig
import pandas as pd

def save_to_bigwig(bw_df, config_name, cell_type):
    # setup header
    chrom_lengths = []
    with open("./bigwig_utilities/hg38.chrom.sizes", "r") as f:
        for line in f:
            chrom, length = line.strip().split()
            chrom_lengths.append((chrom, int(length)))


    bw = pyBigWig.open(f"./results/{config_name}/{cell_type}_epAllmer{config_name}Zeta.bw", "w")
    bw.addHeader(chrom_lengths)

    bw.addEntries(bw_df['Chr'].tolist(), bw_df['Start'].tolist(), ends=bw_df['End'].tolist(), values=bw_df['Value'].tolist())

    bw.close()
