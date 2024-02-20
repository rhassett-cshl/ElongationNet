import pandas as pd
import numpy as np
from natsort import natsorted, index_natsorted

# Remove minus strand when positive strand at same position
def filter_strands(group):
    # Mask for rows with Strand == 1
    mask_strand_1 = group['Strand'] == 1
    # Identify unique Start positions for Strand == 1 within the group
    starts_with_strand_1 = group.loc[mask_strand_1, 'Start'].unique()
    # Mask for rows with Strand == 0 and Start not in starts_with_strand_1, within the group
    mask_strand_0_unique_starts = (group['Strand'] == 0) & (~group['Start'].isin(starts_with_strand_1))
    # Combine masks to filter the group
    filtered_group = group[mask_strand_1 | mask_strand_0_unique_starts]
    return filtered_group

def process_bigwig_dataframe(bw_df):
    # check that minus strand has only positive values
    # Filter the DataFrame to include only rows where Strand is 0
    minus_strand_df = bw_df[bw_df["Strand"] == 0]

    # Check if all values in the Value column of this subset are greater than 0
    all_values_positive = (minus_strand_df["Value"] > 0).all()

    if not all_values_positive:
        print("Minus strand contains negative values before being processed")
    # throw error if all_values_positive is false
    

    bw_df = bw_df.groupby('Chr').apply(filter_strands).reset_index(drop=True)

    # negate zeta value for minus strand
    bw_df.loc[bw_df["Strand"] == 0, "Value"] = bw_df.loc[bw_df["Strand"] == 0, "Value"] * -1

    # remove strand column when storing to bigwig file
    del bw_df["Strand"]

    # End value is exclusive, add 1 to all end positions
    bw_df['End'] = bw_df['End'] + 1

    # update Chr column values
    bw_df['Chr'] = 'chr' + bw_df['Chr'].astype(str)

    # index_natsorted returns indices to then sort by multiple columns
    bw_df = bw_df.sort_values(by=['Chr', 'Start'], key=lambda x: np.argsort(index_natsorted(zip(bw_df['Chr'], bw_df['Start']))))
    
    return bw_df