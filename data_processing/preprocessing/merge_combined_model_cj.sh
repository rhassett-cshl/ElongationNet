""" Merge combined model c_j with main data file """

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cell_type>"
    exit 1
fi

cell_type=$1

combined_cj_zeta_input_file = "${cell_type}_combined_cj_zeta.csv"
awk_tmp_file = "awk_tmp.csv"
paste_tmp_file = "paste_tmp.csv"
combined_cj_file = "${cell_type}_combined_cj.csv"
main_data_file = "${cell_type}_epAllmer_zeta_norm.csv"


# rename column in cell_combined_cj_zeta to combined_lambda_alphaj

awk -F, -v OFS=, 'NR == 1 {for (i=1; i<=NF; i++) if ($i == "\"lambda_alphaj\"") $i = "combined_lambda_alphaj"} {print}' "$combined_cj_zeta_input_file" > "$awk_tmp_file" && mv "$awk_tmp_file" "$combined_cj_zeta_input_file"

# extract combined_lambda_alphaj column only from cell_combined_cj_zeta file

cut -d, --complement -f2 "$combined_cj_zeta_input_file" > "$combined_cj_file"

# copy combined_lambda_alphaj column to main data file
paste -d, "$main_data_file" "$combined_cj_file" > "$paste_tmp.csv"

mv "$paste_tmp.csv" "$main_data_file"

rm "$combined_cj_file"