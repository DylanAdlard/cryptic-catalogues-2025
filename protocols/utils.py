import piezo
import json
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.ticker as ticker
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from upsetplot import UpSet
from upsetplot import from_indicators
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.stats import norm


# Function to extract numeric values from MIC strings
def extract_numeric(values):
    nums = []
    for val in values:
        try:
            nums.append(float(val))
        except ValueError:
            try:
                nums.append(float(val[1:]))  # Remove '>' or '<='
            except ValueError:
                nums.append(float(val[2:]))  # Handle double-prefixed values
    return np.array(nums)


def cap_mic_to_float(df, mic_column="METHOD_MIC", plate_column="METHOD_3"):
    """
    Converts MIC values to floats while automatically detecting and collapsing tail values 
    based on the two plate designs in the dataset.

    - Detects two plate designs from `METHOD_3`
    - Identifies upper and lower tail MICs using '>' and '<=' symbols
    - Collapses upper and lower tails to the more restrictive range
    - Caps all non-censored values to the upper and lower bounds

    Parameters:
    df (pd.DataFrame): DataFrame containing MIC values.
    mic_column (str): Column name for MIC values.
    plate_column (str): Column name for plate design.

    Returns:
    pd.Series: Converted MIC values with collapsed tails and capped values.
    """

    # Ensure there are exactly two plate designs
    unique_plates = df[plate_column].unique()
    if len(unique_plates) != 2:
        raise ValueError(f"Expected exactly 2 plate designs, found {len(unique_plates)}: {unique_plates}")

    # Split data into the two plates
    plate_1, plate_2 = unique_plates
    df1 = df[df[plate_column] == plate_1]
    df2 = df[df[plate_column] == plate_2]


    # Identify numeric MIC values (ignoring '>' and '<=')
    ref1_numeric = extract_numeric([x for x in df1[mic_column] if not (">" in x or "<=" in x)])
    ref2_numeric = extract_numeric([x for x in df2[mic_column] if not (">" in x or "<=" in x)])

    # Identify upper and lower tail values
    upper_tail_1 = max([float(x[1:]) for x in df1[mic_column] if x.startswith(">")], default=None)
    upper_tail_2 = max([float(x[1:]) for x in df2[mic_column] if x.startswith(">")], default=None)
    lower_tail_1 = min([float(x[2:]) for x in df1[mic_column] if x.startswith("<=")], default=None)
    lower_tail_2 = min([float(x[2:]) for x in df2[mic_column] if x.startswith("<=")], default=None)

    # Determine the more restrictive upper and lower bounds
    final_upper_tail = min(upper_tail_1, upper_tail_2) if upper_tail_1 and upper_tail_2 else upper_tail_1 or upper_tail_2
    final_lower_tail = max(lower_tail_1, lower_tail_2) if lower_tail_1 and lower_tail_2 else lower_tail_1 or lower_tail_2

    # Process MIC values
    def adjust_mic(mic):
        if mic.startswith(">"):  # Upper boundary
            num = float(mic[1:])
            if final_upper_tail and num > final_upper_tail:
                return final_upper_tail  # Collapse upper tail
            return num
        elif mic.startswith("<="):  # Lower boundary
            num = float(mic[2:])
            if final_lower_tail and num < final_lower_tail:
                return final_lower_tail  # Collapse lower tail
            return num
        else:  # Convert normal numeric values
            try:
                num = float(mic)
                # Cap values that exceed the upper/lower bounds
                if final_upper_tail and num > final_upper_tail:
                    return final_upper_tail  # Cap to upper bound
                if final_lower_tail and num < final_lower_tail:
                    return final_lower_tail  # Cap to lower bound
                return num
            except ValueError:
                return mic  # Keep as is if conversion fails

    return df[mic_column].apply(adjust_mic)


def filter_multiple_phenos(group):
    """
    If a sample contains more than one phenotype,
    keep the highest priority phenotype in order: R > S > U.
    Prefer rows with MIC values if available.

    Parameters:
    group (pd.DataFrame): A dataframe containing sample data with phenotypes.

    Returns:
    pd.DataFrame: A filtered dataframe prioritizing resistant phenotypes.
    """
    if len(group) == 1:
        return group
    
    # Define phenotype priority order
    priority_order = {"R": 1, "S": 2, "U": 3}
    
    # Sort by phenotype priority (lower is better)
    group = group.sort_values(by='PHENOTYPE', key=lambda x: x.map(priority_order))
    
    # Keep only rows of the highest priority phenotype
    highest_priority = group.iloc[0]['PHENOTYPE']
    filtered_group = group[group['PHENOTYPE'] == highest_priority]
    
    # Check for rows with METHOD_MIC values
    with_mic = filtered_group.dropna(subset=['METHOD_MIC'])
    
    return with_mic.iloc[0:1] if not with_mic.empty else filtered_group.iloc[0:1]

#drop duplicate  entries
def filter_multiple_phenos_all_drugs(group):
    """
    If a (UNIQUEID, DRUG) contains more than one phenotype,
    keep the resistant phenotype (preferably with MIC) if there is one.

    Parameters:
    group (pd.DataFrame): A dataframe containing sample data with phenotypes.

    Returns:
    pd.DataFrame: A filtered dataframe prioritizing resistant phenotypes.
    """
    if len(group) == 1:
        return group

    # Prioritize rows with 'R' phenotype
    prioritized_group = (
        group[group["PHENOTYPE"] == "R"] if "R" in group["PHENOTYPE"].values else group
    )

    # Check for rows with METHOD_MIC values
    with_mic = prioritized_group.dropna(subset=["METHOD_MIC"])
    return with_mic.iloc[0:1] if not with_mic.empty else prioritized_group.iloc[0:1]

def piezo_predict(iso_df, catalogue_file, drug, U_to_R=False, U_to_S=False, Print=True):
    """
    Predicts drug resistance based on genetic mutations using a resistance catalogue.

    Parameters:
    iso_df (pd.DataFrame): DataFrame containing isolate data with UNIQUEID, PHENOTYPE, and GENE_MUT columns.
    catalogue_file (str): Path to the resistance catalogue file.
    drug (str): The drug for which resistance predictions are to be made.
    U_to_R (bool, optional): If True, treat 'U' predictions as 'R'. Defaults to False.
    U_to_S (bool, optional): If True, treat 'U' predictions as 'S'. Defaults to False.
    Print (bool, optional): If True, prints the confusion matrix, coverage, sensitivity, and specificity. Defaults to True.

    Returns:
    list: Confusion matrix, isolate coverage, sensitivity, specificity, and false negative IDs.
    """
    # Load and parse the catalogue with piezo
    catalogue = piezo.ResistanceCatalogue(catalogue_file)

    # Ensure the UNIQUEID and PHENOTYPE columns are used correctly
    ids = iso_df['UNIQUEID'].unique().tolist()
    labels = iso_df.groupby('UNIQUEID')['PHENOTYPE'].first().reindex(ids).tolist()
    predictions = []

    for id_ in ids:
        # For each sample
        df = iso_df[iso_df['UNIQUEID'] == id_]
        # Predict phenotypes for each mutation via lookup
        mut_predictions = []
        for var in df['MUTATION']:
            if pd.isna(var):
                predict = 'S'
            else:
                try:
                    predict = catalogue.predict(var)
                except ValueError:
                    predict = "U"
            if isinstance(predict, dict):
                if drug in predict.keys():
                    mut_predictions.append(predict[drug])
            else:
                mut_predictions.append(predict)

        # Make sample-level prediction from mutation-level predictions. R > U > S
        if "R" in mut_predictions:
            predictions.append("R")
        elif "U" in mut_predictions:
            if U_to_R:
                predictions.append("R")
            elif U_to_S:
                predictions.append("S")
            else:
                predictions.append("U")
        else:
            predictions.append("S")

    # Log false negative samples
    FN_id = [
        id_
        for id_, label, pred in zip(ids, labels, predictions)
        if pred == "S" and label == "R"
    ]

    FP_id = [
        id_
        for id_, label, pred in zip(ids, labels, predictions)
        if pred == "R" and label == "S"
    ]

    # Generate confusion matrix for performance analysis
    cm = confusion_matrix(labels, predictions, classes=["R", "S", "U"])

    if "U" not in predictions:
        cm = cm[:2, :2]
    else:
        cm = cm[:2, :]

    if Print:
        print(cm)
    
    # Calculate performance metrics
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    isolate_cov = (len(labels) - predictions.count("U")) / len(labels)

    if Print:
        print("Catalogue coverage of isolates:", isolate_cov)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

    return [cm, isolate_cov, sensitivity, specificity, FN_id, FP_id]

def confusion_matrix(labels, predictions, classes):
    """
    Creates a confusion matrix for given labels and predictions with specified classes.

    Parameters:
    labels (list): Actual labels.
    predictions (list): Predicted labels.
    classes (list): List of all classes.

    Returns:
    np.ndarray: Confusion matrix.
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for label, prediction in zip(labels, predictions):
        if label in class_to_index and prediction in class_to_index:
            cm[class_to_index[label], class_to_index[prediction]] += 1

    return cm

def plot_truthtables(truth_table, U_to_S=False, fontsize=10, colors=None, save=None):
    """
    Plots a truth table as a confusion matrix to denote each cell with perfect squares or proportional rectangles.

    Parameters:
    truth_table (pd.DataFrame): DataFrame containing the truth table values.
                                The DataFrame should have the following structure:
                                - Rows: True labels ("R" and "S")
                                - Columns: Predicted labels ("R", "S", and optionally "U")
    U_to_S (bool): Whether to separate the "U" values from the "S" column. If True,
                   an additional column for "U" values will be used.
    fontsize (int): Font size for the text in the plot.
    colors (list): List of four colors for the squares.
                   Defaults to red and green for the diagonal, pink and green for the off-diagonal.

    Returns:
    None
    """

    # Default colors if none provided
    if colors is None:
        if U_to_S:
            colors = ["#e41a1c", "#4daf4a", "#fc9272", "#4daf4a"]
        else:
            colors = ["#e41a1c", "#4daf4a", "#fc9272", "#4daf4a", "#4daf4a", "#4daf4a"]

    # Determine the number of columns for U_to_S condition
    num_columns = 3 if not U_to_S else 2
    num_rows = 2

    # Adjust the figure size to ensure square cells
    figsize = (
        (num_columns / 1.8, num_rows / 1.8)
        if num_columns == 2
        else (num_columns * 1.5 / 1.8, num_rows / 1.8)
    )

    fig = plt.figure(figsize=figsize)
    axes = plt.gca()

    if not U_to_S:
        assert (
            len(colors) == 6
        ), "The length of supplied colors must be 6, one for each cell"
        axes.add_patch(Rectangle((2, 0), 1, 1, fc=colors[4], alpha=0.5))
        axes.add_patch(Rectangle((2, 1), 1, 1, fc=colors[5], alpha=0.5))

        axes.set_xlim([0, 3])
        axes.set_xticks([0.5, 1.5, 2.5])
        axes.set_xticklabels(["S", "R", "U"], fontsize=9)
    else:
        assert (
            len(colors) == 4
        ), "The length of supplied colors must be 4, one for each cell"
        axes.set_xlim([0, 2])
        axes.set_xticks([0.5, 1.5])
        axes.set_xticklabels(["S+U", "R"], fontsize=9)

    # Apply provided colors for the squares
    axes.add_patch(Rectangle((0, 0), 1, 1, fc=colors[0], alpha=0.8))
    axes.add_patch(Rectangle((1, 0), 1, 1, fc=colors[1], alpha=0.8))
    axes.add_patch(Rectangle((1, 1), 1, 1, fc=colors[2], alpha=0.8))
    axes.add_patch(Rectangle((0, 1), 1, 1, fc=colors[3], alpha=0.8))

    axes.set_ylim([0, 2])
    axes.set_yticks([0.5, 1.5])
    axes.set_yticklabels(["R", "S"], fontsize=9)

    # Add text to the plot
    axes.text(
        1.5,
        0.5,
        int(truth_table["R"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        1.5,
        1.5,
        int(truth_table["R"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        1.5,
        int(truth_table["S"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        0.5,
        int(truth_table["S"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    if not U_to_S:
        axes.text(
            2.5,
            0.5,
            int(truth_table["U"]["R"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        axes.text(
            2.5,
            1.5,
            int(truth_table["U"]["S"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    axes.set_aspect("equal")  # Ensure squares remain squares

    if save != None:
        plt.savefig(save, format="pdf", bbox_inches="tight")

    plt.show()

def str_to_dict(val):
    if isinstance(val, str):  # Only convert if it's a valid string
        return json.loads(val)
    return val  # Keep as is (e.g., NaN values)
    

def expand_catalogue_pair(cat1, cat2, drugs, model, cat_names):
    """This function takes 2 catalogues with rules in, and expands them to include any rows in the other
    catalogue that fall under that rule - this allows one to compare the effective contents of each catalogues,
    not just specific rows."""

    cat1_no_rules = cat1[~cat1['MUTATION'].str.contains(r'[*?=]', regex=True)]
    cat1_rules_only = cat1[cat1['MUTATION'].str.contains(r'[*?=]', regex=True)]
    cat2_no_rules = cat2[~cat2['MUTATION'].str.contains(r'[*?=]', regex=True)]
    cat2_rules_only = cat2[cat2['MUTATION'].str.contains(r'[*?=]', regex=True)]

    expanded_catalogues = {}

    row = {
        'GENBANK_REFERENCE': 'NC00962.3',
        'CATALOGUE_NAME': '-',
        'CATALOGUE_VERSION': 0,
        'CATALOGUE_GRAMMAR': 'GARC1',
        'PREDICTION_VALUES': model,
        'DRUG': None,
        'MUTATION': None,
        'PREDICTION': None,
        'SOURCE': {},
        'EVIDENCE': {},
        'OTHER': {}
    }

    for drug in drugs:

        #catalogues filtered by drug
        cat1_drug = cat1[cat1.DRUG==drug]
        cat2_drug = cat2[cat2.DRUG==drug]
        #catalogues with rules removed, filtered for drug
        cat1_no_rules_drug = cat1_no_rules[cat1_no_rules.DRUG==drug]
        cat2_no_rules_drug = cat2_no_rules[cat2_no_rules.DRUG==drug]
        #catalogue expert rules (not defaults) filtered for drug
        cat2_rules_drug = cat2_rules_only[cat2_rules_only.DRUG==drug]
        cat1_rules_drug = cat1_rules_only[cat1_rules_only.DRUG==drug]
        #add placeholder rules to rule catalogues so avoid piezo error
        for i in model:
            if i != 'U':
                row['PREDICTION'] = i
                row['MUTATION'] = 'placeholder@A1A'
                row['DRUG'] = drug
                cat2_rules_drug = pd.concat([cat2_rules_drug, pd.DataFrame([row])], ignore_index=True)
                cat1_rules_drug = pd.concat([cat1_rules_drug, pd.DataFrame([row])], ignore_index=True)
    
        genes = set(cat2_drug['MUTATION'].apply(lambda x: x.split('@')[0]).tolist() + cat1_drug['MUTATION'].apply(lambda x: x.split('@')[0]).tolist())
        #add a default wildcard U rule to rule catalogues so that a U is thrown if a mutation is not shared
        for gene in genes:
            for mut in [f'{gene}@*?', f'{gene}@-*?']:
                row['PREDICTION'] = 'U'
                row['MUTATION'] = mut
                row['DRUG'] = drug
                cat2_rules_drug = pd.concat([cat2_rules_drug, pd.DataFrame([row])], ignore_index=True)
                cat1_rules_drug = pd.concat([cat1_rules_drug, pd.DataFrame([row])], ignore_index=True)

        #write out rule catalogues so piezo can read them in and scan the other non-rule catalogue
        cat1_rules_drug['EVIDENCE'] = cat1_rules_drug['EVIDENCE'].to_json()
        cat1_rules_drug.to_csv(f'./catalogues/temp/cat1_rules_only.csv')
        cat1_rules_piezo = piezo.ResistanceCatalogue(f'./catalogues/temp/cat1_rules_only.csv')

        cat2_rules_drug['EVIDENCE'] = cat2_rules_drug['EVIDENCE'].to_json()
        cat2_rules_drug.to_csv(f'./catalogues/temp/cat2_rules_only.csv')
        cat2_rules_piezo = piezo.ResistanceCatalogue(f'./catalogues/temp/cat2_rules_only.csv')

        #use cat2 catalogue to scan cat1 non-rule catalogue to find variants that fall under that rule
        vars = []
        for var in cat1_no_rules_drug.MUTATION:
            try:
                prediction = cat2_rules_piezo.predict(var)
                if prediction[drug] in ['R', 'S']:
                    vars.append((var, prediction[drug]))
            except ValueError:
                continue
        #add variants from cat1 catalogue that fall under cat2 rules to cat2 non-rule catalogue
        for m, p in vars:
            row['MUTATION'] = m
            row['PREDICTION'] = p
            row['EVIDENCE'] = {'expanded_rule'}
            cat2_no_rules_drug = pd.concat([cat2_no_rules_drug, pd.DataFrame([row])], ignore_index=True)
        #use cat1rule catalogue to scan cat2 non-rule catalogue to find variants that fall under that rule
        vars = []
        for var in cat2_no_rules_drug.MUTATION:
            try:
                prediction = cat1_rules_piezo.predict(var)
                if prediction[drug] in ['R', 'S']:
                    vars.append((var, prediction[drug]))
            except ValueError:
                continue
        #add variants from cat2 catalogue that fall under cat1 rules to cat1 non-rule catalogue
        for m, p in vars:
            row['MUTATION'] = m
            row['PREDICTION'] = p
            row['EVIDENCE'] = {'expanded_rule'}
            cat1_no_rules_drug = pd.concat([cat1_no_rules_drug, pd.DataFrame([row])], ignore_index=True)

        expanded_catalogues[drug] = {
            cat_names[0]: cat1_no_rules_drug.drop_duplicates('MUTATION'), 
            cat_names[1]:cat2_no_rules_drug.drop_duplicates('MUTATION'), 
            'merged':pd.merge(cat1_no_rules_drug, cat2_no_rules_drug, on='MUTATION', how='outer', suffixes=(f'_{cat_names[0]}', f'_{cat_names[1]}'))
            }
        
    return expanded_catalogues

def back2back_sens_spec(data, palette, savefig=None):
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 7), gridspec_kw={'width_ratios': [1, 1]})

        # Sensitivity Plot (Left Side)
        sns.barplot(
            data=data,
            y="DRUG", x="SENSITIVITY", hue="catalogue",
            dodge=True, ax=ax1, palette=palette
        )
        ax1.set_xlabel("Sensitivity")
        ax1.invert_xaxis()  # Flip the x-axis for back-to-back effect
        ax1.legend_.remove()  # Remove duplicate legend
        ax1.set_ylabel("")  # Remove the left-side y-axis label
        ax1.set_yticklabels([])  # Remove y-tick labels
        ax1.spines['top'].set_visible(False)
        ax1.set_yticks([])
        ax1.spines['left'].set_visible(False)

        # Specificity Plot (Right Side)
        sns.barplot(
            data=data,
            y="DRUG", x="SPECIFICITY", hue="catalogue",
            dodge=True, ax=ax2, palette=palette
        )
        ax2.legend().remove()
        ax2.set_xlabel("Specificity")
        ax2.set_ylabel("")  # Remove the right-side y-axis label
        ax2.set_yticklabels([])  # Remove y-tick labels
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Add the drug names in the middle between the plots
        middle_positions = range(1, len(data["DRUG"].unique())+1)
        for y_pos, label in zip(middle_positions, data["DRUG"].unique()):
            fig.text(
                0.5,  # x-coordinate: centered between the plots
                (y_pos+1.05) / (len(middle_positions)+2.25),  # Normalize y-coordinate
                label,  # Text label
                ha='center', va='center', fontsize=12, rotation=0
            )

        for drug in data["DRUG"].unique():
            drug_data = data[data["DRUG"] == drug]
            y_position = middle_positions[list(data["DRUG"].unique()).index(drug)]-1

            ax1.scatter(
                drug_data["COVERAGE"],
                [y_position - 0.3 if cat == "WHOv1" else 
                y_position - 0.1 if cat == "WHOv2" else 
                y_position + 0.1 if cat == "catomatic_v1" else 
                y_position + 0.3 for cat in drug_data["catalogue"]],
                s=40, c="black", edgecolors="white", label=None  # Adjust marker size (s) and color (c) as needed
            )
            ax2.scatter(
                drug_data["COVERAGE"],
                [y_position - 0.3 if cat == "WHOv1" else 
                y_position - 0.1 if cat == "WHOv2" else 
                y_position + 0.1 if cat == "catomatic_v1" else 
                y_position + 0.3 for cat in drug_data["catalogue"]],
                s=40, c="black" ,edgecolors='white', label=None  # Adjust marker size (s) and color (c) as needed
            )

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1)  
        handles, labels = ax1.get_legend_handles_labels()
        handles, labels = ax1.get_legend_handles_labels()

        # Create a custom handle for the 'coverage' scatter points
        coverage_handle = Line2D([], [], marker='o', linestyle='None',
                                markersize=7, markerfacecolor='black',
                                markeredgecolor='white', label='coverage')

        # Append the custom handle and its label
        handles.append(coverage_handle)
        labels.append('coverage')
        fig.legend(
        handles, labels, loc="lower center", 
        ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04)
)
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

def classify_predictions(row, suffixes=('cat1', 'cat2')):

    if not pd.isna(row[f'PREDICTION_{suffixes[1]}']) and not pd.isna(row[f'PREDICTION_{suffixes[0]}']):
        return f"{row[f'PREDICTION_{suffixes[0]}']}.{row[f'PREDICTION_{suffixes[1]}']}"
    elif pd.isna(row[f'PREDICTION_{suffixes[1]}']) and not pd.isna(row[f'PREDICTION_{suffixes[0]}']):
        return f"{row[f'PREDICTION_{suffixes[0]}']}.U"
    elif not pd.isna(row[f'PREDICTION_{suffixes[1]}']) and pd.isna(row[f'PREDICTION_{suffixes[0]}']):
        return f"U.{row[f'PREDICTION_{suffixes[1]}']}"
    else:
        return "U.U"  # Default case if both are NaN
    

def sum_solo_counts(df, suffixes=['cat', 'who']):
    df = df.copy()
    df['x'] = df[f'solo_R_{suffixes[0]}'] + df[f'solo_S_{suffixes[0]}']
    df['y'] = df[f'solo_R_{suffixes[1]}'] + df[f'solo_S_{suffixes[1]}']
    return df

def read_data(file_path):
    """Reads .pkl, .pkl.gz, .csv, .csv.gz, or .parquet files automatically."""
    file_path = Path(file_path)
    ext = ''.join(file_path.suffixes).lower()  # Get full extension, e.g., ".pkl.gz"

    read_funcs = {
        ".pkl": pd.read_pickle,
        ".pkl.gz": pd.read_pickle,  # Supports compressed pickle
        ".csv": pd.read_csv,
        ".csv.gz": pd.read_csv,  # Supports compressed CSV
        ".parquet": pd.read_parquet
    }

    if ext in read_funcs:
        return read_funcs[ext](file_path)
    
    raise ValueError(f"Unsupported file type: {ext}")

def flatten_grid_results(grid):
    '''Flattens parameter search grid results into a DataFrame'''
    return pd.DataFrame(
        [
            {
                "DRUG": drug,
                "BACKGROUND_RATE": background_rate,
                "p_value": p_value,
                "SENSITIVITY": metrics.get("sens"),
                "SPECIFICITY": metrics.get("spec"),
                "COVERAGE": metrics.get("cov"),
            }
            for (drug, background_rate, p_value), metrics in grid.items()
        ]
    )


def plot_grid_results(df, height=4):
    for drug in df["DRUG"].unique():
        drug_data = df[df["DRUG"] == drug]
        g = sns.FacetGrid(drug_data, col="Metric", col_wrap=3, sharey=False, height=height)
        g.map_dataframe(
            sns.lineplot, x="BACKGROUND_RATE", y="Value", hue="p_value", marker="o"
        )
        g.set_axis_labels("Background Rate", "")
        g.add_legend(title="P Value")
        g.figure.suptitle(drug, y=1.05)
        g.tight_layout()
        plt.show()

def weighted_score(df, weights=(0.5, 0.3, 0.2)):
    w1, w2, w3 = weights
    df['Score'] = (
        w1 * df['SENSITIVITY'] +
        w2 * df['SPECIFICITY'] +
        w3 * df['COVERAGE']
    )
    return df.sort_values(by="Score", ascending=False)

def plot_FRS_vs_perf(df):
    for drug in df["DRUG"].unique():
        drug_data = df[df["DRUG"] == drug]
        g = sns.FacetGrid(drug_data, col="Metric", col_wrap=3, sharey=False, height=2.5)
        g.map_dataframe(sns.lineplot, x="THRESHOLD", y="Value", marker=".", color='black')
        g.set_axis_labels("FRS (Threshold)", "")
        g.figure.suptitle(drug, y=1.05)
        g.tight_layout()
        g.set(ylim=(0, 1))
        plt.show()


def plot_grid_counts(df, valid_drugs, prediction_colors, savefig=None):
    n_cols = 4
    n_rows = int(np.ceil(len(valid_drugs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))

    axes = axes.flatten()
    all_handles = {}

    # Compute global min and max values for log-scale axes
    all_x_values = df["x"].values
    all_y_values = df["y"].values

    # Remove non-positive values for log scale
    all_x_values = all_x_values[all_x_values > 0]
    all_y_values = all_y_values[all_y_values > 0]

    if all_x_values.size > 0 and all_y_values.size > 0:
        global_min = min(np.nanmin(all_x_values), np.nanmin(all_y_values)) * 0.9
        global_max = max(np.nanmax(all_x_values), np.nanmax(all_y_values)) * 1.1

        # Ensure min is positive for log scale
        global_min = max(global_min, 1e-5)

        # Avoid identical limits issue
        if global_min == global_max:
            global_min /= 1.5
            global_max *= 1.5
    else:
        global_min, global_max = 1e-5, 10  # Default range if no valid data

    idx = 0
    for drug in valid_drugs:
        ax = axes[idx]
        scatter = sns.scatterplot(
            data=df[df.DRUG == drug], 
            x="x", y="y", 
            hue="PREDICTION_PAIR", palette=prediction_colors, 
            alpha=0.6, ax=ax
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        ax.set_title(drug)
        ax.set_xlabel("log(training isolate count)")
        ax.set_ylabel("log(WHO isolate count)")
        sns.despine(ax=ax)

        # Collect legend handles while avoiding duplicates
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            all_handles[l] = h  

        if ax.get_legend():
            ax.get_legend().remove()  
        idx += 1

    # Hide any unused subplots
    for j in range(idx, len(axes)):
        axes[j].axis('off')

    # Create a shared legend at the bottom
    if all_handles:
        fig.legend(all_handles.values(), all_handles.keys(), 
                title="Prediction Pairs", loc='lower center', 
                bbox_to_anchor=(0.5, -0.1), ncol=len(all_handles), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit legend
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def extract_value(value, *keys):
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, IndexError, TypeError):
        return None
    
def extract_unclassified_rows(cat):
    u_cat = cat[~cat.PREDICTION.isin(['R', 'S'])].copy()
    for field in ['PROPORTION', 'CONFIDENCE']:
        u_cat[field] = u_cat['EVIDENCE'].apply(lambda x: extract_value(x, field.lower()))
    return u_cat

    
def expand_and_classify_cats(cat1, cat2, drug):
    '''Expands the rules of 2 catalogues wrt to oneanother. Then classifies prediction pairs.
    Returns the merged, expanded and cleaned catalogue, as well as just the merged '''

    u_cat1, u_cat2 = extract_unclassified_rows(cat1), extract_unclassified_rows(cat2)
    cat1 = cat1[cat1.PREDICTION.isin(['R', 'S'])]
    cat2 = cat2[cat2.PREDICTION.isin(['R', 'S'])]


    expanded_catalogues = expand_catalogue_pair(cat1, cat2, [drug], 'RUS', ('cat1', 'cat2'))
    merged = expanded_catalogues[drug]['merged']
    
    for field in ['PROPORTION', 'CONFIDENCE']:
        for cat in ['cat1', 'cat2']:
            merged[f'{field}_{cat}'] = merged[f'EVIDENCE_{cat}'].apply(lambda x: extract_value(x, field.lower()))
    
    merged['PREDICTION_PAIR'] = merged.apply(classify_predictions, axis=1)
    merged['DRUG'] = merged['DRUG_cat1'].combine_first(merged['DRUG_cat2'])
    merged_as_is = merged

    df = update_missing_proportions(merged, [u_cat1, u_cat2], suffixes=('cat1', 'cat2'))

    return df, merged_as_is


def update_missing_proportions(merged_cats, u_cats, suffixes=('cat1', 'cat2')):
    """
    Updates missing proportions in the merged catalogue using provided suffixes.

    Parameters:
    merged_cats (pd.DataFrame): The expanded, merged catalogue.
    u_cats (list of pd.DataFrame): List of dataframes containing missing values.
    suffixes (list of str): List of suffixes corresponding to each u_cat.

    Returns:
    pd.DataFrame: Updated merged_cats dataframe with missing proportions filled.
    """
    for i in merged_cats.index:
        mutation = merged_cats.at[i, 'MUTATION']
        for suffix, u_cat in zip(suffixes, u_cats):
            prop_col = f'PROPORTION_{suffix}'
            conf_col = f'CONFIDENCE_{suffix}'

            if pd.isna(merged_cats.at[i, prop_col]) and mutation in u_cat['MUTATION'].tolist():
                merged_cats.at[i, prop_col] = u_cat.loc[u_cat['MUTATION'] == mutation, 'PROPORTION'].values[0]
                merged_cats.at[i, conf_col] = u_cat.loc[u_cat['MUTATION'] == mutation, 'CONFIDENCE'].values[0]

    # Drop rows where any of the proportion or confidence columns are still NaN
    drop_columns = [f'PROPORTION_{suffix}' for suffix in suffixes] + [f'CONFIDENCE_{suffix}' for suffix in suffixes]
    return merged_cats.dropna(subset=drop_columns)


def abs_err_to_rel(values, errors):
    """Convert absolute errors to relative errors, ensuring valid shapes and bounds."""
    errors = [np.ravel(e) for e in errors]  # Ensures everything is 1D
    # Stack into a 2D array
    errors = np.array(errors).T   
    if errors.shape[0] != 2:
        errors = errors.reshape(2, -1)
    return np.vstack([
        np.maximum(0, values - errors[0, :]),  # Lower error
        np.minimum(1, errors[1, :] - values)   # Upper error
    ]) if values.size > 0 and errors.size > 0 else np.array([])


def load_catomatic_catalogue(drug, background, p, frs, dir):
    '''loads catomatic atalogue, coverts evidence to dict, and removes default rows'''
    cat = pd.read_csv(f"{dir}{drug.lower()}/bg_{background}_p_{p}_FRS_{frs}.csv", index_col=0)
    cat['CATALOGUE_VERSION'], cat['CATALOGUE_NAME'] = 0, '-'
    cat['EVIDENCE'] = cat['EVIDENCE'].apply(str_to_dict)
    return cat[~cat['EVIDENCE'].apply(lambda x: isinstance(x, dict) and 'default_rule' in x)]


def plot_cat_comp_proportions(twoD_data, oneD_data, ax_labels={'x':'Catalogue 1', 'y':'Catalogue 2'}, legend='prediction_pair', max_err=1, category_colors={}, figpath=None):

    if len(category_colors)==0:
        # Assign colors
        unique_categories = np.unique(np.concatenate([data["categories"] for d in (twoD_data, oneD_data) for data in d.values() if "categories" in data]))
        if legend == 'prediction_pair':
            cmap = cm.get_cmap('tab10', len(unique_categories))
        else:
            cmap = cm.get_cmap('tab20', len(unique_categories))
        category_colors = {category: mcolors.to_hex(cmap(i)) for i, category in enumerate(unique_categories)}

    # Jitter parameters
    buffer = 0.04
    jitter_strength = 0.09

    # Keep track of categories that were actually plotted
    plotted_categories = set()

    def plot_errorbar(x, y, xerr, yerr, color, category, alpha=0.7, max_err=1):
        """Helper function to plot an error bar."""

        # Ensure xerr and yerr are absolute values
        xerr = np.abs(xerr) if xerr is not None else None
        yerr = np.abs(yerr) if yerr is not None else None

        if (xerr is not None and np.any(xerr >= max_err)) or (yerr is not None and np.any(yerr >= max_err)):
            return  # Skip plotting

        # If the point is plotted, store its category
        plotted_categories.add(category)

        # Reshape xerr and yerr correctly for plt.errorbar
        xerr = xerr[:, None] if xerr is not None else None
        yerr = yerr[:, None] if yerr is not None else None

        plt.errorbar(
            x, y, xerr=xerr, yerr=yerr, 
            fmt='o', color=color, ecolor=color, alpha=alpha, capsize=2, markersize=2.5, linewidth=0.65
        )

    for drug, data in twoD_data.items():
        plt.figure(figsize=(2.5, 2.5))
        ax = plt.gca()

        x_err = abs_err_to_rel(data["x"], data["xerr"])
        y_err = abs_err_to_rel(data["y"], data["yerr"])

        # Plot main points
        for i in range(len(data["x"])):
            plot_errorbar(
                data["x"][i], data["y"][i], x_err[:, i], y_err[:, i], 
                category_colors[data["categories"][i]], data["categories"][i],
                max_err=max_err
            )

        # Extract missing x/y values for jittered scatter plots
        oneD_data[drug]["x"] = np.array(oneD_data[drug]["x"], dtype=float)
        oneD_data[drug]["y"] = np.array(oneD_data[drug]["y"], dtype=float)

        missing_x = np.isnan(oneD_data[drug]["x"]) & ~np.isnan(oneD_data[drug]["y"])
        missing_y = np.isnan(oneD_data[drug]["y"]) & ~np.isnan(oneD_data[drug]["x"])

        missing_x_vals = oneD_data[drug]["y"][missing_x]
        missing_y_vals = oneD_data[drug]["x"][missing_y]
        missing_x_err = oneD_data[drug]["yerr"][missing_x] if len(oneD_data[drug]['x']) > 1 else oneD_data[drug]["yerr"]
        missing_y_err = oneD_data[drug]["xerr"][missing_y] if len(oneD_data[drug]['x']) > 1 else oneD_data[drug]["xerr"]

        missing_x_err = abs_err_to_rel(missing_x_vals, missing_x_err)
        missing_y_err = abs_err_to_rel(missing_y_vals, missing_y_err)

        missing_x_cats = oneD_data[drug]["categories"][missing_x]  
        missing_y_cats = oneD_data[drug]["categories"][missing_y]

        # Plot missing X values (jittered along y-axis)
        if missing_x_vals.size > 0 and missing_x_err.size > 0:
            for i in range(len(missing_x_vals)):
                plot_errorbar(
                    -buffer - np.random.uniform(0, jitter_strength), missing_x_vals[i], 
                    None, missing_x_err[:, i], category_colors[missing_x_cats[i]], missing_x_cats[i],
                    max_err=max_err
                )

        # Plot missing Y values (jittered below x-axis)
        if missing_y_vals.size > 0 and missing_y_err.size > 0:
            for i in range(len(missing_y_vals)):
                plot_errorbar(
                    missing_y_vals[i], -buffer - np.random.uniform(0, jitter_strength), 
                    missing_y_err[:, i], None, category_colors[missing_y_cats[i]], missing_y_cats[i],
                    max_err=max_err
                )
        
        plt.xlim(-buffer * 3.2, 1.05)
        plt.ylim(-buffer * 3.2, 1.05)
        plt.xlabel(f"Proportion R in {ax_labels['x']}", fontsize=6)
        plt.ylabel(f"Proportion R in {ax_labels['y']}", fontsize=6)

        # Draw background lines
        if 'background_1' in data:
            plt.axvline(data['background_1'], color='gray', linestyle='--', linewidth=0.5)
        if 'background_2' in data:
            plt.axhline(data['background_2'], color='gray', linestyle='--', linewidth=0.5)

        # Adjust axis appearance
        ax.spines['left'].set_bounds(0, 1.05)
        ax.spines['bottom'].set_bounds(0, 1.05)
        ax.axvline(0, ymax=1, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(0, xmax=1, color='black', linestyle='-', linewidth=0.5)
        plt.title(f'{drug}', fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # **Update the legend only with plotted categories**
        filtered_categories = sorted(plotted_categories)  # Ensure consistent order
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=category_colors[cat], markersize=5) 
                   for cat in filtered_categories]
        
        if handles:
            legend = plt.legend(
                handles, filtered_categories, 
                loc='upper center',  # Places legend at the bottom
                bbox_to_anchor=(0.5, -0.2),  # Adjusts position below the plot
                fontsize=6, frameon=False,  # Small font, no frame
                ncol=min(len(filtered_categories), 5),  # Wrap after 5 columns (adjust as needed)
                handletextpad=0.4, columnspacing=1.0  # Adjust spacing for readability
    )


        # Shade non-plottable areas
        ax.fill_betweenx([0, 1.05], -buffer * 3.2, 0, color='gray', alpha=0.18)
        ax.fill_between([0, 1.05], -buffer * 3.2, 0, color='gray', alpha=0.18)

        ax.text(-buffer * 3.5 / 2, -buffer * 3.2 / 2, "None", ha='center', va='center', fontsize=6, color='black')
        if figpath is not None:
            plt.savefig(f'{figpath}{drug}_cat_comp.pdf')
        plt.show()


def plot_perf_heatmaps(performance_df):
    '''Iterates through a single performance dataframe and plots one drug at a time.'''
    
    # Define custom colormaps
    red_gray_cmap = mcolors.LinearSegmentedColormap.from_list("red_gray", ["#D3D3D3", "#8B0000"])  # Dark red → Light gray
    blue_gray_cmap = mcolors.LinearSegmentedColormap.from_list("blue_gray", ["#D3D3D3", "#4682B4"])  # Dark blue → Light gray
    green_gray_cmap = mcolors.LinearSegmentedColormap.from_list("green_gray", ["#D3D3D3", "#2E8B57"])  # Dark green → Light gray
    
    metrics = ["Sensitivity", "Specificity", "Coverage"]
    colormaps = [red_gray_cmap, blue_gray_cmap, green_gray_cmap]
    
    for drug in performance_df['Drug'].unique():
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))
        
        for i, (metric, cmap) in enumerate(zip(metrics, colormaps)):
            subset_df = performance_df[performance_df['Drug'] == drug].pivot(
                index='Build_FRS', columns='Test_FRS', values=metric
            )
            
            # Format annotations to remove scientific notation
            annot_values = subset_df.map(lambda x: f"{x:.0f}")
            
            ax = sns.heatmap(
                subset_df, 
                annot=annot_values,  # Use formatted values
                fmt="",  # Prevent scientific notation
                cmap=cmap, 
                ax=axes[i], 
                vmin=0, vmax=90, 
                cbar=False
            )
            
            ax.set_title(f"{metric} - {drug}", fontsize=6)
            ax.set_xlabel("Test min FRS", fontsize=6)
            ax.set_ylabel("Build min FRS", fontsize=6)
            ax.invert_yaxis()
            
            # Force normal notation on axis tick labels
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{subset_df.columns[int(x)]:.2f}" if int(x) < len(subset_df.columns) else ""
            ))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, _: f"{subset_df.index[int(y)]:.2f}" if int(y) < len(subset_df.index) else ""
            ))
        plt.savefig(f'figs/frs/{drug}.pdf')
        plt.tight_layout()
        plt.show()

# Function to plot error bars with jittered x-values
def plot_mutation_error_bars(frs_prop_data, color_map={}, min_err= 1, label_cutoff=15, figpath=None):
    np.random.seed(0)

    for drug, mutations_dict in frs_prop_data.items():
        plt.figure(figsize=(6.6, 3))

        unique_mutations = list(mutations_dict.keys())
        num_mutations = len(unique_mutations)
        if len(color_map)==0:
            mutation_colors = dict(zip(unique_mutations, sns.color_palette("tab20", num_mutations)))  # More distinct colors
        else:
            mutation_colors=color_map[drug]
        truncated_labels = {mutation: (mutation[:label_cutoff] + '...' if len(mutation) > label_cutoff else mutation) 
                    for mutation in unique_mutations}

        mutation_jitter = {mutation: np.random.uniform(-0.045, 0.045) for mutation in unique_mutations}

        for mutation, data in mutations_dict.items():

            x_values = np.array(data['frs'])  # FRS values
            y_values = np.array(data['y'])  # Proportion values
            y_errors = data['error']  # Error bars
            jitter = mutation_jitter[mutation]  # Use the fixed jitter value per mutation
            background = data['background']
            y_errors = abs_err_to_rel(y_values, y_errors)
            filt = np.abs(np.array(y_errors[0]) - np.array(y_errors[1])) <= min_err
            x_values = x_values[filt]
            y_values = y_values[filt]
            y_errors = (np.array(y_errors[0])[filt], np.array(y_errors[1])[filt])  # Keep same structure

            # Scatter points with error bars, applying consistent jitter
            if len(x_values) > 0:
                plt.errorbar(
                    x=x_values + jitter, y=y_values, 
                    yerr=y_errors, 
                    fmt='o', capsize=0, linewidth=1.25, markersize=2.5, alpha=0.65, 
                    label=truncated_labels[mutation], color=mutation_colors[mutation]
                )
        for i, start in enumerate(np.arange(0.05, 1.0, 0.1)):  # Iterate over x-ranges
            if i % 2 == 0:  # Shade every other section
                plt.axvspan(start, start + 0.1, color='lightgrey', alpha=0.3)
        plt.axhline(background, linewidth=1)
        plt.xlabel("min FRS (Binned, with Jitter)")
        plt.ylabel("Proportion R")
        plt.title(f"Proportion vs FRS - {drug}", fontsize=7)
        plt.xticks(np.arange(0.1, 1.05, 0.1))  # Minor ticks at every 0.05
        plt.ylim(-0.05, 1.05)  # Keep proportions in range
        plt.grid(True, linestyle="--", alpha=0.5)
        num_cols = min(num_mutations, 7)  # Adjust '5' as needed for your plot width
        plt.legend(title="Mutation", loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=5, ncol=num_cols)
        plt.tight_layout()
        sns.despine()
        plt.grid(False)
        if figpath is not None:
            plt.savefig(f'{figpath}{drug}_frs_vs_prop.pdf')
        plt.show()



def plot_frs_vs_mic(df_mic, color_map={}, figpath=None, min_n=0):
    y_axis_orders = {
        "INH": ["0.025", "0.05", "0.1", "0.2", "0.4", "0.8", "1.6"],
        "AMI": ["0.25", "0.5", "1.0", "2.0", "4.0", "8.0"],
        "EMB": ["0.25", "0.5", "1.0", "2.0", "4.0", "8.0"],
        "ETH": ["0.25", "0.5", "1.0", "2.0", "4.0", "8.0"],
        "LEV": ["0.12", "0.25", "0.5", "1.0", "2.0", "4.0", "8.0"],
        "MXF": ["0.06", "0.12", "0.25", "0.5", "1.0", "2.0", "4.0"],
        "RIF": ["0.06", "0.12", "0.25", "0.5", "1.0", "2.0", "4"],
        "STM": [],
        "KAN": ["1", "2.0", "4.0", "8.0", "16"],
        "DLM": ["0.015", "0.03", "0.06", "0.12", "0.25", "0.5"],
        "CAP": [],
        "LZD": ["0.06", "0.12", "0.25", "0.5", "1.0", "2"]
    }

    for drug, v in df_mic.items():    
        fig, axes = plt.subplots(figsize=(4, 2))
        y_axis_order = y_axis_orders.get(drug, [])

        # Count occurrences of each mutation
        mutation_counts = v["MUTATION"].value_counts()
        total_unique_mutations = len(mutation_counts)

        # If more than 20 mutations exist, filter out those with ≤3 occurrences - makes plot too messy
        filter_applied = False
        if total_unique_mutations > 10:
            filtered_mutations = mutation_counts[mutation_counts > min_n].index
            v = v[v["MUTATION"].isin(filtered_mutations)]
            filter_applied = True  # Indicate that filtering was done

        # Generate a custom color palette from color_map if available
        if drug in color_map and color_map[drug]:  
            unique_mutations = v["MUTATION"].unique()
            custom_palette = {mutation: color_map[drug][mutation] for mutation in unique_mutations if mutation in color_map[drug]}
        else:
            custom_palette = "muted"  # Default seaborn palette if no custom colors

        # Plot
        sns.stripplot(
            x="FRS", y="MIC", data=v, ax=axes, jitter=0.2, order=y_axis_order, size=3, 
            hue="MUTATION", palette=custom_palette
        )

        # Set the title, adding (n > 3) if filtering was applied
        if min_n > 0:
            title = f"{drug} (n > {min_n})" if filter_applied else drug
        else:
            title = drug
        axes.set_title(title)
        
        axes.invert_yaxis()
        axes.set_ylabel("MIC (mg/L)")
        axes.set_xlabel("")
        axes.tick_params(axis="both")
        axes.set_xlim(0, 1.02)

        # Modify legend labels to truncate long names
        legend = plt.legend(title="Mutation", bbox_to_anchor=(1, 1.2), loc='upper left', frameon=False, fontsize=5.5)
        for text in legend.get_texts():
            label = text.get_text()
            if len(label) > 20:  # Truncate if longer than 20 characters
                text.set_text(label[:17] + "...")  # Keep first 17 chars + "..."

        plt.tight_layout()
        sns.despine()
        if figpath is not None:
            plt.savefig(f'{figpath}{drug}_frs_vs_mic.pdf')
        plt.show()

def plot_pheno_counts(phenotypes, title, savefig):
    # Compute the count for each (DRUG, PHENOTYPE)
    barplot = (
        phenotypes.groupby(['DRUG', 'PHENOTYPE'])['UNIQUEID']
        .nunique()
        .reset_index()
        .rename(columns={'UNIQUEID': 'count'})
    )

    # Compute total count per DRUG (sum of R, S, U)
    total_counts = barplot.groupby('DRUG')['count'].sum().reset_index()
    
    # Order DRUGs by total count descending
    plot_order = total_counts.sort_values('count', ascending=False)['DRUG'].tolist()

    # Create the bar plot
    plt.figure(figsize=(6.69, 3.5))
    sns.barplot(
        data=barplot,
        x='DRUG', y='count', hue='PHENOTYPE',
        hue_order=['S', 'R', 'U'], order=plot_order, dodge=True, palette='muted', alpha=0.9
    )

    # Customize the plot
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel("# unique samples", fontsize=7)
    plt.xlabel("Drug", fontsize=7)
    plt.title(f"{title}: {phenotypes.UNIQUEID.nunique()} samples", fontsize=8)
    plt.legend(title="Phenotype")
    sns.set_theme(style="whitegrid")

    # Annotate bars with values
    for p in plt.gca().patches:
        if p.get_height() > 0:  # Only label non-zero bars
            plt.text(
                p.get_x() + p.get_width() / 2, p.get_height() + 0.5, f"{int(p.get_height())}",
                ha='center', va='bottom', fontsize=5.5
            )

    plt.legend(frameon=False, fontsize=7)
    plt.grid(False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savefig)
    # Show the plot
    plt.show()


# Helper function to create binary data for upset plots
def prepare_upset_data(data, filters, plot_order, min_sample_count=5):
    # Apply the filter for the subset
    filtered_data = data.query(filters)

    # Convert relevant columns to binary (1 for 'R', 0 otherwise)
    binary_data = filtered_data.assign(
        **{col: filtered_data[col].apply(lambda x: 1 if x == "R" else 0) for col in plot_order}
    )[plot_order]

    # Count the unique patterns
    pattern_counts = binary_data.groupby(plot_order).size()

    # Filter out patterns that appear less than `min_sample_count` times
    valid_patterns = pattern_counts[pattern_counts >= min_sample_count].index
    filtered_binary_data = binary_data[binary_data[plot_order].apply(tuple, axis=1).isin(valid_patterns)]

    # Number of remaining samples
    n_samples = filtered_binary_data.shape[0]

    return filtered_binary_data, n_samples



def create_upset_plot(data, title, n_samples, file_name=None):
    # Ensure columns are boolean
    data_boolean = data.astype(bool)

    # Convert to UpSet-compatible format
    data_for_upset = from_indicators(indicators=data_boolean.columns, data=data_boolean)

    # Create the UpSet plot with custom styling
    upset = UpSet(
        data_for_upset,
        subset_size="count",
        sort_by="cardinality",
        sort_categories_by="cardinality",
        intersection_plot_elements=10,  # Show top 10 most common intersections
        show_percentages=True,  # Show percentages on bars
        facecolor="darkblue",  # Change bar colors
        element_size=40,  # Adjust matrix dot size
    )

    # Create the plot - figure size seems to do nothing
    fig = plt.figure(figsize=(6.69, 2))  # Set figure size
    upset.plot(fig=fig)

    # Add additional text and styling
    plt.suptitle(title, fontsize=7, fontweight="bold")  # Bigger title
    plt.figtext(0.5, 0.92, f"n = {n_samples} samples", ha="center", fontsize=7, fontweight="bold")
    plt.figtext(0.5, 0.89, "Only interactions with >5 isolates displayed", ha="center", fontsize=7, color="gray")

    # Modify subplot aesthetics
    for ax in fig.axes:
        ax.grid(False)  # Remove grid lines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    # Show the plot
    plt.show()


def wilson_ci(row, R_col, S_col):
    """
    Calculates the Wilson confidence interval for a given row based on the specified column names
    for the counts of successes (R) and failures (S).

    Parameters:
        row (pd.Series): The row of the DataFrame.
        R_col (str): The name of the column for successes (R).
        S_col (str): The name of the column for failures (S).
        
    Returns:
        list: A list containing the lower and upper bounds of the Wilson confidence interval.
    """
    z = norm.ppf(0.975)  # 95% confidence level (z = 1.96)
    
    # Extract the values for R and S based on the provided column names
    R = row[R_col]
    S = row[S_col]
    if np.isnan(R) or np.isnan(S) or (R + S == 0):
        return [np.nan, np.nan]
    n = R + S
    p_hat = R / n if n > 0 else 0

    # Wilson confidence interval calculation
    denominator = 1 + z**2 / n
    center_adjusted = p_hat + z**2 / (2 * n)

    margin_of_error = z * np.sqrt((p_hat * (1 - p_hat) / n) + z**2 / (4 * n**2))
    
    lower_bound = (center_adjusted - margin_of_error) / denominator
    upper_bound = (center_adjusted + margin_of_error) / denominator
    
    return [abs(lower_bound), abs(upper_bound)]


def extract_errors(df, prop_col, conf_col, suffixes=['cat1', 'cat2']):

    x, y = df[f'{prop_col}_{suffixes[0]}'], df[f'{prop_col}_{suffixes[1]}']
    # Ensure errors are lists, and apply capping
    def clean_ci(ci):
        if isinstance(ci, list) and len(ci) == 2:
            return [max(0, float(ci[0])), min(1, float(ci[1]))] 
        return [np.nan, np.nan]

    x_err, y_err = df[f'{conf_col}_{suffixes[0]}'].apply(clean_ci), df[f'{conf_col}_{suffixes[1]}'].apply(clean_ci)

    if len(x) > 1:
        return (np.array(x).flatten(), np.array(y).flatten(), np.vstack(x_err), np.vstack(y_err))
    elif len(x) == 1:
        return (np.array(x).flatten(), np.array(y).flatten(), np.stack(x_err.values).T, np.stack(y_err.values).T)
    else:
        return (np.array(x).flatten(), np.array(y).flatten(), x_err.to_numpy(), y_err.to_numpy())