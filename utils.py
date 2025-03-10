# %%
import csv
import numpy as np

from piezo import catalogue
import json
import piezo
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
import json
import os
import pandas as pd
from pathlib import Path
from matplotlib.patches import Rectangle



def get_separator(filename):
    """
    Ascertain file separator
    """
    with open(filename, "r") as file:
        sample = file.read(2048)  # Read a larger sample of the file
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except csv.Error:
            print(f"CSV Sniffer failed. Sample read from file: {sample[:200]}...")
            print("Falling back to common delimiters.")
            if "," in sample:
                return ","
            elif "\t" in sample:
                return "\t"
            elif ";" in sample:
                return ";"
            else:
                raise ValueError(
                    f"Unable to determine the separator for file '{filename}'"
                )


def check_gene_file_columns(df):
    """
    Check that the columns in the dataframe from genes file are correct
    """
    required_columns = {"drug", "gene"}
    df.columns = df.columns.str.lower()  # Convert column names to lowercase
    if not set(df.columns) == required_columns:
        required_columns_lower = {col.lower() for col in required_columns}
        if not required_columns_lower <= set(df.columns):
            raise ValueError("Genes file must have the following columns: drug, gene")
    else:
        print("all good with genes file requirements")


def check_mutations_file_columns(df):
    """
    Check that the columns in the dataframe from mutations file are correct
    """
    required_columns = {"ena_run", "gene", "mutation"}
    df.columns = df.columns.str.lower()  # Convert column names to lowercase
    if not set(df.columns) == required_columns:
        required_columns_lower = {col.lower() for col in required_columns}
        if not required_columns_lower <= set(df.columns):
            raise ValueError(
                "Hi genes file must have the following columns: drug, gene"
            )
    else:
        print("all good with mutations file requirements")


def prepare_catomatic_mutations_input(df):
    # Define the required columns and new names compatible with catomatic input requirement
    required_columns = ["ena_run", "GENE_MUT", "frs"]
    new_column_names = ["UNIQUEID", "MUTATION", "FRS"]

    # Check if all required columns exist in the DataFrame
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing: {', '.join(missing_columns)}"
        )

    # Select and rename the columns
    df_selected = df[required_columns].copy()
    df_selected.columns = new_column_names

    return df_selected


def check_phenotypes_file_columns(df):
    """
    Check that the columns in the dataframe from phenotypes file are correct
    """
    required_columns = {"ena_run", "drug", "phenotype", "phenotype_quality"}
    df.columns = df.columns.str.lower()  # Convert column names to lowercase
    if not set(df.columns) == required_columns:
        raise ValueError(
            "Phenotypes file must have the following columns: ena_run, drug, phenotype, phenotype_quality"
        )


def prepare_catomatic_phenotypes_input(df):
    # Define the required columns and new names compatible with catomatic input requirement
    required_columns = ["ena_run", "phenotype"]
    new_column_names = ["UNIQUEID", "PHENOTYPE"]

    # Check if all required columns exist in the DataFrame
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing: {', '.join(missing_columns)}"
        )

    # Select and rename the columns
    df_selected = df[required_columns].copy()
    df_selected.columns = new_column_names

    return df_selected


def get_genes_of_interest_from_genes_file(filename, drug):
    """
    Get list of genes by filtering genes list df by drug of interest
    """

    # if not isinstance(filename, str) or not os.path.exists(filename):
    #     raise ValueError("Filename must be a valid file path.")

    # Get separator or assume
    genes_separator = get_separator(filename)
    try:
        genes_df = pd.read_csv(filename, sep=genes_separator, header=0)
    except Exception as e:
        raise ValueError(f"Hi Error reading genes file: {e}")

    # Check columns
    check_gene_file_columns(genes_df)
    print("getting genes")

    genes_gene_column = next(
        (col for col in genes_df.columns if "gene" in col.lower()), None
    )
    genes_drug_column = next(
        (col for col in genes_df.columns if "drug" in col.lower()), None
    )

    if genes_gene_column and genes_drug_column:
        # Convert drug column to lowercase (or uppercase)
        genes_df[genes_drug_column] = genes_df[genes_drug_column].str.upper()

        # Convert user input to lowercase (or uppercase)
        drug = drug.upper()

        # Filter genes based on case-insensitive comparison
        filtered_genes = genes_df.loc[
            genes_df[genes_drug_column] == drug, genes_gene_column
        ].tolist()
        return filtered_genes
    else:
        raise ValueError("Missing 'gene' or 'drug' column in the genes file")


def filter_multiple_phenos(group):
    """
    If a sample contains more than 1 phenotype,
    keep the resistant phenotype if there is one.
    (Developer: DA)
    """
    if len(group) > 1:
        if "R" in group["phenotype"].values:
            return group[group["phenotype"] == "R"].iloc[0:1]
        else:
            return group.iloc[0:1]
    else:
        return group


def RSIsolateTable(df, genes):
    """returns df of number of R and S isolates"""
    table = {}
    table["Total"] = {
        "R": df[df.phenotype == "R"].ena_run.nunique(),
        "S": df[df.phenotype == "S"].ena_run.nunique(),
        "Total": df.ena_run.nunique(),
    }
    for i in genes:
        d = df[df.gene == i]
        table[i] = {
            "R": d[d.phenotype == "R"].ena_run.nunique(),
            "S": d[d.phenotype == "S"].ena_run.nunique(),
            "Total": d[d.phenotype == "R"].ena_run.nunique()
            + d[d.phenotype == "S"].ena_run.nunique(),
        }

    return pd.DataFrame.from_dict(table).T


def RSVariantTable(df, genes):
    """returns df of number of R and S variants"""
    table = {}
    table["Total"] = {
        "R": df[df.phenotype == "R"].ena_run.count(),
        "S": df[df.phenotype == "S"].ena_run.count(),
        "Total": df.ena_run.count(),
    }
    for i in genes:
        d = df[df.gene == i]
        table[i] = {
            "R": d[d.phenotype == "R"].ena_run.count(),
            "S": d[d.phenotype == "S"].ena_run.count(),
            "Total": d[d.phenotype == "R"].ena_run.count()
            + d[d.phenotype == "S"].ena_run.count(),
        }

    return pd.DataFrame.from_dict(table).T


def CombinedDataTable(all):

    df = RSIsolateTable(all, all.GENE.unique())
    df1 = RSIsolateTable(all[all.FRS < 0.9], all.GENE.unique())
    df2 = RSVariantTable(all, all.GENE.unique())
    df3 = RSVariantTable(all[all.FRS < 0.9], all.GENE.unique())
    df = pd.concat([df, df1, df2, df3], axis=1)

    df.columns = pd.MultiIndex.from_tuples(
        zip(
            [
                "All",
                "",
                "",
                "Minor alleles",
                "",
                "",
                "All",
                "",
                "",
                "Minor alleles",
                "",
                "",
            ],
            df.columns,
        )
    )

    return df


def build_S_arr(self):
    # remove mutations predicted as susceptible from df (to potentially proffer additional, effective solos)
    mutations = self.all_data_frs_filtered[
        ~self.all_data_frs_filtered.GENE_MUT.isin(i["mut"] for i in self.S)
    ]
    # mutations = self.all_data_frs_filtered[
    # (~self.all_data_frs_filtered.GENE_MUT.isin(i["mut"] for i in self.S)) &
    # (~self.all_data_frs_filtered.GENE_MUT.isin(i["mut"] for i in self.R)) &
    # (~self.all_data_frs_filtered.GENE_MUT.isin(i["mut"] for i in self.U))
    # ]
    # extract samples with only 1 mutation
    solos = mutations.groupby("ena_run").filter(lambda x: len(x) == 1)

    # method is jammed - end here.
    if len(solos) == 0:
        self.run = False

    mut_count = 0
    s_iters = 0
    S_count = 0
    # U_count = 0
    # R_count = 0
    # for non WT or synonymous mutations
    for mut in solos[~solos.GENE_MUT.isna()].GENE_MUT.unique():
        mut_count += 1
        # print(f"this mutation: {mut}")
        # determine phenotype of mutation using Fisher's test
        # print(f"{s_iters}: {mut}")
        pheno = fisher_binary(solos, mut, self.run_OR)
        # print(f"this prediction: {pheno}")
        # print(f"evid: {pheno['evid']}")
        if pheno["pred"] == "S":
            # print(f"adding to self.S")
            # if susceptible, add mutation to phenotype array
            self.S.append({"mut": mut, "evid": pheno["evid"]})
            s_iters += 1
        # if pheno["pred"] == "S":
        #     S_count += 1
        # if pheno["pred"] == "U":
        #     self.U.append({"mut": mut, "evid": pheno["evid"]})
        #     s_iters += 1
        #     U_count += 1
        # if pheno["pred"] == "R":
        #     self.R.append({"mut": mut, "evid": pheno["evid"]})
        #     s_iters += 1
        #     R_count += 1
    print(
        f"The number of unique mutations is {len(solos[~solos.GENE_MUT.isna()].GENE_MUT.unique())} for this pass"
    )

    if s_iters == 0:
        # if no susceptible solos (ie jammed) - move to mop up
        self.run = False


def mop_up(self):
    # remove mutations predicted as susceptible from df (to potentially proffer additional, effective solos)
    no_S_mutations = self.all_data_frs_filtered[
        ~self.all_data_frs_filtered.GENE_MUT.isin(i["mut"] for i in self.S)
    ]

    print(f" There are {len(self.S)} mutations in self.S")
    # extract samples with only 1 mutation
    mop_solos = no_S_mutations.groupby("ena_run").filter(lambda x: len(x) == 1)

    print(f"Number of mutations considered for finding R and U: {len(mop_solos)}")
    # for non WT or synonymous mutations
    for mut in mop_solos[~mop_solos.GENE_MUT.isna()].GENE_MUT.unique():
        # determine phenotype of mutation using Fisher's test and add mutation to phenotype array (should be no S)
        pheno = fisher_binary(mop_solos, mut, self.run_OR)
        if pheno["pred"] == "R":
            self.R.append({"mut": mut, "evid": pheno["evid"]})
        elif pheno["pred"] == "U":
            # print(F"length or U before adding {mut}: {len(self.U)}")
            self.U.append({"mut": mut, "evid": pheno["evid"]})
            # print(F"length or U after adding {mut}: {len(self.U)}")
            # print(f"{pheno['evid']}")


def fisher_binary(solos, mut, run_OR=False):
    # Count occurrences of "R" and "S" phenotypes for the mutation and without the mutation
    R_count = len(solos[(solos["phenotype"] == "R") & (solos["GENE_MUT"] == mut)])
    S_count = len(solos[(solos["phenotype"] == "S") & (solos["GENE_MUT"] == mut)])
    R_count_no_mut = len(
        solos[
            (~solos["GENE_MUT"].isna())
            & (solos["GENE_MUT"] != mut)
            & (solos["phenotype"] == "R")
        ]
    )
    S_count_no_mut = len(
        solos[
            (~solos["GENE_MUT"].isna())
            & (solos["GENE_MUT"] != mut)
            & (solos["phenotype"] == "S")
        ]
    )

    # Build contingency table: ((R count, S count), (background R count, background S count))
    data = [[R_count, S_count], [R_count_no_mut, S_count_no_mut]]

    # Calculate Fisher's exact test p-value
    _, p_value = stats.fisher_exact(data)

    # Run fisher exact test, calculate OR and CIs and classify according to OR > 1 and ci_low and ci_high > 1 == R or
    if run_OR:
        # print("Running fisher exact with OR and CIs at 95%")
        odds_ratio, ci_low, ci_high = calculate_odds_ratio_and_ci(
            R_count, S_count, R_count_no_mut, S_count_no_mut, alpha=0.05
        )
        if (
            odds_ratio is not None and ci_low is not None and ci_high is not None
        ):  # Check if any value is None
            if p_value < 0.05 or solos[solos.GENE_MUT == mut].phenotype.nunique() == 1:
                # print(f"OR, CIs: {odds_ratio}, {ci_low}, {ci_high}")
                if odds_ratio > 1 and ci_low > 1 and ci_high > 1:
                    prediction = "R"
                elif odds_ratio < 1 and ci_low < 1 and ci_high < 1:
                    prediction = "S"
                else:
                    prediction = "U"
            else:
                prediction = "U"
        else:
            prediction = "U"  # Handle the case where any of the values is None

        return {
            "pred": prediction,
            "evid": [
                [R_count, S_count],
                [R_count_no_mut, S_count_no_mut],
                [p_value, odds_ratio],
                [ci_low, ci_high],
            ],
        }
    # Run fisher exact test and assign phenotype by majority count
    else:
        # print("Running fisher exact without OR and CIs at 95%")
        if p_value < 0.05 or solos[solos.GENE_MUT == mut].phenotype.nunique() == 1:
            # if variant frequency is 1 simply call the phenotype, otherwise call phenotype at 95% confidence
            prediction = "R" if R_count > S_count else "S"
        else:
            prediction = "U"

        return {
            "pred": prediction,
            "evid": [
                [R_count, S_count],
                [R_count_no_mut, S_count_no_mut],
                [p_value],
            ],
        }


def calculate_odds_ratio_and_ci(
    R_count, S_count, R_count_no_mut, S_count_no_mut, alpha=0.05
):
    # Check if any of the denominators == 0
    if S_count == 0 or R_count_no_mut == 0:
        return None, None, None
    # Calculate odds ratio
    odds_ratio = (R_count * S_count_no_mut) / (S_count * R_count_no_mut)

    # Calculate confidence intervals for odds ratio
    ci_low, ci_high = proportion_confint(
        R_count, R_count + S_count, alpha=alpha, method="beta"
    )

    return odds_ratio, ci_low, ci_high


def construct_catalogue(self):
    catalogue = {}
    for i in self.S:
        catalogue[i["mut"]] = {"pred": "S", "evid": i["evid"]}
    for i in self.R:
        catalogue[i["mut"]] = {"pred": "R", "evid": i["evid"]}
    for i in self.U:
        catalogue[i["mut"]] = {"pred": "U", "evid": i["evid"]}

    return catalogue


def return_catalogue(self):
    return {
        mutation: {"phenotype": data["pred"]}
        for mutation, data in self.catalogue.items()
    }


def insert_wildcards(self, wildcards):
    if self.catalogue is not None and wildcards is not None:
        self.catalogue = {**self.catalogue, **wildcards}
    elif self.catalogue is None and wildcards is not None:
        self.catalogue = wildcards
    elif self.catalogue is not None and wildcards is None:
        # Do nothing or handle the case where wildcards is None
        pass
    else:
        # Both self.catalogue and wildcards are None, handle as needed
        pass


def return_piezo(
    self,
    genbank_file,
    catalogue_name,
    catalogue_version,
    drug,
    piezo_wildcards,
    grammar="GARC1",
    values="RUS",
):
    # insert piezo wildcards into catalogue object
    insert_wildcards(self, piezo_wildcards)

    piezo = (
        pd.DataFrame.from_dict(self.catalogue, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "MUTATION",
                "pred": "PREDICTION",
                "evid": "EVIDENCE",
                "p": "p_value",
            }
        )
    )
    piezo["GENBANK_REFERENCE"] = genbank_file
    piezo["CATALOGUE_NAME"] = catalogue_name
    piezo["CATALOGUE_VERSION"] = catalogue_version
    piezo["CATALOGUE_GRAMMAR"] = grammar
    piezo["PREDICTION_VALUES"] = values
    piezo["DRUG"] = drug
    piezo["SOURCE"] = json.dumps({})
    piezo["EVIDENCE"] = [
        (
            json.dumps(
                {
                    "solo_R": i[0][0],
                    "solo_S": i[0][1],
                    "background_R": i[1][0],
                    "background_S": i[1][1],
                    "p_value": i[2][0],
                    "odds_ratio": i[2][1],
                    "CI_low": i[3][0],
                    "CI_high": i[3][1],
                }
            )
            if i
            else json.dumps({})
        )
        for i in piezo["EVIDENCE"]
    ]
    piezo["OTHER"] = json.dumps({})

    piezo = piezo[
        [
            "GENBANK_REFERENCE",
            "CATALOGUE_NAME",
            "CATALOGUE_VERSION",
            "CATALOGUE_GRAMMAR",
            "PREDICTION_VALUES",
            "DRUG",
            "MUTATION",
            "PREDICTION",
            "SOURCE",
            "EVIDENCE",
            "OTHER",
        ]
    ]

    return piezo


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


def DisplayPerformance(cm, columns):
    sns.set_context("talk")
    # cov = 100 * (len(val_samples) - cm[0][2] - cm[1][2]) / len(val_samples)
    df_cm = pd.DataFrame(cm, index=["R", "S"], columns=columns)
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        df_cm,
        annot=True,
        cbar=False,
        fmt="g",
        cmap="Greens",
        annot_kws={"fontsize": 24},
    )
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.show()

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


def str_to_dict(s):
    """Convert strings to dictionary - helpful for evidence column"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s

# %%
