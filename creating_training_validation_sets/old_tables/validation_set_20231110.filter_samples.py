#!/usr/bin/env python3

# This script filters the runs in validation_set_20231110.csv. It
# also uses CRyPTIC_reuse_table_20231107.csv and ENA metadata from
# validation_set_20231110.ena_meta.json.gz.
#
# It only keeps paired runs, then aggregates the runs into samples
# (some samples have >1 run).
#
# Samples are removed where:
# - when there is >1 run, the drprg phenotypes disagree between runs
# - runs within a sample belong to >1 project
# - the drprg sample is different from the ENA lookup of run -> sample
# - the drprg phenotypes are not the same as the cryptic phenotypes
#
# Output files:
# - validation_set_20231110.fail.json.gz. Failed samples in gzipped JSON format
# - validation_set_20231110.pass.json.gz. Pass samples in gzipped JSON format
# - validation_set_20231110.pass.tsv. Pass samples in TSV format



import csv
import gzip
import json

DRPRG_DRUGS = [
    "amikacin",
    "bedaquiline",
    "capreomycin",
    "ciprofloxacin",
    "clofazimine",
    "cycloserine",
    "delamanid",
    "ethambutol",
    "ethionamide",
    "gatifloxacin",
    "isoniazid",
    "kanamycin",
    "levofloxacin",
    "linezolid",
    "moxifloxacin",
    "ofloxacin",
    "para-aminosalicylic_acid",
    "pyrazinamide",
    "rifabutin",
    "rifampicin",
    "streptomycin",
    "thioacetazone",
]

CRYPTIC_DRUGS = {
    "AMI_BINARY_PHENOTYPE": "amikacin",
    "BDQ_BINARY_PHENOTYPE": "bedaquiline",
    "CFZ_BINARY_PHENOTYPE": "clofazimine",
    "DLM_BINARY_PHENOTYPE": "delamanid",
    "EMB_BINARY_PHENOTYPE": "ethambutol",
    "ETH_BINARY_PHENOTYPE": "ethionamide",
    "INH_BINARY_PHENOTYPE": "isoniazid",
    "KAN_BINARY_PHENOTYPE": "kanamycin",
    "LEV_BINARY_PHENOTYPE": "levofloxacin",
    "LZD_BINARY_PHENOTYPE": "linezolid",
    "MXF_BINARY_PHENOTYPE": "moxifloxacin",
    "RIF_BINARY_PHENOTYPE": "rifampicin",
    "RFB_BINARY_PHENOTYPE": "rifabutin",
}


# This is a dict of ENA run -> basic ENA metadata (run, sample, project
# accessions etc)
with gzip.open("validation_set_20231110.ena_meta.json.gz", "rt") as f:
    all_ena_meta = json.load(f)


# Load the potential validation runs. Since some have more than one
# run per sample, gather them by sample. Use all sample accessions of the
# form "ERS"/"SRS" for keys, not the "SAM..." ones
validation_samples = {}
with open("validation_set_20231110.csv") as f:
    for d in csv.DictReader(f):
        run_meta = all_ena_meta[d["run"]]
        sample = run_meta["secondary_sample_accession"]
        assert sample[:3] in {"ERS", "SRS"}
        if sample not in validation_samples:
            validation_samples[sample] = {"runs": {}, "cryptic": None, "fails": {}}
        assert d["run"] not in validation_samples[sample]["runs"]
        validation_samples[sample]["runs"][d["run"]] = {"drprg": d, "ena": run_meta}

print("Total samples:", len(validation_samples))


# Add cryptic data to the validation samples, where possible.
# Check that the cryptic sample is "ERS", so we're not accidentally trying
# to match a "SAM" sample identifier instead
cryptic_samples = {}
with open("../data/CRyPTIC_reuse_table_20231107.csv") as f:
    for d in csv.DictReader(f):
        sample = d["ENA_SAMPLE"]
        assert sample[:3] == "ERS"
        if sample in validation_samples:
            validation_samples[sample]["cryptic"] = d


failed_samples = {}

# Sanity checks of all the metadata
for sample, sample_d in validation_samples.items():
    in_cryptic = "no" if sample_d["cryptic"] is None else "yes"

    # Some samples only have unpaired reads. We can't use them
    paired_runs = [
        r for r, d in sample_d["runs"].items() if d["ena"]["library_layout"] == "PAIRED"
    ]
    if len(paired_runs) == 0:
        sample_d["fails"]["No_paired_runs"] = True
        print("Exclude no paired runs", sample, sep="\t")

    # If there's more than one run for this sample, then check that all
    # runs have the same drprg phenotype calls
    if len(sample_d["runs"]) > 1:
        bad_phenos = []
        for drug in DRPRG_DRUGS:
            phenos = {d["drprg"][drug] for d in sample_d["runs"].values()}
            if len(phenos) > 1:
                bad_phenos.append(f"{drug}:{','.join(sorted(list(phenos)))}")

        if len(bad_phenos) > 0:
            print(
                "Exclude drprg pheno mismatch",
                f"in_cryptic={in_cryptic}",
                sample,
                ",".join(sample_d["runs"]),
                *bad_phenos,
                sep="\t",
            )
            sample_d["fails"]["drprg_pheno_self_inconsistent"] = "; ".join(bad_phenos)

    # If the drprg phenotype calls are ok, then we can make a single dictionary
    # of them
    if "drprg_pheno_self_inconsistent" not in sample_d["fails"]:
        sample_d["drprg_pheno"] = list(sample_d["runs"].values())[0]["drprg"]

    # Check that the sample from drprg is the same as what we got when
    # we lookup run -> sample in the ENA.
    # And check the project accessions are sane - don't want >1 project accession
    drprg_samples = {d["drprg"]["biosample"] for d in sample_d["runs"].values()}
    assert len(drprg_samples) == 1
    drprg_sample = drprg_samples.pop()
    ena_samples = set()
    projects = set()
    for d in sample_d["runs"].values():
        ena_samples.add(d["ena"]["sample_accession"])
        ena_samples.add(d["ena"]["secondary_sample_accession"])
        assert d["ena"]["study_accession"].startswith("PRJ")
        projects.add(d["ena"]["study_accession"])

    if len(projects) > 1:
        projects = sorted(list(projects))
        sample_d["fails"]["more_than_on_project"] = projects
        print("Exclude multiple projects", sample, *projects, sep="\t")
    else:
        sample_d["project"] = projects.pop()

    if drprg_sample not in ena_samples:
        ena_samples = ",".join(sorted(list(ena_samples)))
        sample_d["fails"][
            "samples_mismatch"
        ] = f"drprg={drprg_sample}, ena={ena_samples}"
        print(
            "Exclude sample mismatch",
            f"in_cryptic={in_cryptic}",
            f"drprg={drprg_sample}",
            f"ena={ena_samples}",
            sep="\t",
        )

    # If the sample is in cryptic, and the drprg phenotypes are self-consistent,
    # then we can check that the cryptic phenotypes agree with the drprg
    # phenotypes. There's no point doing this if we've already found that
    # the drprg phenotypes are not consistent, since we're after "the"
    # phenotypes for each sample.
    # Cryptic drugs are a subset of drprg drugs, so for each cryptic pheno
    # we check the corresponding drprg call
    if sample_d["cryptic"] is not None and "drprg_pheno" in sample_d:
        bad_phenos = []

        for cryptic_key, drprg_key in CRYPTIC_DRUGS.items():
            # there's a mix of empty strings and "NA" in the pheno calls.
            # Make the empty ones NA, so can compare
            cryptic_pheno = sample_d["cryptic"][cryptic_key]
            if len(cryptic_pheno) == "":
                cryptic_pheno = "NA"
            drprg_pheno = sample_d["drprg_pheno"][drprg_key]
            if drprg_pheno == "":
                drprg_pheno = "NA"
            if cryptic_pheno != drprg_pheno:
                bad_phenos.append(
                    f"{drprg_key}:cryptic={cryptic_pheno},drprg={drprg_pheno}"
                )

        if len(bad_phenos) > 0:
            sample_d["fails"]["cryptic_drprg_pheno_mismatch"] = bad_phenos
            print(
                "Exclude cryptic/drprg pheno mismatch",
                f"in_cryptic={in_cryptic}",
                sample,
                *bad_phenos,
                sep="\t",
            )

    if len(sample_d["fails"]) > 0:
        failed_samples[sample] = sample_d


# Split the samples into pass and fail, then write JSON files
for sample in failed_samples:
    del validation_samples[sample]

with gzip.open("validation_set_20231110.pass.json.gz", "wt") as f:
    json.dump(validation_samples, f, indent=2)

with gzip.open("validation_set_20231110.fail.json.gz", "wt") as f:
    json.dump(failed_samples, f, indent=2)


print("Total pass samples:", len(validation_samples))
print("Total fail samples:", len(failed_samples))


# Make a TSV of the pass samples
with open("validation_set_20231110.pass.tsv", "w") as f:
    print("sample", "run", "project", *DRPRG_DRUGS, sep="\t", file=f)
    for sample, sample_d in validation_samples.items():
        print(
            sample,
            ".".join(sorted(sample_d["runs"])),
            sample_d["project"],
            *(sample_d["drprg_pheno"][x] for x in DRPRG_DRUGS),
            sep="\t",
            file=f,
        )
