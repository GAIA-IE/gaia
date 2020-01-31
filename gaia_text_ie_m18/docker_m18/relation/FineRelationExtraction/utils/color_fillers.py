def read_filler_tab(fname, start_idx, target_types=("COL",)):
    filler_tab_data = []
    filler_fields = ["source", "temp_id", "text", "provenance", "link", "filler_type", "level", "confidence", "idx"]
    curr_idx = start_idx + 1
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                temp = line.split("\t")

                if temp[5] in target_types:
                    temp.append(str(curr_idx).zfill(7))
                    filler_tab_data.append(dict(zip(filler_fields, temp)))
                    curr_idx += 1
    return filler_tab_data


def read_filler_cs(fname):
    filler_cs_data = []
    max_idx = 0
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                temp = line.split("\t")
                curr_idx = int(temp[0].split("_")[-1])
                if curr_idx > max_idx:
                    max_idx = curr_idx

                filler_cs_data.append(line)
    return filler_cs_data, max_idx


def format_color_str(data_dict, add_prefix=False):
    rdf_prefix = "https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology"
    if add_prefix:
        type_cs = ':Entity_Filler_COL_{idx}\ttype\t{rdf_prefix}#{filler_type}\n'.format(
            rdf_prefix=rdf_prefix,
            **data_dict
        )
    else:
        type_cs = ':Entity_Filler_COL_{idx}\ttype\t{filler_type}\n'.format(**data_dict)

    canon_cs = ':Entity_Filler_COL_{idx}\tcanonical_mention\t"{text}"\t{provenance}\t{confidence}\n'.format(**data_dict)
    ment_cs = ':Entity_Filler_COL_{idx}\tmention\t"{text}"\t{provenance}\t{confidence}\n'.format(**data_dict)
    return type_cs + canon_cs + ment_cs


def combine_filler_sources(cs_fillers, tab_fillers, combo_outfname):
    with open(combo_outfname, "w", encoding="utf-8") as outf:
        for line in cs_fillers:
            outf.write(line + "\n")

        for tab_fil in tab_fillers:
            cs_str = format_color_str(tab_fil)
            outf.write(cs_str)


def isolate_colors(color_fillers, outfname):
    with open(outfname, "w", encoding="utf-8") as outf:
        for col_fil in color_fillers:
            if col_fil["filler_type"] == "COL":
                col_fil["filler_type"] = "VAL"
                cs_str = format_color_str(col_fil, add_prefix=True)
                outf.write(cs_str)


def lang_main(fill_cs_fname, fill_tab_fname, outfname, color_only_outfname=None):
    di_filler, max_idx = read_filler_cs(fill_cs_fname)
    other_filler = read_filler_tab(fill_tab_fname, max_idx)
    combine_filler_sources(di_filler, other_filler, outfname)
    if color_only_outfname:
        isolate_colors(other_filler, color_only_outfname)


if __name__ == "__main__":
    # ======= I1 Demo 2019 ======
    en_fill_cs_fname = "/nas/data/m1/lud2/AIDA/dryrun/20190724/filler/en/filler_cleaned.cs"
    en_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/en_all/en.col.tab"
    en_combo_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/en_all/en_all.fillers.cs"
    en_color_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/en_all/en_all.color_filler.cs"
    lang_main(en_fill_cs_fname, en_color_fill_tab_fname, en_combo_outfname, en_color_outfname)

    # # ======= TA1b EVAL 2019 ======
    # en_fill_cs_fname = "/nas/data/m1/lud2/AIDA/eval/y1/v1/filler/en_hypo_0710/filler_cleaned.cs"
    # en_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/en_all/en.col.tab"
    # en_combo_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/en_all/en_all.fillers.cs"
    # en_color_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/en_all/en_all.color_filler.cs"
    # lang_main(en_fill_cs_fname, en_color_fill_tab_fname, en_combo_outfname, en_color_outfname)
    #
    # ru_fill_cs_fname = "/nas/data/m1/lud2/AIDA/eval/y1/v1/filler/ru_hypo/filler_cleaned.cs"
    # ru_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/ru_all/ru.col.tab"
    # ru_combo_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/ru_all/ru_all.fillers.cs"
    # ru_color_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/ru_all/ru_all.color_filler.cs"
    # lang_main(ru_fill_cs_fname, ru_color_fill_tab_fname, ru_combo_outfname, ru_color_outfname)
    #
    # uk_fill_cs_fname = "/nas/data/m1/lud2/AIDA/eval/y1/v1/filler/uk_hypo/filler_cleaned.cs"
    # uk_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/uk/uk.col.tab"
    # uk_combo_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/uk/uk.fillers.cs"
    # uk_color_outfname = "/nas/data/m1/whites5/AIDA/M18/eval/TA1b/results/E101_PT003/uk/uk.color_filler.cs"
    # lang_main(uk_fill_cs_fname, uk_color_fill_tab_fname, uk_combo_outfname, uk_color_outfname)


    # # ======= DRYRUN 2019/07/04 ======
    # en_fill_cs_fname = "/nas/data/m1/lud2/AIDA/dryrun/20190704/filler/en/filler_cleaned.cs"
    # en_color_fill_tab_fname = "/data/m1/liny9/aida/result/dryrun3_0704/en_all/en.col.tab"
    # en_combo_outfname = "/data/m1/whites5/AIDA/M18/dryrun20190704/results/en_all/en_all.fillers.cs"
    # en_color_outfname = "/data/m1/whites5/AIDA/M18/dryrun20190704/results/en_all/en_all.color_filler.fix.cs"
    # lang_main(en_fill_cs_fname, en_color_fill_tab_fname, en_combo_outfname, en_color_outfname)

    # ru_fill_cs_fname = "/nas/data/m1/lud2/AIDA/dryrun/20190704/filler/ru/filler_cleaned.cs"
    # ru_color_fill_tab_fname = "/data/m1/liny9/aida/result/dryrun3_0704/ru_all/ru.col.tab"
    # ru_combo_outfname = "/data/m1/whites5/AIDA/M18/dryrun20190704/results/ru_all/ru_all.fillers.cs"
    # ru_color_outfname = "/data/m1/whites5/AIDA/M18/dryrun20190704/results/ru_all/ru_all.color_filler.cs"
    # lang_main(ru_fill_cs_fname, ru_color_fill_tab_fname, ru_combo_outfname, ru_color_outfname)
    #
    # uk_fill_cs_fname = "/nas/data/m1/lud2/AIDA/dryrun/20190704/filler/uk/filler_cleaned.cs"
    # uk_color_fill_tab_fname = "/data/m1/liny9/aida/result/dryrun3_0704/uk/uk.col.tab"
    # uk_combo_outfname = "/data/m1/whites5/AIDA/M18/dryrun20190704/results/uk/uk.fillers.cs"
    # uk_color_outfname = "/data/m1/whites5/AIDA/M18/dryrun20190704/results/uk/uk.color_filler.cs"
    # lang_main(uk_fill_cs_fname, uk_color_fill_tab_fname, uk_combo_outfname, uk_color_outfname)


    # # ======= TA1a EVAL 2019 ======
    # en_fill_cs_fname = "/nas/data/m1/lud2/AIDA/eval/y1/v1/filler/en_all/filler_cleaned.cs"
    # en_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/en_all/en.col.tab"
    # en_combo_outfname = "/data/m1/whites5/AIDA/M18/eval/results/en_all/en_all.fillers.cs"
    # en_color_outfname = "/data/m1/whites5/AIDA/M18/eval/results/en_all/en_all.color_filler.cs"
    # lang_main(en_fill_cs_fname, en_color_fill_tab_fname, en_combo_outfname, en_color_outfname)

    # ru_fill_cs_fname = "/nas/data/m1/lud2/AIDA/eval/y1/v1/filler/ru_all/filler_cleaned.cs"
    # ru_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/ru_all/ru.col.tab"
    # ru_combo_outfname = "/data/m1/whites5/AIDA/M18/eval/results/ru_all/ru_all.fillers.cs"
    # ru_color_outfname = "/data/m1/whites5/AIDA/M18/eval/results/ru_all/ru_all.color_filler.cs"
    # lang_main(ru_fill_cs_fname, ru_color_fill_tab_fname, ru_combo_outfname, ru_color_outfname)

    # uk_fill_cs_fname = "/nas/data/m1/lud2/AIDA/eval/y1/v1/filler/uk/filler_cleaned.cs"
    # uk_color_fill_tab_fname = "/data/m1/liny9/aida/result/eval_0628/uk/uk.col.tab"
    # uk_combo_outfname = "/data/m1/whites5/AIDA/M18/eval/results/uk/uk.fillers.cs"
    # uk_color_outfname = "/data/m1/whites5/AIDA/M18/eval/results/uk/uk.color_filler.cs"
    # lang_main(uk_fill_cs_fname, uk_color_fill_tab_fname, uk_combo_outfname, uk_color_outfname)
