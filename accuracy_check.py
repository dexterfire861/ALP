"""
accuracy_check.py — Evaluate LLM responses against ground-truth data.

Reads model output spreadsheets from the results/ directory and computes an
accuracy score for each question based on its type:

  * **Numerical** (Q6, Q7, Q8): Exact match after regex extraction.
  * **Dates** (Q5): Exact four-digit year match.
  * **Binary** (Q11): Exact Yes/No match.
  * **Categorical** (Q3, Q4, Q10): Levenshtein similarity with penalties for
    extra or missing items.

Results are written back to Excel files with an appended ``Accuracy`` column.
"""

import os
import ast
import Levenshtein
import pandas as pd
import numpy as np
import regex as re


# TODO
# refine extraction of 'yes' and 'no' in q_id 11
    # done
# account for q_id 4 not outputting into a python list
    # e.g. 
        # - BTM
        # - IOU (serving more than 25,000 customers)
        # - Co-op
        # - Municipal
        # - State
    # i dont think i need to
# q_id 3 needs to account for dictionaries not being in python format
    # could be the same case as above, mark as incorrect
# llama models not querying
# keep levenshtein and omission rates seperate
    # present both scores
# error metric between numerical and dates

# unified accuracy across questions

# for q_id 6 check rps if null check ces




def compute_accuracy_on_directory(directory):
    """Walk a directory of result spreadsheets and compute per-row accuracy.

    For each Excel file the function iterates over rows, applies the
    appropriate accuracy metric for the question type, and writes a new
    file prefixed with ``accuracy_check_`` containing an ``Accuracy`` column.

    Args:
        directory: Path to the folder containing model output ``.xlsx`` files.
    """
    def percent_diff(num1, num2):
        """Return the symmetric percentage difference between two numbers."""
        return abs(num1 - num2) / ((num1 + num2) / 2)

    reldir = directory

    output_df = pd.DataFrame()

    for subdir, dirs, files in os.walk(reldir):
        for file in files:
            print(f"checking {file}...")
            if file.endswith('.DS_Store'):
                continue
            output_df = pd.read_excel(reldir + "/" + file)
            accuracies = []
            for i, row in output_df.iterrows():
                question_id = row['Question ID']
                question_type = row['Type']
                model_response = row['Response']
                ground_truth = row['Ground Truth']
                # print(f"question_id: {question_id}")
                # print(f"question_type: {question_type}")
                # print(f"model response: {model_response}")
                # print(type(model_response))
                # print(f"ground truth: {ground_truth}")
                # print(type(ground_truth))
                # print()

                if not pd.notna(ground_truth):
                    ground_truth = "None"
                if not pd.notna(model_response):
                    model_response = "None"

                if question_type == "Numerical": # all or nothing
                    if question_id == 6:
                        model_response = re.search(r"\d*\.\d*:\d*}", model_response)
                        if model_response is None: # no match
                            model_response = "None"
                        else:
                            model_response = model_response.group()
                        model_response = [0,0] if model_response == "None" else model_response.split(":")
                        ground_truth = [0,0] if ground_truth == "None" else ground_truth.split(":")
                        for i in range(2):
                            if model_response[i] == "None":
                                model_response[i] = 0
                            if ground_truth[i] == "None":
                                ground_truth[i] = 0
                        if float(model_response[0]) == float(ground_truth[0]) and int(model_response[1]) == int(ground_truth[1]):
                            # print(1)
                            accuracies.append(1)
                        else:
                            # print(0)
                            accuracies.append(0)

                    elif question_id == 7:
                        model_response = re.search(r"\b\d*\.\d+\b", model_response)
                        if model_response is None: # no match
                            model_response = "None"
                        else:
                            model_response = model_response.group()
                        if model_response == "None":
                            model_response = 0
                        else:
                            model_response = float(model_response)
                        if ground_truth == "None":
                            ground_truth = 0
                        else:
                            ground_truth = float(ground_truth)
                        if model_response == ground_truth:
                            # print(1)
                            accuracies.append(1)
                        else:
                            # print(0)
                            accuracies.append(0)

                    elif question_id == 8:
                        model_response = re.search(r"\b\d*\.\d+\b", model_response)
                        if model_response is None: # no match
                            model_response = "None"
                        else:
                            model_response = model_response.group()
                        if model_response == "None":
                            model_response = 0
                        else:
                            model_response = float(model_response)
                        if ground_truth == "None":
                            ground_truth = 0
                        else:
                            ground_truth = float(ground_truth)
                        if model_response == ground_truth:
                            # print(1)
                            accuracies.append(1)
                        else:
                            accuracies.append(0)
                            # print(0)

                elif question_type == "Dates": # all or nothing
                    # question id 5
                    model_response = re.search(r"\b\d{4}\b", model_response)
                    if model_response is None: # no match
                        model_response = "None"
                    else:
                        model_response = model_response.group()
                    if model_response == ground_truth:
                        # print(1)
                        accuracies.append(1)
                    else:
                        # print(0)
                        accuracies.append(0)

                elif question_type == "Binary": # all or nothing
                    model_response = re.search(r"\b(?:Yes|No)\b", model_response, re.IGNORECASE)
                    if model_response is None: # no match
                        model_response = "None"
                    else:
                        model_response = model_response.group()
                    # question id 11
                    if model_response == ground_truth:
                        # print(1)
                        accuracies.append(1)
                    else:
                        # print(0)
                        accuracies.append(0)

                elif question_type == "Categorical": # using levenshtein distance (fuzzy matching) metric of 0-1 for retreival
                    if question_id == 3:
                        similarities = []
                        ground_truth = ast.literal_eval(ground_truth) # convert string representations to dicts
                        ground_truth = ground_truth[0]
                        model_response = re.search(r"\[(?:[^\[\]]*(?:\[[^\[\]]*\])?)*\]", model_response) #  find the list within the inputted string
                        if model_response is None: # no match
                            model_response = ["None"]
                        else:
                            model_response = ast.literal_eval(model_response.group()) # convert string representations to dicts
                        if pd.notna(model_response) != True:
                            model_response = {}
                        if pd.notna(ground_truth) != True:
                            ground_truth = {}
                        if ground_truth == model_response:
                            accuracies.append(1)
                        if model_response == ["None"] and ground_truth != ["None"]:
                            accuracies.append(0)
                        else:
                            for m_r_key in model_response: # match each key in m_r to g_t, save max similiarity between entries
                                if m_r_key in ground_truth:
                                    if model_response[m_r_key] == ground_truth[m_r_key]:
                                        similarities.append(1)
                                    else:
                                        max_similarity = Levenshtein.ratio(m_r_key, g_t_key)
                                        perc_diff = percent_diff(model_response[m_r_key], ground_truth[g_t_key])
                                        score = max_similarity - perc_diff
                                        similarities.append(score)
                                else:
                                    max_similarity = 0
                                    score = 0
                                    for g_t_key in ground_truth:
                                      ratio = Levenshtein.ratio(m_r_key, g_t_key)
                                      if ratio > max_similarity:
                                          max_similarity = ratio
                                          perc_diff = percent_diff(model_response[m_r_key], ground_truth[g_t_key])
                                          score = max_similarity - (perc_diff * 0.5) # difference in retrieved energy multiplier
                                    similarities.append(score)
                                # print(similarities)
                                avg = np.average(similarities)
                                # print(avg)
                                accuracies.append(avg)

                    elif question_id == 4 or question_id == 10:
                        ground_truth = ast.literal_eval(ground_truth)
                        model_response = re.search(r"\[(?:[^\[\]]*(?:\[[^\[\]]*\])?)*\]", model_response) #  find the list within the inputted string
                        if model_response is None: # no match
                            model_response = ["None"]
                        else:
                            model_response = ast.literal_eval(model_response.group())
                        # print(model_response)
                        ground_truth = [string.lower() for string in ground_truth]
                        model_response = [string.lower() for string in model_response]
                        similarities = []
                        if pd.notna(ground_truth).all() != True:
                            ground_truth = ["none"] # make sure lower case
                        if ground_truth == model_response:
                            accuracies.append(1)
                            # print(1)
                        else:
                            for g_t in ground_truth:
                                if g_t in model_response:
                                    similarities.append(1)
                                    model_response.remove(g_t)
                                else:
                                    max_similarity = 0
                                    best_match = ""
                                    for m_r in model_response:
                                      ratio = Levenshtein.ratio(m_r, g_t)
                                      if ratio > max_similarity:
                                          max_similarity = ratio
                                          best_match = m_r
                                    if best_match != "" and max_similarity > 0.5:
                                        model_response.remove(best_match)
                                        similarities.append(max_similarity)
                                    else:
                                        similarities.append(0)
                            # print(similarities)
                            # print(model_response)
                            # print(ground_truth)
                            avg = np.average(similarities)
                            # print(similarities)
                            addition_penalty = len(model_response) / len(ground_truth)
                            total = avg - addition_penalty # subtract for omission
                            total = 0 if total < 0 else total
                            # print(avg, addition_penalty, total)
                            accuracies.append({"average":avg,"addition_penalty":addition_penalty,"total":total})

                # print()
                # also consider if the answer is within the retrieved text
                # this is a retrieval task and can use top-k accuracy

            output_df['Accuracy'] = accuracies
            with pd.ExcelWriter(reldir + "/" + "accuracy_check_" + file ) as writer: # overwrite file
                output_df.to_excel(writer)
            

compute_accuracy_on_directory("results")

