import pandas as pd

# Define file paths
extracted_data_path = "VA Code § 56-585.2.pdf (gpt-3.5-turbo).xlsx - Categorical.csv"
database_path = "Database June 20 2022.xlsx - master.csv"

# Load the extracted data from the Excel file
extracted_data = pd.read_csv(extracted_data_path)

print(extracted_data['Response'])

database_categorical_1 = "Virginia"
#name of the ref law
database_categorical_2 = "VA Code § 56-585.2"

#legal commitment name
database_categorical_3 = "Renewable Portfolio Standard"

#date enacted
database_categorical_4 = "2007/04/04"

#RPS Commitment
database_categorical_5 = "15:2025"

#list of the different renewable energy sources
database_categorical_6 = "biomass, hydroelectric, solar, ocean tidal wave thermal, wind, geothermal geoelectric, landfill gas, anaerobic digestion"

#credit multiplier description
database_categorical_7 = """Solar:2, 
Wind:2, 
Animal Waste:2,
Offshore Wind:3"""

import spacy



nlp = spacy.load('en_core_web_md')

# Define the function to calculate the accuracy of the extracted data

def calculate_accuracy(data_1, data_2):
    doc1 = nlp(data_1)
    doc2 = nlp(data_2)
    return doc1.similarity(doc2)

# Calculate the accuracy of the extracted data with the database

accuracy_name = calculate_accuracy(extracted_data['Response'][0], database_categorical_1)

print(accuracy_name)

accuracy_ref_law = calculate_accuracy(extracted_data['Response'][1], database_categorical_2)

print(accuracy_ref_law)

accuracy_legal_commitment = calculate_accuracy(extracted_data['Response'][2], database_categorical_3)

print(accuracy_legal_commitment)

accuracy_energy_sources = calculate_accuracy(extracted_data['Response'][3], database_categorical_6)

print(accuracy_energy_sources)

# Mapping of terms to a common standard
term_mapping = {
    "Sunlight": "Solar",
    "Onshore wind": "Wind",
    "Facilities in the Commonwealth fueled primarily by animal waste": "Animal Waste",
    "Offshore wind": "Offshore Wind"
}


def parse_input_string(input_string):
    parsed_dict = {}
    for line in input_string.split("\n"):
        key, value = line.split(":")
        parsed_dict[key.strip()] = value.strip()
    return parsed_dict


parsed_dict = parse_input_string(extracted_data['Response'][4])
print("parsed dict: ", parsed_dict)




# Normalize input list using the term mapping
normalized_input = {term_mapping[key]: value for key, value in parsed_dict.items()}

print(normalized_input)

#convert the dictionary to a string
def dict_to_string(input_dict):
    return ", \n".join([f"{key}:{value}" for key, value in input_dict.items()])


normalized_input_string = dict_to_string(normalized_input)
accuracy_credit_multiplier = calculate_accuracy(normalized_input_string, database_categorical_7)

print(accuracy_credit_multiplier)


