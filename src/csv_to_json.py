import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Read the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        # Convert each row to a dictionary and add to a list
        data = [row for row in csv_reader]

    # Write the list of dictionaries to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    csv_path = r"C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\dataset\test_jira_data_set.csv"  # Path to your CSV file
    json_path = r"C:\Sanjeev\VNIT_CLASSES\NLP_PROJ\dataset\output.json"            # Output JSON file path
    csv_to_json(csv_path, json_path)
    print(f"CSV data has been converted to JSON and saved to {json_path}")
