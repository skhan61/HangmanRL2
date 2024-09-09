import json
# Function to read metadata from the JSON file
def read_metadata(metadata_file_path):
    """Read metadata from a JSON file."""
    if metadata_file_path.exists():
        with open(metadata_file_path, 'r') as file:
            metadata = json.load(file)
            return metadata
    else:
        print(f"No metadata file found at {metadata_file_path}")
        return None