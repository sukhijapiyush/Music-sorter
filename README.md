# Music Sorter

This project sorts music files based on the detected language of the audio. It uses the Whisper model for language detection and organizes the files accordingly.

## Requirements

- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:

    ```sh
    git clone <repository_url>
    cd Music sorter
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Download the Whisper model:

    ```sh
    whisper.download_model("large-v3", download_root="whisper_models")
    ```

## Usage

1. Place your music files in the `Music` directory.

2. Run the script to sort the music files:

    ```sh
    python music_sorter.py
    ```

3. To move files based on the CSV results, set the `move_files_flag` to `True`:

    ```python
    move_files_flag = True
    ```

4. To reverse the file movement, set the `reverse_move_flag` to `True`:

    ```python
    reverse_move_flag = True
    ```

## Features

- Detects the language of audio files using the Whisper model.
- Organizes files into folders based on the detected language.
- Supports reversing the file movement.
- Provides progress tracking with `tqdm`.
- Updates results in a CSV file.

## Agent Approach

The project uses a multi-agent approach to identify the language of the music files:

1. **Web Agent**: Searches for the song language on the web and returns details from the web search.
2. **Language Parser Agent**: Parses the language from the details provided by the Web Agent.
3. **Song Language Agent**: Coordinates the Web Agent and Language Parser Agent to identify the language of the song. The language is returned in title case and is one of the following: English, Hindi, Japanese, or Other.

## Agent Script Instructions

1. Ensure you have the required Python packages installed as mentioned in the Installation section.

2. Place your music files in the `Music` directory.

3. Run the agent script to identify the language of the music files:

    ```sh
    python music_sorter_ai_agent.py
    ```

4. The script will process the files and update the `language_identification_results.csv` file with the identified languages.

5. To move files based on the CSV results, set the `move_files_flag` to `True` in the script:

    ```python
    move_files_flag = True
    ```

6. To reverse the file movement, set the `reverse_move_flag` to `True` in the script:

    ```python
    reverse_move_flag = True
    ```

## Notes

- Ensure you have enough disk space for temporary files and the Whisper model.
- The script uses `torch` for model quantization to optimize performance.

## License

This project is licensed under the MIT License.
