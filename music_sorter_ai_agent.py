from phi.agent import Agent
from phi.agent import RunResponse
from phi.model.ollama.chat import Ollama
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from pydantic import BaseModel, Field
from phi.utils.pprint import pprint_run_response


import os
import shutil
import time
import pandas as pd  # Add pandas for table formatting
from tqdm import tqdm  # Add tqdm for progress tracking
import platform  # Add platform to check the OS


# Class for music file language identification results
class MusicFile(BaseModel):
    Language: str = Field(
        ...,
        description="Language of the music file. Should be one of: English, Hindi, Japanese, Other",
    )


# Define the agent
web_agent = Agent(
    name="Web Agent",
    model=Ollama(id="qwen2.5:7b"),
    instructions="Search for the song language on the web and return the details from the web search.",
    tools=[GoogleSearch(), DuckDuckGo()],
    show_tool_calls=True,
)


language_parser_agent = Agent(
    name="Language Parser Agent",
    model=Ollama(id="qwen2.5:7b"),
    instructions="Parse the language from the passed details",
    show_tool_calls=True,
)

song_language_agent = Agent(
    name="Song Language Agent",
    model=Ollama(id="qwen2.5:7b"),
    team=[web_agent, language_parser_agent],
    instructions=[
        "First run the Web Agent to get details about the song and identify the language.",
        "Then run the Language Parser Agent to parse the language from the details as per the given format.",
        "The language should only be out of the following options: English, Hindi, Japanese. If the language is Punjabi, Panjabi or Urdu, then it should be retruned as Hindi and for all other languages which are not in the list, it should be returned as Other.",
        "Return only one language for each song and the language should be in title case.",
    ],
    show_tool_calls=True,
    structured_outputs=True,
    response_model=MusicFile,
)


# Add sound alert function
def sound_alert():
    if platform.system() == "Windows":
        import winsound

        winsound.Beep(1000, 500)  # Frequency 1000 Hz, Duration 500 ms
    else:
        os.system('echo -e "\a"')  # ASCII Bell character


def move_files_from_csv(directory, csv_path="language_identification_results.csv"):
    df = pd.read_csv(csv_path, index_col=0)
    for _, row in tqdm(df.iterrows(), desc="Moving files", total=df.shape[0]):
        fname, lang = row["Filename"], row["Language"]
        language_folder = os.path.join("sorted", lang)
        if not os.path.exists(language_folder):
            os.makedirs(language_folder)
        shutil.move(
            os.path.join(directory, fname), os.path.join(language_folder, fname)
        )


def reverse_file_movement(directory, csv_path="language_identification_results.csv"):
    df = pd.read_csv(csv_path, index_col=0)
    for _, row in tqdm(
        df.iterrows(), desc="Reversing file movement", total=df.shape[0]
    ):
        fname, lang = row["Filename"], row["Language"]
        language_folder = os.path.join("sorted", lang)
        if os.path.exists(os.path.join(language_folder, fname)):
            shutil.move(
                os.path.join(language_folder, fname), os.path.join(directory, fname)
            )


def update_csv(queue, csv_path="language_identification_results.csv"):
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, index_col=0)
        df_new = pd.DataFrame(queue, columns=["Filename", "Language"])
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = pd.DataFrame(queue, columns=["Filename", "Language"])

    df.index += 1  # Start index from 1
    df.to_csv(csv_path, index=True)
    print("\nLanguage identification results updated in CSV file.")


def identify_language(audio_path):
    # Get audio name
    audio_name = os.path.basename(audio_path).split(".")
    audio_name = " ".join(audio_name[: len(audio_name) - 1])
    # Get the language from the agent
    response = song_language_agent.run(audio_name)
    language = response.content.Language
    while language == "":
        response = song_language_agent.run(audio_name)
        language = response.content.Language
    while language not in ["English", "Hindi", "Japanese", "Other"]:
        response = song_language_agent.run(audio_name + f" Language is not {language}")
        language = response.content.Language
    print(f"\nLanguage of {audio_name} is identified as {language}.")
    return language


def sort_music_files(directory, move_files=False, reverse_move=False):
    files = os.listdir(directory)
    queue = []

    def process_file(filename):
        file_path = os.path.join(directory, filename)
        language = identify_language(file_path)
        return (filename, language)

    processed_files = set()
    csv_path = "language_identification_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        processed_files = set(df["Filename"].tolist())

    files_to_process = [f for f in files if f not in processed_files]

    if reverse_move:
        reverse_file_movement(directory)
    elif move_files:
        move_files_from_csv(directory)
    else:
        for i in range(0, len(files_to_process), 4):
            batch_files = files_to_process[i : i + 4]
            for filename in tqdm(batch_files, desc="Processing files", unit="file"):
                queue.append(process_file(filename))
                time.sleep(5)  # Add a delay to avoid rate limiting

            update_csv(queue)
            sound_alert()
            queue.clear()  # Clear the queue for the next batch
            # break  # Remove this line to process all files


if __name__ == "__main__":
    music_directory = "Music"
    if not os.path.exists(music_directory):
        os.makedirs(music_directory)
        print(f"Please add music files to the {music_directory} directory.")
        exit(0)
    move_files_flag = True  # Set this flag to True to move files as per CSV
    reverse_move_flag = False  # Set this flag to True to reverse file movement
    sort_music_files(
        music_directory, move_files=move_files_flag, reverse_move=reverse_move_flag
    )
    print("All files processed.")
    # shutdown the computer
    # os.system("shutdown /s /t 1")  # Uncomment this line to shutdown the computer
